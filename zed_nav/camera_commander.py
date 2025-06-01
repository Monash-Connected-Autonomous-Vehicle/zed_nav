#!/usr/bin/env python3
"""
camera_commander.py
-------------------
ZED-2i → BEV lane detection + depth → 2-D OccupancyGrid
and discrete nav commands.

Publishes
    /nav_command      std_msgs/String   {"stop","forward","turn left","turn right","back"}
    /local_costmap    nav_msgs/OccupancyGrid
"""

# ---------- std / ROS ----------
import math, time, numpy as np, cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg       import String
from sensor_msgs.msg    import Image, CameraInfo
from geometry_msgs.msg  import Pose, Point, Quaternion
from nav_msgs.msg       import Odometry, OccupancyGrid, MapMetaData
from cv_bridge          import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_matrix
import tf2_ros, image_geometry

# ---------- lane-detector helpers ----------
from .lanedetection.bev_tools     import transform_to_bev
from .lanedetection.lanedet_tools import generate_lane_mask
from .lanedetection.util          import Info   # dict→dot access


class CameraCommander(Node):
    # ===== knob block =====
    GOAL          = (2.0, 0.0)     # m (odom)
    YAW_TOL       = math.radians(20)
    DEPTH_STOP    = 0.60           # m
    GRID_RES      = 0.05           # m / cell
    GRID_XY       = 10.0           # grid width [m]
    LOOP_RATE     = 10.0           # Hz (behaviour)
    MAP_RATE      = 2.0            # Hz (grid publish)
    LANE_RATE     = 4.0            # Hz (heavy BEV)
    PIX_STRIDE    = 8              # downsample depth
    # ======================

    def __init__(self):
        super().__init__("camera_commander")

        # ------- ROS params (overridable) ------
        self.declare_parameters(
            '',
            [
                ("focal_x",      2300), ("focal_y",      2930),
                ("opt_center_x", 640),  ("opt_center_y", 360),
                ("cam_height",   2500), ("pitch", 0.0), ("yaw", 0.0), ("roll", 0.0),
                ("ipm_left", 125), ("ipm_right", 1155), ("ipm_top", 650), ("ipm_bottom", 720),
                ("lane_h_low", 0), ("lane_s_low", 0), ("lane_v_low", 200),
                ("lane_h_high", 255), ("lane_s_high", 50), ("lane_v_high", 255),
            ]
        )
        g = lambda k: self.get_parameter(k).value
        self.cameraInfo = Info(dict(
            focalLengthX=g("focal_x"), focalLengthY=g("focal_y"),
            opticalCenterX=g("opt_center_x"), opticalCenterY=g("opt_center_y"),
            cameraHeight=g("cam_height"), pitch=g("pitch"), yaw=g("yaw"), roll=g("roll"),
        ))
        self.ipmInfo = Info(dict(
            left=g("ipm_left"), right=g("ipm_right"),
            top=g("ipm_top"),  bottom=g("ipm_bottom"),
        ))
        self.lane_lower = np.array([g("lane_h_low"), g("lane_s_low"), g("lane_v_low")])
        self.lane_upper = np.array([g("lane_h_high"), g("lane_s_high"), g("lane_v_high")])

        # ------- pubs / subs -------
        self.bridge = CvBridge()
        qos = rclpy.qos.qos_profile_sensor_data
        self.create_subscription(Image,  '/zed/zed_node/left/image_rect_color', self.rgb_cb, qos)
        self.create_subscription(Image,  '/zed/zed_node/depth/depth_registered', self.depth_cb, qos)
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.odom_cb, 10)
        self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.cinfo_cb, 10)
        self.pub_cmd = self.create_publisher(String, '/nav_command', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/local_costmap', 1)

        # ------- state -------
        self.rgb = self.depth = None
        self.x = self.y = self.yaw = 0.0
        self.cam_ready = False
        self.prev_cmd = ""
        self.last_map_t = self.last_lane_t = 0.0
        self.lane_bias_cache = 0.0
        self._lane_idx = None            # (rows, cols) for lane obstacles

        # ------- TF + camera model -------
        self.cam_model = image_geometry.PinholeCameraModel()
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf, self, spin_thread=True)

        self.create_timer(1.0 / self.LOOP_RATE, self.loop)

    # ---------- callbacks ----------
    def cinfo_cb(self, m):
        if not self.cam_ready:
            self.cam_model.fromCameraInfo(m)
            self.cam_ready = True

    def rgb_cb(self, m):   self.rgb   = self.bridge.imgmsg_to_cv2(m, 'bgr8')
    def depth_cb(self, m): self.depth = self.bridge.imgmsg_to_cv2(m, '32FC1')

    def odom_cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ---------- main loop ----------
    def loop(self):
        if self.rgb is None or self.depth is None or not self.cam_ready:
            return
        now = self.get_clock().now().seconds_nanoseconds()[0]

        if now - self.last_map_t > 1.0 / self.MAP_RATE:
            self.publish_costmap()
            self.last_map_t = now

        if now - self.last_lane_t > 1.0 / self.LANE_RATE:
            self.lane_bias_cache = self.update_lane_cache(self.rgb)
            self.last_lane_t = now

        if self.is_blocked_ahead():
            self.emit("stop")
            return

        if abs(self.lane_bias_cache) > 0.35:
            self.emit("turn left" if self.lane_bias_cache < 0 else "turn right")
            return

        gx, gy = self.GOAL
        dist = math.hypot(gx - self.x, gy - self.y)
        if dist < 0.15:
            self.emit("stop")
            return
        heading = math.atan2(gy - self.y, gx - self.x)
        err = self.wrap(heading - self.yaw)
        if err > self.YAW_TOL:
            self.emit("turn left")
        elif err < -self.YAW_TOL:
            self.emit("turn right")
        else:
            self.emit("forward")

    # ---------- lane detector ----------
    def update_lane_cache(self, bgr):
        try:
            bev = transform_to_bev(bgr, self.cameraInfo, self.ipmInfo)
        except Exception as e:
            self.get_logger().warn(f"BEV transform failed: {e}")
            self._lane_idx = None
            return 0.0

        if bev.ndim == 3 and bev.shape[2] == 4:
            bev = cv2.cvtColor(bev.astype(np.uint8), cv2.COLOR_BGRA2BGR)

        mask, _ = generate_lane_mask(bev, self.lane_lower, self.lane_upper)
        if mask is None or mask.sum() < 5000:
            self._lane_idx = None
            return 0.0

        # centroid → steering bias
        m = cv2.moments(mask)
        cx = m['m10'] / m['m00'] if m['m00'] else bev.shape[1] / 2
        bias = (cx - bev.shape[1] / 2) / (bev.shape[1] / 2)

        # -------- rasterise lane pixels to grid indices --------
        ys, xs = np.where(mask > 0)
        # In BEV 1 pixel ≈ GRID_RES m; +X right, +Y forward
        x_cam = (xs - bev.shape[1] / 2) * self.GRID_RES
        y_cam = ys * self.GRID_RES
        pts = np.stack([x_cam, np.zeros_like(x_cam), y_cam, np.ones_like(x_cam)], 1)

        try:
            tfm = self.tf_buf.lookup_transform('odom', 'zed_left_camera_optical_frame',
                                               rclpy.time.Time())
            T = quaternion_matrix([tfm.transform.rotation.x,
                                   tfm.transform.rotation.y,
                                   tfm.transform.rotation.z,
                                   tfm.transform.rotation.w])
            T[:3, 3] = [tfm.transform.translation.x,
                        tfm.transform.translation.y,
                        tfm.transform.translation.z]
            pts_odom = (T @ pts.T).T[:, :2]
            half = self.GRID_XY / 2
            msk = (np.abs(pts_odom[:, 0]) < half) & (np.abs(pts_odom[:, 1]) < half)
            rows = ((pts_odom[msk, 1] + half) / self.GRID_RES).astype(np.int32)
            cols = ((pts_odom[msk, 0] + half) / self.GRID_RES).astype(np.int32)
            self._lane_idx = (rows, cols)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self._lane_idx = None

        return float(bias)

    # ---------- cost-map ----------
    def publish_costmap(self):
        try:
            tfm = self.tf_buf.lookup_transform('odom', 'zed_left_camera_optical_frame',
                                               rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return
        T = quaternion_matrix([tfm.transform.rotation.x,
                               tfm.transform.rotation.y,
                               tfm.transform.rotation.z,
                               tfm.transform.rotation.w])
        T[:3, 3] = [tfm.transform.translation.x,
                    tfm.transform.translation.y,
                    tfm.transform.translation.z]

        depth = self.depth
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(0, w, self.PIX_STRIDE),
                           np.arange(0, h, self.PIX_STRIDE))
        z = depth[v, u]
        ok = (z > 0.20) & (z < self.GRID_XY / 2) & np.isfinite(z)
        if not ok.any() and self._lane_idx is None:
            return
        z, u, v = z[ok], u[ok].astype(np.float32), v[ok].astype(np.float32)
        x_opt = (u - w / 2) * z / self.cam_model.fx()
        y_opt = (v - h / 2) * z / self.cam_model.fy()
        pts_odom = (T @ np.stack([x_opt, y_opt, z, np.ones_like(z)], 1).T).T[:, :2]

        cells = int(self.GRID_XY / self.GRID_RES)
        grid = np.full((cells, cells), -1, np.int8)
        half = self.GRID_XY / 2
        mask = (np.abs(pts_odom[:, 0]) < half) & (np.abs(pts_odom[:, 1]) < half)
        rows = ((pts_odom[mask, 1] + half) / self.GRID_RES).astype(np.int32)
        cols = ((pts_odom[mask, 0] + half) / self.GRID_RES).astype(np.int32)
        grid[rows, cols] = 100                       # depth obstacles

        # ---- lane → obstacles ----
        if self._lane_idx is not None:
            r, c = self._lane_idx
            grid[r, c] = 100

        # ---- publish ----
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'odom'
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        meta = MapMetaData()
        meta.resolution = self.GRID_RES
        meta.width = meta.height = cells
        meta.origin = Pose(position=Point(x=-half, y=-half, z=0.0),
                           orientation=Quaternion(w=1.0))
        grid_msg.info = meta
        grid_msg.data = grid.flatten().tolist()
        self.pub_map.publish(grid_msg)

    # ---------- helpers ----------
    def is_blocked_ahead(self):
        h, w = self.depth.shape
        roi = self.depth[h//2:h//2+40, w//2-40:w//2+40]
        finite = np.isfinite(roi) & (roi > 0.01)
        return finite.any() and roi[finite].min() < self.DEPTH_STOP

    @staticmethod
    def wrap(a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def emit(self, cmd):
        if cmd != self.prev_cmd:
            self.pub_cmd.publish(String(data=cmd))
            self.prev_cmd = cmd


def main():
    rclpy.init()
    rclpy.spin(CameraCommander())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
