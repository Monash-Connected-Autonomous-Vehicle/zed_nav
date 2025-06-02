#!/usr/bin/env python3
"""
camera_commander.py
-------------------
ZED-2i → BEV lane detection + LIDAR obstacles → 2-D OccupancyGrid
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
from sensor_msgs.msg    import Image, CameraInfo, PointCloud2
from geometry_msgs.msg  import Pose, Point, Quaternion
from nav_msgs.msg       import Odometry, OccupancyGrid, MapMetaData
from cv_bridge          import CvBridge
from tf_transformations import euler_from_quaternion, quaternion_matrix
import tf2_ros, image_geometry
import sensor_msgs_py.point_cloud2 as pc2

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
    # ======================

    def __init__(self):
        super().__init__("camera_commander")

        # ------- ROS params (overridable) ------
        self.declare_parameters(
            '',
            [
                ("focal_x",      954.76), ("focal_y",      955.065),
                ("opt_center_x", 640//2),  ("opt_center_y", 360//2),
                ("cam_height",   48.3), ("pitch", 0.0), ("yaw", 0.0), ("roll", 0.0),
                ("ipm_left", 100), ("ipm_right", 640-100), ("ipm_top", 260), ("ipm_bottom", 360),
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
        self.create_subscription(PointCloud2,  '/velodyne_voxel_filtered', self.lidar_cb, qos)
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.odom_cb, 10)
        self.create_subscription(CameraInfo, '/zed/zed_node/left/camera_info', self.cinfo_cb, 10)
        self.pub_cmd = self.create_publisher(String, '/nav_command', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/local_costmap', 1)

        # ------- state -------
        self.rgb = None
        self.lidar_pc = None  # Stores latest Lidar point cloud
        self.x = self.y = self.yaw = 0.0
        self.cam_ready = False
        self.prev_cmd = ""
        self.last_map_t = self.last_lane_t = 0.0
        self.lane_bias_cache = 0.0
        self._lane_pts_odom = None  # Store lane points as absolute positions

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
    def lidar_cb(self, m): self.lidar_pc = m  # Store Lidar point cloud

    def odom_cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ---------- main loop ----------
    def loop(self):
        if self.rgb is None or not self.cam_ready or self.lidar_pc is None:
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

    # ---------- obstacle detection (LIDAR) ----------
    def is_blocked_ahead(self):
        """Check if obstacles exist in front using Lidar"""
        if self.lidar_pc is None:
            return False

        try:
            # Get transform from Lidar to base_link
            tf = self.tf_buf.lookup_transform(
                'base_link', 
                self.lidar_pc.header.frame_id, 
                rclpy.time.Time()
            )
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException):
            return False

        # Build transform matrix
        T = quaternion_matrix([
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ])
        T[:3, 3] = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ]

        # Process points
        for p in pc2.read_points(self.lidar_pc, field_names=("x", "y", "z"), skip_nans=True):
            p_hom = np.array([p[0], p[1], p[2], 1.0])
            p_base = T @ p_hom
            
            # Check if in collision box (front of robot)
            if (0 <= p_base[0] <= self.DEPTH_STOP and 
                -0.25 <= p_base[1] <= 0.25 and 
                -0.1 <= p_base[2] <= 0.5):
                return True
                
        return False

    # ---------- lane detector ----------
    def update_lane_cache(self, bgr):
        try:
            bev = transform_to_bev(bgr, self.cameraInfo, self.ipmInfo)
        except Exception as e:
            self.get_logger().warn(f"BEV transform failed: {e}")
            self._lane_pts_odom = None
            return 0.0

        if bev.ndim == 3 and bev.shape[2] == 4:
            bev = cv2.cvtColor(bev.astype(np.uint8), cv2.COLOR_BGRA2BGR)

        mask, _ = generate_lane_mask(bev, self.lane_lower, self.lane_upper)
        if mask is None or mask.sum() < 5000:
            self._lane_pts_odom = None
            return 0.0

        # centroid → steering bias
        m = cv2.moments(mask)
        cx = m['m10'] / m['m00'] if m['m00'] else bev.shape[1] / 2
        bias = (cx - bev.shape[1] / 2) / (bev.shape[1] / 2)

        # -------- store lane points in odom frame --------
        ys, xs = np.where(mask > 0)
        x_cam = (xs - bev.shape[1] / 2) * self.GRID_RES
        y_cam = ys * self.GRID_RES
        pts = np.stack([x_cam, np.zeros_like(x_cam), y_cam, np.ones_like(x_cam)], 1)

        try:
            tfm = self.tf_buf.lookup_transform('odom', 'zed_left_camera_optical_frame',
                                               rclpy.time.Time())
            T = quaternion_matrix([
                tfm.transform.rotation.x,
                tfm.transform.rotation.y,
                tfm.transform.rotation.z,
                tfm.transform.rotation.w
            ])
            T[:3, 3] = [
                tfm.transform.translation.x,
                tfm.transform.translation.y,
                tfm.transform.translation.z
            ]
            pts_odom = (T @ pts.T).T[:, :2]  # Transform to odom
            self._lane_pts_odom = pts_odom  # Store absolute positions
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self._lane_pts_odom = None

        return float(bias)
    
    def publish_costmap(self):
        half = self.GRID_XY / 2
        cells = int(self.GRID_XY / self.GRID_RES)
        
        # Initialize grid as free space (0)
        grid = np.zeros((cells, cells), dtype=np.int8)
        
        # Grid origin (bottom-left corner in odom)
        origin_x = self.x - half
        origin_y = self.y - half

        # ------ Create obstacle mask ------
        obstacle_mask = np.zeros((cells, cells), dtype=np.uint8)

        # ------ LIDAR obstacles ------
        if self.lidar_pc:
            try:
                # Get transform from Lidar to odom
                tf_odom = self.tf_buf.lookup_transform(
                    'odom', 
                    self.lidar_pc.header.frame_id, 
                    rclpy.time.Time()
                )
            except (tf2_ros.LookupException, 
                    tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException):
                tf_odom = None

            if tf_odom:
                # Build transform matrix
                T_odom = quaternion_matrix([
                    tf_odom.transform.rotation.x,
                    tf_odom.transform.rotation.y,
                    tf_odom.transform.rotation.z,
                    tf_odom.transform.rotation.w
                ])
                T_odom[:3, 3] = [
                    tf_odom.transform.translation.x,
                    tf_odom.transform.translation.y,
                    tf_odom.transform.translation.z
                ]

                # Process points and expand to 3x3 blocks
                points = []
                for p in pc2.read_points(self.lidar_pc, field_names=("x", "y", "z"), skip_nans=True):
                    p_hom = np.array([p[0], p[1], p[2], 1.0])
                    p_odom = T_odom @ p_hom
                    points.append(p_odom[:3])  # Store x,y,z

                if points:
                    points = np.array(points)
                    x_odom = points[:, 0]
                    y_odom = points[:, 1]
                    
                    # Convert to grid coordinates
                    x_rel = x_odom - origin_x
                    y_rel = y_odom - origin_y
                    cols = (x_rel / self.GRID_RES).astype(int)
                    rows = (y_rel / self.GRID_RES).astype(int)
                    
                    # Expand each point to 3x3 grid cells
                    offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),  (0, 0),  (0, 1),
                            (1, -1),  (1, 0),  (1, 1)]
                    
                    all_rows = []
                    all_cols = []
                    
                    for r, c in zip(rows, cols):
                        for dr, dc in offsets:
                            rr = r + dr
                            cc = c + dc
                            all_rows.append(rr)
                            all_cols.append(cc)
                    
                    # Convert to arrays for vectorized operations
                    all_rows = np.array(all_rows)
                    all_cols = np.array(all_cols)
                    
                    # Filter valid points
                    valid = (all_rows >= 0) & (all_rows < cells) & \
                            (all_cols >= 0) & (all_cols < cells)
                    
                    # Update grid with obstacles
                    grid[all_rows[valid], all_cols[valid]] = 100
                    obstacle_mask[all_rows[valid], all_cols[valid]] = 1
        # ------ Lane obstacles ------
        if self._lane_pts_odom is not None:
            x_odom = self._lane_pts_odom[:, 0]
            y_odom = self._lane_pts_odom[:, 1]
            
            # Convert to grid coordinates
            x_rel = x_odom - origin_x
            y_rel = y_odom - origin_y
            cols = (x_rel / self.GRID_RES).astype(int)
            rows = (y_rel / self.GRID_RES).astype(int)
            
            # Filter valid points
            valid = (rows >= 0) & (rows < cells) & (cols >= 0) & (cols < cells)
            grid[rows[valid], cols[valid]] = 100
            obstacle_mask[rows[valid], cols[valid]] = 1


        # ------ Apply inflation using distance transform ------
        if np.any(obstacle_mask):
            # Compute distance to nearest obstacle
            dist_transform = cv2.distanceTransform(
                src=1 - obstacle_mask,  # Invert: obstacles become 0, free becomes 1
                distanceType=cv2.DIST_L2,
                maskSize=3
            )
            
            # Normalize distances (0-1 range)
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                normalized_dist = dist_transform / max_dist
            else:
                normalized_dist = dist_transform
                
            # Inflation parameters
            inflation_radius = 0.5  # meters
            inflation_cells = int(inflation_radius / self.GRID_RES)
            max_cost = 100
            
            # Create cost grid based on distance
            for i in range(cells):
                for j in range(cells):
                    if obstacle_mask[i, j] == 1:
                        # Obstacle cell - maximum cost
                        grid[i, j] = max_cost
                    else:
                        # Calculate distance in grid cells
                        dist = dist_transform[i, j]
                        
                        if dist < inflation_cells:
                            # Calculate cost based on proximity to obstacle
                            # Quadratic decay: cost = max_cost * (1 - (d/d_max)^2)
                            decay_factor = 1.0 - (dist / inflation_cells)**2
                            cost = int(max_cost * decay_factor)
                            grid[i, j] = max(grid[i, j], cost)

        # ---- publish ----
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'odom'
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        meta = MapMetaData()
        meta.resolution = self.GRID_RES
        meta.width = meta.height = cells
        meta.origin = Pose(
            position=Point(x=origin_x, y=origin_y, z=0.0),
            orientation=Quaternion(w=1.0)
        )
        grid_msg.info = meta
        grid_msg.data = grid.flatten().tolist()
        self.pub_map.publish(grid_msg)

    # ---------- helpers ----------
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