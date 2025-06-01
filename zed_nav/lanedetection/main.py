import cv2
import numpy as np
from util import Info
from bev_tools import transform_to_bev 
from lanedet_tools import generate_lane_mask

def ensure_3channel_uint8(img):
    """Convert grayscale or BGRA images to 3-channel BGR uint8."""
    if img.ndim == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.astype(np.uint8)

# Open video
cap = cv2.VideoCapture("test_vid.mp4")
if not cap.isOpened():
    raise IOError("Could not open video file.")

# Read one frame to get dimensions
ret, frame = cap.read()
if not ret:
    raise ValueError("Could not read frame from video.")
height, width = frame.shape[:2]

# Define camera parameters
camera_params = {
    "focalLengthX": 2300,
    "focalLengthY": 2930,
    "opticalCenterX": width // 2,
    "opticalCenterY": height // 2,
    "cameraHeight": 2500,
    "pitch": 0,
    "yaw": 0,
    "roll": 0
}
cameraInfo = Info(camera_params)

# Define IPM parameters
ipm_params = {
    "left": 125,
    "right": width - 125,
    "top": 650,
    "bottom": height
}
ipmInfo = Info(ipm_params)

# Define HSV thresholds for lane detection (tune as needed)
lower = np.array([0, 0, 200])
upper = np.array([255, 50, 255])

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Target size for display
target_size = (640, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BEV transformation
    bevImg = transform_to_bev(frame, cameraInfo, ipmInfo)

    # Generate lane masks
    mask, msk = generate_lane_mask(bevImg, lower, upper)

    # Convert all to 3-channel BGR and uint8
    frame_bgr = ensure_3channel_uint8(frame)
    bev_bgr = ensure_3channel_uint8(bevImg)
    mask_bgr = ensure_3channel_uint8(mask)
    msk_bgr = ensure_3channel_uint8(msk)

    # Resize to uniform size
    frame_resized = cv2.resize(frame_bgr, target_size)
    bev_resized = cv2.resize(bev_bgr, target_size)
    mask_resized = cv2.resize(mask_bgr, target_size)
    msk_resized = cv2.resize(msk_bgr, target_size)

    # Create 2x2 grid
    top_row = cv2.hconcat([frame_resized, bev_resized])
    bottom_row = cv2.hconcat([mask_resized, msk_resized])
    grid = cv2.vconcat([top_row, bottom_row])

    # Show grid
    cv2.imshow("Lane Detection Grid", grid)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

