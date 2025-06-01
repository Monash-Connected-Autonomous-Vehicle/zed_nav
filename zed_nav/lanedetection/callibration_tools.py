import cv2
from bev_tools import transform_to_bev  # Your BEV transformation function

class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]

# Load image
I = cv2.imread('input/frame0.jpg')
if I is None:
    raise FileNotFoundError("Image 'frame0.jpg' not found.")

height, width = I.shape[:2]

# Default values
camera_params = {
    "focalLengthX": 3000,
    "focalLengthY": 4000,
    "opticalCenterX": width // 2,
    "opticalCenterY": height // 2,
    "cameraHeight": 1500,
    "pitch": 4,
    "yaw": 0,
    "roll": 0
}

ipm_params = {
    "left": 300,
    "right": width - 300,
    "top": 2300,
    "bottom": height
}

# Create window
cv2.namedWindow("BEV Tuner", cv2.WINDOW_NORMAL)

# Trackbar callback (dummy)
def nothing(x): pass

# Camera sliders
cv2.createTrackbar("FocalX", "BEV Tuner", camera_params["focalLengthX"], 8000, nothing)
cv2.createTrackbar("FocalY", "BEV Tuner", camera_params["focalLengthY"], 8000, nothing)
cv2.createTrackbar("Pitch", "BEV Tuner", -90, 90, nothing)
cv2.createTrackbar("Height", "BEV Tuner", 0, 5000, nothing)

# IPM sliders
cv2.createTrackbar("Width_Thresh", "BEV Tuner", 0, width, nothing)
cv2.createTrackbar("Height_Thresh", "BEV Tuner", 0, height, nothing)

while True:
    # Read slider values
    focalX = cv2.getTrackbarPos("FocalX", "BEV Tuner")
    focalY = cv2.getTrackbarPos("FocalY", "BEV Tuner")
    pitch = cv2.getTrackbarPos("Pitch", "BEV Tuner")
    cam_height = cv2.getTrackbarPos("Height", "BEV Tuner")
    Width_Thresh = cv2.getTrackbarPos("Width_Thresh", "BEV Tuner")
    Height_Thresh = cv2.getTrackbarPos("Height_Thresh", "BEV Tuner")

    cameraInfo = Info({
        "focalLengthX": focalX,
        "focalLengthY": focalY,
        "opticalCenterX": width // 2,
        "opticalCenterY": height // 2,
        "cameraHeight": cam_height,
        "pitch": pitch,
        "yaw": 0,
        "roll": 0
    })

    ipmInfo = Info({
        "inputWidth": width,
        "inputHeight": height,
        "left": Width_Thresh,
        "right": width-Width_Thresh,
        "top": Height_Thresh,
        "bottom": height,
    })

    # BEV transform
    outImage = transform_to_bev(I, cameraInfo, ipmInfo)
    outImage = outImage/255
    if outImage is not None:
        cv2.imshow("BEV Tuner", outImage)

    key = cv2.waitKey(100) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
