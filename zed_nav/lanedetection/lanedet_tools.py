import cv2
import numpy as np

# Function input transformed_frame, th1, th2
# Function output mask, msk

def generate_lane_mask(transformed_frame, th1, th2):
    """
    Generate a binary mask and visual overlay mask from a transformed frame using HSV thresholding and contour tracing.

    Args:
        transformed_frame (np.ndarray): The input BGR image frame.
        th1 (list or np.ndarray): Lower HSV threshold [l_h, l_s, l_v].
        th2 (list or np.ndarray): Upper HSV threshold [u_h, u_s, u_v].

    Returns:
        mask (np.ndarray): Binary mask from HSV thresholding.
        msk (np.ndarray): Visual mask with lane rectangles.
    """
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    lower = np.array(th1)
    upper = np.array(th2)
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        # Left threshold
        img_left = mask[max(y-40,0):y, max(left_base-50,0):left_base+50]
        contours, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        # Right threshold
        img_right = mask[max(y-40,0):y, max(right_base-50,0):right_base+50]
        contours, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    return mask, msk