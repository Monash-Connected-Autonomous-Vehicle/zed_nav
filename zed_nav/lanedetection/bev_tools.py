import cv2
import numpy as np
from .util import GetVanishingPoint, TransformImage2Ground, TransformGround2Image

def mono_cam_coord_bev(frame, input_coord, output_coord, resize=None):
    '''
    Transforms a specified quadrilateral region in an image to a bird's-eye view.

    Parameters:
        frame (ndarray): Input image.
        input_coord (list of tuples): Source points [(tl), (bl), (tr), (br)].
        output_coord (list of tuples): Destination points corresponding to input_coord.
        resize (tuple, optional): Desired output size as (width, height).

    Returns:
        tuple: Original image with marked points, transformed image.
    '''
    if len(input_coord) != 4 or len(output_coord) != 4:
        raise ValueError("Both input_coord and output_coord must contain exactly four points.")
    if not all(len(pt) == 2 for pt in input_coord + output_coord):
        raise ValueError("Each point must be a tuple of two coordinates (x, y).")

    if resize is not None:
        frame = cv2.resize(frame, resize)

    for point in input_coord:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    matrix = cv2.getPerspectiveTransform(np.float32(input_coord), np.float32(output_coord))

    if resize is not None:
        transformed_frame = cv2.warpPerspective(frame, matrix, resize)
    else:
        # Compute size based on output_coord
        width = int(max([pt[0] for pt in output_coord]))
        height = int(max([pt[1] for pt in output_coord]))
        transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))

    return frame, transformed_frame

'''
def transform_to_bev(frame, cameraInfo, ipmInfo):

    vpp = GetVanishingPoint(cameraInfo)
    vp_x = vpp[0][0]
    vp_y = vpp[1][0]
    ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
    uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
                [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)

    xyLimits = TransformImage2Ground(uvLimitsp, cameraInfo)
    row1 = xyLimits[0, :]
    row2 = xyLimits[1, :]
    xfMin = min(row1)
    xfMax = max(row1)
    yfMin = min(row2)
    yfMax = max(row2)
    outImage = np.zeros((640,960,4), np.float32)
    outImage[:,:,3] = 255
    outRow = int(outImage.shape[0])
    outCol = int(outImage.shape[1])
    stepRow = (yfMax - yfMin)/outRow
    stepCol = (xfMax - xfMin)/outCol
    xyGrid = np.zeros((2, outRow*outCol), np.float32)
    y = yfMax-0.5*stepRow

    for i in range(0, outRow):
        x = xfMin+0.5*stepCol
        for j in range(0, outCol):
            xyGrid[0, (i-1)*outCol+j] = x
            xyGrid[1, (i-1)*outCol+j] = y
            x = x + stepCol
        y = y - stepRow

    # TransformGround2Image
    uvGrid = TransformGround2Image(xyGrid, cameraInfo)
    # mean value of the image
    RR = frame.astype(float)/255
    for i in range(0, outRow):
        for j in range(0, outCol):
            ui = uvGrid[0, i*outCol+j]
            vi = uvGrid[1, i*outCol+j]
            #print(ui, vi)
            if ui < ipmInfo.left or ui > ipmInfo.right or vi < ipmInfo.top or vi > ipmInfo.bottom:
                outImage[i, j] = 0.0
            else:
                x1 = np.int32(ui)
                x2 = np.int32(ui+0.5)
                y1 = np.int32(vi)
                y2 = np.int32(vi+0.5)
                x = ui-float(x1)
                y = vi-float(y1)
                outImage[i, j, 0] = float(RR[y1, x1, 0])*(1-x)*(1-y)+float(RR[y1, x2, 0])*x*(1-y)+float(RR[y2, x1, 0])*(1-x)*y+float(RR[y2, x2, 0])*x*y
                outImage[i, j, 1] = float(RR[y1, x1, 1])*(1-x)*(1-y)+float(RR[y1, x2, 1])*x*(1-y)+float(RR[y2, x1, 1])*(1-x)*y+float(RR[y2, x2, 1])*x*y
                outImage[i, j, 2] = float(RR[y1, x1, 2])*(1-x)*(1-y)+float(RR[y1, x2, 2])*x*(1-y)+float(RR[y2, x1, 2])*(1-x)*y+float(RR[y2, x2, 2])*x*y

    outImage[-1,:] = 0.0 
    # show the result

    outImage = outImage * 255

    return outImage
'''

def transform_to_bev(frame, cameraInfo, ipmInfo):
    # Vanishing point
    vpp = GetVanishingPoint(cameraInfo)
    vp_x, vp_y = vpp[0][0], vpp[1][0]
    ipmInfo.top = float(max(int(vp_y), ipmInfo.top))

    # Define BEV output size
    out_h, out_w = 640, 960

    # Compute IPM boundaries
    uvLimits = np.array([
        [vp_x, ipmInfo.right, ipmInfo.left, vp_x],
        [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]
    ], dtype=np.float32)

    # Image → ground
    xyLimits = TransformImage2Ground(uvLimits, cameraInfo)
    x_min, x_max = xyLimits[0].min(), xyLimits[0].max()
    y_min, y_max = xyLimits[1].min(), xyLimits[1].max()

    # Generate BEV xy grid
    x_vals = np.linspace(x_min, x_max, out_w)
    y_vals = np.linspace(y_max, y_min, out_h)  # reversed y
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Ground → image
    uvGrid = TransformGround2Image(xy_grid, cameraInfo)
    map_x = uvGrid[0, :].reshape(out_h, out_w).astype(np.float32)
    map_y = uvGrid[1, :].reshape(out_h, out_w).astype(np.float32)

    # Mask out-of-bounds areas
    mask = (
        (map_x >= ipmInfo.left) & (map_x < ipmInfo.right) &
        (map_y >= ipmInfo.top) & (map_y < ipmInfo.bottom)
    ).astype(np.uint8)

    # Perform remapping
    bev = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = np.full((out_h, out_w, 1), 255, dtype=np.uint8)
    bev_rgba = np.concatenate([bev, alpha], axis=2)
    bev_rgba[mask == 0] = 0  # mask out invalid regions

    return bev_rgba.astype(np.float32)
