# Code Taken: https://github.com/JamesLiao714/IPM-master/blob/master/TransformImage2Ground.py

import numpy as np
from math import cos, sin, pi

def TransformGround2Image(xyGrid,cameraInfo):
    inPoints2 = xyGrid[0:2]
    inPointsr3 = np.ones((1,len(xyGrid[1])))*(-cameraInfo.cameraHeight)
    inPoints3 = np.concatenate((inPoints2, inPointsr3), axis=0)
    c1 = cos(cameraInfo.pitch*pi/180)
    s1 = sin(cameraInfo.pitch*pi/180)
    c2 = cos(cameraInfo.yaw*pi/180)
    s2 = sin(cameraInfo.yaw*pi/180)

    matp = [[cameraInfo.focalLengthX * c2 + c1*s2* cameraInfo.opticalCenterX,
        -cameraInfo.focalLengthX * s2 + c1*c2* cameraInfo.opticalCenterX,
        -s1 * cameraInfo.opticalCenterX],
        [s2 * (-cameraInfo.focalLengthY * s1 + c1* cameraInfo.opticalCenterY),
        c2 * (-cameraInfo.focalLengthY * s1 + c1* cameraInfo.opticalCenterY),
        -cameraInfo.focalLengthY * c1 - s1* cameraInfo.opticalCenterY],
        [c1*s2, c1*c2, -s1]]
    inPoints3 = np.array(matp).dot(np.array(inPoints3))
    inPointsr3 = inPoints3[2,:]
    div = inPointsr3
    inPoints3[0,:] = inPoints3[0,:]/div
    inPoints3[1,:] = inPoints3[1,:]/div
    inPoints2 = inPoints3[0:2,:]
    uvGrid = inPoints2
    return uvGrid

def TransformImage2Ground(uvLimits, cameraInfo):
    row, col = uvLimits.shape[0:2]
    inPoints4 = np.zeros((row + 2, col), np.float32)
    inPoints4[0:2] = uvLimits
    inPoints4[2] =[1,1,1,1]
    inPoints3 = np.array(inPoints4)[0:3,:]
    
    c1 = cos(cameraInfo.pitch*pi/180)
    s1 = sin(cameraInfo.pitch*pi/180)
    c2 = cos(cameraInfo.yaw*pi/180)
    s2 = sin(cameraInfo.yaw*pi/180)

    matp= [
        [-cameraInfo.cameraHeight*c2/cameraInfo.focalLengthX,
        cameraInfo.cameraHeight*s1*s2/cameraInfo.focalLengthY,
        (cameraInfo.cameraHeight*c2*cameraInfo.opticalCenterX/cameraInfo.focalLengthX)
        -(cameraInfo.cameraHeight *s1*s2* cameraInfo.opticalCenterY/ cameraInfo.focalLengthY)
        -cameraInfo.cameraHeight *c1*s2],
        [cameraInfo.cameraHeight *s2/cameraInfo.focalLengthX,
        cameraInfo.cameraHeight *s1*c2/cameraInfo.focalLengthY,
        (-cameraInfo.cameraHeight *s2* cameraInfo.opticalCenterX
            /cameraInfo.focalLengthX)-(cameraInfo.cameraHeight *s1*c2*
            cameraInfo.opticalCenterY /cameraInfo.focalLengthY) - 
            cameraInfo.cameraHeight *c1*c2],
        [0, cameraInfo.cameraHeight *c1/cameraInfo.focalLengthY, (-cameraInfo.cameraHeight *c1* cameraInfo.opticalCenterY/cameraInfo.focalLengthY)+cameraInfo.cameraHeight*s1],
        [0, -c1 /cameraInfo.focalLengthY,(c1* cameraInfo.opticalCenterY /cameraInfo.focalLengthY) - s1]]

    inPoints4 = np.array(matp).dot(np.array(inPoints3))
    inPointsr4 = inPoints4[3,:]
    div = inPointsr4
    inPoints4[0,:] = inPoints4[0,:]/div
    inPoints4[1,:] = inPoints4[1,:]/div
    inPoints2 = inPoints4[0:2,:]
    xyLimits = inPoints2
    return xyLimits

def GetVanishingPoint(cameraInfo):
    vpp = [[sin(cameraInfo.yaw*pi/180)/cos(cameraInfo.pitch*pi/180)],
        [cos(cameraInfo.yaw*pi/180)/cos(cameraInfo.pitch*pi/180)],
        [0]]

    tyawp = [[cos(cameraInfo.yaw*pi/180), -sin(cameraInfo.yaw*pi/180), 0],
            [sin(cameraInfo.yaw*pi/180), cos(cameraInfo.yaw*pi/180), 0],
            [0, 0, 1]]
			
    tpitchp = [[1, 0, 0],
            [0, -sin(cameraInfo.pitch*pi/180), -cos(cameraInfo.pitch*pi/180)],
            [0, cos(cameraInfo.pitch*pi/180), -sin(cameraInfo.pitch*pi/180)]]

    t1p = [[cameraInfo.focalLengthX, 0, cameraInfo.opticalCenterX],
		  [0, cameraInfo.focalLengthY, cameraInfo.opticalCenterY],
		  [0, 0, 1]]

    transform = np.array(tyawp).dot(np.array(tpitchp))
    transform = np.array(t1p).dot(transform)
    vp = transform.dot(np.array(vpp))
    return vp

#function [ cameraInfo, ipmInfo ] = GetInfo
# camera Info
class CameraInfo():
    def __init__(self):
        # focal length
        self.focalLengthX=600
        self.focalLengthY=600
        # optical center
        self.opticalCenterX=638.1608
        self.opticalCenterY=738.8648
        # height of the camera in mm
        self.cameraHeight=1879.8
        # 393.7 + 1786.1
        # pitch of the camera
        self.pitch=15.5
        # yaw of the camera
        self.yaw=0.0
        # imag width and height
        self.imageWidth=1280
        self.imageHeight=1024

# ipmInfo
# settings for stop line perceptor
class IpmInfo:
    def __init__(self):
        #128
        self.ipmWidth = 640
        #160#320#160 
        #96
        self.ipmHeight = 480
        #120#240#120
        self.ipmLeft = 256
        #80 #90 #115 #140 #50 #85 #100 #85
        self.ipmRight = 1024
        #500 #530 #500 #590 #550
        self.ipmTop = 500
        #220 #200 #50
        self.ipmBottom = 1000
        #360 #350 #380
        #0 bilinear, 1: NN
        self.ipmInterpolation = 0
        self.ipmVpPortion = 0
        #.09 #0.06 #.05 #.125 #.2 #.15 #.075#0.1 #.05

def GetInfo():
    return CameraInfo(), IpmInfo()

class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]