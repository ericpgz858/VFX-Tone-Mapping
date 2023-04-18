import cv2
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

"""
add this from Adaptive_Logarithmic_Mapping import * 
call Adaptive_Logarithmic_Mapping(HDR_image, b = 0.85, L_dmax = 100, gamma_correction_type = 1)
HDR_image : 需要做 tone mapping 的 image
b, L_dmax : hyper parameter
gamma_correction_type : 用哪個方式修正圖片 default = 1 (type 1 比較亮, type 2 比較暗)
"""
def gamma_correction(pixel, type):
    if type == 1:
        if pixel <= 0.018 : 
            return 4.5*pixel
        else:
            return 1.099*pixel**0.8-0.099
    else :
        if pixel <= 0.0031308:
            return pixel * 12.92
        else :
            return 1.055 * (pixel**1/2.4) - 0.055
    
def Adaptive_Logarithmic_Mapping(HDR_image, b = 0.85, L_dmax = 100, gamma_correction_type = 1):
    width, height, _ = HDR_image.shape
    CIE_image = cv2.cvtColor(HDR_image, cv2.COLOR_BGR2XYZ)
    #CIE_image = HDR_image
    L_avg = np.sum(CIE_image[:,:,1])/(width*height)
    _, L_wmax, _, _ = cv2.minMaxLoc(CIE_image[:,:,1])

    L_wmax /= L_avg
    multiplier = L_dmax*0.01/math.log(L_wmax+1, 10)
    bias = math.log(b)/math.log(0.5)

    LDR_image = HDR_image.copy()
    for i in range(width):
        for j in range(height):
            Y_w = CIE_image[i][j][1]/L_avg
            L_d = multiplier * math.log(Y_w + 1) * math.log(2 + ((Y_w/L_wmax)**bias) * 8) 
            scale = L_d/CIE_image[i][j][1]
            LDR_image[i][j][0] = gamma_correction(scale*CIE_image[i][j][0], gamma_correction_type)
            LDR_image[i][j][1] = gamma_correction(scale*CIE_image[i][j][1], gamma_correction_type)
            LDR_image[i][j][2] = gamma_correction(scale*CIE_image[i][j][2], gamma_correction_type)

    LDR_image = cv2.cvtColor(LDR_image, cv2.COLOR_XYZ2BGR)
    return LDR_image


