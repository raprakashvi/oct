#%%

import cv2
import matplotlib
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt
import ipympl


#%%

img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-width-SGL-20220218-142926 (5).jpg",0)
img_1 = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-width-SGL-20220218-142926 (5).jpg",1)

# applying bilateral filter

bi1 = cv2.bilateralFilter(img,5,75,75)
# cv2.imshow("Bilaternal Filter", bi1)
# k = cv2.waitKey(0)

## Adaptive Histogram Equalization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
cl1 = clahe.apply(bi1)




# #Canny edge detection 
# edges_2 = cv2.Canny (cl1,50,200,apertureSize=3)
# cedge = np.copy(img_1)

# # cv2.imshow("Canny Edge", edges_2)
# # k = cv2.waitKey(0)


# # P.Hough Transform
# minLineLength = 250
# maxLineGap = 10
# lineP = cv2.HoughLinesP(edges_2, 1,np.pi/180,10,minLineLength,maxLineGap)

# if lineP is not None:
#   for i in range(0, len(lineP)):
#     l = lineP[i][0]
#     cv2.line(cedge, (l[0], l[1]), (l[2], l[3]), (255,0,0),4)
# # plt.imshow( cedge)
# # plt.show()
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cedge)
# k = cv2.waitKey(0)

edges_2 = cv2.Canny (cl1,75,150,apertureSize=3)
cedge = np.copy(img_1)
lineP = cv2.HoughLinesP(edges_2, 1,np.pi/180,10,minLineLength=100,maxLineGap=5)

if lineP is not None:
  for i in range(0, len(lineP)):
    l = lineP[i][0]
    cv2.line(cedge, (l[0], l[1]), (l[2], l[3]), (255,0,0),4)
plt.imshow( cedge)
plt.show()
# %%