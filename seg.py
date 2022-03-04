#%%
import cv2
import matplotlib
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt
import ipympl

#%%

#img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-width-SGL-20220218-142926 (5).jpg",0)
#img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-depth-SGL-20220218-142518 (2).jpg",1)
img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220302/220302/20220302-154133/uniform-075.jpg",0)

#convert to gray
#img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# %%
# slicing images

img_top = img[0:256 , 0:512]
cv2.imshow("top img", img_top)
cv2.waitKey(0)

img_bot = img[256:512 , 0:512]
# cv2.imshow("bottom img", img_bot)
# cv2.waitKey(0)

# %%
# Joining images
new_img = np.concatenate((img_top, img_bot), axis=0) # axis = 1 for horizontal joining
# cv2.imshow("new img", new_img)
# cv2.waitKey(0)

# %%
