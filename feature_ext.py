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
#img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-depth-SGL-20220218-142518 (2).jpg",1)
#img = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/capture.jpg",1)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# %%

# ------------------------------------------------------
## Watershed Segmentation
ret, thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,255]

cv2.imshow("watershed", img)
k = cv2.waitKey(0)


# %%

# ------------------------------------------------------
## Harris Corner Detection 
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow("dst", img)
k = cv2.waitKey(0)

# %%

# corner with subpixel accuracy

# Finding Harris Corner Detection first

dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret , labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# Draw them

res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:,1], res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imshow("pic", img)
k = cv2.waitKey(0)

#%%
#---------------------------------------------------------
## Shi-Tomasi Corner Detector

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow("pic", img)
k = cv2.waitKey(0)

# %%

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
print("Threshold: ", fast.getInt('threshold'))
print ( "nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print ("neighborhood: ", fast.getInt('type'))
print ("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imshow("pic", img2)
k = cv2.waitKey(0)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imshow("pic", img3)
k = cv2.waitKey(0)

# %%
