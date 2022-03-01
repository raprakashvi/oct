#%%
import cv2
import matplotlib
import numpy as np
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt
import ipympl

# %%
img_1 = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220201_Zach/Results_/BSCAN-SGL-20220201-130343.tif",1)
img_2 = cv2.imread("C:/1. Documents_220111/Duke/1. BTL/Data/220218_Zach/Images_with_less_noise_and_gradient/B-width-SGL-20220218-142926 (6).jpg",0)
# %%
# image shape 
print(img.shape)
h, w, c = img.shape
print("Dimensions of the image is:nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c)
# %%
# Displaying the image

cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)
# if k == 27 or k == ord('q'):
#     cv2.destroyAllWindows()
# %%
# Filterning

kernel1 = np.ones((5,5), np.float32)/25
kernel2 = np.ones((3,3), np.float32)/9
dst1 = cv2.filter2D(img, -1, kernel1)
dst2 = cv2.filter2D(img, -1, kernel2)
plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(dst1),plt.title('Averaging 5*5')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(dst2),plt.title('Averaging 3*3')
plt.xticks([]), plt.yticks([])
plt.show()
# %%

# Gaussian Filtering

blur5 = cv2.GaussianBlur(img,(5,5),0)
blur3 = cv2.GaussianBlur(img,(3,3),0)
# cv2.imshow('Full_Length', img)
# k = cv2.waitKey(0)

# cv2.imshow('Gaussian 3', blur3)
# k = cv2.waitKey(0)

# cv2.imshow('Gaussian 5', blur5)
# k = cv2.waitKey(0)

plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur3),plt.title('G3')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(blur5),plt.title('G5')
plt.xticks([]), plt.yticks([])
plt.show()
# %%


## Median Filter

median3 = cv2.medianBlur(img,3)
median5 = cv2.medianBlur(img,5)

# Comparative view
cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)

cv2.imshow('Median 3', median3)
k = cv2.waitKey(0)

cv2.imshow('Median 5', median5)
k = cv2.waitKey(0)

cv2.imshow('Gaussian 3', blur3)
k = cv2.waitKey(0)

cv2.imshow('Gaussian 5', blur5)
k = cv2.waitKey(0)
  # %%

# Bilateral Filtering (img, d, sigmaColor, sigmaSpace)
# Apply bilateral filter with d = 9,
# sigmaColor = sigmaSpace = 75.
bi1 = cv2.bilateralFilter(img,5,75,75)
bi2 = cv2.bilateralFilter(img,9,75,75)
bi3 = cv2.bilateralFilter(img,5,50,50)
bi4 = cv2.bilateralFilter(img,9,100,100)

# Comparative view
cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)

cv2.imshow('Bilateral d=5', bi1)
k = cv2.waitKey(0)

cv2.imshow('Bilateral d=9', bi2)
k = cv2.waitKey(0)

cv2.imshow(' d=5', bi3)
k = cv2.waitKey(0)

cv2.imshow('d=9', bi4)
k = cv2.waitKey(0)

# cv2.imshow('Gaussian 3', blur3)
# k = cv2.waitKey(0)

# cv2.imshow('Gaussian 5', blur5)
# k = cv2.waitKey(0)



# %%
##Morphology

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)

cv2.imshow('Closing', closing)
k = cv2.waitKey(0)

cv2.imshow('Opening', opening)
k = cv2.waitKey(0)

cv2.imshow('Gradient', gradient)
k = cv2.waitKey(0)
 # %%


## Image Gradient

# Applying Gaussian Blur
# img = cv2.GaussianBlur(img, (3, 3), 0)

# These images are in higher forms.
laplacian = cv2.Laplacian(img,cv2.CV_64F)
abs_lap64f = np.absolute(laplacian)
lap_8u = np.uint8(abs_lap64f)

# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# abs_sobelx64f = np.absolute(sobelx)
# sobel_8ux = np.uint8(abs_sobelx64f)


# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# abs_sobely64f = np.absolute(sobely)
# sobel_8uy = np.uint8(abs_sobely64f)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_

cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)

cv2.imshow('Laplacian', laplacian)
k = cv2.waitKey(0)

cv2.imshow('SobelX', sobel_8ux)
k = cv2.waitKey(0)

cv2.imshow('SobelY', sobel_8uy)
k = cv2.waitKey(0)
# %%

## Canny Edge Detection 
edges = cv2.Canny(img,100,200)

cv2.imshow('Full_Length', img)
k = cv2.waitKey(0)

cv2.imshow('Canny', edges)
k = cv2.waitKey(0)
# %%

## Histogram
## cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histcv = cv2.calcHist([img],[0],None,[256],[0,256])
histnp1,bins = np.histogram(img.ravel(),256,[0,256])
histnp2 = np.bincount(img.ravel(),minlength=256)

plt.hist(img.ravel(),256,[0,256]); plt.show()
# %%

## Histogram Equalization numpy

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.figure("Original Histogram")
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])

cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()
plt.figure("Equalized Histogram")
plt.plot(cdf_normalized2, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cv2.imshow('Full_Image', img)
k = cv2.waitKey(0)

cv2.imshow('Equalized_Image', img2)
k = cv2.waitKey(0)
# %%

# ## Histogram Equalization opencv
img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow('Equalized_Image', equ)
k = cv2.waitKey(0)
# %%
## Adaptive Histogram Equalization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
cl1 = clahe.apply(img_2)

cv2.imshow('Full_Image', img_2)
k = cv2.waitKey(0)
cv2.imshow('Equalized_Image Grid Size 16*16', cl1)
k = cv2.waitKey(0)
 # %%


hist = cv2.calcHist([img_2], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist,interpolation = 'nearest')
plt.show()

 # %%

## Hugh Transform
## Debatable results
edges_1 = cv2.Canny (img, 50,150,apertureSize=3)
lines = cv2.HoughLines(edges_1,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('Hugh', img)
k = cv2.waitKey(0)



# %%

## Probabilistic Hough Transform
edges_2 = cv2.Canny (img_2,100,200,apertureSize=3)
cedge = np.copy(img_2)
minLineLength = 100
maxLineGap = 10
lineP = cv2.HoughLinesP(edges_2, 1,np.pi/180,10,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines_1[0]:
#     cv2.line(cdstP,(x1,y1),(x2,y2),(0,255,0),3,cv2.LINE_AA)

#Draw the line
if lineP is not None:
  for i in range(0, len(lineP)):
    l = lineP[i][0]
    cv2.line(cedge, (l[0], l[1]), (l[2], l[3]), (0,0,255),4)
#cv2.imshow("Source", img_2)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cedge)
k = cv2.waitKey(0)
# %%
