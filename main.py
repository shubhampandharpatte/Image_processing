import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Reading and displaying image
img=cv.imread('car.jpg')
cv.imshow('car', img)

# Creating duplicate
duplicate=cv.imwrite('Duplicate_img.png',img)
cv.imshow('Duplicate img', duplicate)

# read info about image
img=cv.imread('car.jpg')
print(img.shape)  # ( height,width,depth) in pixels

# GRAYSCALE IMAGE
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('GRAY SCALE IMAGE', gray)

# average blur
average=cv.blur(img, (7, 7))
cv.imshow('blur',average)

# gaussian blur
gauss=cv.GaussianBlur(img, (7, 7), 0)
cv.imshow('GaussianBlur',gauss)

# median blur
median=cv.medianBlur(img, 3)
cv.imshow('medianBlur', median)

# Bilateral filter
bilateral=cv.bilateralFilter(img, 10, 35, 35)
cv.imshow('bilateral', bilateral)

# edge detection
canny=cv.Canny(img,  125, 125)
cv.imshow('canny edge', canny)

# dilating the image
dilate=cv.dilate(canny, (4, 4), iterations=1)
cv.imshow('dilate img', dilate)

# BINARY IMAGE CONVERSION  (high Contrast image)
ret, binary=cv.threshold(img, 120, 255, cv.THRESH_BINARY)
# CV2.THRESHOLD(SRC,THRESH,MAX_VALUE,CONVERSION TYPE)
cv.imshow('BINARY IMAGE', binary)

# scaling down to 75%
img1=cv.resize(img, None, fx=0.75, fy=0.75)
cv.imshow('SCALED DOWN IMAGE', img1)

# scaling upto 150%
img2=cv.resize(img, None, fx=1.5, fy=1.5)
cv.imshow('SCALED UP IMAGE', img2)

# scaling using custom dimensions
img3=cv.resize(img, (1000, 500))
cv.imshow('CUSTOM DIMENSIONS', img3)

# masked image:

# circular masked image:
circle=np.zeros(img.shape[:2], dtype='uint8')     # this creates image with blank background screen
cv.imshow('Blank', circle)
# Create a circular mask
mask_circle=cv.circle(circle,(img.shape[1]//2,img.shape[0]//2),90,255,-1)
cv.imshow('mask', mask_circle)
# create masked-circular image and using bitwise operator
masked_circle=cv.bitwise_and(img, img, mask=mask_circle)
cv.imshow('masked', masked_circle)

#  rectangular masked image:
blank=np.zeros(img.shape[:2], dtype='uint8')
# create a circular mask
mask_rectangle=cv.rectangle(blank,  (30, 30), (550, 400), 255, -1)
cv.imshow('mask', mask_rectangle)
# create masked-circular image and using bitwise operator
masked_rectangle=cv.bitwise_and(img, img, mask=mask_rectangle)
cv.imshow('masked', masked_rectangle)

# Histogram
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('GRAY', gray)
gray_hist=cv.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title('Histogram')
plt.xlabel('bins')
plt.ylabel('pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

# simple thresholding
ret, thresh=cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
cv.imshow('THRESH_BINARY',thresh)
ret, thresh1=cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
cv.imshow('THRESH_BINARY_INV', thresh1)

# Otsu thresholding
ret,thresh2=cv.threshold(gray,120,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('Otsu thresholding', thresh2)


cv.waitKey(0)
cv.destroyAllWindows()












