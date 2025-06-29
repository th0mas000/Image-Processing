import cv2
import numpy as np
import matplotlib.pyplot as plt

def median_filter(img, ksize):
    return cv2.medianBlur(img, ksize)

# Load the image
filename = 'myCats.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Display the original image
cv2.imshow('Grayscale', img)

# Apply Average Filter
h, w = img.shape
out1 = np.zeros_like(img, dtype=np.uint8)

for i in range(1, h-1):
    for j in range(1, w-1):
        out1.itemset((i, j), np.mean(img[i-1:i+2, j-1:j+2]))

cv2.imshow('Average Filter', out1)

# Apply Weighted Average Filter
mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
out2 = np.zeros_like(img, dtype=np.uint8)

for i in range(1, h-1):
    for j in range(1, w-1):
        out2.itemset((i, j), (mask * img[i-1:i+2, j-1:j+2]).sum())

cv2.imshow('Weighted Average Filter', out2)

median_filtered_3x3 = median_filter(img, 3)
median_filtered_5x5 = median_filter(img, 5)
median_filtered_7x7 = median_filter(img, 7)

cv2.imshow('Median Filter (3x3)', median_filtered_3x3)
cv2.imshow('Median Filter (5x5)', median_filtered_5x5)
cv2.imshow('Median Filter (7x7)', median_filtered_7x7)

cv2.waitKey(0)
cv2.destroyAllWindows()