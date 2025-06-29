
import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_filter(img, mask):
    return cv2.filter2D(img, cv2.CV_64F, mask)

# Load the image
filename = 'myCats.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Display the original image
cv2.imshow('Grayscale', img)

laplacian_mask_negative_4dir = np.array([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]])
laplacian_negative_4dir = laplacian_filter(img, laplacian_mask_negative_4dir)
cv2.imshow('Laplacian (Negative Mask - 4 Directions)', laplacian_negative_4dir)

laplacian_mask_positive_8dir = np.array([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]])
laplacian_positive_8dir = laplacian_filter(img, laplacian_mask_positive_8dir)
cv2.imshow('Laplacian (Positive Mask - 8 Directions)', laplacian_positive_8dir)

cv2.waitKey(0)
cv2.destroyAllWindows()
