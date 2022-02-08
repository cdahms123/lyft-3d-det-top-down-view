# temp.py

import numpy as np
import cv2

image = np.zeros((400, 400, 3), np.uint8)

image[200][200] = (270, 270, 270)

print(np.min(image))
print(np.max(image))

cv2.imshow('image', image)
cv2.waitKey()

