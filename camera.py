import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)

ret, frame = cap.read()

hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

mask = cv.inRange(hsv, lower_blue, upper_blue)

res = cv.bitwise_and(frame, frame, mask=mask)

# Convert BGR to RGB for Matplotlib
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
res_rgb = cv.cvtColor(res, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(frame_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(res_rgb)
plt.title("Blue Objects")
plt.axis("off")

plt.show()

cap.release()