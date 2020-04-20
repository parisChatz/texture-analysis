from matplotlib import pyplot as plt
import numpy as np
import cv2

path = '/home/paris/PycharmProjects/Texture-Analysis/docu/ImgPIA.jpg'
# Using cv2.imread() method
image = cv2.imread(path)

# Displaying the image
cv2.imshow('image', image)
# Maintain output window utill user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
