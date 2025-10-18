import cv2
import matplotlib.pyplot as plt
import numpy as np

# ex 2) open, display its size, plot the image
img = cv2.imread("lena.tif", cv2.IMREAD_COLOR)
if img is None:
    print("Couldn't read the image ")
# using GUI on cv2
cv2.imshow("Lena", img)
cv2.waitKey(0)

#using plot
# need rgb for plot
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.title("Lena")
plt.axis("off")
plt.show()

# ex 3) apply blur/sharpen filters

blur3 = cv2.blur(img, (3,3))
blur8 = cv2.blur(img, (5,5))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blur3, cv2.COLOR_BGR2RGB))
plt.title("Blur (3x3)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blur8, cv2.COLOR_BGR2RGB))
plt.title("Blur (5x5)")
plt.axis("off")

plt.show()

blurGauss3 = cv2.GaussianBlur(img, (3,3), 0.1)
blurGauss9 = cv2.GaussianBlur(img, (5,5), 1)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blurGauss3, cv2.COLOR_BGR2RGB))
plt.title("Gaussian blur (3x3)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blurGauss9, cv2.COLOR_BGR2RGB))
plt.title("Gaussian blur (5x5)")
plt.axis("off")

plt.show()

blurMedian3 = cv2.medianBlur(img, 3)
blurMedian5 = cv2.medianBlur(img,21)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blurMedian3, cv2.COLOR_BGR2RGB))
plt.title("Median blur (3)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blurMedian5, cv2.COLOR_BGR2RGB))
plt.title("Median blur (5)")
plt.axis("off")

plt.show()

blurBilateral1 = cv2.bilateralFilter(img, 3, 50, 50)
blurBilateral2 = cv2.bilateralFilter(img, 5, 100, 100)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blurBilateral1, cv2.COLOR_BGR2RGB))
plt.title("Bilateral blur (3, 50, 50)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blurBilateral1, cv2.COLOR_BGR2RGB))
plt.title("Bilateral blur (5, 100, 100)")
plt.axis("off")

plt.show()

#ex 4) Apply the w matrix filter
a = -2
b = 5
w = np.array( [[0,a ,0], [-2, 9, b], [0,-2,0]])

wFilterImg = cv2.filter2D(img, ddepth= -1, kernel = w)
cv2.imshow("w matrix filter", wFilterImg)
cv2.waitKey(0)






