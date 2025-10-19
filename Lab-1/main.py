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
blur15 = cv2.blur(img, (15,15))

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
plt.imshow(cv2.cvtColor(blur15, cv2.COLOR_BGR2RGB))
plt.title("Blur (15x15)")
plt.axis("off")

plt.show()

blurGauss3 = cv2.GaussianBlur(img, (3,3), 0.1)
blurGauss11 = cv2.GaussianBlur(img, (11,11), 4)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(blurGauss3, cv2.COLOR_BGR2RGB))
plt.title("Gaussian blur (3x3) and sigma = 0.1")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(blurGauss11, cv2.COLOR_BGR2RGB))
plt.title("Gaussian blur (11x11) and sigma = 4")
plt.axis("off")

plt.show()

blurMedian3 = cv2.medianBlur(img, 3)
blurMedian15 = cv2.medianBlur(img,15)

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
plt.imshow(cv2.cvtColor(blurMedian15, cv2.COLOR_BGR2RGB))
plt.title("Median blur (15)")
plt.axis("off")

plt.show()

blurBilateral1 = cv2.bilateralFilter(img, 3, 50, 50)
blurBilateral2 = cv2.bilateralFilter(img, 15, 100, 100)

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
plt.imshow(cv2.cvtColor(blurBilateral2, cv2.COLOR_BGR2RGB))
plt.title("Bilateral blur (21, 100, 100)")
plt.axis("off")

plt.show()

kernelSharp1 = cv2.filter2D(img, ddepth = -1 , kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]))
kernelSharp2 = cv2.filter2D(img, ddepth = -1, kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(kernelSharp1, cv2.COLOR_BGR2RGB))
plt.title("Sharpen 1")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(kernelSharp2, cv2.COLOR_BGR2RGB))
plt.title("Sharpen 2")
plt.axis("off")

plt.show()


#ex 4) Apply the w matrix filter
a = 1
b = -1
w = np.array( [[0,a ,0], [-2, 9, b], [0,-2,0]])

wFilterImg = cv2.filter2D(img, ddepth= -1, kernel = w)
cv2.imshow("w matrix filter", wFilterImg)
cv2.waitKey(0)

#ex 5) rotate

rotate90c = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotate90cc = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate180c = cv2.rotate(img, cv2.ROTATE_180)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(rotate90c, cv2.COLOR_BGR2RGB))
plt.title("Rotate by 90 clockwise")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(rotate90cc, cv2.COLOR_BGR2RGB))
plt.title("Rotate by 90 counterclockwise ")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(rotate180c, cv2.COLOR_BGR2RGB))
plt.title("Rotate by 180")
plt.axis("off")

plt.show()

# rotate by any angle
def rotateByAngle(img, angle : float):
    h, w = img.shape[:2]
    rotationMatrix = cv2.getRotationMatrix2D((w // 2, h // 2),angle,1.0)

    return cv2.warpAffine(img, rotationMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

angle = 23.5
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(rotateByAngle(img,angle), cv2.COLOR_BGR2RGB))
plt.title(f"Rotate by {angle}°")
plt.axis("off")

plt.show()


#ex 6 crop function

def cropImage(img, x : int, y : int, width : int, height : int):
    h,w = img.shape[:2]

    xEnd : int = min(x + width, w)
    yEnd : int = min(y + height, h)

    return img[y:yEnd, x:xEnd]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(imgRGB)
plt.title("The original photo")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(cropImage(img, 0,50, 300, 300), cv2.COLOR_BGR2RGB))
plt.title(f"Cropped image")
plt.axis("off")

plt.show()


