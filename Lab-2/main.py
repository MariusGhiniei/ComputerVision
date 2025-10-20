import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("bkg.jpg", cv2.IMREAD_COLOR)

grayScale1 = ((img1[:,:,0] + img1[:,:,1] + img1[:,:,2])/3).astype("uint8")
#better to divide each, no overflow
grayScale2 = (img1[:,:,0] / 3 + img1[:,:,1] / 3 + img1[:,:,2] / 3).astype("uint8")

plt.figure("Arithmetic gray scale",figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(grayScale1, cmap = "gray")
plt.title("Gray = (R+G+B)/3")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(grayScale2, cmap = "gray")
plt.title("Gray = R/3 + G/3 + B/3")
plt.axis("off")

plt.show()

#ex 2

B, G, R = img1[:,:,0], img1[:,:,1], img1[:,:,2]

gray1 = 0.3 * R + 0.59 * G + 0.11 * B
gray2 = 0.2126 * R + 0.7152 * G + 0.0722 * B
gray3 = 0.299 * R + 0.587 * G + 0.114 * B

plt.figure("Ratio gray scale", figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray1, cmap = "gray")
plt.title("0.3R + 0.59G + 0.11B")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(gray2, cmap = "gray")
plt.title("0.2126R + 0.7152G + 0.0722B")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(gray3, cmap = "gray")
plt.title("0.299R + 0.587G + 0.114B")
plt.axis("off")

plt.show()

#ex3 Desaturation
## Gray = (min(R,G,B) + max(R,G,B))/2

grayDesaturation = (np.minimum(B, np.minimum(G,R)) + np.maximum(B,np.maximum(G,R)))/2

plt.figure("Desaturation", figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(grayDesaturation, cmap = "gray")
plt.title("Gray desaturation")
plt.axis("off")

plt.show()

#ex-4 Decomposition

# max gray = max (R,G,B)
# min gray = min (R,G,B)

minGray = np.minimum(B, np.minimum(G,R))
maxGray = np.maximum(B, np.maximum(G,R))


plt.figure("Decomposition", figsize=(8,4))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(minGray, cmap = "gray")
plt.title("Minimum gray - dark")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(maxGray, cmap = "gray")
plt.title("Maximum gray - light")
plt.axis("off")

plt.show()

#ex 5 - Single colour channel

grayBlue, grayGreen, grayRed = B, G, R

plt.figure("Single colour channel", figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(grayBlue, cmap = "gray")
plt.title("Blue channel - dark")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(grayGreen, cmap = "gray")
plt.title("Green channel")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(grayRed, cmap = "gray")
plt.title("Red channel - light")
plt.axis("off")
plt.show()

#ex 6 - Custom number of grey shades

image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

p = 8
length = 256 // p

a = np.arange(0, 256, length)
res = np.zeros_like(image)

for i in range(len(a)-1):
    mask = (image >= a[i]) & (image < a[i+1])
    res[mask] = int((a[i]+a[i+1]) / 2)

plt.figure("Shades of gray", figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(res, cmap="gray")
plt.title(f"Gray of {p} shades")
plt.axis("off")

plt.show()

# ex 7

# Floyd-Steinberg algorithm
h, w = image.shape
cimage = image.astype(float).copy()

for y in range(h-1):
    for x in range(1, w-1):
        old = cimage[y,x]
        if old < 128:
            new = 0
        else: new = 255

        cimage[y, x] = new
        err = old - new

        cimage[y, x + 1] = cimage[y, x + 1] + err * 7 / 16
        cimage[y + 1, x - 1] =  cimage[y + 1, x - 1] + err * 3 / 16
        cimage[y + 1, x] = cimage[y + 1, x] + err * 5 / 16
        cimage[y + 1, x + 1] = cimage[y + 1, x + 1] + err * 1 / 16

imgFloyd = np.clip(cimage, 0, 255).astype("uint8")

#Burkes Dithering algorithm

h, w = image.shape
cimage = image.astype(float).copy()

for y in range(h-1):
    for x in range(2, w-2):
        old = cimage[y, x]
        if old < 128:
            new = 0
        else: new = 255

        cimage[y, x] = new
        err = old - new

        #row y
        cimage[y, x + 1] = cimage[y, x + 1] + err * 8 / 32
        cimage[y, x + 2] = cimage[y, x + 2] + err * 4 / 32

        #row y + 1

        cimage[y + 1, x - 2] = cimage[y + 1, x - 2] + err * 2 / 32
        cimage[y + 1, x - 1] = cimage[y + 1, x - 1] + err * 4 / 32
        cimage[y + 1, x] = cimage[y + 1, x] + err * 8 / 32
        cimage[y + 1, x + 1] = cimage[y + 1, x + 1] + err * 4 / 32
        cimage[y + 1, x + 2] = cimage[y + 1, x +2] + err * 2 / 32

imgBurkes = np.clip(cimage, 0, 255).astype("uint8")

plt.figure("Error-diffusuin dithering", figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(imgFloyd, cmap="gray")
plt.title("Floyd–Steinberg algorithm ")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(imgBurkes, cmap="gray")
plt.title("Burkes Dithering algorithm")
plt.axis("off")

plt.tight_layout()
plt.show()


