import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("grayMan.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
colored1 = np.zeros((h,w,3), dtype = np.uint8)
colored2 = np.zeros((h,w,3), dtype = np.uint8)
colored3 = np.zeros((h, w, 3), dtype=np.uint8)

colored1[:,:,0] = (img * 0.3).astype(np.uint8)
colored1[:,:,1] =  (img * 0.5).astype(np.uint8)
colored1[:,:,2] = (img * 1.0).astype(np.uint8)

colored2[:,:,0] = 255 - img
colored2[:,:,1] = img
colored2[:,:,2] = img // 2 + 80

colored3[:, :, 0] = (255 - img // 2)
colored3[:, :, 1] = (img // 1.5).astype(np.uint8)
colored3[:, :, 2] = (img // 2 + 80).astype(np.uint8)

plt.figure("grayToRGB", figsize = (8,8))
plt.subplot(2,2,1)
plt.imshow(img, cmap = "gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(colored1, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(colored2, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(colored3, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
