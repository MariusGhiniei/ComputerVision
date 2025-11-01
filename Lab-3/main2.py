import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


## TO DO: pentru fiecare dintre cele 2 foldere cu imagini(face and family) trebuie sa calculez maska de pixeli folosind
# fiecare metoda RGB, HSV, YCbCr si le compar cu mask - urile din Ground_Truth apoi sa fac matriciele TP,FN,FP,TN si
# la final accuratetea.

def rgbPixelMethod(R, G, B):
    R = int(R)
    G = int(G)
    B = int(B)
    return R > 95 and G > 40 and B > 20 and max(R,G,B) - min(R,G,B) > 15 and abs(R - G) > 15 and R > G and R > B

def hsvPixelMethod(R,G,B):
    pixel = np.uint8([[[R,G,B]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[0,0]

    H = int(H) * 2
    S = int(S) / 255.0
    V = int(V) / 255.0

    return H >= 0 and H <= 50 and S >= 0.23 and S <= 0.68 and V >= 0.35 and V <= 1

def yCbCrPixelMethod(R,G,B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return Y > 80 and Cb > 85 and Cb < 135 and Cr > 135 and Cr < 180

dataset = "/Users/marius/PycharmProjects/CV-lab1/Lab-3/Face_Dataset"

# Face photo
facePhotoDataset = os.path.join(dataset,"Pratheepan_Dataset/FacePhoto")
familyPhotoDataset = os.path.join(dataset, "Pratheepan_Dataset/FamilyPhoto")

facePhotoDatasetMasks = "/Users/marius/PycharmProjects/CV-lab1/Lab-3/Face_Dataset/Pratheepan_Dataset/facePhotoMasks"
familyPhotoDatasetMasks = "/Users/marius/PycharmProjects/CV-lab1/Lab-3/Face_Dataset/Pratheepan_Dataset/familyPhotoMasks"

for image in os.listdir(facePhotoDataset):
    if image is None:
        raise ValueError("Cannot read image: " + image)
    elif image.endswith(".jpg") or image.endswith(".jpeg"):
        imagePath = os.path.join(facePhotoDataset, image)

        # RGB - method
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                r,g,b = img[y,x]
                if rgbPixelMethod(r,g,b):
                    mask[y,x] = 255
                else: mask[y,x] = 0

        name = os.path.splitext(image)[0] + "_maskRGB.png"
        outputPath = os.path.join(facePhotoDatasetMasks, name)
        cv2.imwrite(outputPath, mask)

        # HSV - method

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                r, g, b = img[y, x]
                if hsvPixelMethod(r, g, b):
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0

        name = os.path.splitext(image)[0] + "_maskHSV.png"
        outputPath = os.path.join(facePhotoDatasetMasks, name)
        cv2.imwrite(outputPath, mask)

        #YCbCr - method

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                r, g, b = img[y, x]
                if yCbCrPixelMethod(r, g, b):
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0

        name = os.path.splitext(image)[0] + "_maskYCbCr.png"
        outputPath = os.path.join(facePhotoDatasetMasks, name)
        cv2.imwrite(outputPath, mask)




