import os
import cv2
import numpy as np


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
facePhotoDatasetMasks = os.path.join(dataset, "Pratheepan_Dataset/facePhotoMasks")

# Family photo
familyPhotoDataset = os.path.join(dataset, "Pratheepan_Dataset/FamilyPhoto")
familyPhotoDatasetMasks = os.path.join(dataset, "Pratheepan_Dataset/familyPhotoMasks")

# Ground Truth
groundTruthFaceDataset = os.path.join(dataset, "Ground_Truth/GroundT_FacePhoto")
groundTruthFamilyDataset = os.path.join(dataset, "Ground_Truth/GroundT_FamilyPhoto" )


def getMasks(dirPhoto, dirMasks):
    for image in os.listdir(dirPhoto):
        if image is None:
            raise ValueError("Cannot read the image: " + image)
        elif image.endswith(".jpg") or image.endswith(".jpeg"):
            imagePath = os.path.join(dirPhoto, image)

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
            outputPath = os.path.join(dirMasks, name)
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
            outputPath = os.path.join(dirMasks, name)
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
            outputPath = os.path.join(dirMasks, name)
            cv2.imwrite(outputPath, mask)
## 4:30 run time for both
# calculate the masks for face
# getMasks(facePhotoDataset, facePhotoDatasetMasks)
# calculate the masks for family
# getMasks(familyPhotoDataset, familyPhotoDatasetMasks)

def evaluateMask(predMask, gtMask):
    pred = (predMask >= 128).astype(np.uint8)
    gt = (gtMask >= 128).astype(np.uint8)

    TP = np.sum((gt == 1) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-14)
    return TP, FN, FP, TN, acc

def compareMask(predDir, gtDir, method, accPerImage):
    predAcc = []
    TP = FN = FP = TN = 0

    methodExt = method + ".png"
    for image in os.listdir(predDir):
        if not image.endswith(methodExt):
            continue

        predImagePath = os.path.join(predDir, image)
        predImage = cv2.imread(predImagePath, cv2.IMREAD_GRAYSCALE)

        if predImage is None:
            print("Cannot read the image:", predImagePath)
            continue

        #pozaNume_maskRGB.png -> pozaNume
        gtName = image[:-(len(method) + 5)]
        gtPath = os.path.join(gtDir, gtName + ".png")

        if not os.path.exists(gtPath):
            print("Path incorect", gtPath)
            continue

        gtImage = cv2.imread(gtPath, cv2.IMREAD_GRAYSCALE)
        if gtImage is None:
            print("Cannot read the image:", gtPath);
            continue

        tp, fn, fp, tn, acc = evaluateMask(predImage, gtImage)
        TP = TP + tp
        FN = FN + fn
        FP = FP + fp
        TN = TN + tn
        predAcc.append(acc)
        accPerImage.setdefault(gtName, []).append(acc)
        #print(f"{gtName}: ACC={acc:.5f}")

    meanAcc = float(np.mean(predAcc))

    print(f"\n=== {method} SUMMARY ===")
    print(f"Mean ACC = {meanAcc:.4f}")

    return predAcc


results = {}
print("Evaluate on RGB method: ")
compareMask(facePhotoDatasetMasks, groundTruthFaceDataset, "maskRGB", results)

print("Evaluate on HSV method: ")
compareMask(facePhotoDatasetMasks, groundTruthFaceDataset, "maskHSV", results)

print("Evalate on YCbCr method:")
compareMask(facePhotoDatasetMasks, groundTruthFaceDataset, "maskYCbCr", results)

def printWorstBest(data, results):
    print(f"\n--- {data} summary ---\n")
    methods = ["RGB", "HSV", "YCbCr"]

    for name, accs in sorted(results.items()):

        temp = []
        for acc in accs:
            if acc is not None:
                temp.append(acc)
            else:
                temp.append(np.nan)

        values = np.array(temp, dtype=float)

        bestIndex = np.nanargmax(values)
        worstIndex = np.nanargmin(values)

        bestAcc = values[bestIndex]
        worstAcc = values[worstIndex]

        print(f"{name}: worst = {worstAcc:.4f} ({methods[worstIndex]}) | "
              f"best = {bestAcc:.4f} ({methods[bestIndex]})")

printWorstBest("facePhoto", results)

# --- family ----

results = {}
print("Evaluate on RGB method: ")
compareMask(familyPhotoDatasetMasks, groundTruthFamilyDataset, "maskRGB", results)

print("Evaluate on HSV method: ")
compareMask(familyPhotoDatasetMasks, groundTruthFamilyDataset, "maskHSV", results)

print("Evalate on YCbCr method:")
compareMask(familyPhotoDatasetMasks, groundTruthFamilyDataset, "maskYCbCr", results)

printWorstBest("familyPhoto", results)