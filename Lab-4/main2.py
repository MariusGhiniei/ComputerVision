import os
import glob
import random
import numpy as np
from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt

def createDetector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    elif method == "AKAZE":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Metodă necunoscută: {method}")

def detectKeyPoints(gray, method="SIFT"):
    detector = createDetector(method)
    keypoints = detector.detect(gray, None)
    return keypoints

def computeDescriptors(gray, keypoints, method="SIFT"):
    detector = createDetector(method)
    if keypoints is None or len(keypoints) == 0:
        return None
    _, descriptors = detector.compute(gray, keypoints)
    return descriptors

def getNormMethod(method):
    if method.upper() in ["SIFT", "SURF"]:
        return cv2.NORM_L2
    else:
        return cv2.NORM_HAMMING

def matchDescriptors(des1, des2, method="SIFT", knn=False, k=2, ratio=0.75):
    norm = getNormMethod(method)
    bf = cv2.BFMatcher(norm)

    if des1 is None or des2 is None:
        return [], []

    if knn:
        matches_knn = bf.knnMatch(des1, des2, k=k)
        good = []
        for m, n in matches_knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        return matches_knn, good
    else:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, matches

# == ex 2 ==
datasetPath   = "/Users/marius/PycharmProjects/CV-lab1/Lab-4/IIT_DB_selectie"
trainDirPath  = os.path.join(datasetPath, "train")
testDirPath   = os.path.join(datasetPath, "test")

method      = "SIFT"
matcherType = "knn"  # "bf"


def loadImages(baseDir):
    pathsLabels = []
    for labelDir in sorted(os.listdir(baseDir)):
        fullDir = os.path.join(baseDir, labelDir)
        if not os.path.isdir(fullDir):
            continue

        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
            for imgPath in glob.glob(os.path.join(fullDir, ext)):
                pathsLabels.append((imgPath, labelDir))
    return pathsLabels


trainList = loadImages(trainDirPath)
testList  = loadImages(testDirPath)

print(f"Train images: {len(trainList)}")
print(f"Test images : {len(testList)}")


def getFeatureList(imgList, method="SIFT"):
    data = []
    for path, label in imgList:
        img = cv2.imread(path)
        if img is None:
            print(f" Nu pot citi imaginea: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # keypoints
        kp = detectKeyPoints(gray, method)
        if kp is None or len(kp) == 0:
            print(f" {method} nu a gasit niciun keypoint în {path}")
            des = None
        else:
            # descriptors
            des = computeDescriptors(gray, kp, method)
            if des is None:
                print(f"{method} nu a putut calcula descriptori pentru {path}")

        data.append({
            "path": path,
            "label": label,
            "image": img,
            "gray": gray,
            "keypoints": kp,
            "descriptors": des
        })
    return data


print("\n -> Calculam keypoints + descriptori pentru TRAIN <-")
train_data = getFeatureList(trainList, method)

print("\n -> Calculam keypoints + descriptori pentru TEST <-")
test_data  = getFeatureList(testList, method)


def countGoodMatches(desTest, desTrain, method="SIFT", matcherType="knn"):

    if desTest is None or desTrain is None:
        return 0

    if matcherType == "knn":
        _, good = matchDescriptors(desTest, desTrain, method=method, knn=True, ratio=0.75)
        return len(good)
    elif matcherType == "bf":
        matches, good = matchDescriptors(desTest, desTrain, method=method, knn=False)
        return len(good)
    else:
        raise ValueError("matcherType nu este 'knn' sau 'bf'")


def classify(testItem, trainData, method="SIFT", matcherType="knn"):
    desTest  = testItem["descriptors"]
    pathTest = testItem["path"]

    if desTest is None:
        print(f"Test image fara descriptori: {pathTest}")
        return None, []

    scores = []  # (num_matches, label_train, path_train)
    for tr in trainData:
        desTr = tr["descriptors"]
        if desTr is None:
            continue
        n_matches = countGoodMatches(desTest, desTr, method=method, matcherType=matcherType)
        scores.append((n_matches, tr["label"], tr["path"]))

    if not scores:
        return None, []

    # 2c - max de match uri
    maxMatches = max(s[0] for s in scores)
    best = [s for s in scores if s[0] == maxMatches]

    return best, scores



def decideLabelBest(bestList):

    # bestList = [(num_matches, label, path), ...] toate cu același nr de match-uri maxim
    # - un singur element -> label-ul direct
    # - altfel -> votăm după label; în caz de paritate -> random

    if len(bestList) == 0:
        return None, False, []

    if len(bestList) == 1:
        return bestList[0][1], False, []

    labels = [b[1] for b in bestList]
    cnt = Counter(labels)
    maxVotes = max(cnt.values())
    candidateLabels = [l for l, c in cnt.items() if c == maxVotes]

    if len(candidateLabels) == 1:
        return candidateLabels[0], False, []
    else:
        chosen = random.choice(candidateLabels)
        tiePaths = [b[2] for b in bestList if b[1] in candidateLabels]
        return chosen, True, tiePaths


# 2e – clasificare + acuratețe

yTrue = []
yPred = []
parityCases = []
noKpCases = []
labelsSet = set()

print(f"\n=== CLASIFICARE folosind metoda {method} și matcher {matcherType} ===\n")

for testItem in test_data:
    trueLabel = testItem["label"]
    labelsSet.add(trueLabel)
    pathTest = testItem["path"]

    # folosim train_data, nu trainList
    best, scores = classify(testItem, train_data, method=method, matcherType=matcherType)

    if best is None:
        print(f" {pathTest} NU poate fi clasificat (nu are descriptori). Atribui eticheta random.")
        noKpCases.append(pathTest)
        trainLabels = [tr["label"] for tr in train_data if tr["descriptors"] is not None]
        if not trainLabels:
            continue
        predLabel = random.choice(trainLabels)
    else:
        predLabel, parityFlag, tiePaths = decideLabelBest(best)
        if parityFlag:
            print(f" Pentru imaginea de test {pathTest}")
            print(" Imagini de train implicate în paritate:")
            for p in tiePaths:
                print("     ", p)
            parityCases.append(pathTest)

    yTrue.append(trueLabel)
    yPred.append(predLabel)

correct  = sum(1 for t, p in zip(yTrue, yPred) if t == p)
accuracy = correct / len(yTrue) if yTrue else 0.0
print(f"\nAcuratețe (metoda={method}, matcher={matcherType}): {accuracy * 100:.2f}%")

#confusion matrix
labels = sorted(list(labelsSet))
label_to_idx = {l: i for i, l in enumerate(labels)}
cm = np.zeros((len(labels), len(labels)), dtype=int)

for t, p in zip(yTrue, yPred):
    i = label_to_idx[t]
    j = label_to_idx[p]
    cm[i, j] += 1

print("\nEtichete (ordine în matrice):", labels)
print("Confusion matrix (rows = TRUE, cols = PREDICTED):")
print(cm)

print("\n Imagini de test cu PARITATE la vot (dacă exista):")
for p in parityCases:
    print("  ", p)

print("\n Imagini de test fara keypoints/descriptori (dacă exista):")
for p in noKpCases:
    print("  ", p)
