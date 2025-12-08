import os
import glob
import random
import numpy as np
from collections import Counter
import cv2
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

globalResults = []

def createDetector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    elif method == "AKAZE":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Metoda necunoscuta: {method}")


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

def matchDescriptors(des1, des2, method="SIFT", knn=False, k=2):
    norm = getNormMethod(method)

    if des1 is None or des2 is None:
        return [], []

    if knn:
        method_up = method.upper()
        if method_up in ["SIFT", "SURF"]:
            ratio = 0.75
        else:
            # descriptorii binari (ORB/AKAZE)
            ratio = 0.6

        bf = cv2.BFMatcher(norm)
        matches_knn = bf.knnMatch(des1, des2, k=k)
        good = []
        for m, n in matches_knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        return matches_knn, good
    else:
        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, matches

datasetPath   = "/Users/marius/PycharmProjects/CV-lab1/Lab-4/IIT_DB_selectie"
trainDirPath  = os.path.join(datasetPath, "train")
testDirPath   = os.path.join(datasetPath, "test")

outputRoot    = os.path.join(datasetPath, "output_results_v2")
os.makedirs(outputRoot, exist_ok=True)

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
            print(f"Nu pot citi imaginea: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp = detectKeyPoints(gray, method)
        if kp is None or len(kp) == 0:
            print(f"{method} nu a găsit niciun keypoint în {path}")
            des = None
        else:
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

def countGoodMatches(desTest, desTrain, method="SIFT", matcherType="knn"):
    if desTest is None or desTrain is None:
        return 0

    if matcherType == "knn":
        _, good = matchDescriptors(desTest, desTrain, method=method, knn=True)
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
        print(f"[INFO] Test image fără descriptori: {pathTest}")
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

    maxMatches = max(s[0] for s in scores)
    best = [s for s in scores if s[0] == maxMatches]

    return best, scores


def decideLabelBest(bestList):
    """
    bestList = [(num_matches, label, path), ...]
    """
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
        chosen   = random.choice(candidateLabels)
        tiePaths = [b[2] for b in bestList if b[1] in candidateLabels]
        return chosen, True, tiePaths

def save_matches_figure(testItem, scores, trainImagePath, predLabel,
                        method, matcherType, out_dir, top_k=3):
    os.makedirs(out_dir, exist_ok=True)

    pathTest   = testItem["path"]
    trueLabel  = testItem["label"]
    imgTest   = testItem["image"]

    scoresSorted = sorted(scores, key=lambda x: x[0], reverse=True)
    top_scores = scoresSorted[:top_k]

    cols = top_k + 1
    plt.figure(figsize=(4 * cols, 4))

    # subplot 1: imaginea de test
    plt.subplot(1, cols, 1)
    plt.imshow(cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB))
    plt.title(f"TEST\ntrue={trueLabel}\npred={predLabel}")
    plt.axis('off')

    # subplots: top_k imagini din train
    for idx, (numMatches, trainLabel, trainPath) in enumerate(top_scores, start=2):
        imgTrain = trainImagePath.get(trainPath, None)
        if imgTrain is None:
            continue
        plt.subplot(1, cols, idx)
        plt.imshow(cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB))
        base = os.path.basename(trainPath)
        plt.title(f"TRAIN\n{trainLabel}\n{numMatches} matches\n{base}")
        plt.axis('off')

    base_test = os.path.basename(pathTest)
    out_path = os.path.join(out_dir, f"{os.path.splitext(base_test)[0]}_matches.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_summary_txt(method, matcherType, accuracy, labels, cm,
                     parityCases, noKpCases, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summaryPath = os.path.join(out_dir, "summary.txt")
    with open(summaryPath, "w") as f:
        f.write(f"Metoda: {method}\n")
        f.write(f"Matcher: {matcherType}\n")
        f.write(f"Acuratete: {accuracy * 100:.2f}%\n\n")
        f.write("Etichete (ordine in matrice):\n")
        f.write(", ".join(labels) + "\n\n")
        f.write("Matrice de confuzie (rows = TRUE, cols = PREDICTED):\n")
        f.write(str(cm) + "\n\n")

        f.write("Imagini de test cu PARITATE la vot:\n")
        if parityCases:
            for p in parityCases:
                f.write("  " + p + "\n")
        else:
            f.write("  (niciuna)\n")

        f.write("\nImagini de test fara keypoints/descriptori:\n")
        if noKpCases:
            for p in noKpCases:
                f.write("  " + p + "\n")
        else:
            f.write("  (niciuna)\n")

methods      = ["SIFT", "ORB", "AKAZE"]
matcherTypes = ["knn", "bf"]

for method in methods:
    print("\n" + "=" * 70)
    print(f"=== FEATURES pentru metoda {method} ===")
    print("=" * 70)

    trainData = getFeatureList(trainList, method)
    testData  = getFeatureList(testList, method)

    # dicționar pentru acces rapid la imaginile din train după path
    trainImagePath = {d["path"]: d["image"] for d in trainData}

    for matcherType in matcherTypes:
        print("\n" + "-" * 70)
        print(f"=== CLASIFICARE folosind metoda {method} și matcher {matcherType} ===")
        print("-" * 70)

        yTrue       = []
        yPred       = []
        parityCases = []
        noKpCases   = []
        labelsSet   = set()

        total_matches_all_tests = 0
        total_tests_with_matches = 0

        combo_out_dir = os.path.join(outputRoot, f"{method}_{matcherType}")
        os.makedirs(combo_out_dir, exist_ok=True)

        for testItem in testData:
            trueLabel = testItem["label"]
            labelsSet.add(trueLabel)
            pathTest = testItem["path"]

            best, scores = classify(testItem, trainData, method=method, matcherType=matcherType)

            if scores:
                max_test_matches = max(s[0] for s in scores)
                total_matches_all_tests += max_test_matches
                total_tests_with_matches += 1

            if best is None:
                print(f"{pathTest} NU poate fi clasificat(nu avem descriptori), atribui eticheta aleator.")
                noKpCases.append(pathTest)
                trainLabels = [tr["label"] for tr in trainData if tr["descriptors"] is not None]
                if not trainLabels:
                    continue
                predLabel = random.choice(trainLabels)
            else:
                predLabel, parityFlag, tiePaths = decideLabelBest(best)
                if parityFlag:
                    print(f"[PARITATE] pentru imaginea de test {pathTest}")
                    print("   Imagini de train implicate în paritate:")
                    for p in tiePaths:
                        print("     ", p)
                    parityCases.append(pathTest)

            yTrue.append(trueLabel)
            yPred.append(predLabel)

            # salvăm figura cu test + top matches
            if scores:
                save_matches_figure(
                    test_item=testItem,
                    scores=scores,
                    train_image_by_path=trainImagePath,
                    predLabel=predLabel,
                    method=method,
                    matcherType=matcherType,
                    out_dir=combo_out_dir,
                    top_k=3
                )

        correct  = sum(1 for t, p in zip(yTrue, yPred) if t == p)
        accuracy = correct / len(yTrue) if yTrue else 0.0
        print(f"\nAcuratețe (metoda={method}, matcher={matcherType}): {accuracy * 100:.2f}%")

        avg_matches = (total_matches_all_tests / total_tests_with_matches
                       if total_tests_with_matches > 0 else 0)

        globalResults.append({
            "method": method,
            "matcher": matcherType,
            "accuracy": accuracy,
            "avg_matches": avg_matches
        })

        labels = sorted(list(labelsSet))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)

        for t, p in zip(yTrue, yPred):
            i = label_to_idx[t]
            j = label_to_idx[p]
            cm[i, j] += 1

        print("Etichete (ordine în matrice):", labels)
        print("Confusion matrix (rows = TRUE, cols = PREDICTED):")
        print(cm)

        print("\nImagini de test cu PARITATE la vot:")
        for p in parityCases:
            print("  ", p)

        print("\nImagini de test fără keypoints/descriptori :")
        for p in noKpCases:
            print("  ", p)

        save_summary_txt(
            method=method,
            matcherType=matcherType,
            accuracy=accuracy,
            labels=labels,
            cm=cm,
            parityCases=parityCases,
            noKpCases=noKpCases,
            out_dir=combo_out_dir
        )

best_by_accuracy = max(globalResults, key=lambda x: x["accuracy"])
best_by_matches  = max(globalResults, key=lambda x: x["avg_matches"])

best_path = os.path.join(outputRoot, "best_overall_results.txt")

with open(best_path, "w") as f:
    f.write("=== CELE MAI BUNE REZULTATE GLOBALE ===\n\n")

    f.write(">> Cea mai buna combinatie dupa ACURATETE:\n")
    f.write(f"Metoda: {best_by_accuracy['method']}\n")
    f.write(f"Matcher: {best_by_accuracy['matcher']}\n")
    f.write(f"Acuratete: {best_by_accuracy['accuracy'] * 100:.2f}%\n")
    f.write(f"Nr mediu match-uri: {best_by_accuracy['avg_matches']:.2f}\n\n")

    f.write(">> Cea mai buna combinatie dupa NR MEDIU DE MATCH-URI:\n")
    f.write(f"Metoda: {best_by_matches['method']}\n")
    f.write(f"Matcher: {best_by_matches['matcher']}\n")
    f.write(f"Acuratete: {best_by_matches['accuracy'] * 100:.2f}%\n")
    f.write(f"Nr mediu match-uri: {best_by_matches['avg_matches']:.2f}\n\n")

    f.write("=== TOATE COMBINATIILE TESTATE ===\n")
    for r in globalResults:
        f.write(
            f"{r['method']} + {r['matcher']} | "
            f"Acuratete = {r['accuracy'] * 100:.2f}% | "
            f"Match-uri medii = {r['avg_matches']:.2f}\n"
        )

print("\nbBest results saved in:", best_path)
