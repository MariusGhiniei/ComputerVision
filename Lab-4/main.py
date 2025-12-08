import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage(img, title="image"):
    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

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

def harrisKeypoints(gray, block_size=2, ksize=3, k=0.04, thresh=0.01):
    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    threshold = thresh * dst.max()
    pts = np.argwhere(dst > threshold)  # (y, x)
    keypoints = [cv2.KeyPoint(float(x), float(y), 3) for (y, x) in pts]
    return keypoints


#ex1
img = cv2.imread("/Users/marius/PycharmProjects/CV-lab1/Lab-4/Photos/palatul2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

showImage(img, "Original")
showImage(gray, "Grayscale")

methods = ["SIFT", "ORB", "AKAZE"]

# descriptori
for m in methods:
    kp = detectKeyPoints(gray, m)
    print(f"{m}: {len(kp)} keypoints (doar detect)")

    img_kp_simple = cv2.drawKeypoints(
        gray, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    showImage(img_kp_simple, f"{m} - keypoints - default")

    des = computeDescriptors(gray, kp, m)
    if des is not None:
        print(f"{m}: descriptori calculați, shape = {des.shape}")
    else:
        print(f"{m}: NU s-au putut calcula descriptori")

    img_kp_rich = cv2.drawKeypoints(
        gray, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    showImage(img_kp_rich, f"{m} - keypoints - rich")

# harris method
kp_harris = harrisKeypoints(gray)
print(f"HARRIS: {len(kp_harris)} keypoints")
img_harris = cv2.drawKeypoints(gray, kp_harris, None)
showImage(img_harris, "HARRIS keypoints")

# ex 2 (grid 8 x 8)
def keypointsGridStats(kp_list, img_shape, grid_size=8):
    h, w = img_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    counts = np.zeros((grid_size, grid_size), dtype=int)

    for kp in kp_list:
        x, y = kp.pt
        col = int(x // cell_w)
        row = int(y // cell_h)
        row = min(row, grid_size - 1)
        col = min(col, grid_size - 1)
        counts[row, col] += 1

    return counts

for m in methods:
    kp = detectKeyPoints(gray, m)
    counts = keypointsGridStats(kp, gray.shape, grid_size=8)
    print(f"Distribuție pe grid 8x8 pentru {m}:")
    print(counts, "\n")

counts_harris = keypointsGridStats(kp_harris, gray.shape, grid_size=8)
print("Distribuție pe grid 8x8 pentru HARRIS:")
print(counts_harris, "\n")

# ======= image functions ========
def blurMean(img, size=5):
    return cv2.blur(img, (size, size))

def blurGaussian(img, size=5):
    if size % 2 == 0:
        size += 1
    return cv2.GaussianBlur(img, (size, size), 0)

def rotateImage(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def addGaussianNoise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

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


def drawMatches(img1, kp1, img2, kp2, matches, title, max_draw=50):
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:max_draw],
        None,
        flags=2
    )
    showImage(img_matches, title)

def pair(transform_name, img2, method="SIFT"):
    print(f"\n{transform_name} cu metoda {method}")

    gray1 = gray
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. detect
    kp1 = detectKeyPoints(gray1, method)
    kp2 = detectKeyPoints(gray2, method)

    # 3. compute descriptors
    des1 = computeDescriptors(gray1, kp1, method)
    des2 = computeDescriptors(gray2, kp2, method)

    # 4. matching BF
    m_bf, good_bf = matchDescriptors(des1, des2, method=method, knn=False)
    # 4. matching kNN
    m_knn, good_knn = matchDescriptors(des1, des2, method=method, knn=True)

    print(f"BF matches: {len(good_bf)}")
    print(f"kNN good matches: {len(good_knn)}")

    # opțional: afișăm câteva match-uri kNN
    if len(good_knn) > 0:
        drawMatches(gray1, kp1, gray2, kp2, good_knn,
                     f"{transform_name} - kNN good matches")
#blur method
blur_sizes = [5, 9, 15]
method_for_matching = "SIFT"

for size in blur_sizes:
    blurred_mean = blurMean(img, size=size)
    blurred_gauss = blurGaussian(img, size=size)

    showImage(blurred_mean, f"Blur mediu, size={size}")
    showImage(blurred_gauss, f"Blur gaussian, size={size}")

    pair(f"Mean blur size={size}", blurred_mean, method_for_matching)
    pair(f"Gaussian blur size={size}", blurred_gauss, method_for_matching)


#rotation method
angles = [15, 30, 60]

for angle in angles:
    rotated = rotateImage(img, angle)
    showImage(rotated, f"Rotated {angle}°")
    pair(f"Rotation {angle}°", rotated, method_for_matching)

# gaussian noise
sigmas = [10, 20, 30]

for sigma in sigmas:
    noisy = addGaussianNoise(img, sigma=sigma)
    showImage(noisy, f"Noisy (sigma={sigma})")
    pair(f"Gaussian noise sigma={sigma}", noisy, method_for_matching)
