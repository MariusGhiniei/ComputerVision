import cv2
import numpy as np

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

def converToMask(image, method):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            r, g, b = image[y, x]
            if(method == "RGB"):
                if rgbPixelMethod(r, g, b):
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0
            elif(method == "HSV"):
                if hsvPixelMethod(r, g, b):
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0
            elif(method == "YCbCr"):
                if yCbCrPixelMethod(r, g, b):
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0
            else: raise ValueError("Unknown method " + str(method))
    return mask

def rectToSquare(x,y, w, h, W, H):
    side = int(max(w, h))
    cx, cy = x + w//2, y + h//2
    sx, sy = cx - side//2, cy - side//2
    sx = max(0, min(sx, W - side))
    sy = max(0, min(sy, H - side))
    return sx, sy, side

def faceSquare(mask):
    h,w = mask.shape[:2]

    tMask = mask.copy()

    ys, xs = np.nonzero(tMask)

    if len(xs) == 0: return None

    x, y = xs.min(), ys.min()
    wS, hS = xs.max() - x + 1, ys.max() - y + 1
    return rectToSquare(x, y, wS, hS, w, h)

def detectSquare(image):
    mask = converToMask(image, "RGB")
    cv2.imshow("mask", mask)
    #mask = converToMask(image, "HSV")
    # mask = converToMask(image, "YCbCr")

    square = faceSquare(mask)
    return square

def drawSquare(image, square):
    out = image.copy()
    if square is not None:
        x, y, s = square
        cv2.rectangle(out,(x, y), (x + s , y + s), color=(255, 0, 0), thickness = 2)

    return out

# img = cv2.imread("photos/5.jpg", cv2.COLOR_BGR2RGB)
# img = cv2.imread("photos/2.jpg", cv2.COLOR_BGR2RGB)
# img = cv2.imread("/Users/marius/PycharmProjects/CV-lab1/Lab-3/Face_Dataset/Pratheepan_Dataset/FacePhoto/124511719065943_2.jpg", cv2.COLOR_BGR2RGB)
img = cv2.imread("/Users/marius/PycharmProjects/CV-lab1/Lab-3/photos/portrait.jpg", cv2.COLOR_BGR2RGB)

square = detectSquare(img)
imageDraw = drawSquare(img, square)
cv2.imshow("Detect Face", imageDraw)
cv2.waitKey(0)