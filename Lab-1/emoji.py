import cv2
import numpy as np

def emoji(filename):
    size = 512
    img = np.full((size, size, 3), 255, np.uint8)

    # bgr colors
    ORANGE = (0, 165, 255)
    BLACK  = (0, 0, 0)
    WHITE  = (255, 255, 255)
    PINK   = (180, 105, 255)

    cx = size // 2
    cy = size // 2

    r = int(size * 0.4)

    # face
    cv2.circle(img, center = (cx, cy), radius = r, color = ORANGE, thickness= -1, lineType=cv2.LINE_AA)

    # ears
    leftEar = np.array([
        [cx - int(0.90*r), cy - int(0.30*r)],
        [cx - int(0.55*r), cy - int(1.10*r)],
        [cx - int(0.40*r), cy - int(0.30*r)]
    ], np.int32)

    rightEar = np.array([
        [cx + int(0.90*r), cy - int(0.30*r)],
        [cx + int(0.55*r), cy - int(1.10*r)],
        [cx + int(0.40*r), cy - int(0.30*r)]
    ], np.int32)

    cv2.fillPoly(img, [leftEar], color = ORANGE, lineType=cv2.LINE_AA)
    cv2.fillPoly(img, [rightEar], color = ORANGE, lineType=cv2.LINE_AA)

    # eyes
    eyeY  = cy - int(r * 0.2)
    eyeX = int(r * 0.28)
    eyeRadius  = int(r * 0.1)
    eyeWhiteRadius   = max(2, int(eyeRadius * 0.35))
    eyeWhiteEye = int(eyeRadius * 0.45)

    leftEye  = (cx - eyeX, eyeY)
    rightEye = (cx + eyeX, eyeY)

    cv2.circle(img, center = leftEye, radius =  eyeRadius, color = BLACK, thickness = -1, lineType = cv2.LINE_AA)
    cv2.circle(img, center = rightEye, radius = eyeRadius , color = BLACK, thickness = -1, lineType = cv2.LINE_AA)
    cv2.circle(img, center = (leftEye[0]  - eyeWhiteEye, leftEye[1]  - eyeWhiteEye), radius = eyeWhiteRadius,
               color = WHITE, thickness =  -1, lineType = cv2.LINE_AA)
    cv2.circle(img, center = (rightEye[0] - eyeWhiteEye, rightEye[1] - eyeWhiteEye), radius =  eyeWhiteRadius,
               color = WHITE, thickness = -1, lineType = cv2.LINE_AA)

    # mouth
    mouthCenter = (cx, cy + int(r * 0.35))
    cv2.ellipse(img, center = mouthCenter,  axes = (int(r * 0.25), int(r * 0.15)),
               angle = 0, startAngle = 0, endAngle = 180, color = BLACK, thickness = 3, lineType = cv2.LINE_AA)

    # tongue
    tongueCenter = (cx, cy + int(r * 0.55))
    tongueAxes  = (int(r * 0.12), int(r * 0.11))
    cv2.ellipse(img, center = tongueCenter, axes = tongueAxes, angle = 0, startAngle = 0,  endAngle = 360,
                color = PINK, thickness =  -1, lineType =  cv2.LINE_AA)

    cv2.line(img, pt1 = (tongueCenter[0], tongueCenter[1] - int(tongueAxes[1] * 0.2)),
                pt2 = (tongueCenter[0], tongueCenter[1] + int(tongueAxes[1] * 0.9)),
            color =  (160, 70, 200), thickness = 2, lineType = cv2.LINE_AA)

    # whiskers
    whiskerLen = int(r * 0.9)
    for side in (-1, 1):
        x = cx + int(side * r * 0.42)
        for dy in (-14, 0, 14):
            y = cy + int(r * 0.30) + dy

            if side == -1:
                cv2.line(img, pt1 = (x, y),
                     pt2 = (x + int(side * whiskerLen), y + side * 10),
                     color = BLACK, thickness = 2, lineType = cv2.LINE_AA)
            else:  cv2.line(img, pt1 = (x, y),
                     pt2 = (x + int(side * whiskerLen), y - side * 10),
                     color = BLACK, thickness = 2, lineType = cv2.LINE_AA)
    cv2.imwrite(filename, img)
    return img

cat = emoji("Ghiniei_Marius_Iulian.jpg")

