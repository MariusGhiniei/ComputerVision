import os
import glob
import cv2
from ultralytics import YOLO

DATA_ROOT = "./car_yolo_dataset"
VAL_IMG_DIR = os.path.join(DATA_ROOT, "images/val")
VAL_LBL_DIR = os.path.join(DATA_ROOT, "labels/val")

model = YOLO("../runs/detect/train/weights/best.pt")
IOU_TH = 0.5
CONF_TH = 0.25

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def yolo_lbl_to_xyxy(line, w, h):
    cid, xc, yc, bw, bh = line.split()
    cid = int(cid)
    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
    x1 = (xc - bw/2) * w
    y1 = (yc - bh/2) * h
    x2 = (xc + bw/2) * w
    y2 = (yc + bh/2) * h
    return cid, (x1,y1,x2,y2)

tp = 0  # correct
fn = 0  # missed
fp = 0
gt_total = 0

img_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.*")))

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    lbl_path = os.path.join(VAL_LBL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    if not os.path.exists(lbl_path):
        continue

    gt = []
    for line in open(lbl_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        cid, box = yolo_lbl_to_xyxy(line, w, h)
        gt.append((cid, box))
    gt_total += len(gt)

    # predict
    res = model.predict(source=img, conf=CONF_TH, verbose=False)[0]
    preds = []
    for b in res.boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        preds.append((int(b.cls[0]), (x1,y1,x2,y2), float(b.conf[0])))

    # match GT
    used = set()
    for gt_cid, gt_box in gt:
        best_iou = 0.0
        best_j = -1
        for j, (pcid, pbox, pconf) in enumerate(preds):
            if j in used:
                continue
            if pcid != gt_cid:
                continue
            s = iou_xyxy(gt_box, pbox)
            if s > best_iou:
                best_iou = s
                best_j = j

        if best_iou >= IOU_TH:
            tp += 1
            used.add(best_j)
        else:
            fn += 1
    fp = fp + len(preds) - len(used)

jaccard = tp / (tp + fp + fn + 1e-9)
dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)


print("Total predictions: ", tp + fp)
print("TP (correct identified):", tp)
print("FP (false possitive):", fp)
print("FN (missed identified):", fn)
print("\n===========================\n")
print("Jaccard / IoU:", jaccard)
print("Dice coefficient:", dice)
