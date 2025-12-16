import os, random, shutil
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH   = "./Car-tracking/archive/annotations.xml"
IMAGES_DIR = "./Car-tracking/archive/images"

OUT_ROOT   = "./car_yolo_dataset"
TRAIN_RATIO = 0.7
SEED = 42

CLASS_MAP = {"car": 0, "minivan": 1}

random.seed(SEED)

def parse_cvat_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    frame_dets = {}  # frame -> list of (label, xtl, ytl, xbr, ybr)
    for track in root.findall(".//track"):
        label = track.get("label")
        if label not in CLASS_MAP:
            continue
        for box in track.findall(".//box"):
            outside = int(box.get("outside", "0"))
            if outside == 1:
                continue
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl")); ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr")); ybr = float(box.get("ybr"))
            frame_dets.setdefault(frame, []).append((label, xtl, ytl, xbr, ybr))
    return frame_dets

def find_image_for_frame(images_dir, frame_idx):
    for ext in (".png",".jpg",".jpeg",".bmp"):
        p = os.path.join(images_dir, f"frame_{frame_idx:06d}{ext}")
        if os.path.exists(p): return p
    for ext in (".png",".jpg",".jpeg",".bmp"):
        p = os.path.join(images_dir, f"{frame_idx}{ext}")
        if os.path.exists(p): return p
    return None

def yolo_line(label, xtl, ytl, xbr, ybr, w, h):
    # clamp în [0,w/h]
    xtl = max(0.0, min(xtl, w-1))
    xbr = max(0.0, min(xbr, w-1))
    ytl = max(0.0, min(ytl, h-1))
    ybr = max(0.0, min(ybr, h-1))

    xc = ((xtl + xbr) / 2.0) / w
    yc = ((ytl + ybr) / 2.0) / h
    bw = (xbr - xtl) / w
    bh = (ybr - ytl) / h

    cid = CLASS_MAP[label]
    return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

frame_dets = parse_cvat_boxes(XML_PATH)
frames = sorted(frame_dets.keys())
print("Annotated frames:", len(frames), "| first:", frames[:5])


random.shuffle(frames)
n_train = int(TRAIN_RATIO * len(frames))
train_frames = frames[:n_train]
val_frames   = frames[n_train:]
print("Train frames:", len(train_frames), "Val frames:", len(val_frames))

out = Path(OUT_ROOT)
for split in ["train","val"]:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

def export_split(frames_list, split):
    for f in frames_list:
        img_path = find_image_for_frame(IMAGES_DIR, f)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # copy image
        dst_img = out / "images" / split / Path(img_path).name
        shutil.copy2(img_path, dst_img)

        # write label file (same stem)
        dst_lbl = out / "labels" / split / (Path(img_path).stem + ".txt")
        lines = []
        for (label, xtl, ytl, xbr, ybr) in frame_dets[f]:
            lines.append(yolo_line(label, xtl, ytl, xbr, ybr, w, h))
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

export_split(train_frames, "train")
export_split(val_frames, "val")

# 4) write data.yaml
yaml_text = f"""path: {Path(OUT_ROOT).resolve()}
train: images/train
val: images/val
names:
  0: car
  1: minivan
"""
(Path(OUT_ROOT) / "data.yaml").write_text(yaml_text)
print("Wrote:", Path(OUT_ROOT) / "data.yaml")
print("Done. Dataset ready in:", OUT_ROOT)
