import os
import cv2
import xml.etree.ElementTree as ET

XML_PATH   = "./Car-tracking/archive/annotations.xml"
IMAGES_DIR = "./Car-tracking/archive/images"
SHOW_FRAMES = True
MAX_FRAMES = 500

def draw_bbox(img, bbox, text, color=(0,255,0), thick=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
    cv2.putText(img, text, (x1, max(20, y1-7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def parse_annotations_boxes(xml_path):
#      <track id="..." label="car|minivan">
#        <box frame=".." xtl=".." ytl=".." xbr=".." ybr=".." outside="0|1" ... />
#      </track>
    # -> frame_dets[frame] = [{track_id,label,bbox=(x1,y1,x2,y2)}]

    root = ET.parse(xml_path).getroot()
    frame_dets = {}

    for track in root.findall(".//track"):
        track_id = int(track.get("id"))
        label = track.get("label")

        for box in track.findall(".//box"):
            frame = int(box.get("frame"))
            outside = int(box.get("outside", "0"))
            if outside == 1:
                continue  # obiectul e "în afara" frame-ului

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            frame_dets.setdefault(frame, []).append({
                "track_id": track_id,
                "label": label,
                "bbox": (xtl, ytl, xbr, ybr)
            })

    return frame_dets

def find_image_for_frame(images_dir, frame_idx):
    for ext in (".png", ".PNG"):
        p = os.path.join(images_dir, f"frame_{frame_idx:06d}{ext}")
        if os.path.exists(p):
            return p

    for ext in (".png", ".PNG"):
        p = os.path.join(images_dir, f"{frame_idx}{ext}")
        if os.path.exists(p):
            return p

    return None

#xml parse
frame_dets = parse_annotations_boxes(XML_PATH)
frames_sorted = sorted(frame_dets.keys())
print("\nTotal frames with annotations:", len(frames_sorted))
if not frames_sorted:
    raise RuntimeError("Nu am găsit niciun <box> în XML (verifică XML_PATH).")

#count car/frame
counts_per_frame = {}
for f in frames_sorted:
    car_cnt = sum(d["label"] == "car" for d in frame_dets[f])
    minivan_cnt = sum(d["label"] == "minivan" for d in frame_dets[f])
    counts_per_frame[f] = {"car": car_cnt, "minivan": minivan_cnt}

#tracked_id - red
tracked_id = 21
tracked_frames = [f for f in frames_sorted if any(d["track_id"] == tracked_id for d in frame_dets[f])]
enter_frame, exit_frame = min(tracked_frames), max(tracked_frames)

print("\nTRACKED VEHICLE")
print("Track ID:", tracked_id)
print("Entering frame:", enter_frame)
print("Exiting frame:", exit_frame)

#detecting
if SHOW_FRAMES:
    shown = 0
    for f in frames_sorted:
        if shown >= MAX_FRAMES:
            break

        img_path = find_image_for_frame(IMAGES_DIR, f)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        cv2.putText(img,
                    f"Frame {f} | car={counts_per_frame[f]['car']} minivan={counts_per_frame[f]['minivan']}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        for d in frame_dets[f]:
            color = (0, 255, 0)
            text = f"{d['label']} id={d['track_id']}"
            if d["track_id"] == tracked_id:
                color = (0, 0, 255)
                text = f"TRACKED {d['label']} id={d['track_id']}"
            draw_bbox(img, d["bbox"], text, color=color)

        cv2.imshow("Tracking", img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        shown += 1

    cv2.destroyAllWindows()
