from ultralytics import YOLO
import cv2
model = YOLO("yolo11n.pt")   # sau yolov8n.pt etc.

def draw_dets(img, boxes, names):
    # boxes: list de (x1,y1,x2,y2,conf,cls_id)
    for x1,y1,x2,y2,conf,cls_id in boxes:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        label = f"{names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(20,y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

#ex-1 -> image and box detection
#imagePath = "./Images/photo1.jpg"
imagePath = "./Images/photo2.jpg"
img = cv2.imread(imagePath)

res = model.predict(source=img, conf=0.25, verbose=False)[0]
names = model.names

boxes = []
for b in res.boxes:
    x1,y1,x2,y2 = b.xyxy[0].tolist()
    conf = float(b.conf[0])
    cls_id = int(b.cls[0])
    boxes.append((x1,y1,x2,y2,conf,cls_id))

out = draw_dets(img.copy(), boxes, names)
#outPath = "./Images/YoloOut1.jpg"
outPath = "./Images/YoloOut2.jpg"
cv2.imwrite(outPath, out)

#ex - 2 -> 3 people in a room
#imagePath = "./Images/3people.jpg"
imagePath = "./Images/3people2.jpg"
img = cv2.imread(imagePath)

res = model.predict(source=img, conf=0.25, verbose=False)[0]
names = model.names

person_id = [k for k,v in names.items() if v == "person"][0]
count = int((res.boxes.cls == person_id).sum())

out = img.copy()

boxes = []
for b in res.boxes:
    if int(b.cls[0]) == person_id:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        boxes.append((x1,y1,x2,y2, float(b.conf[0]), person_id))

out = draw_dets(out, boxes, names)
#outPath = "./Images/3people_YoloOut1.jpg"
outPath = "./Images/3people_YoloOut2.jpg"
cv2.putText(out, f"Persons: {count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imwrite(outPath, out)
print("Persons:", count)

#ex - 3 -> video with moving cars

#cap = cv2.VideoCapture("Videos/pakistan.mp4")
#cap = cv2.VideoCapture("Videos/japan.mp4")
cap = cv2.VideoCapture("Videos/Iasi_Airport.mp4")

directions = 4
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
writer = cv2.VideoWriter("./Videos/out1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(source=frame, conf=0.25, verbose=False)[0]
    dets = []
    for b in res.boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        dets.append((x1,y1,x2,y2, float(b.conf[0]), int(b.cls[0])))

    dets.sort(key=lambda x: x[4], reverse=True)
    dets = dets[:directions]

    frame_out = draw_dets(frame, dets, model.names)
    writer.write(frame_out)

    cv2.imshow("Video with YOLO", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
