from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # backbone pretrained
model.train(
    data="./car_yolo_dataset/data.yaml",
    epochs=50,
    imgsz=512,
    batch=8
)
