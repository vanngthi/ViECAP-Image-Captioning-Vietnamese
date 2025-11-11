from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model('../dataset/Flick_sportball/images/111796099.jpg', conf=0.5)

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = model.names[cls_id]
    print(f"{label}: {conf:.2f}")