from ultralytics import YOLO

model = YOLO("skinalyzer2.pt")

model.predict(source = 0, show=True, save=True, conf=0.4)