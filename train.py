from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(data = "data_custom.yaml", imgsz = 640, batch = 8, epochs = 100, workers = 1, device = 0)