from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent  # the Skinalyzer2 folder

model = YOLO(HERE / "skinalyzer2.pt")

results = model.predict(source=HERE / "captures/EXP-001", conf=0.25, imgsz=1024, save=True,
                        project=HERE / "runs/detect", name="EXP-001", exist_ok=True)

for r in results:
    print(r.path, len(r.boxes))