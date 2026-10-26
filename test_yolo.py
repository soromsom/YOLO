from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Detect objects in a sample image
results = model(
    "https://ultralytics.com/images/bus.jpg",
    save=True
)

print("Object detection completed!")