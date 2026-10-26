from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Display model information
model.info()

print("YOLOv8 model loaded successfully!")