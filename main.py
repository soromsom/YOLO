from pathlib import Path
from ultralytics import YOLO
import cv2

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent

# Define input and output locations
IMAGE_PATH = BASE_DIR / "images" / "yolo 1.jpg"
RESULTS_DIR = BASE_DIR / "results"

# Create the results folder if necessary
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the YOLOv8 Nano model
print("Loading YOLOv8 model...")

model = YOLO("yolov8n.pt")

# Check whether the image exists
if not IMAGE_PATH.exists():
    print("ERROR: sample.jpg was not found.")
    print("Put your image inside the images folder.")
    raise SystemExit(1)

# Run object detection
print("Detecting objects...")

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    verbose=False
)

# Get the detection result
result = results[0]

# Display detected objects
print("\nOBJECT DETECTION RESULTS")
print("-" * 35)

if result.boxes is None or len(result.boxes) == 0:
    print("No objects detected.")
else:
    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"Object: {object_name} | "
            f"Confidence: {confidence:.2%}"
        )

# Draw bounding boxes and labels
annotated_image = result.plot()

# Save the processed image
output_path = RESULTS_DIR / "detected_image.jpg"

success = cv2.imwrite(
    str(output_path),
    annotated_image
)

if success:
    print(f"\nProcessed image saved to: {output_path}")
else:
    print("ERROR: Could not save the processed image.")

print("Detection completed successfully.")