from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Configure the web page
st.set_page_config(
    page_title="YOLOv8 AI Object Detector",
    page_icon="📷",
    layout="wide"
)

st.title("YOLOv8 AI Object Detection System")

st.write(
    "Upload an image to detect people, vehicles, "
    "and other objects using artificial intelligence."
)

# Load the model once and reuse it
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Allow the user to upload an image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Detection confidence setting
confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("AI is analysing your image..."):
            # Run YOLOv8
            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # Convert annotated image from BGR to RGB
            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

            st.subheader("Detection Results")

            st.image(
                annotated_image,
                caption="Objects detected by YOLOv8",
                use_container_width=True
            )

            # Display the detected objects
            st.subheader("Detected Objects")

            if result.boxes is None or len(result.boxes) == 0:
                st.info("No objects were detected.")
            else:
                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    name = model.names[class_id]
                    score = float(box.conf[0])

                    detections.append({
                        "Object": name,
                        "Confidence": f"{score:.2%}"
                    })

                st.dataframe(
                    detections,
                    use_container_width=True
                )

            # Offer the processed image for download
            import io

            output = io.BytesIO()

            Image.fromarray(annotated_image).save(
                output,
                format="JPEG"
            )

            st.download_button(
                label="Download Processed Image",
                data=output.getvalue(),
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )
else:
    st.info("Upload an image to begin.")