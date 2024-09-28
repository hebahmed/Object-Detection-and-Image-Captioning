import streamlit as st
from PIL import Image
from object_detection import detect_objects
from image_captioning import generate_caption

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import ViTImageProcessor, VisionEncoderDecoderModel, AutoTokenizer
import torch

st.title("Object Detection and Image Captioning")
st.write("Upload an image to analyze.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Object detection
        detected_objects = detect_objects(image)
        st.write("Detected objects:")
        for obj, score in detected_objects:
            st.write(f"{obj}: {score:.2f}")

        # Generate Caption
        caption = generate_caption(image)
        #print("Generated Caption:", caption)
        st.write("Generated Caption:")
        st.write(caption)
    except Exception as e:
        st.error(f"Error: {e}")