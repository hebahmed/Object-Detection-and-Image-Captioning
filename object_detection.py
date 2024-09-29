# object_detection.py
import streamlit as st  
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

def detect_objects(image):
    try:
        # Load Object Detection Model
        print("Loading object detection model...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        print("Model loaded successfully")

        # Image processing
        image = image.convert("RGB")  # تأكد من أن الصورة بصيغة RGB

        # أضيفي هذا السطر لعرض الصورة بعد تحويلها
        st.image(image, caption='Converted Image (RGB)', use_column_width=True)

        inputs = processor(images=image, return_tensors="pt")
        print("Object detection inputs:", inputs)

        # Getting results
        with torch.no_grad():  # لتقليل استخدام الذاكرة
            outputs = model(**inputs)

        # أضيفي هذا السطر لعرض مخرجات النموذج
        print("Object detection outputs:", outputs)

        # Processing the results to get the detected objects
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        detected_objects = []
        for score, label in zip(results["scores"], results["labels"]):
            detected_objects.append((model.config.id2label[label.item()], score.item()))

        print("Detected objects:", detected_objects)
        return detected_objects
    except Exception as e:
        print("Error in detect_objects:", str(e))
        return []
