# object_detection.py
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
#from PIL import Image
#import requests

def detect_objects(image):
    try:
        # Load Object Detection Model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Image processing
        inputs = processor(images=image, return_tensors="pt")
        print("Object detection inputs:", inputs)

        # Getting results
        outputs = model(**inputs)

        # Processing the results to get the detected objects
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detected_objects = []
        for score, label in zip(results["scores"], results["labels"]):
            detected_objects.append((model.config.id2label[label.item()], score.item()))

        print("Detected objects:", detected_objects)
        return detected_objects
    except Exception as e:
        print("Error in detect_objects:", str(e))
        return []

