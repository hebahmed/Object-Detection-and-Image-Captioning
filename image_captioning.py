# image_captioning.py
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

def generate_caption(image):
    try:
        print("Image shape:", image.size)

        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
        processor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

        # تحضير الصورة للمعالجة
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        out = model.generate(**inputs)

        # التأكد من أن هناك ناتج صحيح
        if out is not None and len(out) > 0:
            caption = tokenizer.decode(out[0], skip_special_tokens=True)
            print("Generated Caption:", caption)
            return caption
        else:
            return "No caption generated."
    except Exception as e:
        print("Error in generate_caption:", str(e))
        return None
