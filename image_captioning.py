#image_captioning.py
from transformers import AutoModelForVision2Seq, ViTImageProcessor, AutoTokenizer
import torch

def generate_caption(image):
    try:
        # Ensure the image is in RGB format
        image = image.convert("RGB")  
        
        processor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
        model = AutoModelForVision2Seq.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
        tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

        # Prepare the image for processing
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():  # Reduce memory usage
            pixel_values = inputs['pixel_values']
            generated_ids = model.generate(pixel_values, max_new_tokens=50)

        generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_caption
    except Exception as e:
        print("Error in generate_caption:", str(e))
        return None
