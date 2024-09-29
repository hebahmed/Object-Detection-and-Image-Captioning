from transformers import AutoModelForVision2Seq, ViTImageProcessor, AutoTokenizer
import torch

def generate_caption(image):
    try:
        print("Image shape:", image.size)
        
        # استخدم AutoModelForVision2Seq بدلاً من VisionEncoderDecoderModel
        model = AutoModelForVision2Seq.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
        processor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
        tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

        # تجهيز الصورة للمعالجة
        inputs = processor(images=image, return_tensors="pt")
        print("Image inputs for captioning:", inputs)  # طباعة المدخلات المعالجة
        pixel_values = inputs['pixel_values']

        # إعداد attention_mask إذا كانت البيانات تحتوي على padding
        attention_mask = torch.ones(pixel_values.shape[:2])

        # توليد التسميات التوضيحية مع تحديد الحد الأقصى للطول
        generated_ids = model.generate(pixel_values, attention_mask=attention_mask, max_new_tokens=50)
        generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print("Generated Caption:", generated_caption)
        return generated_caption
    except Exception as e:
        print("Error in generate_caption:", str(e))
        return None
