"""Image caption generation using BLIP model from Hugging Face Transformers."""

import os
import glob
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Constants
MODEL_NAME = "Salesforce/blip-image-captioning-base"
IMAGE_DIR = "C:\\Users\\franc\\Desktop\\imagenes\\Music_art_gallery"
IMAGE_EXTS = ("jpg", "jpeg", "png")
OUTPUT_FILE = "captions2.txt"


def load_model_components():
    """Load and return BLIP model components.
    
    Returns:
        tuple: (processor, model) loaded from pretrained weights
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    return processor, model


def generate_caption(processor, model, image_path):
    """Generate caption for a single image.
    
    Args:
        processor: BLIP image processor
        model: BLIP captioning model
        image_path (str): Path to image file
        
    Returns:
        str: Generated caption text
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(outputs[0], skip_special_tokens=True)


def process_image_directory():
    """Main processing function to generate captions for all images in directory."""
    processor, model = load_model_components()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for ext in IMAGE_EXTS:
            pattern = os.path.join(IMAGE_DIR, f"*.{ext}")
            for img_path in glob.glob(pattern):
                try:
                    caption = generate_caption(processor, model, img_path)
                    file_name = os.path.basename(img_path)
                    output_line = f"{file_name}: {caption}\n"
                    output_file.write(output_line)
                except IOError as error:
                    print(f"Error processing {img_path}: {str(error)}")


if __name__ == "__main__":
    process_image_directory()
