"""Image captioning web application using BLIP model and Gradio interface.

This module provides a web interface to generate captions for images using
Salesforce's BLIP model from Hugging Face Transformers.
"""

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image: np.ndarray) -> str:
    """Generate a caption for the input image using BLIP model.

    Args:
        input_image: Input image array in numpy format

    Returns:
        Generated caption text
    """
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image with prompt text
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # Generate caption with constrained maximum length
    outputs = model.generate(**inputs, max_length=50)

    # Decode and clean the generated caption
    return processor.decode(outputs[0], skip_special_tokens=True)


# Create and configure Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="Web app for generating captions for images using AI"
)

# Launch the application
iface.launch(inbrowser=True)
