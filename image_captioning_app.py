import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Define the image captioning function
def caption_image(input_image: np.ndarray):
    """
    Takes an input image and returns a caption
    """
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    
    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

#Create web app interface

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="Web app for generating captions for images"
)

iface.launch(inbrowser=True)