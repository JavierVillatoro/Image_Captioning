#import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

#Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, 
img_path = "C:\\Users\\franc\\Desktop\\IBM_GENERATIVE_AI\\Building_Generative_AI_APP\\Module1p\\jim.jpg"

# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")   #pt for pytorch tensors 

# Generate a caption for the image #MAX 50 tokens
outputs = model.generate(**inputs, max_length=50)   #** unpacking the inputs dictionary and passing its items as arguments to the model

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)