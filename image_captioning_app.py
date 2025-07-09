import gradio as gr 
import numpy as np 
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model     = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')


def caption_image(input_image: np.array):
    """
    Receive an image as NP.Array and create a caption using transformers
    """
    raw_image = Image.fromarray(input_image).convert('RGB')

    text = "The image of "
    inputs = processor(images=raw_image,text=text,return_tensors='pt')
    outputs = model.generate(**inputs, max_length=50)

    return processor.decode(outputs[0], skip_special_tokens=True)

#create the interface using Gradio
iface = gr.Interface (
    fn=caption_image,
    inputs = gr.Image(),
    outputs='text',
    title='Image Captioning',
    description="This is a simple web app for generating captions for images using a trained model."
)

#launch the application interface
iface.launch(share=True)