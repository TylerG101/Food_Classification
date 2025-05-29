import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Get class names from text file
with open("class_names.txt", "r") as f: # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in  f.readlines()]


# Create the EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101,
)

# Load the saved weights
effnetb2.load_state_dict(
    torch.load(
        f="effnetb2_feature_extractor_food101.pth",
        map_location=torch.device("cpu"),
    )
)


# Create a function to predict on images
def predict(img) -> Tuple[Dict, float]:
    """Predicts food item in provided image
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate time
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time




# Set title, description and article strings
title = "Food Classification App"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food from a set of 101 different kinds."
article = ""

# Make a list of examples
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=101, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    server_name="0.0.0.0", 
                    title=title,
                    description=description,
                    article=article,).launch()