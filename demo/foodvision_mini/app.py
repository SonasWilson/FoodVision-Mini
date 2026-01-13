import gradio as gr
import torch
from PIL import Image
import os
import time
from model import create_model, get_transform
import torch

CLASS_NAMES = ["pizza", "steak", "sushi"]
MODEL_PATH = "models/best_effnetb2_feature_extractor.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# instantiate the model and load
model = create_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

# set to target device
model.to(DEVICE)
# model in eval mode
model.eval()

# create transform
transform = get_transform()

# prediction
def predict(image: Image.Image):

    start_time = time.time()

    # take image and return probs
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        preds = model(image)
        probs = torch.softmax(preds, dim=1)

    probs = probs.squeeze().cpu().numpy()

    pred_time = time.time() - start_time

#     return dictionary
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}, pred_time

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]
### Gradio Interface ###

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
    title = "FoodVision Mini 🍕🥩🍣",
    description="An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi.",
    examples=example_list
)

# launch app
if __name__ == "__main__":
    demo.launch()