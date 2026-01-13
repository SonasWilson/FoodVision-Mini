
# FoodVision Mini 🍕🥩🍣


**A lightweight computer vision project that classifies images of food into pizza, steak, or sushi using PyTorch and EfficientNetB2.**

#### **Note:** This project was developed while following the [Daniel Bourke's FoodVision tutorial](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface) - for learning purposes. The dataset and learning approach are inspired by his course, and I built this project to practice end-to-end model development, deployment, and portfolio creation.
---

## 🔗 Live Demo
Try the model online here: [FoodVision Mini on Hugging Face](https://huggingface.co/spaces/sonawilson/foodvision-mini)


## Features

* Trains an EfficientNetB2 feature extractor on a small food dataset
* Automatically saves the **best-performing model** during training
* Modular, clean code structure for training, data handling, and inference
* Real-time interactive demo using Gradio
* Cross-platform (Windows/Linux/macOS) and GPU-ready

---

## Project Structure

```
FoodVision-Mini/
├── data/                       # Dataset (train/test folders)
├── models/                     # Saved best model
├── demo/
│   └── foodvision_mini/
│       └── app.py              # Gradio interface
│       └── model.py            # Creating model
├── utils/
│   ├── data_download.py        # Dataset check/download
│   ├── data_setup.py           # Dataloaders & transforms
│   └── engine.py               # Train/Test steps
├── train.py                    # Model training
└── README.md
└── requirements.txt
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SonasWilson/FoodVision-Mini.git
cd FoodVision-Mini
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

*(requirements.txt includes: torch, torchvision, gradio, PIL, requests)*

---

## Usage

### 1️⃣ Train the model

```bash
python train.py
```

* Downloads the dataset if not already present
* Trains the model and saves the best checkpoint to `models/`

### 2️⃣ Launch the demo

```bash
python demo/foodvision_mini/app.py
```

* Opens a Gradio interface
* Upload an image to see predicted probabilities for pizza, steak, or sushi

## ⚠️ Disclaimer
Due to visual similarities between some foods (e.g., round sushi rolls and pizza), 
the model may occasionally misclassify images. This is a known limitation of the dataset.


---

## Notes

* The project is **modular**, so you can swap the model, dataset, or transforms easily.
* Designed to be **lightweight** for learning, demo, and portfolio purposes.

> ⚠️ Note: Due to visual similarity between some foods (e.g., round sushi rolls and pizza),
> the model may occasionally misclassify images. This is a known limitation of the dataset.


