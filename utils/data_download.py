import os
import zipfile
import requests
from pathlib import Path

DATA_URL = (
    "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/"
    "pizza_steak_sushi_20_percent.zip"
)

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "pizza_steak_sushi_20_percent.zip"


def download_data():
    """
    Downloads and extracts the dataset if it does not already exist.
    """

    # 1. Check if dataset already exists
    if (DATA_DIR / "train").exists() and (DATA_DIR / "test").exists():
        print("Dataset already exists. Skipping download.")

    else:
        # Create data directory if needed
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        print("⬇️  Dataset not found. Downloading...")

        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()

        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting dataset...")

        # unzipping
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print("Unzipping data...")

        # remove zip file
        ZIP_PATH.unlink()

        print("Dataset downloaded.")
