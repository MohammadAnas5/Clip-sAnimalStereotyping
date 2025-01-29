import os
import pandas as pd
from PIL import Image

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

def load_dataset(path, exclude_folder="ant"):
    data = {"path": [], "label": []}
    for root, _, files in os.walk(path):
        if exclude_folder in root:
            continue
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                if is_valid_image(full_path):
                    label = os.path.basename(root)
                    data["path"].append(full_path)
                    data["label"].append(label)
    return pd.DataFrame(data)

if __name__ == "__main__":
    dataset_path = "/root/.cache/kagglehub/datasets/anas123siddiqui/animals/versions/1"
    dataset = load_dataset(dataset_path)
    dataset.to_csv("dataset.csv", index=False)
    print("Dataset loaded and saved as 'dataset.csv'")
