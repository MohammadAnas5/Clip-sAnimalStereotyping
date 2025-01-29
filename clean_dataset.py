import os
import pandas as pd
import torch
import clip
from PIL import Image
import shutil
import zipfile

def clean_images(model, image_features, prompts, temperature=0.1, top_k=200):
    cleaning_text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        cleaning_text_features = model.encode_text(cleaning_text_tokens)
        cleaning_text_features /= cleaning_text_features.norm(dim=-1, keepdim=True)
    
    logits = image_features @ cleaning_text_features.T
    logits /= temperature
    probs = logits.softmax(dim=-1)
    
    removed_indices = set()
    for i in range(len(prompts)):
        removed_indices.update(probs[:, i].topk(top_k).indices.cpu().numpy())
    return removed_indices

if __name__ == "__main__":
    # Load dataset and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = pd.read_csv("dataset.csv")
    
    # Preprocess and encode images
    image_paths = dataset["path"].tolist()
    preprocessed_images = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device) for p in image_paths]
    image_features = model.encode_image(torch.cat(preprocessed_images)).detach()
    
    # Define cleaning prompts and clean images
    prompts = ["images with humans", "images with text"]
    removed_indices = clean_images(model, image_features, prompts)
    
    # Filter and save dataset
    remaining_images = [image_paths[i] for i in range(len(image_paths)) if i not in removed_indices]
    filtered_path = "filtered_dataset"
    os.makedirs(filtered_path, exist_ok=True)
    for img_path in remaining_images:
        label = dataset.loc[dataset["path"] == img_path, "label"].values[0]
        label_folder = os.path.join(filtered_path, label)
        os.makedirs(label_folder, exist_ok=True)
        shutil.copy(img_path, label_folder)
    
    with zipfile.ZipFile("filtered_dataset.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(filtered_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), filtered_path))
    
    print("Filtered dataset saved and compressed as 'filtered_dataset.zip'")
