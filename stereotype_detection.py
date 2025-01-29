import torch
import clip

def detect_stereotypes(model, image_features, prompts, temperature=0.1):
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    logits = image_features @ text_features.T
    logits /= temperature
    probs = logits.softmax(dim=-1)
    
    return probs

if __name__ == "__main__":
    # Load model and dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    image_features = torch.load("image_features.pt")  # Assume saved features
    
    # Define prompts and detect stereotypes
    prompts = [
        "image of a brave animal", 
        "image of a lazy animal","......more prompts............"
    ] 
  
    results = detect_stereotypes(model, image_features, prompts)
    print("Stereotype detection results:", results)
