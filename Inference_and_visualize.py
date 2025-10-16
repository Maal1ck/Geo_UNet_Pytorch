import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc

def predict_and_visualize(model_path, image_path, device=None, num_classes=2, threshold=0.5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = UNet(in_channels=3, out_channels=num_classes if num_classes>1 else 1).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = np.array(Image.open(image_path).convert("RGB"))
    transform = A.Compose([A.CenterCrop(1024,1024), A.Normalize(), ToTensorV2()])
    aug = transform(image=img)
    inp = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
        if num_classes <= 2:
            prob = torch.sigmoid(out)[:,0,:,:].cpu().numpy()[0]
            pred = (prob > threshold).astype(np.uint8)
        else:
            pred = torch.argmax(out, dim=1).cpu().numpy()[0].astype(np.uint8)

    # show original + mask
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img); plt.title("Image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(pred, cmap="gray"); plt.title("Predicted mask"); plt.axis("off")
    plt.show()
    return pred

    # Now create model with the correct number of classes
    out_channels = 1 if num_classes <= 2 else num_classes
    model = UNet(in_channels=3, out_channels=out_channels).to(device)
    
    # Load weights
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # ... rest of your function remains the same ...
    
    # Load and preprocess image
    img_original = np.array(Image.open(image_path).convert("RGB"))
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    
    aug = transform(image=img_original)
    inp = aug["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        out = model(inp)
        if num_classes <= 2:
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)
        else:
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"Predicted Mask (thresh={threshold})")
    axes[1].axis("off")
    
    # Overlay
    overlay = img_original.copy()
    if num_classes <= 2:
        overlay[pred > 0] = [255, 0, 0]  # Red overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return pred