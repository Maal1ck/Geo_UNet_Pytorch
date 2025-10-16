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

def train_model(
    train_images, train_masks,
    val_images, val_masks,
    num_classes=2,
    img_size=256,
    epochs=25,
    batch_size=8,
    lr=1e-4,
    device=None,
    model_save_path="unet_best.pth"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets and loaders
    train_ds = SegDataset(train_images, train_masks, transforms=get_transforms(img_size, do_aug=True))
    val_ds   = SegDataset(val_images, val_masks, transforms=get_transforms(img_size, do_aug=False))
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True, drop_last=True)  # ← num_workers=0
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)  # ← num_workers=0
    
    # Model
    out_channels = 1 if num_classes <= 2 else num_classes
    model = UNet(in_channels=3, out_channels=out_channels).to(device)
    
    # Loss & optimizer
    if num_classes <= 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_val_iou = -1
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
    
    for epoch in range(1, epochs+1):
        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if num_classes <= 2:
                masks_float = masks.float().unsqueeze(1)
                loss = criterion(outputs, masks_float)
            else:
                loss = criterion(outputs, masks.long())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ---------- Validate ----------
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                outputs = model(imgs)
                
                if num_classes <= 2:
                    masks_float = masks.float().unsqueeze(1)
                    loss = criterion(outputs, masks_float)
                    preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
                else:
                    loss = criterion(outputs, masks.long())
                    preds = torch.argmax(outputs, dim=1)
                
                val_loss += loss.item()
                
                # Calculate metrics
                for pred, mask in zip(preds, masks):
                    iou_scores.append(iou_score(pred, mask))
                    dice_scores.append(dice_score(pred, mask))
        
        val_loss /= len(val_loader)
        mean_iou = float(np.mean(iou_scores))
        mean_dice = float(np.mean(dice_scores))
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(mean_iou)
        history['val_dice'].append(mean_dice)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {mean_iou:.4f}")
        print(f"  Val Dice: {mean_dice:.4f}")
        
        # Learning rate scheduling
        scheduler.step(mean_iou)
        
        # Save best model
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_iou": mean_iou,
                "val_dice": mean_dice,
                "history": history
            }, model_save_path)
            print(f"  ✓ Saved best model (IoU: {mean_iou:.4f})")
        print("-" * 50)
    
    return model, history