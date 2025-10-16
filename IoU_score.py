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


def iou_score(pred, target, smooth=1e-7):
    """Calculate IoU score for binary segmentation"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    
    if union == 0:
        return 1.0  # Perfect score if both are empty
    
    return (intersection + smooth) / (union + smooth)

def dice_score(pred, target, smooth=1e-7):
    """Calculate Dice coefficient"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    intersection = (pred & target).sum()
    
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)