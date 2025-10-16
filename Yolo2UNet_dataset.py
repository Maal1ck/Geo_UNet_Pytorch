import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def yolo_to_unet_masks(images_dir, labels_dir, output_dir, single_class=True, visualize=False):
    """
    Converts YOLO segmentation labels (.txt) to mask images for U-Net.

    Args:
        images_dir (str): Path to folder with images.
        labels_dir (str): Path to folder with YOLO labels (.txt).
        output_dir (str): Path to save mask images.
        single_class (bool): True if dataset is single-class.
        visualize (bool): True to preview masks with matplotlib.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = list(Path(images_dir).glob("*.*"))  # jpg/png/tif

    for img_path in tqdm(image_files):
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Load image to get size
        img = Image.open(img_path)
        w, h = img.size

        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Read YOLO polygons
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # skip invalid lines

                cls = int(parts[0])
                xy = np.array(parts[1:], dtype=float).reshape(-1, 2)
                xy[:, 0] *= w
                xy[:, 1] *= h
                xy = xy.astype(np.int32)

                if single_class:
                    cv2.fillPoly(mask, [xy], color=1)  # binary mask
                else:
                    cv2.fillPoly(mask, [xy], color=cls+1)  # multi-class

        # Optional visualization
        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow(mask, cmap="gray")
            plt.title(f"{img_path.name} mask preview")
            plt.show()

        # Save mask
        save_path = Path(output_dir) / f"{img_path.stem}_mask.png"
        # Multiply by 255 for easier visualization (does not affect training)
        cv2.imwrite(str(save_path), mask * 255)

        # Debug: check unique values
        # print(img_path.name, "unique mask values:", np.unique(mask))