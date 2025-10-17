
# 🌍 Geo_UNet_PyTorch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)  
[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-green.svg)](LICENSE)  
[![Made with ❤️ by Maal1ck](https://img.shields.io/badge/Made%20with%20❤️%20by-Maal1ck-red)](https://github.com/Maal1ck)

---

## 🧠 Overview

**Geo_UNet_PyTorch** is a modular and geospatially-aware implementation of the **U-Net architecture** for **semantic segmentation** on **aerial and satellite imagery**.  
It provides an end-to-end workflow — from data preparation and tiling, to model training, evaluation, and geospatial export of predictions.

The pipeline is optimized for **remote sensing and precision agriculture** use cases, where accurate delineation of features such as **tree crowns**, **crop parcels**, or **land cover types** is required.

---

## 🏗️ Pipeline Architecture

```text
          ┌────────────────────────────┐
          │       Input Imagery        │
          │   (Orthomosaic / Raster)   │
          └──────────────┬─────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Data Tiling &  │
               │ Preprocessing   │
               └─────────────────┘
                         │
                         ▼
          ┌────────────────────────────┐
          │    U-Net Training Loop     │
          │ (Geo_UNet_PyTorch.ipynb)   │
          └────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Model Inference  │
              │  (Segmentation)  │
              └──────────────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │ Post-processing & GeoExport  │
        │ (Raster Mask → GPKG / SHP)   │
        └──────────────────────────────┘
```

---

## 📂 Repository Structure

```
Geo_UNet_PyTorch/
│
├── Geo_UNet_PyTorch.ipynb     # Main training and inference notebook
├── data_preparation.py        # Tiling, normalization, and dataset handling
├── model.py                   # U-Net model definition (PyTorch)
├── utils.py                   # Helper functions (metrics, visualization, etc.)
├── requirements.txt           # Dependencies
├── LICENSE                    # GPL-2.0 license
└── README.md                  # Project documentation
```

---

## ⚙️ Key Features

| Feature | Description |
|----------|--------------|
| 🧩 **U-Net architecture** | Fully convolutional encoder–decoder for segmentation |
| 🗺️ **Geospatial support** | Handles georeferenced TIFFs, exports predictions as `.gpkg` or `.tif` |
| 🧱 **Tile-based training** | Efficiently handles large images through tiling and batching |
| 📈 **Metrics & Visualization** | IoU, F1, precision/recall + matplotlib visualization |
| 🔄 **Flexible inference** | Works with RGB or multispectral inputs |
| 🧠 **Customizable backbone** | Easily swap encoder with ResNet, EfficientNet, etc. |

---

## 🧩 Example Workflow

```bash
# 1️⃣ Prepare data tiles
python data_preparation.py   --input path/to/orthomosaic.tif   --labels path/to/annotations.tif   --tile_size 512

# 2️⃣ Train U-Net model
# (inside Geo_UNet_PyTorch.ipynb)
python train.py --epochs 50 --batch_size 8 --lr 0.0001

# 3️⃣ Run inference and export segmentation map
python inference.py   --model checkpoints/best_model.pth   --input path/to/orthomosaic.tif   --output outputs/segmentation_mask.tif
```

---

## 📊 Model Details

- **Architecture:** U-Net (Configurable encoder)  
- **Framework:** PyTorch  
- **Loss Functions:** BCE, Dice, or hybrid (Dice + BCE)  
- **Optimizers:** Adam / AdamW  
- **Metrics:** IoU, Dice coefficient, Pixel accuracy  
- **Data Augmentation:** Flip, rotation, normalization via Albumentations  

---

## 📸 Applications

| Domain | Example Use Case |
|---------|------------------|
| 🌾 Agriculture | Crop and tree crown segmentation |
| 🌳 Forestry | Canopy cover mapping and tree species delineation |
| 🛰️ Remote Sensing | Land cover classification from orthomosaics |
| 🏙️ Urban Analysis | Building footprint extraction and impervious surface mapping |

---

## 🔬 Research Context

**Geo_UNet_PyTorch** contributes to ongoing research in **geospatial deep learning**, particularly for high-resolution Earth observation data.

It has been used alongside other architectures (YOLOv8, Mask R-CNN) in comparative experiments for:

- Tree crown segmentation  
- Crop parcel delineation  
- Orchard mapping using aerial imagery  

📄 Related publication:  
*High-Precision Mango Orchard Mapping Using a Deep Learning Pipeline Leveraging Object Detection and Segmentation* (2024)

---

## 🧭 Future Work

- [ ] Integration with **DeepLearningSolutions** (YOLO + U-Net hybrid workflow)  
- [ ] Multi-class and multispectral segmentation  
- [ ] Support for **ONNX** / **TorchScript** deployment  
- [ ] Cloud-ready pipeline (AWS / GEE integration)  
- [ ] Semi-supervised learning for limited annotation datasets  

---

## 🧾 Citation

If you use this repository in your research, please cite it as:

```bibtex
@software{dieye2025geounetpytorch,
  author       = {El Hadji Malick Dieye},
  title        = {Geo_UNet_PyTorch: A Geospatial U-Net Implementation for Semantic Segmentation in Remote Sensing},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/Maal1ck/Geo_UNet_PyTorch},
  license      = {GPL-2.0}
}
```

---

## 👨‍💻 Author

**El Hadji Malick DIEYE**  
🎓 Master’s Student in Space Science & Technologies (CRASTE-LF)  
🌍 National Point of Contact – Senegal, **SGAC**  
🔗 [LinkedIn](https://linkedin.com/in/maal1ck) • [GitHub](https://github.com/Maal1ck)

---

## 📜 License

Licensed under the **GNU General Public License v2.0 (GPL-2.0)**.  
You are free to use, modify, and redistribute this work under the same license.
