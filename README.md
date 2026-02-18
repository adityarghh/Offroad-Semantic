# 🏜️ Offroad Semantic Scene Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A semantic segmentation model trained on synthetic desert environments using Duality AI's digital twin platform**

[Report Bug](https://github.com/YOUR_USERNAME/YOUR_REPO/issues) · [Request Feature](https://github.com/YOUR_USERNAME/YOUR_REPO/issues) · [Duality AI Discord](https://discord.com/invite/dualityfalcommunity)

</div>

---

## 🎯 Overview

This project is a submission for the **Duality AI Offroad Autonomy Segmentation Hackathon** — a challenge focused on training robust semantic segmentation models using synthetic data generated from Duality AI's **FalconCloud** geospatial digital twin platform.

The model is trained on annotated desert environment images and evaluated on a novel (unseen) desert scene, demonstrating how synthetic data can effectively bridge real-world data scarcity in off-road autonomy applications.

### Key Highlights

- **🌵 Synthetic Data Training**: Leverages Duality AI's Falcon digital twin platform for high-quality annotated desert scenes
- **🔍 Semantic Segmentation**: Pixel-level classification across 10 desert environment classes
- **🤖 DINOv2 Backbone**: Facebook's self-supervised ViT-S/14 pretrained on 142M images
- **📊 IoU Evaluation**: Model performance benchmarked using Intersection over Union (IoU) score
- **⚡ Optimized Inference**: Target inference speed under 50ms per image
- **📝 Comprehensive Documentation**: Full methodology, failure analysis, and reproducibility guide

---

## 👥 Team Kairo

| Name | Role | GitHub |
|------|------|--------|
| **Aditya Raj** | AI Engineering & Model Training | [@your-handle](https://github.com/your-handle) |
| **Harsh Pal** | AI Engineering & Infrastructure | [@your-handle](https://github.com/your-handle) |
| **Akshita Singh** | Documentation & Analysis | [@your-handle](https://github.com/your-handle) |
| **Fuzailur Rahman** | Documentation & Presentation | [@your-handle](https://github.com/your-handle) |

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Input Image   │
│   (RGB Desert)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │
│  & Augmentation     │
│  448×448 resize     │
│  ImageNet normalize │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────┐
│  DINOv2 ViT-S/14        │◄──── Pretrained (142M images)
│  Backbone               │
│  embed_dim = 384        │
│  patch_size = 14        │
└────────┬────────────────┘
         │  patch tokens (B, N, 384)
         ▼
┌─────────────────┐
│  MLP Decoder    │
│  384→256→128→10 │
│  + BatchNorm    │
│  + Dropout      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bilinear       │
│  Upsample       │
│  → Full Res     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Segmentation   │
│  Mask Output    │
│  (10 Classes)   │
└─────────────────┘
```

---

## ✨ Features

### Core Functionality
- **📄 Data Pipeline**: Automated loading and preprocessing for train/val/test splits
- **🎨 Visualization**: High-contrast color-coded segmentation output
- **📈 Benchmarking**: IoU, Dice, and Accuracy tracking across all training epochs
- **🔁 Checkpointing**: Automatic saving of best model weights by validation IoU
- **🧪 Failure Analysis**: Side-by-side visualizations to identify misclassification cases

### Technical Features
- **Two-Phase Training**: Frozen backbone warmup → full fine-tuning at epoch 5
- **Mixed Precision**: FP16 via `torch.cuda.amp` for faster training
- **Cosine LR Scheduling**: Smooth learning rate decay over all epochs
- **Modular Codebase**: Clean separation between training, evaluation, and visualization
- **Conda Environment**: Reproducible dependency management via the `EDU` environment
- **Cross-platform Setup**: Setup scripts for both Windows (`.bat`) and Mac/Linux (`.sh`)

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch 2.0+ |
| **Backbone** | DINOv2 ViT-S/14 (Facebook Research) |
| **Environment** | Conda (EDU) |
| **Data Source** | Duality AI FalconCloud |
| **Visualization** | Matplotlib / Pillow |
| **Experiment Tracking** | Local logs + metric graphs |

---

## 🏷️ Dataset Classes

All data is generated from Duality AI's FalconEditor across various desert environment digital twins.

| Class ID | Class Name     | Model Index | Description |
|----------|----------------|-------------|-------------|
| 100      | Trees          | 0 | Desert trees (e.g. Joshua trees) |
| 200      | Lush Bushes    | 1 | Dense, green shrubbery |
| 300      | Dry Grass      | 2 | Sparse dry grassland |
| 500      | Dry Bushes     | 3 | Dry desert shrubs |
| 550      | Ground Clutter | 4 | Small debris and mixed ground materials |
| 600      | Flowers        | 5 | Desert wildflowers |
| 700      | Logs           | 6 | Fallen logs and branches |
| 800      | Rocks          | 7 | Boulders and rocky terrain |
| 7100     | Landscape      | 8 | General ground (catch-all) |
| 10000    | Sky            | 9 | Sky pixels |

---

## 📁 Project Structure

```
offroad-segmentation/
│
├── ENV_SETUP/
│   ├── setup_env.bat          # Windows environment setup
│   └── setup_env.sh           # Mac/Linux environment setup
│
├── dataset/
│   ├── Train/
│   │   ├── rgb/               # Training RGB images
│   │   └── seg/               # Training segmentation masks
│   ├── Val/
│   │   ├── rgb/               # Validation RGB images
│   │   └── seg/               # Validation segmentation masks
│   └── testImages/            # Unseen test images (DO NOT use for training)
│
├── train_stats/               # Auto-generated after training
│   ├── training_curves.png    # Train vs Val loss
│   ├── iou_curves.png         # Validation IoU per epoch
│   ├── dice_curves.png        # Validation Dice per epoch
│   ├── all_metrics_curves.png # Combined dashboard
│   └── evaluation_metrics.txt # Final numeric results
│
├── predictions/               # Auto-generated after testing
│   ├── colorized/             # Color-coded segmentation masks
│   ├── visualizations/        # Side-by-side comparison images
│   └── test_results.txt       # Per-image IoU breakdown
│
├── segmentation_head.pth      # Trained model weights (best checkpoint)
├── train.py                   # Model training script
├── test.py                    # Model evaluation & inference script
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- A free [Falcon account](https://falcon.duality.ai/auth/sign-up?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2)
- CUDA-capable GPU recommended (CPU training is supported but slow)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Set up the Conda environment**

   **Windows (Anaconda Prompt):**
   ```bash
   cd ENV_SETUP
   setup_env.bat
   ```

   **Mac/Linux:**
   ```bash
   cd ENV_SETUP
   bash setup_env.sh
   ```
   > This creates a conda environment called `EDU` with all required dependencies.

3. **Download the dataset**

   Download from the [Duality AI Hackathon page](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2) and place the contents into the `dataset/` folder following the structure above.
   > Navigate to the **Segmentation Track** section on the dataset page.

4. **Activate the environment**
   ```bash
   conda activate EDU
   ```

---

## 💡 Usage

### Train the Model
```bash
python train.py
```
Trains the model using images from `dataset/Train/` and `dataset/Val/`. Best model saved to `segmentation_head.pth`. Graphs and metrics saved to `train_stats/`.

**Custom parameters:**
```bash
python train.py --epochs 30 --batch_size 4 --lr 0.0001 --img_size 448
```

### Evaluate on Test Images
```bash
python test.py
```
Runs inference on `dataset/testImages/` — images the model has **never seen** during training. Outputs colorized predictions and visualizations to `predictions/`.

**With ground truth IoU evaluation:**
```bash
python test.py --seg_dir dataset/Val/seg
```

---

## ⚙️ Configuration

Key hyperparameters in `train.py`:

```python
CONFIG = {
    "num_epochs":       30,    # Training epochs
    "batch_size":       4,     # Reduce to 2 if GPU runs out of memory
    "lr":               1e-4,  # Head learning rate (backbone uses lr * 0.1)
    "img_size":         448,   # Must be divisible by 14 (DINOv2 patch size)
    "unfreeze_epoch":   5,     # Epoch to unfreeze backbone for fine-tuning
}
```

---

## 📊 Results

### Baseline (10 Epochs, Frozen Backbone)

| Metric | Value |
|--------|-------|
| Val Loss | 0.8163 |
| **Val IoU** | **0.2921** |
| Val Dice | 0.4364 |
| Val Accuracy | 0.7024 |

### Optimized Run (30 Epochs, Full Fine-tuning)

| Metric | Value |
|--------|-------|
| Val Loss | _TBD_ |
| **Val IoU** | _TBD_ |
| Val Dice | _TBD_ |
| Val Accuracy | _TBD_ |

> 🎯 **Benchmark targets:** Maximize Mean IoU · Inference speed < 50ms per image

---

## 🧪 How It Works

### 1. Data Preparation
- RGB images and corresponding segmentation masks loaded from train/val splits
- Raw class pixel IDs (100, 200, ..., 10000) remapped to contiguous indices 0–9
- Augmentation applied: random flips, color jitter for generalization

### 2. Model Training — Two Phase Strategy
- **Phase 1 (Epochs 1–4):** Backbone frozen, only MLP head trains — prevents catastrophic forgetting
- **Phase 2 (Epoch 5+):** Full fine-tuning with 10× lower LR — adapts DINOv2 to desert domain
- Best model checkpoint saved automatically by validation IoU

### 3. Evaluation
- `test.py` runs inference on held-out `testImages/`
- Outputs per-image predictions, IoU scores, and colorized masks

### 4. Visualization & Failure Analysis
- Side-by-side RGB + predicted mask + ground truth (when available)
- Colorized masks using a distinct high-contrast palette per class
- Failure cases documented in `predictions/visualizations/`

---

## 🐛 Troubleshooting

**Training is slow?**
- Reduce `--batch_size` to 2
- Reduce `--img_size` to 336 (still divisible by 14)
- Verify GPU is active: `python -c "import torch; print(torch.cuda.is_available())"`

**CUDA out of memory?**
- Set `--batch_size 2` and `--img_size 336`

**Segmentation masks not loading?**
- Ensure mask filenames exactly match RGB image filenames
- Verify mask pixel values match the class IDs in the table above

---

## 🔮 Future Enhancements

- [ ] Upgrade to DINOv2 ViT-B/14 for stronger features
- [ ] UPerNet / FPN decoder for multi-scale feature fusion
- [ ] Class-balanced sampling for rare classes (Logs, Flowers)
- [ ] Test-Time Augmentation (TTA) for inference boost without retraining
- [ ] Docker containerization for full reproducibility
- [ ] Multi-environment generalization beyond desert biomes

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Duality AI](https://www.duality.ai/) for the FalconCloud digital twin platform and dataset
- [Facebook Research](https://github.com/facebookresearch/dinov2) for DINOv2
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Anaconda](https://www.anaconda.com/) for environment management

---

## 🔗 Important Links

| Resource | Link |
|----------|------|
| Create a Falcon Account | [Sign Up](https://falcon.duality.ai/auth/sign-up?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2) |
| Download Dataset | [Dataset Page](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2) |
| Discord Community | [Join Server](https://discord.com/invite/dualityfalcommunity) |

---

## 🏆 Judging Criteria

| Criteria | Points |
|----------|--------|
| IoU Score — pixel classification accuracy | 80 pts |
| Structured Findings & Detailed Reporting | 20 pts |
| **Total** | **100 pts** |

---

## 📈 Project Status

**Team**: Kairo
**Current Version**: 1.0.0
**Status**: Active Development
**Last Updated**: February 2026

---

<div align="center">

### Built by Team Kairo for the Duality AI Offroad Autonomy Segmentation Hackathon

Made with ❤️ by Aditya Raj · Harsh Pal · Akshita Singh · Fuzailur Rahman

[Duality AI](https://www.duality.ai/) · [Falcon Platform](https://falcon.duality.ai/)

</div>
