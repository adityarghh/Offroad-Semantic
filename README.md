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
- **📊 IoU Evaluation**: Model performance benchmarked using Intersection over Union (IoU) score
- **⚡ Optimized Inference**: Target inference speed under 50ms per image
- **📝 Comprehensive Documentation**: Full methodology, failure analysis, and reproducibility guide

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Input Image   │
│   (RGB Desert)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  & Augmentation │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Segmentation   │◄──── Pretrained Backbone
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pixel-wise     │
│  Class Logits   │
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
- **📈 Benchmarking**: IoU score tracking across training runs and model versions
- **🔁 Checkpointing**: Automatic saving of model weights and training logs to `runs/`
- **🧪 Failure Analysis**: Tools to identify and document misclassification cases

### Technical Features
- **Modular Codebase**: Clean separation between training, evaluation, and visualization
- **Conda Environment**: Reproducible dependency management via the `EDU` environment
- **Cross-platform Setup**: Setup scripts for both Windows (`.bat`) and Mac/Linux (`.sh`)
- **Configurable Hyperparameters**: Easily tune batch size, learning rate, and model selection

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch |
| **Environment** | Conda (EDU) |
| **Data Source** | Duality AI FalconCloud |
| **Visualization** | Matplotlib / OpenCV |
| **Experiment Tracking** | Local logs + loss graphs |

---

## 🏷️ Dataset Classes

All data is generated from Duality AI's FalconEditor across various desert environment digital twins.

| Class ID | Class Name     | Description |
|----------|----------------|-------------|
| 100      | Trees          | Desert trees (e.g. Joshua trees) |
| 200      | Lush Bushes    | Dense, green shrubbery |
| 300      | Dry Grass      | Sparse dry grassland |
| 500      | Dry Bushes     | Dry desert shrubs |
| 550      | Ground Clutter | Small debris and mixed ground materials |
| 600      | Flowers        | Desert wildflowers |
| 700      | Logs           | Fallen logs and branches |
| 800      | Rocks          | Boulders and rocky terrain |
| 7100     | Landscape      | General ground (catch-all) |
| 10000    | Sky            | Sky pixels |

---

## 📁 Project Structure

```
offroad-segmentation/
│
├── ENV_SETUP/
│   ├── setup_env.bat          # Windows environment setup
│   └── setup_env.sh           # Mac/Linux environment setup
│
├── data/
│   ├── train/                 # Training RGB + segmented image pairs
│   ├── val/                   # Validation RGB + segmented image pairs
│   └── testImages/            # Unseen test images (DO NOT use for training)
│
├── runs/                      # Auto-generated: logs, checkpoints, outputs
│
├── train.py                   # Model training script
├── test.py                    # Model evaluation & inference script
├── visualize.py               # Segmentation visualization tool
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- A free [Falcon account](https://falcon.duality.ai/auth/sign-up?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2)

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

   Download from the [Duality AI Hackathon page](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=GHR2) and place the contents into the `data/` folder.
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
Trains the model using images from `data/train/` and `data/val/`. Logs and checkpoints are saved automatically to `runs/`.

### Evaluate on Test Images
```bash
python test.py
```
Runs inference on `data/testImages/` — images the model has **never seen** during training. Outputs predictions, loss metrics, and IoU score.

### Visualize Segmentation Output
```bash
python visualize.py
```
Generates high-contrast color-coded segmentation overlays for visual inspection and failure analysis.

---

## ⚙️ Configuration

Key hyperparameters can be adjusted directly in `train.py`:

```python
# Data loading
batch_size = 8            # Reduce if training is too slow

# Training
num_epochs = 50
learning_rate = 1e-4

# Model
model_name = "your_model" # Swap in your architecture of choice
```

---

## 📊 Results

| Metric              | Baseline | Best Run |
|---------------------|----------|----------|
| Mean IoU Score      | _TBD_    | _TBD_    |
| Inference Speed     | _TBD_ ms | _TBD_ ms |

> 🎯 **Benchmark targets:** Maximize Mean IoU · Inference speed < 50ms per image

---

## 🧪 How It Works

### 1. Data Preparation
- RGB images and corresponding segmentation masks are loaded from train/val splits
- Data augmentation is applied to improve generalization across unseen environments

### 2. Model Training
- The segmentation model is trained to classify every pixel into one of 10 desert classes
- Training loss and IoU are tracked each epoch and saved to `runs/`

### 3. Evaluation
- `test.py` evaluates the model on the held-out `testImages/` set
- Predictions, loss metrics, and final IoU score are reported

### 4. Visualization & Analysis
- `visualize.py` renders high-contrast color maps for easy visual inspection
- Failure cases (misclassified regions) are documented for iterative improvement

---

## 📊 Performance Considerations

- **Batch Size**: Reduce if you run into memory issues during training
- **Augmentation**: Stronger augmentation improves generalization to novel desert environments
- **Backbone Choice**: Larger backbones improve accuracy but increase inference time
- **Class Imbalance**: Some classes (e.g. Logs, Flowers) are rare — consider weighted loss functions

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Multi-environment support**: Generalize across biomes beyond desert terrain
- [ ] **Conversation / run tracking**: Log and compare multiple training experiments
- [ ] **Source highlighting**: Visualize exact regions contributing to each prediction
- [ ] **Local model support**: Run inference without cloud dependencies
- [ ] **Ensemble methods**: Combine predictions from multiple model architectures
- [ ] **Multilingual documentation**: Broaden accessibility for international contributors

### Technical Improvements
- [ ] Docker containerization for easy reproducibility
- [ ] Unit and integration tests
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Automated failure case report generation
- [ ] Performance monitoring and structured logging

---

## 🐛 Known Limitations

- Training data is limited to synthetic desert environments — no real-world images
- No OCR support; models rely on pixel-level segmentation masks only
- `setup_env.bat` is Windows-only; Mac/Linux users must use the equivalent `.sh` script
- Single-environment processing per training run
- Performance depends on GPU availability

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**

- GitHub: [@your-handle](https://github.com/your-handle)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- [Duality AI](https://www.duality.ai/) for the FalconCloud digital twin platform and dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Anaconda](https://www.anaconda.com/) for environment and dependency management
- [Facebook Research FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search utilities

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

**Current Version**: 1.0.0  
**Status**: Active Development  
**Last Updated**: February 2026

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ for the [Duality AI Offroad Autonomy Segmentation Hackathon](https://www.duality.ai/)

</div>
