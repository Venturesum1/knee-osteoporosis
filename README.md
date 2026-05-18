# Knee Osteoporosis Classification using Hybrid Semi-Supervised Multi-View Vision Transformer with Fuzzy Rule-Based Decision Fusion

## Overview

This repository presents an advanced deep learning framework for automated **knee osteoporosis detection and classification** from knee X-ray images. The proposed model integrates:

* Multi-view image preprocessing
* Pretrained Swin Transformer backbone
* Semi-supervised learning
* Mamdani fuzzy rule-based pseudo-label generation
* Iterative retraining for improved classification

The system is designed to improve early diagnosis of knee osteoporosis while effectively utilizing both labeled and unlabeled radiographic datasets.

---

## Key Features

* **Hybrid Deep Learning Architecture**

  * Multi-view image representations:

    * Original X-ray image
    * Contrast-enhanced image
    * Region-of-interest-focused image

* **Transformer-Based Feature Extraction**

  * Shared pretrained Swin Transformer backbone
  * Hierarchical local-global feature learning

* **Semi-Supervised Learning**

  * Uses both labeled and unlabeled data
  * Confidence-guided pseudo-labeling

* **Fuzzy Rule-Based Decision Fusion**

  * Mamdani fuzzy inference system
  * Improves uncertainty estimation
  * Enhances pseudo-label reliability

---


## Installation

```bash
git clone https://github.com/Venturesum1/knee-osteoporosis.git
cd knee-osteoporosis
pip install -r requirements.txt
```

---

## Usage

### Run 20 Epoch Model

```bash
python 20epochs.py
```

### Run 30 Epoch Model

```bash
python 30epochs.py
```

### Run 40 Epoch Model

```bash
python 40epochs.py
```

---

## Requirements

* Python 3.9+
* PyTorch
* torchvision
* timm
* scikit-learn
* OpenCV
* NumPy
* Matplotlib
* Pillow

---


## License

This project is intended for academic and research purposes.

---

## Contact

For research collaboration or academic inquiries:

**Author:** soumyasis
**GitHub:** [https://github.com/Venturesum1](https://github.com/Venturesum1)

---
