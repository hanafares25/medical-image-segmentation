# Medical Image Segmentation App

> **Important:** This repository is an educational research demo and is **not a medical device**. It must not be used for diagnosis or clinical decisions.

## Features

- Custom U-Net implemented in PyTorch
- Binary pixel-level segmentation
- Grayscale medical-image preprocessing with OpenCV
- Dice score and Intersection-over-Union (IoU) evaluation
- Training and validation pipeline
- Model checkpoint saving
- Image inference CLI
- Streamlit web interface
- Segmentation mask overlay visualization
- Synthetic sample-data generator, so the app can be tested without private patient data
- CPU, NVIDIA CUDA, and Apple MPS device support

## Quick Start

### Windows


```text
run_windows.bat
```

It will create a virtual environment, install dependencies, generate a small synthetic dataset, train a U-Net, and launch the Streamlit application.

### Manual setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the demo dataset:

```bash
python generate_sample_data.py
```

Train:

```bash
python train.py --epochs 12
```

Launch the UI:

```bash
streamlit run streamlit_app.py
```

## Use Your Own Dataset

Store images and corresponding binary segmentation masks with matching filenames:

```text
data/
├── images/
│   ├── scan_001.png
│   ├── scan_002.png
│   └── ...
└── masks/
    ├── scan_001.png
    ├── scan_002.png
    └── ...
```

Masks should use background value `0` and foreground/region-of-interest value `255`.

Then run:

```bash
python train.py --images data/images --masks data/masks --epochs 30
```

## Command-Line Prediction

```bash
python predict.py --image data/images/sample_000.png --checkpoint checkpoints/unet_best.pt
```

The resulting overlay and mask are written to `outputs/`.

## Architecture

The U-Net follows an encoder-decoder design:

```text
Input
  ↓
Encoder: Conv → Conv → Pool
  ↓
Encoder: Conv → Conv → Pool
  ↓
Encoder: Conv → Conv → Pool
  ↓
Encoder: Conv → Conv → Pool
  ↓
Bottleneck
  ↓
Decoder + Skip Connections
  ↓
1×1 Convolution
  ↓
Segmentation Mask
```

Skip connections preserve fine spatial information from the encoder and combine it with higher-level semantic features during decoding.

## Metrics

**Dice coefficient** measures overlap between predicted and reference masks:

`Dice = 2|A ∩ B| / (|A| + |B|)`

**IoU / Jaccard index**:

`IoU = |A ∩ B| / |A ∪ B|`

Higher values indicate stronger segmentation overlap.
