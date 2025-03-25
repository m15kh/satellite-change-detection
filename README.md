# Satellite Change Detection

A project for detecting changes between satellite images using deep learning techniques.

## Overview

This project uses deep learning models, specifically a pre-trained U-Net architecture, to detect changes between satellite images taken at different times. It can identify new structures, modifications, or other changes in satellite imagery.

## Features

- Change detection between before/after satellite images
- Implementation using U-Net architecture with pre-trained ResNet34 encoder
- Autoencoder implementation for additional analysis (in progress)
- Visualization of detected changes

## Installation

1. Clone this repository
```bash
git clone <repository-url>
cd satellite-change-detection
```

2. Install the required dependencies
```bash
pip install torch torchvision opencv-python segmentation-models-pytorch numpy
```

## Usage

### Using the Python Script

Place your before and after satellite images in the project directory and run:

```bash
python unet.py
```

The script will generate a `detected_building.png` file highlighting the changes.

### Using the Jupyter Notebook

You can also use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook main.ipynb
```

## Project Structure

- `unet.py`: Main script for change detection using U-Net
- `main.ipynb`: Jupyter notebook with interactive implementation
- `autoencoder.ipynb`: Implementation of an autoencoder (in progress)
- `.gitignore`: Git ignore file

## Requirements

- Python 3.9+
- PyTorch
- segmentation-models-pytorch
- OpenCV
- NumPy

## How It Works

1. The system loads before and after satellite images
2. Images are processed through a pre-trained U-Net model with ResNet34 backbone
3. The model generates a segmentation mask highlighting changes
4. Post-processing is applied to refine the detected changes
5. Results are visualized with the changes highlighted in the original image

## Future Work

- Complete autoencoder implementation for unsupervised change detection
- Add support for batch processing of multiple image pairs
- Implement additional models for comparison
- Enhance visualization options

## License

[Insert your license information here]

## Acknowledgements

- segmentation_models_pytorch for the pre-trained U-Net implementation
- ResNet34 architecture and ImageNet weights
