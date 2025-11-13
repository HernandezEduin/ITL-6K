# Electrical Impedance Tomography (EIT) Signal Reconstruction

This is the official implementation for the paper:  
**_Data-Driven Phantom Imaging in EIT with Physically Controlled Ground Truth_**

This repository contains code and documentation related to signal reconstruction using Electrical Impedance Tomography (EIT) data. The project involves processing voltage measurements from 16 electrode pins and reconstructing internal structures based on these readings using deep learning.

## 📊 Dataset

The dataset used in this project is publicly available on Kaggle:

👉 [EIT Signal Reconstruction Dataset](https://www.kaggle.com/datasets/eehernandez/eit-signal-reconstruction)

**Dataset Specifications:**
- **Voltage measurements**: 16 electrodes
- **Sampling rate**: 2048 samples per electrode per cycle
- **Phantom configurations**: 1 to 4 circular phantoms
- **Image resolution**: 128×128 pixels
- **Total experiments**: 8 configurations (1022.1-1025.8)
- **Data format**: Labeled and vectorized image data for supervised learning

## 🏗️ Project Structure

```
EIT/
├── data/                      # Raw EIT voltage and image data
│   ├── 1022.1/               # Experiment folders
│   ├── 1022.2/
│   └── ...
├── models/                    # Neural network architectures
│   ├── image_reconstruction.py  # Main reconstruction model (Voltage2Image)
│   └── schedulers.py           # Learning rate schedulers and callbacks
├── utils/                     # Utility functions and helpers
│   ├── dataloader.py          # Dataset loading and caching
│   ├── filters.py             # Signal processing filters (PCA, Savgol, Wavelet, Bandpass)
│   ├── metrics.py             # Evaluation metrics and image processing
│   └── setup.py               # Random seed configuration
├── checkpoints/               # Saved model weights and metrics
├── .cache/                    # Cached preprocessed data
├── sample_train.py            # Main training and evaluation script
├── sample_eval.py             # Standalone evaluation script
├── sample_loading.py          # Data loading demonstration
├── sample_reconstruction.py   # Reconstruction visualization
├── train.sh                   # Training script with multiple configurations
├── eval.sh                    # Evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/leonrdsa/EIT_new.git
cd EIT
```

2. Create a conda environment (recommended):
```bash
conda create -n EIT python=3.9
conda activate EIT
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/` directory.

## 🎯 Usage

### Training a Model

**Basic training:**
```bash
python sample_train.py --use-cache -t -e -s --save-model-folder my_model
```

**Training with preprocessing filters:**
```bash
# PCA filtering
python sample_train.py --use-cache -t -e -s --save-model-folder model_pca --use-pca

# Wavelet denoising
python sample_train.py --use-cache -t -e -s --save-model-folder model_wavelet --use-wavelet

# Savitzky-Golay filtering
python sample_train.py --use-cache -t -e -s --save-model-folder model_savgol --use-savgol

# Combined filters
python sample_train.py --use-cache -t -e -s --save-model-folder model_wavelet_pca --use-wavelet --use-pca
```

**Training on specific phantom configurations:**
```bash
# Train on 1-circle phantoms only
python sample_train.py --use-cache -t -e -s --save-model-folder model_circles1 --training-circles-num 1

# Train on all configurations
python sample_train.py --use-cache -t -e -s --save-model-folder model_all --training-circles-num all
```

**Training with downscaled images:**
```bash
python sample_train.py --use-cache -t -e -s --save-model-folder model_64x64 -d --downscale-resolution 64
```

**Training with periodic checkpoints:**
```bash
python sample_train.py --use-cache -t -e -s --save-model-folder model_benchmark -m --save-every 200
```

### Evaluating a Model

```bash
python sample_train.py --use-cache -l --load-model-folder model_name -e
```

### Batch Training

Use the provided shell scripts to train multiple configurations:

```bash
# Train multiple models with different preprocessing
bash train.sh

# Evaluate all models
bash eval.sh
```

## 🔧 Command-Line Arguments

### Data Loading Parameters
- `--data-path`: Root data directory (default: `./data`)
- `--experiments`: List of experiment folders to load
- `--num-samples`: Total samples per experiment (default: 722)
- `--resolution`: Image resolution (default: 128)

### Data Processing Parameters
- `--batch-size`: Batch size for training (default: 64)
- `--test-size`: Test split proportion (default: 0.2)
- `--binary-threshold`: Threshold to binarize images (default: 0.5)
- `-d, --downscale`: Downscale images for training
- `--downscale-resolution`: Target resolution for downscaling

### Data Filtering Parameters
- `--use-bandpass`: Apply bandpass filtering
- `--use-pca`: Apply PCA dimensionality reduction
- `--pca-components`: Number of PCA components (default: 2048)
- `--use-savgol`: Apply Savitzky-Golay filtering
- `--use-wavelet`: Apply wavelet denoising

### Model Training Parameters
- `-t, --train-model`: Train the model
- `-e, --eval-model`: Evaluate the model
- `--learning-rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 1000)
- `--loss`: Loss function (`mse` or `binary_crossentropy`)
- `--training-circles-num`: Filter by number of circles

### Model Loading/Saving Parameters
- `-l, --load-model`: Load a pre-trained model
- `--load-model-folder`: Folder name to load from
- `-s, --save-model`: Save the trained model
- `-m, --save-multi-checkpoint`: Save periodic checkpoints
- `--save-every`: Checkpoint frequency in epochs

### Caching Options
- `--use-cache`: Load preprocessed data from cache
- `--store-cache`: Store preprocessed data to cache
- `--rebuild-cache`: Rebuild cache from source

## 📈 Evaluation Metrics

The framework computes comprehensive metrics organized in categories:

### Segmentation Metrics (Pixelwise)
- **Accuracy**: Overall pixel-level accuracy
- **IoU (Intersection over Union)**: Jaccard index
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall

### Image-Level Metrics
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **PSNR**: Peak signal-to-noise ratio (dB)
- **SSIM**: Structural similarity index
- **CNR**: Contrast-to-noise ratio

### Object-Level and Boundary-Level Metrics
- **Centroid Error**: Distance between predicted and ground truth centroids (pixels)
- **Area Error**: Percentage difference in object area
- **HD95**: 95th percentile Hausdorff distance (pixels)
- **ASSD**: Average symmetric surface distance (pixels)

### Additional Metrics
- **AUPRC**: Area under precision-recall curve
- **Confusion Matrix**: TP, FP, TN, FN counts

## 🧠 Model Architecture

The `Voltage2Image` model is a hybrid architecture combining:
- **Conv2D + MaxPooling**: Initial feature extraction from voltage data
- **Bidirectional LSTM**: Temporal pattern learning
- **Dense layers with dropout**: Progressive spatial reconstruction
- **Final reshape**: 128×128 image output

Key features:
- Input: (16, 2048, 1) voltage measurements
- Output: (128, 128) binary reconstructed image
- Loss: Binary crossentropy or MSE
- Optimizer: Adam with exponential learning rate decay

## 📊 Results

Trained models and their metrics are saved in the `checkpoints/` directory:
- `model_name.keras`: Saved model weights
- `metrics.json`: Evaluation metrics
- `confusion_matrix.txt`: Confusion matrix
- `history.json`: Training history
- `pr_curve.png/svg`: Precision-recall curve
- `training_validation_loss.png/svg`: Loss curves
- `training_validation_iou.png/svg`: IoU curves

## 🔬 Advanced Features

### Custom Metrics
The repository includes a custom `ThresholdedIoU` metric that:
- Thresholds predictions before computing IoU
- Integrates with Keras training callbacks
- Properly serializes with saved models

### Signal Processing Filters
Multiple preprocessing options:
- **PCA**: Dimensionality reduction
- **Wavelet**: Denoising with wavelet transform
- **Savitzky-Golay**: Smoothing filter
- **Bandpass**: Frequency domain filtering

### Data Augmentation
- Pin skipping (simulate missing electrodes)
- Subset training (train on reduced dataset)
- Circle-specific training (generalization testing)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{salim2026eit,
  title={Data-Driven Phantom Imaging in EIT with Physically Controlled Ground Truth},
  author={Salim, Leonard and Hernandez, Eduin and Rini, Stefano},
  year={2026}
}
```

## 📄 License

This project is released under an **Academic License** for research and educational purposes only. Commercial use is prohibited without explicit permission from the authors.

## 👤 Authors

**Leonard Salim** (Student Member, IEEE)
- GitHub: [@leonrdsa](https://github.com/leonrdsa)

**Eduin Hernandez** (Student Member, IEEE)
- GitHub: [@HernandezEduin](https://github.com/HernandezEduin)
- Kaggle: [EIT Dataset](https://www.kaggle.com/datasets/eehernandez/eit-signal-reconstruction)

**Stefano Rini** (Member, IEEE)

## 🙏 Acknowledgments

- Dataset contributors and collaborators
- TensorFlow and Keras teams
- Scientific Python community