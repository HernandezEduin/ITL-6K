import os
import argparse
import json

import numpy as np
import cv2

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from utils.setup import set_seeds
from utils.dataloader import load_eit_dataset
from utils.filters import bandpass_filter, pca_transform, savitzky_filter, wavelet_filter
from utils.metrics import reconstruct_image, downscale_mask
from utils.metrics import compute_segmentation_metrics, compute_confusion_matrix
from utils.metrics import pr_curve_and_auprc, compute_image_metrics, compute_object_boundary_metrics

import sys

def read_options() -> argparse.Namespace:
    """Parse and return command-line options.

    Returns:
        argparse.Namespace: parsed arguments with attributes matching the
        original hard-coded variables (data_path, experiments, num_samples, ...)
    """

    # Create the parser and register arguments. Defaults preserve original
    # behavior so existing workflows won't be broken.
    parser = argparse.ArgumentParser(
        description="Evaluate a neural network model for EIT image reconstruction (ITL-6k dataset)."
    )

    # =====================
    # Data Loading Parameters
    # =====================
    parser.add_argument('--data-path', type=str, default='./data', help='Root data directory')
    parser.add_argument('-e', '--experiments', nargs='+', 
                        default=['1022.1', '1022.2', '1022.3', '1022.4', '1024.5', '1024.6', '1024.7', '1025.8'],
                        help='List of experiment folder names to load (space separated)')
    parser.add_argument('--num-samples', type=int, default=722, help='Total number of samples per experiment')
    parser.add_argument('--offset-num', type=int, default=2, help='Offset to skip the first N samples')
    parser.add_argument('--num-pins', type=int, default=16, help='Number of pins in the voltage data')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution (height == width)')
    parser.add_argument('--sampling-rate', type=int, default=128, help='Sampling rate for voltage data')
    parser.add_argument('--sample-id', type=int, default=42, help='Index of sample to visualize (0-based)')

    # =====================
    # Data Processing Parameters
    # =====================
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training/testing datasets')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting.')
    parser.add_argument('--binary-threshold', type=float, default=0.5, help='Threshold to binarize the images')
    parser.add_argument('-d', '--downscale', action='store_true', help='Flag to downscale images for training')
    parser.add_argument('--downscale-resolution', type=int, default=64, help='Downscaled target resolution for training/validation masks')
    parser.add_argument('--use-subset', action='store_true', help='Use a smaller subset of the data for training.')
    parser.add_argument('--subset-percentage', type=float, default=0.5, help='Number of samples to use if --use-subset is set.')
    parser.add_argument('--skip-pins', action='store_true', help='Skip certain pins in the voltage data.')
    parser.add_argument('--skip-every-n', type=int, default=2, help='Skip every Nth pin if --skip-pins is set.')

    # =====================
    # Data Filtering Parameters
    # =====================
    parser.add_argument('--use-bandpass', action='store_true', help='Apply Adaptive Bandpass filtering to the voltage data before training.')
    parser.add_argument('--use-pca', action='store_true', help='Apply PCA filtering to the voltage data before training.')
    parser.add_argument('--pca-components', type=int, default=16*128, help='Number of PCA components to keep if PCA filtering is used.')
    parser.add_argument('--use-savgol', action='store_true', help='Apply Savitzky-Golay filtering to the voltage data before training.')
    parser.add_argument('--use-wavelet', action='store_true', help='Apply Wavelet denoising to the voltage data before training.')

    # =====================
    # Model Training Parameters
    # =====================
    parser.add_argument('--training-circles-num', type=str, default='all', choices=['all', '1', '2', '3', '4'], help='Number of circles to include in training data (all or specific number)')
    parser.add_argument('--testing-circles-num', type=str, default='all', choices=['all', '1', '2', '3', '4'], help='Number of circles to include in testing data (all or specific number)')
    
    # =====================
    # Model Loading/Saving Parameters
    # =====================
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save/load model checkpoints')
    parser.add_argument('-l', '--load-model', action='store_true', help='Flag to load a pre-trained model')
    parser.add_argument('--load-model-folder', type=str, default='model_benchmark', help='Folder name to load the pre-trained model from')

    # metrics and evaluation parameters
    parser.add_argument('--save-metrics', action='store_true', help='Flag to save evaluation metrics to a file')

    # =====================
    # Caching Options
    # =====================
    parser.add_argument('--use-cache', action='store_true', help='Load preprocessed data from cache if available')
    parser.add_argument('--store-cache', action='store_true', help='Store preprocessed data to cache after loading')
    parser.add_argument('--rebuild-cache', action='store_true', help='Ignore cache and rebuild it from source files')
    parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory to store/load cached .npz files')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = read_options()

    print('='*20)
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    print('='*20)

    set_seeds(args.seed)

    #------------------------------------------------------------------
    # Loading Samples

    # Load EIT dataset
    voltage_data, images, exp_info = load_eit_dataset(
        data_path=args.data_path,
        experiments=args.experiments,
        num_samples=args.num_samples,
        offset_num=args.offset_num,
        num_pins=args.num_pins,
        resolution=args.resolution,
        sampling_rate=args.sampling_rate,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
        store_cache=args.store_cache,
        cache_dir=args.cache_dir
    )

    voltage_data = voltage_data[..., np.newaxis]  # Add channel dimension for TF

    # Print shapes so the user can verify successful loading
    print(f"Voltage data shape: {voltage_data.shape}")
    print(f"Images shape: {images.shape}")
    print(f"Exp info shape: {exp_info.shape}")

    # ------------------------------------------------------------------
    # Splits and Data Processing

    images = 1*(images > 100) # Binarize images

    x_train, x_test, y_train, y_test, exp_info_train, exp_info_test = train_test_split(
        voltage_data,
        images,
        exp_info,
        test_size=args.test_size,
        random_state=args.seed  # not included by set_seeds function
    )  # split the dataset

    # normalization values (along sample and time axis)
    mean = x_train.mean(axis=(0,2), keepdims = True)
    var = x_train.std(axis=(0,2), keepdims = True)

    x_train = (x_train - mean) / var
    x_test = (x_test - mean) / var

    if args.downscale:
        print(f"Downscaling images from {args.resolution}x{args.resolution} to {args.downscale_resolution}x{args.downscale_resolution}.")

        y_train = downscale_mask(y_train, args.downscale_resolution)
        y_test_original = y_test.copy()
        y_test = downscale_mask(y_test, args.downscale_resolution)

    # Use subset of the data if specified
    if args.use_subset:
        num_train_samples = int(len(x_train) * args.subset_percentage)

        x_train = x_train[:num_train_samples]
        y_train = y_train[:num_train_samples]
        exp_info_train = exp_info_train[:num_train_samples]

        print(f"Using subset of train data ({args.subset_percentage*100:.0f}%): {num_train_samples} samples.")

    if args.skip_pins:
        x_train = x_train[:, ::args.skip_every_n, :]
        x_test = x_test[:, ::args.skip_every_n, :]

        print(f"Skipping every {args.skip_every_n}th pin in the voltage data.")

    data_shape = x_train.shape
    output_shape = y_train.shape

    if args.use_bandpass:
        x_train = bandpass_filter(x_train.copy())
        x_test = bandpass_filter(x_test.copy())
    
    if args.use_savgol:
        x_train = savitzky_filter(x_train)
        x_test = savitzky_filter(x_test)

    if args.use_wavelet:
        x_train = wavelet_filter(x_train)
        x_test = wavelet_filter(x_test)

    if args.use_pca:
        x_train, x_test = pca_transform(x_train, x_test, n=args.pca_components)
        data_shape = x_train.shape

    # circle_index_dict = {1: [], 2: [], 3: [], 4: []}
    circle_index_dict = {i: [] for i in range(1,5)}
    for i0, info in enumerate(exp_info_test):
        num_circles = info['circles']
        circle_index_dict[num_circles].append(i0)

    reconstruct_indexes = [circle_index_dict[i][args.sample_id] for i in range(1,5)]
    x_test_samples = x_test[reconstruct_indexes]
    y_test_samples = y_test[reconstruct_indexes]
    # if args.downscale: y_test_original_samples = y_test_original[reconstruct_indexes]
    info_sample = [exp_info_test[i] for i in reconstruct_indexes]

    print(f"Selected sample indexes for reconstruction: {reconstruct_indexes}")
    print(f"Sample exp info: {info_sample}")
    
    # -------------------------------------------------------------------
    # Loading Model

    load_path = os.path.join(args.checkpoint_dir, args.load_model_folder, f'{args.load_model_folder}.keras')
    
    if load_path is None or not os.path.exists(load_path):
        raise ValueError(f"Error! Model path {load_path} does not exist!")
    print("Loading model from:", load_path)

    model = load_model(load_path)

    print(f"Model loaded. Summary: {model.summary()}")


    # -------------------------------------------------------------------
    # Reconstruction

    reconstructed_images, _ = reconstruct_image(
        model,
        x_test_samples,
        threshold=args.binary_threshold,
        upscale=args.downscale,
        upscale_size=args.resolution,
    )

    # -------------------------------------------------------------------

    path = os.path.join(args.checkpoint_dir, args.load_model_folder)
    for i0 in range(len(x_test_samples)):
        # save reconstructed images
        num_circles = info_sample[i0]['circles']
        cv2.imwrite(
            os.path.join(path, f'reconstructed_{num_circles}_circles.png'),
            (reconstructed_images[i0]*255).astype(np.uint8)
        )

        # cv2.imwrite(
        #     os.path.join(path, f'target_{num_circles}_circles.png'),
        #     (y_test_samples[i0]*255).astype(np.uint8)
        # )
        