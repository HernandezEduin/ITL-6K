import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from utils.setup import set_seeds
from utils.dataloader import load_eit_dataset
from utils.filters import bandpass_filter, pca_transform, savitzky_filter, wavelet_filter
from utils.metrics import reconstruct_image, downscale_mask
from utils.metrics import compute_segmentation_metrics, compute_confusion_matrix
from utils.metrics import pr_curve_and_auprc, compute_image_metrics, compute_object_boundary_metrics
from utils.metrics import ThresholdedIoU
from models.image_reconstruction import Voltage2Image
from models.schedulers import SchedulerandTrackerCallback

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
        description='Load EIT voltage data and corresponding labelled images'
    )

    # Data loading parameters
    parser.add_argument('--data-path', type=str, default='./data', help='Root data directory')
    parser.add_argument('-e', '--experiments', nargs='+', 
                        default=['1022.1', '1022.2', '1022.3', '1022.4', '1024.5', '1024.6', '1024.7', '1025.8'],
                        help='List of experiment folder names to load (space separated)')
    parser.add_argument('--num-samples', type=int, default=722, help='Total number of samples per experiment')
    parser.add_argument('--offset-num', type=int, default=2, help='Offset to skip the first N samples')
    parser.add_argument('--num-pins', type=int, default=16, help='Number of pins in the voltage data')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution (height == width)')
    parser.add_argument('-d', '--downscale', action='store_true', help='Flag to downscale images for training')
    parser.add_argument('--downscale-resolution', type=int, default=64,
                        help='Downscaled target resolution for training/validation masks')

    parser.add_argument('--sampling-rate', type=int, default=128, help='Sampling rate for voltage data')
    parser.add_argument('--sample-id', type=int, default=0, help='Index of sample to visualize (0-based)')

    parser.add_argument('--use-subset', action='store_true', help='Use a smaller subset of the data for training.')
    parser.add_argument('--subset-percentage', type=float, default=0.5, help='Number of samples to use if --use-subset is set.')


    # Data processing parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training/testing datasets')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting.')
    parser.add_argument('--binary-threshold', type=float, default=0.5, help='Threshold to binarize the images')

    # Data filtering parameters
    parser.add_argument('--use-bandpass', action='store_true', help='Apply Adaptive Bandpass filtering to the voltage data before training.')
    parser.add_argument('--use-pca', action='store_true', help='Apply PCA filtering to the voltage data before training.')
    parser.add_argument('--pca-components', type=int, default=16*128, help='Number of PCA components to keep if PCA filtering is used.')
    parser.add_argument('--use-savgol', action='store_true', help='Apply Savitzky-Golay filtering to the voltage data before training.')
    parser.add_argument('--use-wavelet', action='store_true', help='Apply Wavelet denoising to the voltage data before training.')


    # Model training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--loss', type=str, choices=['mse', 'binary_crossentropy'], default='binary_crossentropy',
                        help='Loss function to use during training')
    parser.add_argument('--training-circles-num', type=str, default='all', choices=['all', '1', '2', '3', '4'],
                        help='Number of circles to include in training data (all or specific number)')
    parser.add_argument('--testing-circles-num', type=str, default='all', choices=['all', '1', '2', '3', '4'],
                        help='Number of circles to include in testing data (all or specific number)')
    
    # Model loading parameters
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save/load model checkpoints')
    parser.add_argument('-l', '--load-model', action='store_true', help='Flag to load a pre-trained model')
    parser.add_argument('--load-model-folder', type=str, default='modeloriginal', help='Folder name to load the pre-trained model from')

    parser.add_argument('-s', '--save-model', action='store_true', help='Flag to save the trained model')
    parser.add_argument('-m', '--save-multi-checkpoint', action='store_true', help='Flag to save the model at regular intervals during training')
    parser.add_argument('--save-every', type=int, default=100, help='Save the model every N epochs if --save-multi-checkpoint is set')
    parser.add_argument('--save-model-folder', type=str, default='modeloriginal', help='Folder name to save the trained model to')

    # Caching options to speed up repeated experiments
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

    # Use only samples with specified number of circles for training/testing
    if args.training_circles_num != 'all':
        print(f"Limiting training data to samples with {args.training_circles_num} circles.")
        num_circles = int(args.training_circles_num)
        selected_indices = [i for i, info in enumerate(exp_info_train) if info['circles'] == num_circles]

        x_train = x_train[selected_indices]
        y_train = y_train[selected_indices]
        exp_info_train = exp_info_train[selected_indices]
    
    if args.testing_circles_num != 'all':
        print(f"Limiting testing data to samples with {args.testing_circles_num} circles.")
        num_circles = int(args.testing_circles_num)
        selected_indices = [i for i, info in enumerate(exp_info_test) if info['circles'] == num_circles]

        x_test = x_test[selected_indices]
        y_test = y_test[selected_indices]
        if args.downscale: y_test_original = y_test_original[selected_indices]
        exp_info_test = exp_info_test[selected_indices]

    # convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    print(f"Train dataset samples: {x_train.shape[0]}, Test dataset samples: {x_test.shape[0]}")
    print(f"Train dataset batches: {len(train_dataset)}, Test dataset batches: {len(test_dataset)}")
    
    # -------------------------------------------------------------------
    # Compiling Model and Optimizer

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=500,
        decay_rate=0.9,
        staircase=True
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if args.save_model and args.save_multi_checkpoint:
        # Setup callbacks
        callbacks = [
            SchedulerandTrackerCallback(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(args.checkpoint_dir, f'{args.save_model_folder}_epoch{{epoch:03d}}', f'{args.save_model_folder}_epoch{{epoch:03d}}.keras'),
                save_weights_only=False,
                save_best_only=False,
                save_freq=args.save_every * len(train_dataset),  # Save every N epochs
            )
        ]
    else:
        callbacks = [SchedulerandTrackerCallback()]

    if args.load_model:
        load_path = os.path.join(args.checkpoint_dir, args.load_model_folder, f'{args.load_model_folder}.keras')
        
        if load_path is None or not os.path.exists(load_path):
            raise ValueError(f"Error! Model path {load_path} does not exist!")
        print("Loading model from:", load_path)

        model = load_model(load_path)

        print(f"Model loaded. Summary: {model.summary()}")
    else:
        model = Voltage2Image(
            input_shape=data_shape[1:], 
            output_shape=output_shape[1:]
        )

        custom_iou = ThresholdedIoU(
            num_classes=2,
            target_class_ids=[1], 
            name='seg_iou', 
            threshold=args.binary_threshold
        )

        model.compile(
            loss=args.loss, 
            optimizer=opt, 
            metrics=[
                custom_iou,
                'precision',
                'recall',
                ],
        )

        model.summary()


        train_time_start = time()
        history = model.fit(
            train_dataset, 
            epochs=args.epochs, 
            validation_data=test_dataset,
            callbacks=callbacks,
            shuffle=True
        )
        total_train_time = time() - train_time_start
        # print time in hrs, mins, secs
        hrs, rem = divmod(total_train_time, 3600)
        mins, secs = divmod(rem, 60)
        print(f"Total training time: {int(hrs)} hrs, {int(mins)} mins, {int(secs)} secs.")
        loss = history.history['loss']

        # -------------------------------------------------------------------
        # Saving Model
        if args.save_model:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            if not os.path.exists(os.path.join(args.checkpoint_dir, args.save_model_folder)):
                os.makedirs(os.path.join(args.checkpoint_dir, args.save_model_folder))
            model_path = os.path.join(args.checkpoint_dir, args.save_model_folder, f'{args.save_model_folder}.keras')
            
            print("Saving model to:", model_path)
            model.save(model_path)

            # save history object
            history_path = os.path.join(args.checkpoint_dir, args.save_model_folder, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, indent=4)


    # -------------------------------------------------------------------
    # Evaluation
    metrics = {}

    inference_start_time = time()
    reconstructed_images, binary_reconstructions = reconstruct_image(
        model,
        x_test,
        threshold=args.binary_threshold,
        upscale=args.downscale,
        upscale_size=args.resolution,
    )

    total_inference_time = time() - inference_start_time
    print(f"Total inference time on test set: {total_inference_time:.2f} seconds.")

    if args.downscale:
        y_test = y_test_original

    # ---- Segmentation metrics (Pixelwise) ----
    metrics.update({"Segmentation Metrics": 
        compute_segmentation_metrics(
            binary_reconstructions,
            y_test,
        )
    })

    confusion_matrix = compute_confusion_matrix(
        binary_reconstructions,
        y_test,
    )

    # ---- Image-Level Metrics ----
    metrics.update( {"Image-Level Metrics":
        compute_image_metrics(
            reconstructed_images,
            y_test,
        )
    })

    # --- Object-Level and Boundary-Level Metrics ----
    metrics.update( {"Object-Level and Boundary-Level Metrics":
        compute_object_boundary_metrics(
            binary_reconstructions,
            y_test,
        )
    })

    # ---- PR curve / AUPRC (Pixelwise) ----
    prec, rec, thr, auprc = pr_curve_and_auprc(
        reconstructed_images,
        y_test
    )

    metrics.update({"PR Curve / AUPRC": {
        "AUPRC": auprc,
    }})

    metrics.update(
        {"Time": {
            "Inference Time": total_inference_time,
            "Training Time": total_train_time if 'total_train_time' in locals() else None
        }})

    print("\nEvaluation Metrics:")
    print("="*50)
    
    for category, metric_dict in metrics.items():
        print(f"\n{category}:")
        print("-"*50)
        # Calculate the maximum length for alignment
        max_metric_length = max(len(metric) for metric in metric_dict.keys())
        
        for metric, value in metric_dict.items():
            # Right-align the values and pad metric names for clean columns
            print(f"{metric:<{max_metric_length}} : {value:>10.5f}")
    
    print("\nConfusion Matrix:")
    print("-"*50)
    print(confusion_matrix)

    # save metrics and confusion matrix
    if args.save_model or args.load_model:
        if args.save_model: path = os.path.join(args.checkpoint_dir, args.save_model_folder)
        elif args.load_model: path = os.path.join(args.checkpoint_dir, args.load_model_folder)

        metrics_path = os.path.join(path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        confusion_matrix_path = os.path.join(path, 'confusion_matrix.txt')
        with open(confusion_matrix_path, 'w') as f:
            f.write(np.array2string(confusion_matrix))

        # save PR curve figure
        pos_prev = (y_test.sum() / max(1, y_test.size))
        plt.plot(rec, prec, label=f'PR curve (AP={auprc:.3f})')
        plt.hlines(pos_prev, xmin=0, xmax=1, linestyles='dashed', label=f'Baseline={pos_prev:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'pr_curve.png'), dpi=200)
        plt.savefig(os.path.join(path, 'pr_curve.svg'))  # nice for the paper

        # check if history exists
        if 'history' in locals():
            # save training history figure
            plt.figure()
            plt.plot(history.history['loss'], label = 'Training loss')
            plt.plot(history.history['val_loss'], label = 'Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.tight_layout()
            plt.legend()  # Add legend elements
            plt.grid(True)
            plt.savefig(os.path.join(path, 'training_validation_loss.png'), dpi=200)
            plt.savefig(os.path.join(path, 'training_validation_loss.svg'))  # nice for the paper

            plt.figure()
            plt.plot(history.history['seg_iou'], label = 'Training IoU')
            plt.plot(history.history['val_seg_iou'], label = 'Validation IoU')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('IoU')
            plt.title('Training and Validation IoU')
            plt.tight_layout()
            plt.legend()  # Add legend elements
            plt.grid(True)
            plt.savefig(os.path.join(path, 'training_validation_iou.png'), dpi=200)
            plt.savefig(os.path.join(path, 'training_validation_iou.svg'))  # nice for the paper