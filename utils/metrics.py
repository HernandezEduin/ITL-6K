import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from typing import Dict, Union, Tuple

def reconstruct_image(
        model: tf.keras.Model,
        input_data: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the trained model to reconstruct images from input voltage data.
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to use for prediction.
        input_data (np.ndarray): Input voltage data for prediction.
        threshold (float): Threshold to binarize the model's output.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Reconstructed images and their binary versions.
    """

    reconstructed_images = model.predict(input_data)
    binary_reconstruction = np.where(reconstructed_images >= threshold, 1, 0)
    return reconstructed_images, binary_reconstruction

def compute_segmentation_metrics(
        binary_reconstruction: np.ndarray,
        image_labels: np.ndarray, 
    ) -> Dict[str, float]:
    """
    Compute segmentation metrics between binary reconstruction and ground truth labels.
    Args:
        binary_reconstruction (np.ndarray): Model's binary output images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """

    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)
    accu = np.mean(np.equal(image_labels_flat,binary_reconstruction_flat)) # useless because of background dominance, but kept for completeness
    accu1 = precision_score(image_labels_flat,binary_reconstruction_flat)
    accu2 = recall_score(image_labels_flat,binary_reconstruction_flat)
    f1 = f1_score(image_labels_flat,binary_reconstruction_flat)
    iou = jaccard_score(image_labels_flat,binary_reconstruction_flat, average='binary')

    metrics = {
        "Accuracy": accu,
        "Precision": accu1,
        "Recall": accu2,
        "F1-Score": f1,
        "IoU": iou
    }

    return metrics

def compute_confusion_matrix(
        binary_reconstruction: np.ndarray,
        image_labels: np.ndarray, 
    ) -> np.ndarray:
    """
    Compute confusion matrix between binary reconstruction and ground truth labels.
    Args:
        binary_reconstruction (np.ndarray): Model's binary output images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        np.ndarray: Confusion matrix as a 2D numpy array.
    """


    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)

    cm = confusion_matrix(image_labels_flat, binary_reconstruction_flat)

    return cm

def compute_MSE(
        reconstructed_images: np.ndarray,
        image_labels: np.ndarray
    ) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.
    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: MSE value between the two images.
    """
    mse_value = np.mean((image_labels -  reconstructed_images) ** 2)
    return mse_value

def compute_MAE(
        reconstructed_images: np.ndarray,
        image_labels: np.ndarray
    ) -> float:
    """
    Compute Mean Absolute Error (MAE) between two images.
    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: MAE value between the two images.
    """
    mae_value = np.mean(np.abs(image_labels -  reconstructed_images))
    return mae_value

def compute_CNR(
        binary_reconstruction: np.ndarray,
        image_labels: np.ndarray,
    ) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR) between two images.
    Args:
        binary_reconstruction (np.ndarray): Model's binary output images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: CNR value between the two images.
    """

    signal_region = image_labels[binary_reconstruction == 1]
    background_region = image_labels[binary_reconstruction == 0]

    mean_signal = np.mean(signal_region)
    mean_background = np.mean(background_region)
    std_background = np.std(background_region)

    cnr_value = np.abs(mean_signal - mean_background) / (std_background + 1e-8)  # Avoid division by zero

    return cnr_value

def compute_PSNR(
        reconstructed_images: np.ndarray,
        image_labels: np.ndarray
    ) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: PSNR value between the two images.
    """
    psnr_value = peak_signal_noise_ratio(image_labels, reconstructed_images, data_range=image_labels.max() - image_labels.min())
    return psnr_value

def compute_SSIM(
        reconstructed_image: np.ndarray,
        image_labels: np.ndarray
    ) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two images.
    Args:
        reconstructed_image (np.ndarray): Model's reconstructed image.
        image_label (np.ndarray): Ground truth labelled image.
    Returns:
        float: SSIM value between the two images.
    """
    ssim_value = structural_similarity(image_labels, reconstructed_image, data_range=image_labels.max() - image_labels.min())
    return ssim_value

def compute_SSIM_batch(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
    threshold: float = 0.5,
    use_threshold_for_prediction: bool = False,
    reduction: str = "mean"  # "mean" or "none"
) -> Union[float, np.ndarray]:
    """
    Compute SSIM for a batch of images produced by `model`.

    Args:
        model: Keras model used for prediction.
        input_data: input tensor/array shaped (batch, ...).
        image_labels: ground-truth images shaped (batch, H, W) or (batch, H, W, 1).
        threshold: if `use_threshold_for_prediction` True, binarize predictions at this value.
        use_threshold_for_prediction: if True apply thresholding to predictions before SSIM.
                                      If False, use raw model outputs (recommended).
        reduction: 'mean' to return average SSIM (float), 'none' to return per-image SSIM (np.ndarray).
    Returns:
        mean SSIM (float) if reduction == 'mean', else np.ndarray with shape (batch,)
    """

    # Ensure arrays (numpy) and float32 dtype
    preds = np.asarray(reconstructed_images, dtype=np.float32)
    labels = np.asarray(image_labels, dtype=np.float32)

    # If label images are (batch, H, W), add channel dim
    if labels.ndim == 3:
        labels = np.expand_dims(labels, -1)
    # If preds are (batch, H, W), add channel dim
    if preds.ndim == 3:
        preds = np.expand_dims(preds, -1)

    # Optionally threshold predictions to binary {0,1}
    if use_threshold_for_prediction:
        preds = (preds >= threshold).astype(np.float32)

    # Ensure both are in same value range: SSIM needs max_val param below.
    # Here we assume labels are 0/1 (binary). If they are 0..255, scale to 0..1 first.
    # If needed, normalize:
    # if labels.max() > 1.0:
    #     labels = labels / 255.0
    # if preds.max() > 1.0:
    #     preds = preds / 255.0

    # Convert to tf tensors
    t_preds = tf.convert_to_tensor(preds, dtype=tf.float32)
    t_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    # max_val should match the value range of images (1.0 for binary/normalized)
    max_val = 1.0

    # Compute SSIM: returns shape (batch,)
    ssim_per_image = tf.image.ssim(t_labels, t_preds, max_val=max_val).numpy()

    if reduction == "mean":
        return float(np.mean(ssim_per_image))
    else:
        return ssim_per_image