import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from sklearn.metrics import precision_recall_curve as sk_pr_curve, average_precision_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import label as sp_label, center_of_mass as sp_com
from scipy.ndimage import binary_erosion, distance_transform_edt

from typing import Dict, Union, Tuple, List, Optional

#----------------------------------------------------------------
'Image Reconstruction Metrics and Utilities using Numpy, SciPy, Scikit-learn, and Skimage'

# ---- Image Reconstruction Utility ----
def reconstruct_image(
    model: tf.keras.Model,
    input_data: np.ndarray,
    threshold: float = 0.5,
    upscale: bool = False,
    upscale_size: Optional[int] = None,
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
    if upscale:
        assert upscale_size is not None, "upscale_size must be provided when upscale is True"
        reconstructed_images = upscale_images(reconstructed_images, upscale_size)
    binary_reconstruction = np.where(reconstructed_images >= threshold, 1, 0)
    return reconstructed_images, binary_reconstruction


def downscale_mask(
    masks: np.ndarray, 
    size: int
) -> np.ndarray:
    """
    Downscale masks to the target resolution.

    Args:
        masks (np.ndarray): Input binary masks of shape (N, H, W) or (N, H, W, 1).
        size (int): Target size for both height and width.
    Returns:
        np.ndarray: Downscaled binary masks of shape (N, size, size).
    """
    t = tf.convert_to_tensor(masks, dtype=tf.float32)
    if t.shape.rank == 3:  # (N,H,W) -> (N,H,W,1)
        t = t[..., None]
    t = tf.image.resize(t, [size, size], method='area')
    t = tf.cast(t >= 0.5, tf.float32)
    return t.numpy().squeeze(3)

# Upscale predictions to original resolution for evaluation/plots
def upscale_images(
    rec: np.ndarray, 
    size: int, 
) -> np.ndarray:
    """
    Upscale the input images to the target resolution.

    Args:
        rec (np.ndarray): Input reconstructed images of shape (N, H, W) or (N, H, W, 1).
        size (int): Target size for both height and width.
    Returns:
        np.ndarray: Upscaled reconstructed images of shape (N, size, size) or (N, size, size, 1).
    """
    target = [size, size]
    rec_up = tf.image.resize(tf.convert_to_tensor(rec[..., None], dtype=tf.float32), target, method='bilinear').numpy()
    return rec_up.squeeze(3)

# ---- Segmentation Metrics ----
def compute_segmentation_metrics(
    binary_reconstruction: np.ndarray,
    image_labels: np.ndarray, 
) -> Dict[str, float]:
    """
    Compute segmentation metrics between binary reconstruction and ground truth labels.
    
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    TN = sum((y_true == 0) & (y_pred == 0))
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    IoU = TP / (TP + FP + FN)

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
    iou = jaccard_score(image_labels_flat,binary_reconstruction_flat, average='binary') # pixel-wise IoU

    return {
        "Accuracy": accu,
        "IoU": iou,
        "Precision": accu1,
        "Recall": accu2,
        "F1-Score": f1,
    }


def compute_confusion_matrix(
    binary_reconstruction: np.ndarray,
    image_labels: np.ndarray, 
) -> np.ndarray:
    """
    Compute confusion matrix between binary reconstruction and ground truth labels.

    Confusion Matrix Format:
                    Predicted Positive   Predicted Negative
    Actual Positive        TP                  FN
    Actual Negative        FP                  TN

    Args:
        binary_reconstruction (np.ndarray): Model's binary output images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        np.ndarray: Confusion matrix as a 2D numpy array.
    """


    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)

    return confusion_matrix(image_labels_flat, binary_reconstruction_flat)


# ---- Image Quality Metrics -----
def _to_bhw(
    x: np.ndarray
) -> np.ndarray:
    """
    Convert input array to (N,H,W) format and clip to [0,1].
    
    Args:
        x (np.ndarray): Input array of shape (H,W), (N,H,W),
                        or (N,H,W,1).
    Returns:
        np.ndarray: Converted array of shape (N,H,W) with values in [0,1].
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:          # (H,W) -> (1,H,W)
        x = x[None, ...]
    elif x.ndim == 4 and x.shape[-1] == 1:  # (N,H,W,1) -> (N,H,W)
        x = x[..., 0]
    assert x.ndim == 3, f"Expected (N,H,W) or (N,H,W,1); got {x.shape}"
    return np.clip(x, 0.0, 1.0)


def _pick_ssim_win_size(
    h: int, 
    w: int
) -> int:
    """
    SSIM requires an odd win_size (default 7). If the image is small,
    choose the largest odd <= min(h,w). Minimum valid is 3.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
    Returns:
        int: Appropriate window size for SSIM computation.
    """
    m = max(3, min(h, w))
    if m % 2 == 0:
        m -= 1
    return max(3, m)


def compute_image_metrics(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute various image quality metrics between reconstructed images and ground truth labels.
    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        dict: A dictionary containing MSE, PSNR, SSIM, and CNR values.
    """

    mse_value = compute_MSE(reconstructed_images, image_labels)
    mae_value = compute_MAE(reconstructed_images, image_labels)
    psnr_value = compute_PSNR_batch(reconstructed_images, image_labels)
    ssim_value = compute_SSIM_batch(reconstructed_images, image_labels)
    cnr_value = compute_CNR(reconstructed_images, image_labels)

    return {
        "MSE": mse_value,
        "MAE": mae_value,
        "PSNR (dB)": psnr_value,
        "SSIM": ssim_value,
        "CNR": cnr_value,
    }


def compute_MSE(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.

    Mean Squared Error (MSE) is calculated as:
        MSE = (1/n) * Σ (Y_true - Y_pred)^2
    where n is the number of pixels, Y_true is the ground truth image, and Y_pred is the reconstructed image.

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
    image_labels: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Error (MAE) between two images.

    Mean Absolute Error (MAE) is calculated as:
        MAE = (1/n) * Σ |Y_true - Y_pred|
    where n is the number of pixels, Y_true is the ground truth image, and Y_pred is the reconstructed image.

    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: MAE value between the two images.
    """
    mae_value = np.mean(np.abs(image_labels -  reconstructed_images))
    return mae_value


def compute_PSNR(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is calculated as:
        PSNR = 10 * log10((MAX_I^2) / MSE)
    where MAX_I is the maximum possible pixel value (255 for 8-bit images) and MSE is the mean squared error between the two images.

    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: PSNR value between the two images.
    """
    psnr_value = peak_signal_noise_ratio(image_labels, reconstructed_images, data_range=1.0)
    return psnr_value


def compute_PSNR_batch(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
    reduction: str = "mean",  # "mean" or "none"
):
    """
    Per-image PSNR with data_range=1.0 (because inputs are in [0,1]).
    Returns float (mean) or np.ndarray (N,) if reduction == 'none'.

    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
        reduction (str): Reduction method, either "mean" or "none".
    Returns:
        Union[float, np.ndarray]: PSNR value(s) between the two images.
    """
    preds = _to_bhw(reconstructed_images)
    gts   = _to_bhw(image_labels)
    assert preds.shape == gts.shape, f"Shape mismatch: {preds.shape} vs {gts.shape}"

    N = preds.shape[0]
    psnrs = np.empty(N, dtype=np.float32)
    for i in range(N):
        psnrs[i] = peak_signal_noise_ratio(
            gts[i], preds[i], data_range=1.0
        )
    return float(psnrs.mean()) if reduction == "mean" else psnrs


def compute_SSIM(
    reconstructed_image: np.ndarray,
    image_labels: np.ndarray,
) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two images.

    SSIM is calculated based on luminance, contrast, and structure comparisons between the two images.
        Specifically, it considers the following components:
        - Luminance: The brightness of the images.
        - Contrast: The contrast of the images.
        - Structure: The structural information of the images.

    Args:
        reconstructed_image (np.ndarray): Model's reconstructed image.
        image_label (np.ndarray): Ground truth labelled image.
    Returns:
        float: SSIM value between the two images.
    """
    ssim_value = structural_similarity(
        image_labels, 
        reconstructed_image, 
        data_range=1.0
    )
    return ssim_value


def compute_SSIM_batch(
    reconstructed_images: np.ndarray,
    image_labels: np.ndarray,
    reduction: str = "mean",  # "mean" or "none"
) -> Union[float, np.ndarray]:
    """
    Per-image SSIM (skimage), grayscale (channel_axis=None), data_range=1.0.
    Returns float (mean) or np.ndarray (N,) if reduction == 'none'.

    Args:
        reconstructed_images (np.ndarray): Model's reconstructed images.
        image_labels (np.ndarray): Ground truth labelled images.
        reduction (str): Reduction method, either "mean" or "none".
    Returns:
        Union[float, np.ndarray]: SSIM value(s) between the two images.
    """
    preds = _to_bhw(reconstructed_images)
    gts   = _to_bhw(image_labels)
    assert preds.shape == gts.shape, f"Shape mismatch: {preds.shape} vs {gts.shape}"

    N, H, W = preds.shape
    win_size = _pick_ssim_win_size(H, W)

    ssim_vals = np.empty(N, dtype=np.float32)
    for i in range(N):
        ssim_vals[i] = structural_similarity(
            gts[i], preds[i],
            data_range=1.0,
            channel_axis=None,     # grayscale
            win_size=win_size,     # safe for small images
            gaussian_weights=True  # common & stable
        )
    return float(ssim_vals.mean()) if reduction == "mean" else ssim_vals


def compute_CNR(
    reconstructed_images: np.ndarray, 
    gt_mask: np.ndarray,
) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR) between two images.

    CNR is calculated as:
        CNR = (mean_signal - mean_background) / (std_background + 1e-8)

    Args:
        binary_reconstruction (np.ndarray): Model's binary output images.
        image_labels (np.ndarray): Ground truth labelled images.
    Returns:
        float: CNR value between the two images.
    """

    rec = np.asarray(reconstructed_images, dtype=np.float32)
    gt  = (np.asarray(gt_mask) > 0)

    # If batched, compute per-image then mean
    if rec.ndim == 4 and rec.shape[-1] == 1:
        rec = rec[..., 0]
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    if rec.ndim == 3:  # (N,H,W)
        vals = []
        for r, g in zip(rec, gt):
            sig = r[g]
            bg  = r[~g]
            if sig.size == 0 or bg.size == 0:
                continue
            vals.append(abs(sig.mean() - bg.mean()) / (bg.std() + 1e-8))
        return float(np.mean(vals)) if vals else float("nan")
    else:              # (H,W)
        sig = rec[gt]; bg = rec[~gt]
        return float(abs(sig.mean() - bg.mean()) / (bg.std() + 1e-8)) if sig.size and bg.size else float("nan")


# ---- Object-level and Boundary-level Metrics ----
def _area(
    mask: np.ndarray,
    spacing=(1.0, 1.0)
) -> float:
    """
    Area of LCC (SciPy) or of all foreground (fallback).

    Args:
        mask (np.ndarray): Binary mask of the object.
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm).
    Returns:
        float: Area of the largest connected component in physical units.
    """
    pix_area = spacing[0] * spacing[1]
    lcc = _largest_cc(mask)
    return float(lcc.sum()) * pix_area


def _centroid(
    mask: np.ndarray, 
    spacing=(1.0, 1.0)
) -> Union[Tuple[float, float], None]:
    """
    Centroid in (row, col) with optional physical spacing (row_mm, col_mm).
    If SciPy is available and there are multiple components, centroid of LCC; otherwise centroid of all foreground pixels.

    Args:
        mask (np.ndarray): Binary mask of the object.
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm).
    Returns:
        Union[Tuple[float, float], None]: Centroid coordinates or None if no foreground.
    """
    mask = (mask > 0)
    if mask.sum() == 0:
        return None  # no foreground
    # center_of_mass expects weights; mask.astype(float) is fine
    r, c = sp_com(mask.astype(float))
    # convert to physical space if spacing provided
    return (r * spacing[0], c * spacing[1])


def _directed_surface_distances(
    A: np.ndarray, 
    B: np.ndarray, 
    spacing=(1.0,1.0),
) -> np.ndarray:
    """
    Directed minimal distances from surface A to surface B.

    Args:
        A (np.ndarray): Binary mask of object A.
        B (np.ndarray): Binary mask of object B.
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm).
    Returns:
        np.ndarray: Array of distances from each surface point in A to the nearest surface point in B.
    """
    A_surf = _surface(A)
    B_surf = _surface(B)
    if A_surf.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if B_surf.sum() == 0:
        # all points in A to "infinite" distance; treat as large
        return np.array([np.inf], dtype=np.float32)

    # Compute distance to B surface via EDT on ~B_surf (distance to nearest True in B_surf)
    # Trick: EDT gives distance to zeros. We invert B_surf so that B_surf==False becomes 0 “targets”.
    # Simpler: distance to B_surf can be computed by EDT on ~B_surf and sample at A_surf.
    # But we want distance to nearest True pixel; an alternative is EDT on (~B_surf) with sampling rules:
    # Instead, compute EDT on the complement of B_surf==False? Use the standard approach:
    # Make an array where 0 at B_surf pixels; 1 elsewhere. EDT computes distance to zeros.
    target = np.ones_like(B_surf, dtype=bool)
    target[B_surf] = False
    dt = distance_transform_edt(target, sampling=spacing)
    return dt[A_surf].astype(np.float32)


def _largest_cc(
    mask: np.ndarray
) -> np.ndarray:
    """
    Return the largest connected component (LCC) of the binary mask.

    Args:
        mask (np.ndarray): Binary mask of the object.
    Returns:
        np.ndarray: Binary mask of the largest connected component.
    """
    m = (mask > 0).astype(np.uint8)
    lbl, n = sp_label(m)
    if n == 0:
        return np.zeros_like(m, dtype=bool)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return (lbl == sizes.argmax())


def _surface(
    mask: np.ndarray
) -> np.ndarray:
    """
    Return a boolean mask of boundary pixels.

    Args:
        mask (np.ndarray): Binary mask of the object.
    Returns:
        np.ndarray: Binary mask of the boundary pixels.
    """
    mask = (mask > 0)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask)
    return mask ^ er


def compute_object_boundary_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute object-level and boundary-level metrics between predicted and ground truth masks.

    Args:
        pred_mask (np.ndarray): Predicted binary masks of shape (N, H, W)
        gt_mask (np.ndarray): Ground truth binary masks of shape (N, H, W)
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm) for distance calculations.
    Returns:
        dict: A dictionary containing mean centroid error, area error percentage
                HD95, and ASSD across all images.
    """
    # ---- Object-level metrics and boundary metrics (per-image, then mean) ----
    centroid_errors, area_err_pcts, hd95s, assds = [], [], [], []
    for i in range(gt_mask.shape[0]):
        gt_i   = gt_mask[i]
        pred_i = pred_mask[i]

        obj = object_metrics_from_masks(pred_i, gt_i, spacing=spacing)  # set spacing if known
        bd  = hd95_assd(pred_i, gt_i, spacing=spacing)

        centroid_errors.append(obj['centroid_error'])
        area_err_pcts.append(obj['area_error_pct'])
        hd95s.append(bd['hd95'])
        assds.append(bd['assd'])

    mean_centroid_err = float(np.mean(np.array([x for x in centroid_errors if np.isfinite(x)]))) if len(centroid_errors) else float('nan')
    mean_area_err_pct = float(np.mean(np.array([x for x in area_err_pcts if np.isfinite(x)])))   if len(area_err_pcts) else float('nan')
    mean_hd95 = float(np.mean(np.array([x for x in hd95s if np.isfinite(x)])))                   if len(hd95s) else float('nan')
    mean_assd = float(np.mean(np.array([x for x in assds if np.isfinite(x)])))                   if len(assds) else float('nan')
    
    return {
        "Centroid Error (px)": mean_centroid_err,
        "Area Error (%)": mean_area_err_pct,
        "HD95 (px)": mean_hd95,
        "ASSD (px)": mean_assd
    }


def object_metrics_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing=(1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute object-level localization & size errors.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm).
    
    Returns:
        {
          'centroid_error': float,
          'area_error_pct': float,
          'pred_area': float,
          'gt_area': float,
        }
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask   = (gt_mask   > 0).astype(np.uint8)

    gt_c = _centroid(gt_mask, spacing=spacing)
    pr_c = _centroid(pred_mask, spacing=spacing)

    if (gt_c is None) and (pr_c is None):
        centroid_err = 0.0
    elif (gt_c is None) or (pr_c is None):
        centroid_err = float('inf')  # one is missing
    else:
        dr = pr_c[0] - gt_c[0]
        dc = pr_c[1] - gt_c[1]
        centroid_err = float(np.hypot(dr, dc))

    gt_area = _area(gt_mask, spacing=spacing)
    pr_area = _area(pred_mask, spacing=spacing)

    if gt_area == 0.0:
        area_err_pct = float('inf') if pr_area > 0 else 0.0
    else:
        area_err_pct = 100.0 * (pr_area - gt_area) / gt_area

    return {
        'centroid_error': centroid_err,
        'area_error_pct': float(area_err_pct),
        'pred_area': float(pr_area),
        'gt_area': float(gt_area),
    }


def hd95_assd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing=(1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute symmetric HD95 and ASSD between binary masks.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.
        spacing (Tuple[float, float]): Physical spacing (row_mm, col_mm).
    
    Returns:
        {
          'hd95': float,
          'assd': float
        }
    """
    pred = (pred_mask > 0)
    gt   = (gt_mask   > 0)

    dists_p2g = _directed_surface_distances(pred, gt, spacing=spacing)
    dists_g2p = _directed_surface_distances(gt, pred, spacing=spacing)

    # symmetric sets of surface distances
    both = np.concatenate([dists_p2g, dists_g2p])

    # HD95: 95th percentile of the symmetric surface distances
    if np.isfinite(both).any():
        hd95 = float(np.percentile(both[np.isfinite(both)], 95))
        assd = float(np.mean(both[np.isfinite(both)]))
    else:
        hd95, assd = float('inf'), float('inf')

    return {
        'hd95': hd95,
        'assd': assd
    }


# ---- Precision-Recall Curve and AUPRC ----
def pr_curve_and_auprc(
    y_prob_flat: np.ndarray,
    y_true_flat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return precision array, recall array, thresholds array, and AUPRC.
    Inputs must be 1D and aligned.

    Args:
        y_prob_flat (np.ndarray): Flattened predicted probabilities.
        y_true_flat (np.ndarray): Flattened ground truth binary labels.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Precision, recall, thresholds, and AUPRC value.
    """
    y_true_flat = y_true_flat.astype(np.uint8).ravel()
    y_prob_flat = y_prob_flat.astype(np.float32).ravel()

    precision, recall, thresholds = sk_pr_curve(y_true_flat, y_prob_flat)
    auprc = float(average_precision_score(y_true_flat, y_prob_flat))
    return precision, recall, thresholds, auprc


#----------------------------------------------------------------
'Custom Keras Metric'

@tf.keras.utils.register_keras_serializable()
class ThresholdedIoU(tf.keras.metrics.IoU):
    """
    Custom IoU metric that thresholds predictions before computing IoU.

    This wrapper registers the metric with Keras so saved models that use
    ThresholdedIoU can be deserialized with `tf.keras.models.load_model`.
    The metric thresholds `y_pred` using the provided `threshold` and then
    delegates to the base `tf.keras.metrics.IoU` implementation which
    accumulates TP/FP/FN across batches.
    """

    def __init__(
            self,
            num_classes: int = 2,
            target_class_ids: Optional[List[int]] = None,
            name: str = 'seg_iou',
            threshold: float = 0.5,
            dtype: Optional[tf.DType] = None,
            **kwargs
        ) -> None:
        if target_class_ids is None:
            target_class_ids = [1]
        # Accept and forward extra kwargs from older saved configs (e.g., ignore_class, sparse_y_true).
        super().__init__(num_classes=num_classes, target_class_ids=target_class_ids, name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)  # Define your desired threshold

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
        ) -> None:
        # Explicitly threshold the predictions to get binary values (0 or 1)
        y_pred_thr = tf.cast(tf.math.greater(y_pred, self.threshold), dtype=tf.float32)

        # Ensure y_true is in a compatible dtype (float32 works with parent implementation)
        y_true_cast = tf.cast(y_true, dtype=tf.float32)

        # Call the parent update_state with the now-discrete values
        return super().update_state(y_true_cast, y_pred_thr, sample_weight)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'threshold': float(self.threshold),
        })
        return base_config