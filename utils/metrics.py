import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from typing import Dict

def compute_segmentation_metrics(
        model: tf.keras.Model, 
        input_data: np.ndarray, 
        image_labels: np.ndarray, 
        threshold=0.5
    ) -> Dict[str, float]:
    """
    Compute segmentation metrics for a trained model on given input data
    Args:
        model (tf.keras.Model): Trained TensorFlow Keras model to evaluate.
        input_data (np.ndarray): Input voltage data for evaluation.
        image_labels (np.ndarray): Ground truth labelled images.
        threshold (float): Threshold to binarize the model's output.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """

    res = model.predict(input_data)
    binary_reconstruction = np.where(res >= threshold, 1, 0)
    image_labels_flat = image_labels.reshape(-1)
    binary_reconstruction_flat = binary_reconstruction.reshape(-1)
    accu = np.mean(np.equal(image_labels_flat,binary_reconstruction_flat))
    accu1 = precision_score(image_labels_flat,binary_reconstruction_flat)
    accu2 = recall_score(image_labels_flat,binary_reconstruction_flat)
    f1 = f1_score(image_labels_flat,binary_reconstruction_flat)

    metrics = {
        "accuracy": accu,
        "precision": accu1,
        "recall": accu2,
        "f1-score": f1
    }

    return metrics