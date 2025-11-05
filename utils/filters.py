import numpy as np
import matplotlib.pyplot as plt

import pywt
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter

from typing import Tuple, Any

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[Any, Any]:
    """
    Create a Butterworth bandpass filter (returns numerator b and denominator a).

    Parameters
    - lowcut: lower cutoff frequency (Hz)
    - highcut: upper cutoff frequency (Hz)
    - fs: sampling frequency (Hz)
    - order: filter order

    Returns
    - b, a: filter coefficients (numpy arrays)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a 1-D signal `data`.

    Parameters
    - data: 1D numpy array of samples
    - lowcut/highcut/fs/order: filter parameters

    Returns
    - filtered signal (numpy array)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data: np.ndarray, bandwidth: float = 2.0, order: int = 4, fs: float = 128.0) -> np.ndarray:
        """
        Adaptive bandpass filter that finds a dominant frequency per-sample
        and applies a narrow bandpass around it.

        Notes
        - This function mutates `data` in-place (it assigns into data[...,0]).
            If you need to keep the original unfiltered signals, call this on a
            copy: `bandpass_filter(data.copy())`.
        - The current implementation assumes a sampling rate `fs=128` and
            that the input shape is (experiments, pins, samples, channels)
            where the time-series to filter live at index `[..., 0]`.

        Parameters
        - data: 4-D numpy array (N, pins, samples, channels)
        - bandwidth: width of the bandpass filter (Hz)
        - order: filter order
        - fs: sampling frequency (Hz)

        Returns
        - filtered data (same shape as input)
        """
        # Iterate per-sample and per-pin; compute FFT to locate dominant freq
        for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                        signal = data[i, j, :, 0]
                        fft_vals = np.fft.fft(signal)
                        freqs = np.fft.fftfreq(len(signal), 1 / fs)
                        power_spectrum = np.abs(fft_vals)
                        # Find dominant frequency component (index of max power)
                        max_power_idx = np.argmax(power_spectrum)
                        freq_of_interest = freqs[max_power_idx]
                        lowcut = freq_of_interest - bandwidth / 2
                        highcut = freq_of_interest + bandwidth / 2

                        # Apply bandpass filter based on identified frequency range
                        data[i, j, :, 0] = butter_bandpass_filter(signal, lowcut, highcut, fs, order)

        return data

"""
NOTE: this module previously contained multiple different
implementations of `pca_transform`. Only the final implementation below
is active. PCA is a learned, stateful transform (it calls
`PCA.fit_transform`) so it WILL memorize dataset-wide structure. Fit
PCA on training data only to avoid leakage.
"""

def pca_transform(
    data_train: np.ndarray, 
    data_test: np.ndarray, 
    n: int = 3, 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on the provided dataset and return the first `n` principal
    components in time.

    Warning: this function calls `PCA.fit_transform` on the entire
    `data` array. Do NOT call it on the full dataset before splitting
    into train/test if you want to avoid information leakage. Instead,
    fit PCA on training data and call `transform` on validation/test.
    Additionally, time signal = sampling_rate * num_pins.

    Parameters
    - data: 4-D array shaped (experiments, num_pins, sampling_rate * num_pins, channels) or
        shape that can be squeezed into (experiments, num_pins, sampling_rate * num_pins).
    - n: number of PCA components to keep

    Returns
    - reshaped_data_train: numpy array with shape (n, pins, (sampling_rate * pins))
      containing the top-n components for each pin.
    - reshaped_data_test: numpy array with shape (n, pins, (sampling_rate * pins))
      containing the top-n components for each pin.
    """
    experiments_train, num_pins, time_signal, _ = data_train.shape
    experiments_test, _, _, _ = data_test.shape
    data_train = data_train.squeeze()
    data_test = data_test.squeeze()

    data_train = data_train.reshape(experiments_train, num_pins * time_signal)
    data_test = data_test.reshape(experiments_test, num_pins * time_signal)

    pca = PCA(n)
    transformed_data_train = pca.fit_transform(data_train)
    transformed_data_test = pca.transform(data_test)

    reshaped_data_train = transformed_data_train.reshape(experiments_train, num_pins, n//num_pins)
    reshaped_data_test = transformed_data_test.reshape(experiments_test, num_pins, n//num_pins)

    return reshaped_data_train[..., np.newaxis], reshaped_data_test[..., np.newaxis]

def savitzky_filter(inputdata: np.ndarray) -> np.ndarray:
    """
    Apply a Savitzky-Golay smoothing filter per sample/pin.

    Parameters
    - inputdata: 4-D array (samples, pins, time, channels)

    Returns
    - smoothed array with the same shape as `inputdata`
    """
    extra_dim = False
    if len(inputdata.shape) == 4:
        if inputdata.shape[3] != 1:
            raise ValueError("Savitzky filter currently only supports single-channel data.")
        extra_dim = True
        inputdata = inputdata.squeeze(3)
    elif len(inputdata.shape) != 3:
        raise ValueError("Input data must be 3-D or 4-D array.")
         
    data = np.zeros(inputdata.shape)
    for i in range(inputdata.shape[0]):
        for j in range(inputdata.shape[1]):
            # Apply filter along time axis for channel 0
            data[i, j] = savgol_filter(inputdata[i, j], window_length=5, polyorder=3)
    
    if extra_dim:
        data = data[..., np.newaxis]

    return data


def wavelet_filter(inputdata: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Simple wavelet denoising (per-sample) using soft thresholding.

    Parameters
    - inputdata: 4-D array (samples, pins, time, channels)
    - threshold: threshold applied to wavelet coefficients

    Returns
    - denoised data with the same shape as input
    """
    extra_dim = False
    if len(inputdata.shape) == 4:
        if inputdata.shape[3] != 1:
            raise ValueError("Wavelet filter currently only supports single-channel data.")
        extra_dim = True
        inputdata = inputdata.squeeze(3)
    elif len(inputdata.shape) != 3:
        raise ValueError("Input data must be 3-D or 4-D array.")


    data = np.zeros(inputdata.shape)
    for i in range(inputdata.shape[0]):
        for j in range(inputdata.shape[1]):
            coeffs = pywt.wavedec(inputdata[i, j], 'db1', level=5)
            coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            data[i, j] = pywt.waverec(coeffs_thresholded, 'db1')

    if extra_dim:
        data = data[..., np.newaxis]

    return data

def paper_rearrangement(data: np.ndarray) -> np.ndarray:
    """
    Reorder input channels according to the pattern used in the paper.

    The function expects `data` to have shape (samples, channels, ...).
    It returns the data with channels rearranged; the final transpose
    places the output into (samples, new_channels, ...).
    """
    rearranged_input = []
    for i in range(8):
        if i == 0:
            rearranged_input.append(data[:, 0])

        rearranged_input.append(data[:, 15 - i])
        if i == 7:
            break
        else:
            rearranged_input.append(data[:, 7 - i])

    rearranged_input = np.array(rearranged_input)

    return rearranged_input.transpose((1, 0, 2, 3))