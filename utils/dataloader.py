import os
import json
import csv
import numpy as np
import cv2

from typing import List, Tuple

def read_voltage(filepath: str, index: List[int]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Read voltage data from a CSV file and return the data for specified indices,
    preserving the order of the indices.

    Args:
        filepath (str): The path to the CSV file containing voltage data.
        index (List[int]): A list of block indices to extract (e.g., [0, 2, 5]).

    Returns:
        Tuple[List[np.ndarray], List[int]]: A list of 2D voltage arrays (float32),
                                            and the indices that were successfully read.
    """
    assert os.path.isfile(filepath), f"Error! {filepath} does not exist or is not a file!"
    print(f"Reading csv file from: {filepath}")

    # Convert list to set for fast lookup
    index_set = set(index)
    block_map = {}
    current_block = []
    block_counter = 0

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for line_idx, line in enumerate(reader):
            if line_idx % 17 == 0:
                # New block header
                if block_counter in index_set and current_block:
                    try:
                        array = np.array(current_block, dtype=np.float32)
                        block_map[block_counter] = array
                    except ValueError:
                        print(f"Warning: Failed to parse block {block_counter}")
                    current_block = []
                block_counter += 1
            else:
                if (block_counter - 1) in index_set:
                    current_block.append(line)

        # Handle last block if unfinished
        if (block_counter - 1) in index_set and current_block:
            try:
                array = np.array(current_block, dtype=np.float32)
                block_map[block_counter - 1] = array
            except ValueError:
                print(f"Warning: Failed to parse block {block_counter - 1}")

    # Reconstruct output in the original index order
    inputdata = []
    final_indices = []
    for idx in index:
        if idx in block_map:
            inputdata.append(block_map[idx])
            final_indices.append(idx)

    return inputdata, final_indices

def load_image(folderpath: str, index: List[int], offset_num: int = 0) -> np.ndarray:
    """
    Load images from a specified directory based on the provided index.
    Args:
        folderpath (str): The directory path where the images are stored.
        index (list): A list of indices corresponding to the images to be loaded.
        offset_num (int): An offset to adjust the image filenames (default is 0).
    Returns:
        np.ndarray: An array of images loaded from the specified directory.
    """
    assert os.path.isdir(folderpath), f"Error! {folderpath} does not exist!"
    images = np.array([cv2.imread(os.path.join(folderpath, f"label{i0 - offset_num}.jpg"), 0) for i0 in index], dtype=int)
    return images

def read_experiment_info(filepath: str) -> dict:
    """
    Read experiment metadata from a JSON file.
    Args:
        filepath (str): The path to the JSON file containing experiment info.
    Returns:
        dict: A dictionary containing the experiment metadata.
    """
    assert os.path.isfile(filepath), f"Error! {filepath} does not exist or is not a file!"
    with open(filepath, 'r') as f:
        info = json.load(f)
    return info

def load_eit_dataset(
    data_path: str, 
    experiments: List[str], 
    num_samples: int, 
    offset_num: int, 
    num_pins: int, 
    resolution: int, 
    sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load voltage time-series and corresponding label images from multiple
    experiment subfolders. This function centralizes the loading logic so
    higher-level scripts can request data by passing a list of experiment
    folder names.

    Inputs
    - data_path: root directory containing experiment subfolders, e.g.
        ./data/<experiment_folder>/voltage.csv and ./data/<experiment_folder>/label_vector/
    - experiments: list of folder names (strings) to load.
    - num_samples: number of samples expected per experiment (used to
        construct the index range to request from `read_voltage` and
        `load_image`). Note: the function will only return indices that
        were actually present in the files.
    - offset_num: offset applied to the index range and to image filenames
        (some datasets may start images at label0 while logical sample
        indexing begins at offset_num).
    - num_pins, resolution, sampling_rate: technical parameters used to
        pre-allocate arrays with correct shapes. They do not change how
        files are read, but they document expected dimensions.

    Returns
    - voltage_data: np.ndarray shaped (N_total, num_pins, sampling_rate*num_pins),
        where N_total is the sum of successfully loaded samples across
        all experiments. Each entry contains the raw voltage time-series
        (float values as read from CSV).
    - images: np.ndarray shaped (N_total, resolution, resolution) containing
        the corresponding label images loaded from the `label_vector` dir.
    - exp_info: np.ndarray shaped (N_total,) containing experiment metadata
        dictionaries loaded from each experiment's `info.json` file. Each
        entry corresponds to one sample in `voltage_data` and `images`.

    Behavior and notes
    - The function constructs an `index` list per experiment using
      `np.arange(offset_num, num_samples + offset_num)` and passes it to
      `read_voltage`. `read_voltage` returns both data and the filtered
      index list of samples that were actually found/parsed. The same
      filtered index list is used to load images so the two arrays stay
      aligned.
    - To accumulate data across multiple experiments we start with
      zero-sized arrays and use `np.vstack`. This keeps the function
      simple but allocates intermediate memory; for very large datasets
      consider preallocating or writing a generator-based loader.
    """

    # ------------------------------------------------------------------
    # Prepare empty containers for stacking loaded data from multiple
    # experiments. We start with zero-sized arrays and vertically stack
    # each experiment's data using `np.vstack` below.
    # ------------------------------------------------------------------
    # Image shape: (N, H, W) where N will grow as we append experiments
    img_shape = (0, resolution, resolution)
    # Voltage shape: (N, pins, samples_per_pin)
    voltage_shape = (0, num_pins, sampling_rate * num_pins)

    images = np.empty(img_shape, dtype=int)
    voltage_data = np.empty(voltage_shape, dtype=int)
    exp_info = np.empty((0,), dtype=object)

    # Loop over requested experiment folders and load data
    for folder in experiments:
        # `index` is a list of sample indices to load for this experiment.
        # `offset_num` is used because some datasets may reserve the first
        # N rows for metadata or a different numbering scheme.
        index = np.arange(offset_num, num_samples + offset_num).tolist()

        exp_path = os.path.join(data_path, folder)
        voltage_path = os.path.join(exp_path, 'voltage.csv')
        img_path = os.path.join(exp_path, 'label_vector')
        info_path = os.path.join(exp_path, 'info.json')
        # Read voltage time-series for the requested indices. `read_voltage`
        # returns a tuple: (list_of_arrays, filtered_index_list). The
        # returned `data` is a list (or array) of shape
        # (n_samples_found, n_pins, n_timepoints). The filtered `index`
        # contains only indices that were present/parsed successfully.
        data, index = read_voltage(voltage_path, index)

        # Stack vertically to append this experiment's samples. We rely on
        # numpy broadcasting of empty arrays initialized above.
        voltage_data = np.vstack((voltage_data, data))

        # Read the binarized label vectors (images) for the same indices.
        # `load_image` returns an np.ndarray shaped (n_samples_found, H, W).
        # We pass the filtered `index` so image ordering matches the
        # voltage data ordering.
        images = np.vstack((images, load_image(img_path, index, offset_num)))

        # Read experiment metadata from the info.json file.
        info = read_experiment_info(info_path)
        info = np.repeat(info, len(data))
        exp_info = np.concatenate((exp_info, info))

    return voltage_data, images, exp_info