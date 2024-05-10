import random
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import image
from os import listdir, makedirs
from os.path import isdir, join
from pathlib import Path
import zipfile

def extract_zip_files(container_path, folders):
    """
    Extracts ZIP files in the specified folders.
    
    Parameters:
    container_path (Path): Path to the container directory.
    folders (list): List of folder names containing ZIP files.
    
    Returns:
    list: List of paths to extracted directories.
    """
    extracted_paths = []
    for folder in folders:
        folder_path = container_path ##join(container_path, folder)
        for item in listdir(folder_path):
            if item.endswith('.zip'):
                zip_path = join(folder_path, item)
                extract_path = join(folder_path, item.replace('.zip', ''))
                if not isdir(extract_path):
                    makedirs(extract_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                extracted_paths.append(extract_path)
    return extracted_paths

def path_to_tensor(img_path, size):
    """
    Convert an image path to a tensor.
    
    Parameters:
    img_path (str): Path to the image file.
    size (int): Size to which the image is to be resized.
    
    Returns:
    ndarray: 4D tensor.
    """
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=50):
    """
    Convert image paths to tensors.
    
    Parameters:
    img_paths (list): List of image file paths.
    size (int): Size to which the images are to be resized.
    
    Returns:
    ndarray: Array of image tensors.
    """
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def load_data(container_path='datasets', folders=['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0):
    """
    Loads sign language dataset from ZIP files.
    
    Parameters:
    container_path (str): Path to the directory containing ZIP files.
    folders (list): List of folder names to process.
    size (int): Number of images to load.
    test_split (float): Proportion of data to be used for testing.
    seed (int): Random seed for shuffling data.
    
    Returns:
    tuple: Training and test datasets.
    """
    filenames, labels = [], []
    container_path = Path(__file__).resolve().parent
    print('container_path:', container_path)

    # Extract ZIP files
    extracted_paths = extract_zip_files(container_path, folders)
    
    for label, extract_path in enumerate(extracted_paths):
        images = [join(extract_path, d) for d in sorted(listdir(extract_path)) if d.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels.extend(len(images) * [label])
        filenames.extend(images)
    
    random.seed(seed)
    data = list(zip(filenames, labels))
    random.shuffle(data)
    data = data[:size]
    filenames, labels = zip(*data)

    # Convert image paths to tensors
    x = paths_to_tensor(filenames).astype('float32') / 255.0
    y = np.array(labels)

    # Split data into training and test sets
    split_index = int(len(x) * (1 - test_split))
    x_train = np.array(x[:split_index])
    y_train = np.array(y[:split_index])
    x_test = np.array(x[split_index:])
    y_test = np.array(y[split_index:])

    return (x_train, y_train), (x_test, y_test)
