import numpy as np


def save_data(images, labels, path):
    with open(f'{path}/images.npy', 'wb') as f:
        np.save(f, images)
    with open(f'{path}/labels.npy', 'wb') as f:
        np.save(f, labels)


def load_data(path) -> tuple:
    """
    load data from path and return it in the following format:
    (images, labels)
    """
    with open(f'{path}/images.npy', 'rb') as f:
        images = np.load(f)
    with open(f'{path}/labels.npy', 'rb') as f:
        labels = np.load(f)
    return images, labels
