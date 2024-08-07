import numpy as np
import json
# from sklearn.model_selection import train_test_split


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


# def save_data(images, labels, path):
#     with open(f'{path}/images.npy', 'wb') as f:
#         np.save(f, images)
#     with open(f'{path}/labels.json', 'w') as f:
#         json.dump(labels, f)


# def load_data(path) -> tuple:
#     """
#     load data from path and return it in the following format:
#     (train_images, train_labels), (test_images, test_labels)
#     """
#     with open(f'{path}/images.npy', 'rb') as f:
#         images = np.load(f)
#     with open(f'{path}/labels.json', 'r') as f:
#         labels = json.load(f)
#     return images, labels
