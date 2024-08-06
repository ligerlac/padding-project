import numpy as np


def save_data(images, labels, path):
    with open(f'{path}/images.npy', 'wb') as f:
        np.save(f, images)
    with open(f'{path}/labels.npy', 'wb') as f:
        np.save(f, labels)


def load_data(path, do_split=False) -> tuple:
    """
    load data from path and return it in the following format:
    (train_images, train_labels), (test_images, test_labels)
    """
    return


