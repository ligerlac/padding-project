import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

from models import Padding, CyclicPadding
from utils import load_data


def main(args):

    X, Y = load_data('data/')

    base_model = Padding().get_model()
    cyclic_model = CyclicPadding().get_model()

    X_base = base_model.predict(X)
    X_cyclic = cyclic_model.predict(X)

    for orig, base, cyclic in zip(X, X_base, X_cyclic):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
        ax0.imshow(orig)
        ax1.imshow(base)
        ax2.imshow(cyclic)
        ax0.set_title('Original')
        ax1.set_title('Zero Padded')
        ax2.set_title('Cyclic Padded')
        plt.show()
        # pred = base_model.predict(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('-i', '--input', type=str, default='data/my_file.npy', help='path to input file (data)')
    parser.add_argument('-o', '--output', type=str, default='data/model', help='path to output file (saved model)')
    parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs', default=100)
    parser.add_argument('-v', '--verbose', action='store_true', help='Output verbosity', default=False)
    args = parser.parse_args()
    main(args)
