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

import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data

from tensorflow.keras.layers import (
    Layer,
    Activation,
    AveragePooling2D,
    MaxPooling2D,
    Conv2D,
    ZeroPadding2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    BatchNormalization,
)

class CircularPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(CircularPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        padding_height, padding_width = self.padding
        if padding_width > 0:
            inputs = tf.concat([inputs[:, :, -padding_width:], inputs, inputs[:, :, :padding_width]], axis=2)
        if padding_height > 0:
            inputs = tf.concat([inputs[:, -padding_height:, :], inputs, inputs[:, :padding_height, :]], axis=1)
        return inputs

class Padding:

    def get_model(self):
        inputs = Input((16, 16, 1), name="inputs_")
        outputs = ZeroPadding2D(padding=(2, 0))(inputs)
        return Model(inputs, outputs, name="padding")


class CyclicPadding:

    def get_model(self):
        inputs = Input((16, 16, 1), name="inputs_")
        outputs = CircularPadding2D(padding=(2, 0))(inputs)
        return Model(inputs, outputs, name="cyclic_padding")



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
