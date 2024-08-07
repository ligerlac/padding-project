import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

from utils import load_data


def main(args):
    images, labels = load_data(args.input)
    feature_names = list(labels.keys())

    for i in range(5):
        x = random.randint(0, len(images)-1)
        im = images[x]
        title_dict = {name: round(labels[name][x], 2) for name in feature_names}
        plt.imshow(im)
        plt.title(title_dict)
        plt.show()

    my_mean = np.mean(images, axis=0)
    plt.imshow(my_mean)
    plt.show()

    for name in feature_names:
        plt.hist(labels[name])
        plt.title(f'distribution of {name}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/', help='path to input file')
    main(parser.parse_args())
    