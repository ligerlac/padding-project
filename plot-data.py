import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data


def main(args):
    images, labels = load_data(args.input)

    print(labels)

    print(dict(labels))

    label_names = labels.keys()
    print()
    
    for im, l in zip(images, labels):
        plt.imshow(im)
        plt.title(labels)
        plt.show()

    my_mean = np.mean(images, axis=0)
    plt.imshow(my_mean)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/', help='path to input file')
    main(parser.parse_args())
    