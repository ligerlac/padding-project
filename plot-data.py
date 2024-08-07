import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

from utils import load_data


def main(args):
    
    images, labels = load_data(args.input)

    for i in range(5):
        x = random.randint(0, len(images)-1)
        im, l = images[x], labels[x]
        plt.imshow(im)
        plt.title(f'w = {round(l[0], 2)}, cx = {round(l[1], 2)}, cy = {round(l[2], 2)}')
        plt.show()

    my_mean = np.mean(images, axis=0)
    plt.imshow(my_mean)
    plt.show()

    for i, name in enumerate(('w', 'cx', 'cy')):
        plt.hist(labels[:, i])
        plt.title(f'distribution of {name}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/', help='path to input file')
    main(parser.parse_args())
    