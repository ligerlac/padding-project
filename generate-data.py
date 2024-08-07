import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import save_data, load_data


n_pixels = 16  # number of rows and columns, images will actually have ()**2 pixels
noise_amplitude = 0    


def bivariate_gaussian(x, y, mu_x=0.5, mu_y=0.5, sig=1):
    """ Calculate the bivariate Gaussian probability density function
    Parameters:
    - x & y coordinates
    - mu_x/y: shift in x/y-direction
    - sig_x/y: with in x/y-direction
    - cor: correlation (skew)
    Returns the value of the PDF at the point (x, y)
    """
    sig /= 10
    mu_x = 2 * mu_x - 1
    mu_y = 2 * mu_y - 1
    x_ = x - mu_x
    y_ = y - mu_y
    dem = 2 * np.pi
    exp = -1/2 * (x_**2/sig + y_**2/sig)
    return 1/dem * np.exp(exp)


def main(args):
    x = np.linspace(-1, 1, n_pixels)
    y = np.linspace(-1, 1, n_pixels)
    xx, yy = np.meshgrid(x, y)
    images, labels = [], []
    for i in range(args.n):
        w = np.random.uniform(0, 1)  # 0: delta peak, 1: rather wide
        cx = np.random.uniform(0, 1)  # 0: left edge, 1: right edge
        cy = np.random.uniform(0, 1)  # 0: top edge, 1: bottom edge

        img = bivariate_gaussian(xx, yy, mu_x=cx, mu_y=cy, sig=w)
        img += bivariate_gaussian(xx, yy - 2, mu_x=cx, mu_y=cy, sig=w)
        img += bivariate_gaussian(xx, yy + 2, mu_x=cx, mu_y=cy, sig=w)
        img /= img.sum()  # normalize
        img += noise_amplitude * np.random.rand(*img.shape)
        images.append(img)
        labels.append((w, cx, cy))

        # plt.imshow(img)
        # plt.show()
    
    save_data(images, labels, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='data/my_file.npy', help='path to output file')
    parser.add_argument('-n', type=int, default=100, help='number of generated examples')
    main(parser.parse_args())
