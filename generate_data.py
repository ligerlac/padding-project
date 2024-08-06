import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import save_data, load_data


n_pixels = 32  # number of rows and columns, images will actually have ()**2 pixels
noise_amplitude = 0    


def get_random_features():
    a = np.random.uniform(1, 2)
    cx = np.random.uniform(-1, 1)
    cy = np.random.uniform(-1, 1)
    wx = np.random.uniform(0.1, 0.2)
    wy = np.random.uniform(0.1, 0.2)
    cor = np.random.uniform(0, 1)        
    return a, cx, cy, wx, wy, cor


def bivariate_gaussian(x, y, mu_x=0, mu_y=0, sig_x=1, sig_y=1, cor=0):
    """ Calculate the bivariate Gaussian probability density function
    Parameters:
    - x & y coordinates
    - mu_x/y: shift in x/y-direction
    - sig_x/y: with in x/y-direction
    - cor: correlation (skew)
    Returns the value of the PDF at the point (x, y)
    """
    x_ = x - mu_x
    y_ = y - mu_y
    cor /= (1/sig_x + 1/sig_y)  # normalize corr to avoid strong squeezing
    dem = 2 * np.pi * sig_x * sig_y * np.sqrt(1 - cor**2)
    exp = -1/(2*(1-cor**2)) *(x_**2/sig_x - 2*cor*x_/sig_x * y_/sig_y + y_**2/sig_y)
    return 1/dem * np.exp(exp)


def main(args):
    x = np.linspace(-1, 1, n_pixels)
    y = np.linspace(-1, 1, n_pixels)
    xx, yy = np.meshgrid(x, y)
    images, labels = [], []
    for i in range(10):
        a, cx, cy, wx, wy, cor = get_random_features()
        zz = bivariate_gaussian(xx, yy, mu_x=cx, mu_y=cy, sig_x=wx, sig_y=wy, cor=cor)
        zz += bivariate_gaussian(xx, yy+2, mu_x=cx, mu_y=cy, sig_x=wx, sig_y=wy, cor=cor)
        zz += bivariate_gaussian(xx, yy-2, mu_x=cx, mu_y=cy, sig_x=wx, sig_y=wy, cor=cor)
        zz *= a / zz.sum()
        zz += noise_amplitude * np.random.rand(*zz.shape)
        images.append(zz)
        labels.append((a, cx, cy, wx, wy, cor))
    
    save_data(images, labels, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='data/my_file.npy', help='path to output file')
    main(parser.parse_args())
