import argparse
import numpy as np
import matplotlib.pyplot as plt

N = 32  # number of rows and columns, images will have N*N pixels

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
xx, yy = np.meshgrid(x, y)


def gen_peak(cx, cy, a=1, w=1):
    return a * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (0.1*w))
    

def main(args):
    data = []
    for i in range(100):
        cx = np.random.uniform(-1, 1)
        cy = np.random.uniform(-1/N, 1/N)  # random between two rows
        y_shift = np.random.randint(-N/2, N/2)  # shift by n rows
        a = 1
        w = np.random.uniform(0.1, 2)
        zz = gen_peak(cx, cy)
        zz = np.roll(zz, y_shift, axis=0)
        true_cx = cx  # x-position of center (between -1 and 1)
        true_cy = cy + y_shift / N * 2  # y-position of center (between -1 and 1)
        print(f'true_cy = {true_cy}')
        plt.imshow(zz)
        plt.show()

        data.append(zz)

    # with open(args.output, 'wb') as f:
    #     np.save(f, np.array(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='data/my_file.npy', help='path to output file')
    main(parser.parse_args())
