import argparse
import numpy as np


x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
xx, yy = np.meshgrid(x, y)


def gen_peak(cx, cy, a=1, w=1):
    return a * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (0.1*w))

def gen_wrapped_peak(cx, cy, a=1, w=1):
    upper_peak = gen_peak(cx, cy, a, w)
    lower_peak = gen_peak(cx, cy - 2, a, w)
    return upper_peak + lower_peak


def main(args):
    data = []
    for i in range(100):
        cx = np.random.uniform(-1, 1)
        cy = np.random.uniform(-1, 1)
        a = 1
        w = np.random.uniform(0.1, 2)
        zz = gen_wrapped_peak(cx, cy)
        data.append(zz)

    with open(args.output, 'wb') as f:
        np.save(f, np.array(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='data/my_file.npy', help='path to output file')
    main(parser.parse_args())


# import matplotlib.pyplot as plt

# numpy.save()

# print(f'data.shape = {data.shape}')
# # with h5py.File("mytestfile.hdf5", "w") as f:
# #     f['dingeling'] = arr


# plt.imshow(zz)
# plt.show()

# exit(0)


# n = 100


# def get_disk_mask(c_x, c_y, r):
#     y,x = np.ogrid[-c_x:n-c_x, -c_y:n-c_y]
#     return (x*x + y*y <= r*r)


# def get_wrapped_disk_mask(c_x, c_y, r):
#     upper_mask = get_disk_mask(c_x, c_y, r)
#     lower_mask = get_disk_mask(c_x + n, c_y, r)
#     return upper_mask + lower_mask


# def generate_image():
#     c_x, c_y, r = 10, 10, 10
#     arr = np.zeros((n, n))
#     mask = get_wrapped_disk_mask(c_x, c_y, r)
#     arr[mask] = 1
#     return arr

# arr = generate_image()

# plt.imshow(arr)
# plt.show()

# with h5py.File("mytestfile.hdf5", "w") as f:
#     f['dingeling'] = arr
#     # dset = f.create_dataset("mydataset", (100,), dtype='i')
