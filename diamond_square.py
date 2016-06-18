# Author: Piotr Sliwa
# Project 1 for ONA
# 3 May 2016

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, help="order of matrix (2^N + 1)", default=7)
parser.add_argument("--sigma", type=float, help="degree of mountainousness", default=0.3)
parser.add_argument("--map_file", type=str, help="name of file in which map will be saved", default="None")
parser.add_argument("--surf_file", type=str, help="name of file in which surface will be saved", default="None")
parser.add_argument("--colormap", type=str, help="colormap from matplotlib", default="gist_earth")
parser.add_argument("--matrix_file", type=str, help="file in which matrix will be saved", default="None")
parser.add_argument("--load_matrix", type=str, help="file from which preprocessed matrix will be loaded", default="None")
parser.add_argument("-K", type=int, help="start generation from step K", default=-1) # required if matrix is loaded
args = parser.parse_args()
n = args.N
k = n
sigma = args.sigma
map_file = args.map_file
surf_file = args.surf_file
colormap = args.colormap
matrix_file = args.matrix_file
load_matrix = args.load_matrix
K = args.K # 1 represents first step

# create matrix
if K == -1:
    # from scratch
    A_size = (1 << n) + 1
    A = np.zeros((A_size, A_size))
    A[0][0] = A[0][A_size - 1] = A[A_size - 1][0] = A[A_size - 1][A_size - 1] = 0
    side_length = A_size - 1
else:
    # when a matrix is loaded
    if ".txt" in load_matrix:
        A = np.loadtxt(load_matrix)
    else:
        A = np.load(load_matrix)
    A_size = A.shape[0]
    n = int(np.log2(A_size - 1) + 0.5)
    k = n - K + 1
    side_length = 2 ** k

# diamond square algorithm for generation of fractal terrain
while (side_length >= 2):
    half_side = side_length // 2

    # diamond step
    for x in range(0, A_size - 1, side_length):
        for y in range(0, A_size - 1, side_length):
            # x, y represent the top left element
            avg = (A[x][y] +                                # top left
                + A[x + side_length][y] +                   # top right
                + A[x][y + side_length] +                   # bottom left
                + A[x + side_length][y + side_length]) * 0.25   # bottom right

            random_val = np.random.normal() * (2 ** k) * sigma
            value = avg + random_val
            A[x + half_side][y + half_side] = value

    # square step
    for x in range(0, A_size - 1, half_side):
        for y in range((x + half_side) % side_length, A_size - 1, side_length):
            # x, y represent the center of the diamond
            avg = (A[(x - half_side + A_size - 1) % (A_size - 1)][y] +  # left
                + A[(x + half_side) % (A_size - 1)][y] +                # right
                + A[x][(y + half_side) % (A_size - 1)] +                # down
                + A[x][(y - half_side + A_size - 1) % (A_size - 1)]) * 0.25 # up
                
            random_val = np.random.normal() * (2 ** k) * sigma
            value = avg + random_val

            A[x][y] = value

            if (x == 0):
                A[A_size - 1][y] = value
            if (y == 0):
                A[x][A_size - 1] = value
    side_length //= 2
    k -= 1

# save matrix
if matrix_file != "None":
    if ".txt" in matrix_file:
        np.savetxt(matrix_file, A)
    else:
        np.save(matrix_file, A)
# draw physical map
fig1 = plt.figure()
plt.imshow(A, cmap=colormap)

# save map image
if map_file != "None":
    plt.savefig(map_file, bbox_inches="tight")
    plt.close()
else:
    plt.show()

# draw 3d surface
fig2 = plt.figure()
ax = fig2.gca(projection='3d')

X = np.arange(0, 2**n + 1)
Y = np.arange(0, 2**n + 1)

X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, A, rstride=1, cstride=1, alpha=1, linewidth=0.05, cmap=colormap)

# save 3d surface image
if surf_file != "None":
    plt.savefig(surf_file, bbox_inches="tight")
    plt.close()
else:
    plt.show()
