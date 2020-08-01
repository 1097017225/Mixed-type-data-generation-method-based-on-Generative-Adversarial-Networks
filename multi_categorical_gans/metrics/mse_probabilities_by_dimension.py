from __future__ import division
from __future__ import print_function
import sys
sys.path.append("/root/pytorch")
import argparse

import numpy as np

from multi_categorical_gans.datasets.formats import data_formats, loaders

from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt

def probability(data):
    return data.sum() / data.shape[0]


def probabilities_by_dimension(data):
    return np.array([probability(data[:, 22+j]) for j in range(63)])


def mse_probabilities_by_dimension(data_x, data_y):
    p_x_by_dimension = probabilities_by_dimension(data_x)
    p_y_by_dimension = probabilities_by_dimension(data_y)
    mse = mean_squared_error(p_x_by_dimension, p_y_by_dimension)
    return p_x_by_dimension, p_y_by_dimension, mse


def main():
    options_parser = argparse.ArgumentParser(description="Plot probabilities by dimension for two samples.")

    #options_parser.add_argument("data_x", type=str, help="Data for x-axis. See 'data_format_x' parameter.")
    #options_parser.add_argument("data_y", type=str, help="Data for y-axis. See 'data_format_y' parameter.")

    options_parser.add_argument("--output", type=str, help="Output numpy data path.")

    options_parser.add_argument("--data_format_x", type=str, choices=data_formats, default="dense")
    options_parser.add_argument("--data_format_y", type=str, choices=data_formats, default="dense")

    options = options_parser.parse_args()

    B = np.load("/root/pytorch/data/fixed_2/count-test.matrix", allow_pickle=True)
    A = np.load("/root/pytorch/samples/medgan/fixed_2/sample.npy", allow_pickle=True)

    p_x_by_dimension, p_y_by_dimension, mse = mse_probabilities_by_dimension(A, B)

    if options.output is not None:
        np.savez(options.output, x=p_x_by_dimension, y=p_y_by_dimension, mse=mse)

    print("MSE: ", mse)
    plt.scatter(p_x_by_dimension, p_y_by_dimension)

    x1 = np.array([0,1])
    y2 = np.array([0,1])

    plt.plot(x1,y2)
    plt.xlabel("Psample")
    plt.ylabel("Ptest")
    plt.show()


if __name__ == "__main__":
    main()
