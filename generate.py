#!/usr/bin/env python
"""Generates 2dimensional Gaussian distribution
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    mean = [4, 4]
    cov = [[0.5, 0],
           [0, 0.5]]
    sample = np.random.multivariate_normal(mean, cov, args.size)
    np.savetxt(args.output, sample, fmt='%g', delimiter=',', header='x,y')

    if args.plot:
        x, y = sample.T
        plt.plot(x, y, 'x')
        plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s', '--size', type=int, default=100,
                           help='number of points to generate')
    argparser.add_argument('-o', '--output', type=str, required=True,
                           help='output file (by default will use stdout as output)')
    argparser.add_argument('-p', '--plot', action='store_true', default=False,
                           help='plot sample')
    args = argparser.parse_args()

    main(args)

