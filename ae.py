#!/usr/bin/env python
"""Detects anomaly points
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm

def main(args):
    sample = np.loadtxt(args.input, delimiter=',')
    detector = svm.OneClassSVM(nu=0.3, gamma=0.1, kernel="rbf")
    detector.fit(sample)

    test = np.loadtxt('test.csv', delimiter=',')
    labels = detector.predict(test)
    np.savetxt('labels.csv', labels, delimiter=',', fmt='%g')

    if args.plot:
        xx1, yy1 = np.meshgrid(np.linspace(-6, 6, 500), np.linspace(-6, 6, 500))
        DF = detector.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        DF = DF.reshape(xx1.shape)
        plt.contour(
        xx1, yy1, DF, levels=[0], linewidths=2, colors='m')
        plt.scatter(sample[:, 0], sample[:, 1], color='black')
        for i in range(len(test)):
            color = 'red' if labels[i] == -1 else 'green'
            plt.scatter(test[i, 0], test[i, 1], color=color)

        plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('input', type=str,
                           help='input file')
    argparser.add_argument('-p', '--plot', action='store_true', default=False,
                           help='plot sample')
    args = argparser.parse_args()

    main(args)    
