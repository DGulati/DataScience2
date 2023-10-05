import argparse
import numpy as np

import sklearn.metrics.pairwise as pairwise

def read_data(filepath):
    Z = np.loadtxt(filepath)
    y = np.array(Z[:, 0], dtype = np.int)  # labels are in the first column
    X = np.array(Z[:, 1:], dtype = np.float)  # data is in all the others
    return [X, y]

def save_data(filepath, Y):
    np.savetxt(filepath, Y, fmt = "%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 3",
        epilog = "CSCI 4360/6360 Data Science II",
        add_help = "How to use",
        prog = "python homework3.py -i <input-data> -o <output-file> [optional args]")

    # Required args.
    parser.add_argument("-i", "--infile", required = True,
        help = "Path to an input text file containing the data.")
    parser.add_argument("-o", "--outfile", required = True,
        help = "Path to the output file where the class predictions are written.")

    # Optional args.
    parser.add_argument("-d", "--damping", default = 0.95, type = float,
        help = "Damping factor in the MRW random walks. [DEFAULT: 0.95]")
    parser.add_argument("-k", "--seeds", default = 1, type = int,
        help = "Number of labeled seeds per class to use in initializing MRW. [DEFAULT: 1]")
    parser.add_argument("-t", "--type", choices = ["random", "degree"], default = "random",
        help = "Whether to choose labeled seeds randomly or by largest degree. [DEFAULT: random]")
    parser.add_argument("-g", "--gamma", default = 0.5, type = float,
        help = "Value of gamma for the RBF kernel in computing affinities. [DEFAULT: 0.5]")
    parser.add_argument("-e", "--epsilon", default = 0.01, type = float,
        help = "Threshold of convergence in the rank vector. [DEFAULT: 0.01]")

    args = vars(parser.parse_args())

    # Read in the variables needed.
    outfile = args['outfile']   # File where output (predictions) will be written. 
    d = args['damping']         # Damping factor d in the MRW equation.
    k = args['seeds']           # Number of (labeled) seeds to use per class.
    t = args['type']            # Strategy for choosing seeds.
    gamma = args['gamma']       # Gamma parameter in the RBF kernel
    epsilon = args['epsilon']   # Convergence threshold in the MRW iteration.
    # For RBF, see: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel

    # Read in the data.
    X, y = read_data(args['infile'])

    # FINISH ME
