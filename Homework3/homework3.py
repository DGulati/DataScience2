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

    #Add index to each data point so that we can keep track of them  
    index = np.arange(0, len(X))
    Data = np.column_stack((index,y,X))

    #Calculate the affinity matrix
    A = pairwise.rbf_kernel(X, gamma = gamma)

    #Calculate the degree matrix
    Sums = np.sum(A, axis = 1)
    Sums_Data = np.column_stack((index,y,Sums))
    D = np.diag(Sums)

    #Calculate the weighted transition matrix
    W = np.dot(np.linalg.inv(D), A)
    
    #Need to calculate seed vectors for each class. 
    classes = np.unique(y)
    u = {}

    if t == "random":
        for i in classes and i != -1:  
            #Get the data points that belong to the class
            i_class = Data[Data[:, 1] == i]      
            indecies_chosen = np.random.choice(Data[0], k, replace = False)
            u[i] = np.zeros(len(X))
            u[i][indecies_chosen] = 1/k
    elif t == "degree":
        for i in classes and i != -1:
            #Get the sums for the points that belong to the class
            i_Sums = Sums_Data[Sums_Data[:, 1] == i]    
            i_Sums = i_Sums[i_Sums[:, 2].argsort()[::-1]]
            indecies_chosen = i_Sums[0:k, 0]
            u[i] = np.zeros(len(X))
            u[i][indecies_chosen] = 1/k

    r = u.deepcopy()
    
    iter = 0
    while iter < 100:
        iter += 1
        r_old = r.deepcopy()
        for i in classes and i != -1:
            r[i] = (1-d)*u[i] + d*np.dot(W, r_old[i])
        if np.linalg.norm(r[i] - r_old[i]) < epsilon:
            break

    if iter == 100:
        print("Did not converge")

    Data[Data[:, 1] == -1][1] = np.argmax(r, axis = 0)[Data[:, 1] == -1]

    save_data(outfile, Data[:, 1])

    print("Done!")

    



    



    


