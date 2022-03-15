from pca import PCA

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def load_iris(filename):

    hlabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2 
    }

    datalist = []
    labellist = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.split(",")
            name = line[-1].strip()
            label = hlabels[name]
            attrs = mcol(np.array([float(i) for i in line[:-1]]))
            datalist.append(attrs)
            labellist.append(label)

    return np.hstack(datalist), np.array(labellist, dtype=np.int32)

def plot_scatter(D, L, folder='/image'):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]
  
    plt.figure()
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(folder + '.pdf' )
    plt.show()


if __name__ == '__main__':
    D, L = load_iris('../data/iris.csv')

    #compute mean
    mu = D.mean(axis=1)
    mu = mu.reshape((mu.size, 1))
    #center the data
    DC = D - mu

    pca = PCA(n_component=2)
    pca.fit(DC)
    DP = pca.trasform(DC)

    plot_scatter(DP, L, "./image/PCA2")