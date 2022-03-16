from asyncio.log import logger
from pca import PCA
from lda import LDA

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

def plot_scatter(D, L, x='',  y='', folder='/image'):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]
  
    plt.figure()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')

    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(folder + '.pdf' )
    plt.show()


if __name__ == '__main__':
    try:
        D, L = load_iris('../data/iris.csv')
        
        #### PCA ####
        pca = PCA(2)
        pca.fit(D)
        DP = pca.trasform(D)

        plot_scatter(DP, L, x="PC1", y='PC2', folder="./image/LDA2")
        
        #### LDA ####
        lda = LDA(2)
        lda.fit2(D, L)
        DP = lda.trasform(D)

        plot_scatter(DP, L, x="LD1", y='LD2', folder="./image/LDA2")

    except Exception as e:
        logger.error(e)