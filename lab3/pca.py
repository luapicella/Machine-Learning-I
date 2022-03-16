import numpy as np
from utils import vcol

class PCA:
    
    def __init__(self, n_component):
        self.n_component = n_component

    def __initFit(self, dataset):
       #compute dataset mean
       self.mu = dataset.mean(axis=1)
       self.mu = vcol(self.mu, self.mu.size)
       #return center data
       return dataset - self.mu
       
    
    def fit(self, dataset):
        if self.n_component > dataset.shape[0]:
                raise Exception("n_componet must be <= of number of feature") 
        
        dataset = self.__initFit(dataset)
        # compute covariance matrix
        C =  np.dot(dataset,dataset.T) / float(dataset.shape[1])
        # compute eigenvector and eigenvalue
        s, U = np.linalg.eigh(C)
        # first n_component element
        self.P = U[:, ::-1][:, 0:self.n_component]
       

    def trasform(self, dataset):
        return  np.dot(self.P.T, dataset)


