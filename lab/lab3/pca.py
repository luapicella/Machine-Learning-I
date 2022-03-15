import numpy as np

class PCA:
    
    def __init__(self, n_component):
        self.n_component = n_component
    
    def fit(self, dataset):
        # compute covariance matrix
        C =  np.dot(dataset,dataset.T) / float(dataset.shape[1])
        # compute eigenvector and eigenvalue
        s, U = np.linalg.eigh(C)
        # first n_component element
        self.P = U[:, ::-1][:, 0:self.n_component]

    def trasform(self, dataset):
        dataset_trasformed = np.dot(self.P.T, dataset)
        return dataset_trasformed


