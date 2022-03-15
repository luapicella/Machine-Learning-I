import numpy as np

class PCA:
    
    def __init__(self, n_component):
        self.n_component = n_component
    
    def fit(self, dataset):
        if self.n_component > dataset.shape[0]:
                raise Exception("n_componet must be <= of number of feature") 
        try:
            # compute covariance matrix
            C =  np.dot(dataset,dataset.T) / float(dataset.shape[1])
            # compute eigenvector and eigenvalue
            s, U = np.linalg.eigh(C)
            # first n_component element
            self.P = U[:, ::-1][:, 0:self.n_component]
        except:
            print("Something went wrong")

    def trasform(self, dataset):
        try:
            dataset_trasformed = np.dot(self.P.T, dataset)
        except:
            print("Something went wrong")
        else:
            return dataset_trasformed


