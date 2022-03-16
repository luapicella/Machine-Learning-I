from matplotlib.pyplot import axis
import numpy as np
from utils import vcol, centerDataset
import scipy.linalg as sc

class LDA:
    
    def __init__(self, n_component):
        self.n_component = n_component
        self.mu_cls = {}
        self.n_cls = {}
        self.sw_cls = {}

    def __initFit(self, dataset, label):
        self.diff_cls = np.unique(label)
        if self.n_component >= self.diff_cls.shape[0]:
                raise Exception("n_componet must be < number of classes") 
        #compute daataset mean
        self.mu = dataset.mean(axis=1)
        self.mu = vcol(self.mu, self.mu.size)
        #compute class mean for each class
        for cls in self.diff_cls:
            #compute mu for a specific class
            mu_n = dataset[:, label==cls].mean(axis=1)
            # Reshape and ad in map 
            self.mu_cls[cls] = vcol(mu_n, mu_n.size)
            # count number of element on specific class
            self.n_cls[cls] = dataset[:, label==cls].shape[1]

    
    def fit(self, dataset, label):
        #init current fit
        self.__initFit(dataset, label)
        #compute SB
        SB = self.__betweenClassCovariance()
        SW = self.__withinClassCovariance(dataset, label)

        #solving the generalized eigenvalue problem
        s, U = sc.eigh(SB, SW)
        self.W = U[:, ::-1][:, 0:self.n_component]

    def fit2(self, dataset, label):
        #init current fit
        self.__initFit(dataset, label)
        #compute SW
        SB = self.__betweenClassCovariance()
        SW = self.__withinClassCovariance(dataset, label)

        U, s, _ = np.linalg.svd(SW)
         # Compute P1 matrix
        P1 = np.dot(np.dot(U, np.diag(1.0/(s**0.5))), U.T)
         # Compute transformed SB
        SBT = np.dot(np.dot(P1, SB), P1.T)
        # Get eigenvectors (columns of U ) from SBT
        _, U = np.linalg.eigh(SBT)

        # Compute P2 (m leading eigenvectors)
        P2 = U[:, ::-1][:, 0:self.n_component]
        # Compute W
        self.W = np.dot(P1.T, P2)

    def trasform(self, dataset):
        try:
            dataset_trasformed = np.dot(self.W.T, dataset)
        except:
            print("Something went wrong")
        else:
            return dataset_trasformed

    def __withinClassCovariance(self, dataset, label):
        s = 0
        #compute SW for each class
        for cls in self.diff_cls:
            datasetC = dataset[:, label==cls]-self.mu_cls[cls]
            self.sw_cls[cls] = (1/self.n_cls[cls])*np.dot(datasetC,datasetC.T)
        
        for cls in self.diff_cls:
            s += self.n_cls[cls]*self.sw_cls[cls] 
        
        return (1/sum(self.n_cls.values())*(s))

    def __betweenClassCovariance(self):
        N = sum(self.n_cls.values())
        s = 0
        for key, value in self.mu_cls.items():
            s += self.n_cls[key]*(np.dot(self.mu_cls[key]-self.mu, (self.mu_cls[key]-self.mu).T))

        return (1/N) * s


