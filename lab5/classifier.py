from matplotlib.pyplot import axis
import numpy as np
from utils import vcol
from utils import vrow
import scipy as sc
import multivariateGaussian



class GaussianClassifier:

    def __init__(self):
        self.mu = [] # classes mean
        self.sigma = [] # classes covariance matrix
        self.classes = []
        self.prior = None # prior prob

    def fit(self, dataset, labels):
        self.classes = np.unique(labels)
        self.n_classes = self.classes.shape[0]

        self.__computeMLestimates(dataset, labels)
        self.__computePrior(dataset, labels)

    def predict(self, dataset):
        # Compute, for each test sample, the MVG log-density
        logS = self.__computeScoreMatrix(dataset, multivariateGaussian.logpdf_GAU_ND)
        # Compute the matrix of joint log-distribution probabilities logSJoint for
        # samples and classes combining the score matrix with prior information.
        logSJoint = multivariateGaussian.joint_log_density(logS,self.prior)
        # Compute the marginal lod density
        marginalLogDensities = multivariateGaussian.marginal_log_densities(logSJoint)
        # Now we can compute the array of class log-posterior probabilities logSPost.
        logSPost = multivariateGaussian.log_posteriors(logSJoint, marginalLogDensities)
        predictedLabels = logSPost.argmax(axis=0)
        return predictedLabels
        
    def __computeMLestimates(self, D, L):
        # Compute classes means over columns of the dataset matrix
        for i in range(self.n_classes):
            _mu = D[:, L == self.classes[i]].mean(axis=1)
            _mu = vcol(_mu, _mu.size)
            self.mu.append(_mu)
            # Count number of elements in each class
            _n = D[:, L == self.classes[i]].shape[1]
            # Subtract classes means from classes datasets with broadcasting
            _DC = D[:, L == self.classes[i]] - _mu
            # Compute classes covariance matrices
            self.sigma.append((1/_n)*(np.dot(_DC, _DC.T)))
    
    def __computePrior(self, D, L):
        self.prior = np.zeros((self.n_classes,1))
        for i in range(self.n_classes):
            self.prior[i] = D[:, L == self.classes[i]].shape[1]/D.shape[1]
    
    def __computeScoreMatrix(self, D, callback):
        S = np.zeros((self.n_classes,D.shape[1]))
        for i in range(self.n_classes):
            S[i] = callback(D, self.mu[i], self.sigma[i])
        return S