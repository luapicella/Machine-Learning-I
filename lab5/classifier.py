from matplotlib.pyplot import axis
import numpy as np
from utils import vcol
from utils import vrow
import utils
import scipy as sc
import multivariateGaussian




class MultivariateGaussianClassifier:

    def fit(self, dataset, labels):
        #self.classes = np.unique(labels)
        #self.n_classes = self.classes.shape[0]

        self.__computeMLestimates(dataset, labels)
        self.__computePrior(dataset, labels)

    def predict(self, dataset):
        # Compute, for each test sample, the MVG log-density
        logS = self.__computeScoreMatrix(dataset, multivariateGaussian.logpdf_GAU_ND)
        # Compute the matrix of joint log-distribution probabilities logSJoint for
        # samples and classes combining the score matrix with prior information.
        logSJoint = multivariateGaussian.joint_log_density(logS,utils.mcol(np.array([np.log(self.pi0), np.log(self.pi1), np.log(self.pi2) ])))
        # Compute the marginal lod density
        marginalLogDensities = multivariateGaussian.marginal_log_densities(logSJoint)
        # Now we can compute the array of class log-posterior probabilities logSPost.
        logSPost = multivariateGaussian.log_posteriors(logSJoint, marginalLogDensities)
        predictedLabels = logSPost.argmax(axis=0)
        return predictedLabels

    def __computeMLestimates(self, D, L):
        # Compute classes means over columns of the dataset matrix
        self.mu0 = utils.mcol(D[:, L == 0].mean(axis=1))
        self.mu1 = utils.mcol(D[:, L == 1].mean(axis=1))
        self.mu2 = utils.mcol(D[:, L == 2].mean(axis=1))
        #Compute class covariance
        self.sigma0 = np.cov(D[:, L == 0])
        self.sigma1 = np.cov(D[:, L == 1])
        self.sigma2 = np.cov(D[:, L == 2])
    
    def __computePrior(self, D, L):
        self.pi0 = D[:, L==0].shape[1]/D.shape[1]
        self.pi1 = D[:, L==1].shape[1]/D.shape[1]
        self.pi2 = D[:, L==2].shape[1]/D.shape[1]
    
    def __computeScoreMatrix(self, D, callback):
        S = np.zeros((3,D.shape[1]))
        S[0] = callback(D, self.mu0, self.sigma0)
        S[1] = callback(D, self.mu1, self.sigma1)
        S[2] = callback(D, self.mu2, self.sigma2)
        return S

class NaiveBayesGaussianClassifier:

        def fit(self, dataset, labels):
            self.__computeMLestimates(dataset, labels)
            self.__computePrior(dataset, labels)

        def predict(self, dataset):
            # Compute, for each test sample, the MVG log-density
            logS = self.__computeScoreMatrix(dataset, multivariateGaussian.logpdf_GAU_ND)
            # Compute the matrix of joint log-distribution probabilities logSJoint for
            # samples and classes combining the score matrix with prior information.
            logSJoint = multivariateGaussian.joint_log_density(logS,utils.mcol(np.array([np.log(self.pi0), np.log(self.pi1), np.log(self.pi2) ])))
            # Compute the marginal lod density
            marginalLogDensities = multivariateGaussian.marginal_log_densities(logSJoint)
            # Now we can compute the array of class log-posterior probabilities logSPost.
            logSPost = multivariateGaussian.log_posteriors(logSJoint, marginalLogDensities)
            predictedLabels = logSPost.argmax(axis=0)
            return predictedLabels

        def __computeMLestimates(self, D, L):
           # Compute classes means over columns of the dataset matrix
           self.mu0 = utils.mcol(D[:, L == 0].mean(axis=1))
           self.mu1 = utils.mcol(D[:, L == 1].mean(axis=1))
           self.mu2 = utils.mcol(D[:, L == 2].mean(axis=1))
           #Compute class covariance
           self.sigma0 = np.cov(D[:, L == 0])
           self.sigma1 = np.cov(D[:, L == 1])
           self.sigma2 = np.cov(D[:, L == 2])
           (self.sigma0, self.sigma1, self.sigma2) = (
               self.sigma0*np.identity(self.sigma0.shape[0]), 
               self.sigma1*np.identity(self.sigma1.shape[0]),
               self.sigma2*np.identity(self.sigma2.shape[0]))
    
        def __computePrior(self, D, L):
            self.pi0 = D[:, L==0].shape[1]/D.shape[1]
            self.pi1 = D[:, L==1].shape[1]/D.shape[1]
            self.pi2 = D[:, L==2].shape[1]/D.shape[1]

        def __computeScoreMatrix(self, D, callback):
            S = np.zeros((3,D.shape[1]))
            S[0] = callback(D, self.mu0, self.sigma0)
            S[1] = callback(D, self.mu1, self.sigma1)
            S[2] = callback(D, self.mu2, self.sigma2)
            return S


class TiedGaussianClassifier:
        def fit(self, dataset, labels):
            self.__computeMLestimates(dataset, labels)
            self.__computePrior(dataset, labels)

        def predict(self, dataset):
            # Compute, for each test sample, the MVG log-density
            logS = self.__computeScoreMatrix(dataset, multivariateGaussian.logpdf_GAU_ND)
            # Compute the matrix of joint log-distribution probabilities logSJoint for
            # samples and classes combining the score matrix with prior information.
            logSJoint = multivariateGaussian.joint_log_density(logS,utils.mcol(np.array([np.log(self.pi0), np.log(self.pi1), np.log(self.pi2) ])))
            # Compute the marginal lod density
            marginalLogDensities = multivariateGaussian.marginal_log_densities(logSJoint)
            # Now we can compute the array of class log-posterior probabilities logSPost.
            logSPost = multivariateGaussian.log_posteriors(logSJoint, marginalLogDensities)
            predictedLabels = logSPost.argmax(axis=0)
            return predictedLabels

        def __computeMLestimates(self, D, L):
           # Compute classes means over columns of the dataset matrix
           self.mu0 = utils.mcol(D[:, L == 0].mean(axis=1))
           self.mu1 = utils.mcol(D[:, L == 1].mean(axis=1))
           self.mu2 = utils.mcol(D[:, L == 2].mean(axis=1))
           #Compute class covariance
           self.sigma0 = np.cov(D[:, L == 0])
           self.sigma1 = np.cov(D[:, L == 1])
           self.sigma2 = np.cov(D[:, L == 2])
           self.sigma0 = self.sigma1 = self.sigma2 = 1/(D.shape[1])*(
               (D[:, L == 0].shape[1]*self.sigma0 +
                D[:, L == 1].shape[1]*self.sigma1 +
                D[:, L == 2].shape[1]*self.sigma2))
    
        def __computePrior(self, D, L):
            self.pi0 = D[:, L==0].shape[1]/D.shape[1]
            self.pi1 = D[:, L==1].shape[1]/D.shape[1]
            self.pi2 = D[:, L==2].shape[1]/D.shape[1]

        def __computeScoreMatrix(self, D, callback):
            S = np.zeros((3,D.shape[1]))
            S[0] = callback(D, self.mu0, self.sigma0)
            S[1] = callback(D, self.mu1, self.sigma1)
            S[2] = callback(D, self.mu2, self.sigma2)
            return S


    
        
    
    

    