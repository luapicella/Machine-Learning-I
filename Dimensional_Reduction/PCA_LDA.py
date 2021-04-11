#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:48:14 2021

@author: luigi
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.linalg as sc

def load_iris():
    '''Import Iris data from skilearn.datasets'''
    X, Y = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    return X, Y

def vcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return vector.reshape(shape0, 1)


def vrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return vector.reshape(1, shape1)

class LDA:
    
    def __init__(self,n_components):
        self.n_components = n_components
        
    def fit(self, X, Y):
        SB = self.computeBetweenClassCovarianceMatrix(X, Y)
        SW = self.computeWithinClassCovarianceMatrix(X, Y)
        self.W = self.computeW(SB, SW, X, Y)
        
        
    def transform(self, X):
        W = self.W
        XP = np.dot(W.T,X)
        return XP
                        
    def computeBetweenClassCovarianceMatrix(self, X, Y):
        # Compute mean of the dataset matrix
        mu = X.mean(axis=1)
        # Reshape the 1-D array mu to a column vector 4x1
        mu = vcol(mu, mu.size)
        # Compute classes means over columns of the dataset matrix
        mu0 = X[:, Y == 0].mean(axis=1)
        mu1 = X[:, Y == 1].mean(axis=1)
        mu2 = X[:, Y == 2].mean(axis=1)
        # Reshape all of them as 4x1 column vectors
        mu0 = vcol(mu0, mu0.size)
        mu1 = vcol(mu1, mu1.size)
        mu2 = vcol(mu2, mu2.size)
        # Count number of elements in each class
        n0 = X[:, Y == 0].shape[1]
        n1 = X[:, Y == 1].shape[1]
        n2 = X[:, Y == 2].shape[1]
        return (1/(n0+n1+n2))*((n0*np.dot(mu0-mu, (mu0-mu).T))+(n1*np.dot(mu1-mu, (mu1-mu).T)) +
                               (n2*np.dot(mu2-mu, (mu2-mu).T)))
    
    def computeWithinClassCovarianceMatrix(self, X, Y):
        # Compute classes means over columns of the dataset matrix
        mu0 = X[:, Y == 0].mean(axis=1)
        mu1 = X[:, Y == 1].mean(axis=1)
        mu2 = X[:, Y == 2].mean(axis=1)
        # Reshape all of them as 4x1 column vectors
        mu0 = vcol(mu0, mu0.size)
        mu1 = vcol(mu1, mu1.size)
        mu2 = vcol(mu2, mu2.size)
        # Count number of elements in each class
        n0 = X[:, Y == 0].shape[1]
        n1 = X[:, Y == 1].shape[1]
        n2 = X[:, Y == 2].shape[1]
        # Compute within covariance matrix for each class
        Sw0 = (1/n0)*np.dot(X[:, Y == 0]-mu0, (X[:, Y == 0]-mu0).T)
        Sw1 = (1/n1)*np.dot(X[:, Y == 1]-mu1, (X[:, Y == 1]-mu1).T)
        Sw2 = (1/n2)*np.dot(X[:, Y == 2]-mu2, (X[:, Y == 2]-mu2).T)
        return (1/(n0+n1+n2))*(n0*Sw0+n1*Sw1+n2*Sw2)
    
    def computeW(self, SB, SW, X, Y):
        # Solve the generalized eigenvalue problem
        s, U = sc.eigh(SB, SW)
        ret =  U[:, ::-1][:, 0:self.n_components]
        return ret
    
    def computeW2(self, SB, SW, X, Y):
        # Second version
        # Compute SVD
        U, s, _ = np.linalg.svd(SW)
        # Compute P1 matrix
        P1 = np.dot(np.dot(U, np.diag(1.0/(s**0.5))), U.T)
        # Compute transformed SB
        SBT = np.dot(np.dot(P1, SB), P1.T)
        # Get eigenvectors (columns of U ) from SBT
        _, U = np.linalg.eigh(SBT)
        # Compute P2 (m leading eigenvectors)
        P2 = U[:, ::-1][:, 0:self.n_components]
        # Compute W
        W = np.dot(P1.T, P2)
        # LDA projection matrix
        return W
    
class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        
    def fit(self, X):
        mu = X.mean(1)
        DC = X - X.mean(1).reshape((mu.size, 1))
        C = np.dot(DC,DC.T)
        s, U = np.linalg.eigh(C)
        P = U[:, ::-1][:, 0:self.n_components]
        self.P = P
        return
    
    def transform(self, X):
        Y = np.dot(self.P.T, X)
        return Y
        
        
        
    

X, Y = load_iris()
target_names = datasets.load_iris()['target_names']

lda = LDA(2)
lda.fit(X, Y)
X_t_lda = lda.transform(X)

pca = PCA(2)
pca.fit(X)
X_t_pca = pca.transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_t_lda[0, Y == i], X_t_lda[1, Y == i], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_t_pca[0, Y == i], X_t_pca[1, Y == i], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()


        
    
        
        
