#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:09:19 2021

@author: luigi
"""


import numpy as np
from sklearn import datasets

def load_iris():
    '''Import Iris data from skilearn.datasets'''
    D, L = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    ''' split the datasets in two parts: 
    the first part will be used for model training, the second for evaluation'''
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def mu(dataset):
    '''Function the compute the mean matrix'''
    return np.mean(dataset,axis=1)

def sigma(dataset):
    '''Function that compute the covariance matrix'''
    diff = dataset-(mu(dataset).reshape(dataset.shape[0],1))
    return np.dot(diff, diff.T)/dataset.shape[1]
    
def computeMaximumLikelihoodEstimates(dataset,label):
    '''Function that compute the parameters of each class that maximize likelihood'''
    D0 = dataset[:, label==0]
    D1 = dataset[:, label==1]
    D2 = dataset[:, label==2]
    
    mu0 = mu(D0)
    mu1 = mu(D1)
    mu2 = mu(D2)

    sigma0 = sigma(D0)
    sigma1 = sigma(D1)
    sigma2 = sigma(D2)
    
    return (mu0,sigma0), (mu1,sigma1), (mu2,sigma2) 

def logpdf_GAU_ND(x, mu, sigma):
    ''' Function that compute the class conditional probabilities'''
    return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)

        

if __name__ == "__main__":
    D, L = load_iris()
    TR, TE = split_db_2to1(D,L)
    par = computeMaximumLikelihoodEstimates(TR[0], TR[1])
    #compute the class-conditional probabilities in a score matrix
    score = np.zeros((3,50))
    for i in range(3):
        x = logpdf_GAU_ND(TE[0],par[i][0].reshape(4,1), par[i][1])
        score[i] = x
    
    
            
    
    
    