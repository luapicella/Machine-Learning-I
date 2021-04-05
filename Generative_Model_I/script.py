#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:09:19 2021

@author: luigi
"""


import numpy as np
from scipy.special import logsumexp
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

class GuassianClassifier:
    """Gaussian Classifier"""
    
    def multivariate_normal(self, x, mu, sigma):
        ''' Function that compute the class conditional probabilities'''
        return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)
multivariate class-conditional density function
    
    def train(self,X,Y):
        
        #class-conditional density - MLE estimate
        self.mean_0 = np.mean( X[:, Y==0], axis=1) .reshape((X.shape[0],1))
        self.mean_1 = np.mean( X[:, Y==1], axis=1) .reshape((X.shape[0],1))
        self.mean_2 = np.mean( X[:, Y==2], axis=1) .reshape((X.shape[0],1))
        self.sigma_0 = np.cov( X[:, Y==0])
        self.sigma_1 = np.cov( X[:, Y==1])
        self.sigma_2 = np.cov( X[:, Y==2])
        
        #class prior
        #self.pi_0 =  Y[Y==0].shape[0]/Y.shape[0]
        #self.pi_1 =  Y[Y==1].shape[0]/Y.shape[0]
        #self.pi_2 =  Y[Y==2].shape[0]/Y.shape[0]
        self.pi_0 =  1/3
        self.pi_1 =  1/3
        self.pi_2 =  1/3
        
        
    def predict(self,X):
            
        #multivariate class-conditiona
        self.prob_0 = self.multivariate_normal(X,self.mean_0,self.sigma_0) .reshape((1,X.shape[1])) + np.log(self.pi_0)
        self.prob_1 = self.multivariate_normal(X,self.mean_1,self.sigma_1). reshape((1,X.shape[1])) + np.log(self.pi_1)
        self.prob_2 = self.multivariate_normal(X,self.mean_2,self.sigma_2). reshape((1,X.shape[1])) + np.log(self.pi_2)
        
        self.SJoint = np.zeros((3,X.shape[1]))
        self.SJoint[0] =  self.prob_0
        self.SJoint[1] =  self.prob_1
        self.SJoint[2] =  self.prob_2
        
        self.marginal = logsumexp(self.SJoint)
        self.SPost = self.SJoint-self.marginal
        
        ret = self.SPost.argmax(axis=0)
        
        return ret 
    
    
    
def accuracy(Y_test, Y_testPred):
        arr = Y_test == Y_testPred
        correct = np.sum(arr)
        accuracy = correct / Y_test.shape[0]
        return accuracy

def error(Y_test, Y_testPred):
    return 1 - accuracy(Y_test, Y_testPred)
    

if __name__ == "__main__":
    #load dataset iris
    dataset, label = load_iris()
    train, test = split_db_2to1(dataset, label)
    classif = GuassianClassifier()
    classif.train(train[0], train[1])
    pred = classif.predict(test[0])
    acc = accuracy(test[1],pred)
    err = error(test[1],pred)
    
    
    
    
            
    
    
    