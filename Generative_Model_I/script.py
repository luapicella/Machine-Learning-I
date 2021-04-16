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
    
    def __init__(self,mode = 'mvg'):
        self.mode = mode;
    
    def multivariate_normal(self, x, mu, sigma):
        ''' Function that compute the class conditional probabilities'''
        return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)
            
    
    def fit(self,X,Y):
        
        #class-conditional density - MLE estimate
        self.mean_0 = np.mean( X[:, Y==0], axis=1) .reshape((X.shape[0],1))
        self.mean_1 = np.mean( X[:, Y==1], axis=1) .reshape((X.shape[0],1))
        self.mean_2 = np.mean( X[:, Y==2], axis=1) .reshape((X.shape[0],1))
        
    
        self.sigma_0 = np.cov( X[:, Y==0])
        self.sigma_1 = np.cov( X[:, Y==1])
        self.sigma_2 = np.cov( X[:, Y==2])
        if (self.mode == 'naive' or self.mode == 'naive tied' ):
            self.sigma_0 = self.sigma_0 * np.identity(self.sigma_0.shape[0])
            self.sigma_1 = self.sigma_1 * np.identity(self.sigma_1.shape[0])
            self.sigma_2 = self.sigma_2 * np.identity(self.sigma_2.shape[0])
        if (self.mode == 'tied' or self.mode == 'naive tied'):
            # Count number of elements in each class
            n0 = X[:, Y == 0].shape[1]
            n1 = X[:, Y == 1].shape[1]
            n2 = X[:, Y == 2].shape[1]
            n = n0+n1+n2
            # Compute within covariance matrix for each class
            self.sigma = (1/(n))*((n0*self.sigma_0)+(n1*self.sigma_1)+(n2*self.sigma_2))
        
        #class prior
        #self.pi_0 =  Y[Y==0].shape[0]/Y.shape[0]
        #self.pi_1 =  Y[Y==1].shape[0]/Y.shape[0]
        #self.pi_2 =  Y[Y==2].shape[0]/Y.shape[0]
        self.pi_0 =  1/3
        self.pi_1 =  1/3
        self.pi_2 =  1/3
        
        
        
        
    def predict(self,X):
            
        #multivariate class-conditiona
        if (self.mode == 'mvg' or self.mode == 'naive'):
            self.prob_0 = self.multivariate_normal(X,self.mean_0,self.sigma_0) .reshape((1,X.shape[1])) + np.log(self.pi_0)
            self.prob_1 = self.multivariate_normal(X,self.mean_1,self.sigma_1). reshape((1,X.shape[1])) + np.log(self.pi_1)
            self.prob_2 = self.multivariate_normal(X,self.mean_2,self.sigma_2). reshape((1,X.shape[1])) + np.log(self.pi_2)
        elif (self.mode == 'tied' or self.mode == 'naive tied'):
            self.prob_0 = self.multivariate_normal(X,self.mean_0,self.sigma) .reshape((1,X.shape[1])) + np.log(self.pi_0)
            self.prob_1 = self.multivariate_normal(X,self.mean_1,self.sigma). reshape((1,X.shape[1])) + np.log(self.pi_1)
            self.prob_2 = self.multivariate_normal(X,self.mean_2,self.sigma). reshape((1,X.shape[1])) + np.log(self.pi_2)
            
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


class Cross_validation:
    
    @staticmethod
    def partition(X, Y, fold, k):
        size = X.shape[1]
        start = (size//k)*fold
        end = (size//k)*(fold+1)
        X_val = X[:, start:end]
        Y_val = Y[start:end]
        X_train = np.concatenate((X[:,0:start],X[:, end:]),1)
        Y_train = np.concatenate((Y[0:start], Y[end:]))
        return X_val, Y_val, X_train, Y_train

    @staticmethod    
    def Cross_validation(learner, k, X, Y):
        validation_folds_score = []
        for fold in range(0, k):
            X_val, Y_val, X_train, Y_train = Cross_validation.partition(X, Y, fold, k)
            
            learner.fit(X_train,Y_train)
            Y_pred = learner.predict(X_val)
            validation_folds_score.append(accuracy(Y_val, Y_pred))
        return validation_folds_score
            
        
        

if __name__ == "__main__":
    #load dataset iris
    dataset, label = load_iris()
    train, test = split_db_2to1(dataset, label)
    
    #mvg classifier valuation
    classif = GuassianClassifier()
    scores = Cross_validation.Cross_validation(classif, train[0].shape[1], train[0], train[1] )
    print('\nCV mvg accuracy : %.3f error: %.3f' %(np.mean(scores), 1-np.mean(scores)))
    
    #naive classifier valuation
    classif = GuassianClassifier('naive')
    scores = Cross_validation.Cross_validation(classif, train[0].shape[1], train[0], train[1] )
    print('\nCV naive accuracy: %.3f error: %.3f' %(np.mean(scores), 1-np.mean(scores)))
    
    #tied classifier valuation
    classif = GuassianClassifier('tied')
    scores = Cross_validation.Cross_validation(classif, train[0].shape[1], train[0], train[1] )
    print('\nCV tied accuracy: %.3f error: %.3f' %(np.mean(scores), 1-np.mean(scores)))
    
    #naive tied valuation
    classif = GuassianClassifier('naive tied')
    scores = Cross_validation.Cross_validation(classif, train[0].shape[1], train[0], train[1] )
    print('\nCV naive tied accuracy: %.3f error: %.3f' %(np.mean(scores), 1-np.mean(scores)))

    
    
    ##########################################################
    # MVG model
    ##########################################################
    classif = GuassianClassifier()
    classif.fit(train[0], train[1])
    pred = classif.predict(test[0])
    acc = accuracy(test[1],pred)
    err = error(test[1],pred)
    print('Accuracy mvg: ' + str(acc))
    print('Error mvg:    ' + str(err))
    ##########################################################
    # MVG naive
    ##########################################################    
    classif = GuassianClassifier('naive')
    classif.fit(train[0], train[1])
    pred = classif.predict(test[0])
    acc = accuracy(test[1],pred)
    err = error(test[1],pred)
    print('Accuracy mvg naive: ' + str(acc))
    print('Error mvg naive   : ' + str(err))
    ##########################################################
    # MVG tied
    ##########################################################
    classif = GuassianClassifier('tied')
    classif.fit(train[0], train[1])
    pred = classif.predict(test[0])
    acc = accuracy(test[1],pred)
    err = error(test[1],pred)
    print('Accuracy mvg tied: ' + str(acc))
    print('Error mvg tied   :    ' + str(err))
    ##########################################################
    # MVG naive tied
    ##########################################################
    classif = GuassianClassifier('naive tied')
    classif.fit(train[0], train[1])
    pred = classif.predict(test[0])
    acc = accuracy(test[1],pred)
    err = error(test[1],pred)
    print('Accuracy mvg naive tied: ' + str(acc))
    print('Error mvg naive tied   :    ' + str(err))
    
            
    
    
    
