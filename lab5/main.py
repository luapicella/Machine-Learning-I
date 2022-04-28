from classifier import MultivariateGaussianClassifier, NaiveBayesGaussianClassifier, TiedGaussianClassifier
import sklearn.datasets
import numpy as np

from sklearn.model_selection import StratifiedKFold

from model_selection import Kfold, LeaveOneOut
from model_selection import cross_val_score

from statistics import mean

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
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

def compute_accuracy(predictedLabels, trueLabels):
    return np.array(predictedLabels == trueLabels).sum()/trueLabels.size*100

def compute_error(predictedLabels, trueLabels):
    return 100 - compute_accuracy(predictedLabels, trueLabels)


if __name__ == '__main__':
    #Load dataset
    D, L = load_iris()
    #Split dataset
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    #Multivariate Gaussian Classifier
    MVG = MultivariateGaussianClassifier()
    MVG.fit(DTR, LTR)
    pred = MVG.predict(DTE)
    accuracy = compute_accuracy(pred, LTE)
    print(accuracy)

    # Naive Bayes Gaussian Classifier
    naiveG = NaiveBayesGaussianClassifier()
    naiveG.fit(DTR, LTR)
    pred = naiveG.predict(DTE)
    accuracy = compute_accuracy(pred, LTE)
    print(accuracy)

    # Tied Covarince Gaussian Classifier
    tiedG = TiedGaussianClassifier()
    tiedG.fit(DTR, LTR)
    pred = tiedG.predict(DTE)
    accuracy = compute_accuracy(pred, LTE)
    print(accuracy)

    # Cross validation scores with MVG
    model = MultivariateGaussianClassifier()
    cv = LeaveOneOut()
    scores = cross_val_score(model, D, L, cv, scoring='error')
    print(mean(scores))

    # Cross validation scores naive guassian classifier
    model = NaiveBayesGaussianClassifier()
    cv = LeaveOneOut()
    scores = cross_val_score(model, D, L, cv, scoring='error')
    print(mean(scores))

    # Cross validation scores with tied gaussian classifier
    model = TiedGaussianClassifier()
    cv = LeaveOneOut()
    scores = cross_val_score(model, D, L, cv, scoring='error')
    print(mean(scores))



    

