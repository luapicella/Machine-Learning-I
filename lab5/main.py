from classifier import GaussianClassifier
import sklearn.datasets
import numpy as np

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


if __name__ == '__main__':
    #Load dataset
    D, L = load_iris()
    #Split dataset
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    guassianClass = GaussianClassifier()
    guassianClass.fit(DTR, LTR)
    pred = guassianClass.predict(DTE)

    accuracy = compute_accuracy(pred, LTE)
    print(accuracy)
