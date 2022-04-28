from abc import abstractmethod
from random import randrange
import numpy as np

def compute_accuracy(predictedLabels, trueLabels):
    return np.array(predictedLabels == trueLabels).sum()/trueLabels.size*100

def cross_validation_split(D, L, seed=0, K=3):
    folds = []  # list that will contain the folds
    labels = [] # list that will contain the labels
    n_sample_fold = int(D.shape[1]/K)
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*n_sample_fold): ((i+1)*(n_sample_fold))]])
        labels.append(L[idx[(i*n_sample_fold): ((i+1)*(n_sample_fold))]])
    return folds, labels

class _BaseCrossValidator:
    """Base class for all cross-validators"""
    
    def cross_validation_split(self, X, Y):
        n_sample = X.shape[1]
        idxs = np.arange(n_sample)
        for test_idx_mask in self._iter_test_masks(X, Y):
            train_idx = idxs[np.logical_not(test_idx_mask)]
            test_idx = idxs[test_idx_mask]
            yield train_idx, test_idx

    def _iter_test_masks(self, X, Y):
        """Generates boolean masks corresponding to test sets"""
        n_sample = X.shape[1]
        for test_index in self._iter_test_idx(X, Y):
            test_mask = np.zeros(n_sample, dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    
    def _iter_test_idx(self, X, Y):
        pass

class Kfold(_BaseCrossValidator):
       
    def __init__(self, K=3):
        super().__init__()
        self.K = K
    
    def _iter_test_idx(self, X, Y):
        train_idxs = []
        test_idx = []
        n_samples = X.shape[1]
        indices = np.arange(n_samples)
        K = self.K
        fold_sizes = np.full(K, n_samples // K, dtype=int)
        fold_sizes[: n_samples % K] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

class StratifiedKfold(_BaseCrossValidator):
    def __init__(self, K=3):
        self.K = K

        def _make_test_folds(self, X, y=None):
            _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
            _, class_perm = np.unique(y_idx, return_inverse=True)
            y_encoded = class_perm[y_inv]
            n_classes = len(y_idx)
            y_counts = np.bincount(y_encoded)
            min_groups = np.min(y_counts)
            if np.all(self.n_splits > y_counts):
                raise ValueError(
                    "n_splits=%d cannot be greater than the"
                    " number of members in each class." % (self.n_splits)
                )

            # Determine the optimal number of samples from each class in each fold,
            y_order = np.sort(y_encoded)
            allocation = np.asarray(
                [
                    np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                    for i in range(self.n_splits)
                ]
            )

            test_folds = np.empty(len(y), dtype="i")
            for k in range(n_classes):
                # since the kth column of allocation stores the number of samples
                # of class k in each test set, this generates blocks of fold
                # indices corresponding to the allocation for class k.
                folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
                test_folds[y_encoded == k] = folds_for_class
            return test_folds

        def _iter_test_masks(self, X, y=None, groups=None):
            test_folds = self._make_test_folds(X, y)
            for i in range(self.n_splits):
                yield test_folds == i


class LeaveOneOut(_BaseCrossValidator):

    def _iter_test_idx(self, X, Y):
        n_samples = X.shape[1]
        return range(n_samples)


def cross_val_score(model, D, L, cv ,scoring='accuracy'):
    scores = []
    for train_idx, test_idx in cv.cross_validation_split(D, L):
        model.fit(D[:, train_idx], L[train_idx])
        pred = model.predict(D[:, test_idx])
        score = compute_accuracy(pred, L[test_idx])
        if scoring == 'error':
            score = 100 - score
        scores.append(score)
    return scores



