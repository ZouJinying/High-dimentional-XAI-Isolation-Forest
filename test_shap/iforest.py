
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.height_limit = np.ceil(np.log2(self.sample_size))
        self.trees = []
        for i in range(self.n_trees):
            X_sample = X[random.sample(range(X.shape[0]), self.sample_size)]
            t = IsolationTree(self.height_limit)
            t.root = t.fit(X_sample, improved)
            self.trees.append(t)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        path_matrix = np.zeros((X.shape[0], self.n_trees))
        indexes = np.array([True] * X.shape[0])
        for j, t in enumerate(self.trees):
            currHeight = 0
            pathLength(path_matrix, indexes, X, j, t.root, currHeight)
        return np.mean(path_matrix, axis=1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_path = self.path_length(X)
        score = 2**(-(avg_path/c(self.sample_size)))
        return score

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores>=threshold)*1

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        if isinstance(X, pd.DataFrame):
            X = X.values

        score = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(score)

    def predictSingleInstance(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        if isinstance(X, pd.DataFrame):
            X = X.values

        score = self.anomaly_score(X)
        return score, (score>=threshold)*1


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.height = 0
        self.root = None

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if not improved:
            n_nodes =1
            self.root, n_nodes = fit2(X, self.height, self.height_limit, n_nodes)
            self.n_nodes = n_nodes
        else:
            n_nodes =1
            self.root, n_nodes = fit_improved(X, self.height, self.height_limit, n_nodes)
            self.n_nodes = n_nodes
        return self.root


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    for threshold in np.arange(1,0,-0.001):
        y_pred = (scores>=threshold)*1
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if desired_TPR*.85<TPR:
            return threshold, FPR


def fit_improved(X: np.ndarray, height, limit, n_nodes):

    if height >= limit or len(X) <2:
        return exNode(len(X)), n_nodes


    else:
        rows, cols = X.shape
        good_split = True

        while good_split:
            choosen_col = np.random.randint(0,cols)
            mx = X[:, choosen_col].max()
            mn = X[:, choosen_col].min()

            if mx==mn:
                return exNode(len(X)), n_nodes


            split = np.random.uniform(mn, mx)
            Xl = X[X[:, choosen_col] < split]
            Xr = X[X[:, choosen_col] >= split]


            if len(Xl)<=0.15*len(Xr) or len(Xr)<0.15*len(Xl) or len(X)<12:
                good_split=False
        left, n_nodes = fit2(Xl, height + 1, limit, n_nodes+1)
        right, n_nodes = fit2(Xr, height + 1, limit, n_nodes+1)
        return inNode(left,
                          right,
                          choosen_col,
                          split), n_nodes


def fit2(X: np.ndarray, height, limit, n_nodes):

    if height >= limit or len(X) <2:
        return exNode(len(X)), n_nodes


    else:
        rows, cols = X.shape
        good_split = True


        choosen_col = np.random.randint(0,cols)
        mx = X[:, choosen_col].max()
        mn = X[:, choosen_col].min()

        if mx==mn:
            return exNode(len(X)), n_nodes


        split = np.random.uniform(mn, mx)
        Xl = X[X[:, choosen_col] < split]
        Xr = X[X[:, choosen_col] >= split]
        left, n_nodes = fit2(Xl, height + 1, limit, n_nodes+1)
        right, n_nodes = fit2(Xr, height + 1, limit, n_nodes+1)
        return inNode(left,
                          right,
                          choosen_col,
                          split), n_nodes


def H(i):
    return np.log(i) + 0.5772156649


def c(size):
    if size > 2:
        return 2 * H(size - 1) - 2 * (size - 1.0) / size
    elif size == 2:
        return 1
    else:
        return 0


class exNode:
    def __init__(self, size):
        self.size = size


class inNode:
    def __init__(self, left, right, splitAtt, splitValue):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitValue = splitValue


def pathLength(path_matrix,obs_indexes,x,tree_index, Tree,currHeight):
    if isinstance(Tree,exNode):
        path_matrix[obs_indexes, tree_index] = currHeight+c(Tree.size)
    else:
        a=Tree.splitAtt
        left_indexes = (obs_indexes)*(x[:,a]<Tree.splitValue)
        pathLength(path_matrix, left_indexes,x,tree_index,Tree.left,currHeight+1)
        right_indexes = (obs_indexes)*(x[:,a]>=Tree.splitValue)
        pathLength(path_matrix, right_indexes,x,tree_index,Tree.right,currHeight+1)