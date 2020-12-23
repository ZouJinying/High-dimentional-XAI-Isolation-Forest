import xgboost
import shap

from sklearn.datasets import load_breast_cancer
import pandas as pd
import pickle
import math
import random

import numpy as np
import itertools
import bisect
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import datetime
import matplotlib.pyplot as plt
import sys
import time
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, \
    confusion_matrix, f1_score, average_precision_score
from iforest import IsolationTreeEnsemble, find_TPR_threshold
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics


if __name__ == "__main__":
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target
    df.loc[df.diagnosis==0,'diagnosis'] = -1
    df.loc[df.diagnosis==1,'diagnosis'] = 0
    df.loc[df.diagnosis==-1,'diagnosis'] = 1
    df.to_csv("cancer.csv", index=False)

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X,y = shap.datasets.boston()

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)


    datafile = 'cancer.csv'

    targetcol = 'diagnosis'
    sample_size = int(5)
    n_trees = int(300)
    desired_TPR = int(80)
    desired_TPR /= 100.0

    X, y = df.drop(targetcol, axis=1), df[targetcol]
    # load JS visualization code to notebook
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)


    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(shap_values)
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

    str = '/Users/jes/Desktop/学/testresult/xgboostresult' + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S") + '.txt'

    suss= np.loadtxt('/Users/jes/Desktop/学/testresult/xgboostresult2020-12-09-01-57-54.txt')
    print("loading",suss)
    np.savetxt(str, shap_values)