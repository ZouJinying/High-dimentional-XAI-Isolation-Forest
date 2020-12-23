import xgboost_xai
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

import bilevel_xai_explainer
import sampling_xai_explianer

if __name__ == "__main__":
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target
    df.loc[df.diagnosis == 0, 'diagnosis'] = -1
    df.loc[df.diagnosis == 1, 'diagnosis'] = 0
    df.loc[df.diagnosis == -1, 'diagnosis'] = 1
    df.to_csv("cancer.csv", index=False)

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()

    samp = np.loadtxt('/Users/jes/Desktop/学/testresult/samplingresult2020-12-19-19-16-07'
                      '.txt')
    xgb = np.loadtxt('/Users/jes/Desktop/学/testresult/imporant/xgboostresult2020-12-09-01-57-54.txt')
    bilev = np.loadtxt('/Users/jes/Desktop/学/testresult/imporant/ResultOfBilevel2020-12-10-09-22-13.txt')

    sumtotalS=[]
    sumtotalB=[]





    ##average
    avesam=samp.copy()
    avexgb=xgb.copy()
    avebile=bilev.copy()

    sampingave=[]
    xgbave=[]
    bilevave=[]

    for i in range(0,30):
        sampingave.append(np.mean(avesam[:, i]))
        xgbave.append(np.mean(avexgb[:,i]))
        bilevave.append(np.mean(avebile[:,i]))


##least square error
    for i in range(0,30):
        sumSamp = 0
        sumBile = 0
        tempS= samp[:, i]-xgb[:,i]
        tempB=bilev[:, i]-xgb[:,i]
        for j in range(len(tempS)):
            sumSamp+=tempS[j]*tempS[j]
            sumBile += tempB[j] * tempB[j]
        sumSamp=sumSamp/len(tempS)
        sumBile=sumBile/len(tempB)
        sumtotalS.append(sumSamp)
        sumtotalB.append(sumBile)
    sumtotalB=np.array(sumtotalB).round(5)
    sumtotalS=np.array(sumtotalS).round(5)
    lqm=[]



