import math
import random
import numpy as np
from sklearn.metrics import mean_squared_error

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

from sklearn.datasets import load_breast_cancer
import datetime
#import k_means_constrained

number_group=5
n_trees = int(300)
desired_TPR = int(80)
desired_TPR /= 100.0

def size_constrained_cluster(X):
    N = len(X)  # 569
    M = len(X[0])  # 30

    fft_X = np.zeros([N, M])
    fft_X_std = np.zeros(M)
    for i in range(M):
        fft_X[:, i] = np.abs(fft(X[:, i])) / N
        #     fft_X_std[i] = np.std(fft_X[:,i])
        fft_X_std[i] = np.mean(fft_X[:, i])

    Matrix_dis = np.zeros([M, M])
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            Matrix_dis[i, j] = np.abs(fft_X_std[j] - fft_X_std[i])
    #         Matrix_dis[i,j] = coe_sim(fft_X[:,j],fft_X[:,i])

    n_clusters = 5
    # clustering2 = k_means_constrained.KMeansConstrained(n_clusters=n_clusters, size_min=6, size_max=29, random_state=0)
    # clustering2 = clustering2.fit_predict(Matrix_dis)


    # print("clustering2:", clustering2)


def pr_anomalies_single_input(it,X, y,threshold, sample_size=256, n_trees = 100, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    N = len(X)
    score, y_pred = it.predictSingleInstance(X, threshold=threshold)
    return score


def pr_anomalies(it,X, y, threshold, sample_size=256, n_trees = 100, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    N = len(X)
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
#     print(y, scores, desired_TPR)
    if desired_TPR is not None:
        try:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        #print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
        except:
            print(y, scores, desired_TPR)
            print(type(y), type(scores), type(desired_TPR))
    else:
        threshold = np.percentile(scores, percentile)
    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)
    confusion = confusion_matrix(y, y_pred)

    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = scores[y==0]

    anomalies = scores[y==1]
    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)

    return PR

def calculatShapley(cFunction,coalition,nPlayer):
    coalition=list(coalition)
    for i in range(0,len(coalition)):
        coalition[i]=list(coalition[i])


    #print("start calculate shapley:")
    shapley_values = []
    for i in range(len(nPlayer)):
        shapley = 0
        for j in coalition:
            if i not in j:
                j=list(j)
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = coalition.index(j)
                k = coalition.index(Cui)
                temp = float(float(cFunction[k]) - float(cFunction[l])) *\
                           float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = coalition.index(Cui)
        temp = float(cFunction[k]) * float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
        shapley += temp

        shapley_values.append(shapley)

    return (shapley_values)


def getcoaltionlist(n):
    coalition = []
    singles = tuple([i for i in range(n)])
    for i in range(1, n + 1):
        for p in itertools.combinations(singles, i):
            coalition.append(p)
    return coalition


def get_Phi_2_level(idx_dic,it,threshold,X, y,local):

    Phi_1level = {}
    playerlist=list(range(0,30))
    groupPlayer=list(range(number_group))
    shapvalue2level=[0 for _ in range(30)]

    plot30playerlist= ['mean radius', 'mean texture', 'mean perimeter','mean area',\
            'mean smoothness', 'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error',\
            'texture error','perimeter error','area error','smoothness error','compactness error',\
            'concavity error','concave points error','symmetry error','fractal dimension error','worst radius',\
            'worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity',\
            'worst concave points','worst symmetry','worst fractal dimension']

    for k in range(number_group):
        characteristic_function = []
        coalition = getcoaltionlist(len(idx_dic[k]))
        subplayerlist=list(range(0,(len(idx_dic[k]))))
       # print("group,coalition:",k,len(coalition))
        for j in range(len(coalition)):
            tcoalition = np.array(coalition[j])
            cX = X.copy()
            cX = np.array(cX)
            cy = y
            removeplayerlist = []
            for i in tcoalition:
                removeplayerlist.append(np.array(idx_dic[k])[i])
            diff = set(playerlist) ^ set(removeplayerlist)

            if (len(diff) != 0):
                cX[:, tuple(diff)] = np.zeros(np.shape(cX[:, tuple(diff)]))

            if local == 1:
                pr = pr_anomalies_single_input(it, cX, y, threshold, n_trees=n_trees, desired_TPR=desired_TPR)
                characteristic_function.extend(pr)
            else:
                pr = pr_anomalies(it, cX, y, threshold, n_trees=n_trees, desired_TPR=desired_TPR)
                characteristic_function.append(pr)

        shap2level=calculatShapley(characteristic_function,coalition,subplayerlist)
        jj=0

        for ii in (idx_dic[k]):
            shapvalue2level[ii]=shap2level[jj]
            jj=jj+1

        forplot=shap2level.copy()
        if local!=1:
            if(k==k):
                plotplayerlist=np.array(plot30playerlist)[np.array(idx_dic[k])]
                print("group, member",k,plotplayerlist)
                plotgraph(plotplayerlist, [a/sum(forplot) for a in forplot], [k,k,k,k,k,k])


       # print("group:",k)


    shapvalue2level=[a/sum(shapvalue2level) for a in shapvalue2level]
    return shapvalue2level


def get_Phi_1_level(idx_dic,it,threshold,X, y,local):
    coalition = getcoaltionlist(number_group)
    characteristic_function=[]
    Phi_1level = {}
    playerlist=list(range(0,30))
    groupPlayer=list(range(number_group))

    for k in range(len(coalition)):
        cX=X.copy()
        cX=np.array(cX)
        cy=y
        removeplayerlist= []
        for j in coalition[k]:
            removeplayerlist.extend(idx_dic[j])
        diff = set(playerlist) ^ set(removeplayerlist)

        if (len(diff) != 0):
            cX[:, tuple(diff)] = np.zeros(np.shape(cX[:, tuple(diff)]))
        if local==1:
            pr = pr_anomalies_single_input(it,cX, y,threshold, n_trees=n_trees, desired_TPR=desired_TPR)
            characteristic_function.extend(pr)
        else:
            pr = pr_anomalies(it,cX, y,threshold, n_trees=n_trees, desired_TPR=desired_TPR)
            characteristic_function.append(pr)

    print("characteristic function", characteristic_function)

    shap1level=calculatShapley(characteristic_function,coalition,(groupPlayer))
    sha1levelnew=[a/sum(shap1level) for a in shap1level]
    return sha1levelnew



def calculateFirstLevelMultipleSecondLevel(X,y,clusters,threshold,it,local):
    idx_dic = {0: [5, 6, 25, 26, 27, 28], 1: [2, 3, 13, 21, 22, 23], 2: [4, 9, 14, 17, 18, 19],
               3: [7, 8, 15, 16, 24, 29], 4: [0, 1, 10, 11, 12, 20]}
    #Shap_2level = shap_2level(X, idx_dic, [0, 1, 2, 3, 4],threshold,it)
    plot30group = [4, 4, 1, 1, 2, 0, 0, 3, 3, 2, 4, 4, 4, 1, 2, 3, 3, 2, 2, 2, 4, 1, 1, 1, 3, 0, 0, 0, 0, 3]
    plot30playerlist= ['mean radius', 'mean texture', 'mean perimeter','mean area',\
            'mean smoothness', 'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error',\
            'texture error','perimeter error','area error','smoothness error','compactness error',\
            'concavity error','concave points error','symmetry error','fractal dimension error','worst radius',\
            'worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity',\
            'worst concave points','worst symmetry','worst fractal dimension']
    plot5playerlist=['Group 0', 'Group 1','Group 2', 'Group 3', 'Group4']
    plotg5roup=[0,1,2,3,4]
    Phi_1level= get_Phi_1_level(idx_dic,it,threshold,X,y,local)
    print("1 level:", Phi_1level)
    phi_1_level_30=[0 for _ in range(30)]

    if local!=1:
        #huatu:
        plotgraph(plot5playerlist, Phi_1level, plotg5roup)
        #####

    Phi_2level= get_Phi_2_level(idx_dic, it, threshold, X, y,local)

    for x in idx_dic:
        for j in idx_dic[x]:
            phi_1_level_30[j]=Phi_1level[x]

  #  print("1 level:",Phi_1level)

    result=[Phi_2level[a]*phi_1_level_30[a] for a in range(0,30)]
    result=[a/sum(result) for a in result]
    if local!=1:
        #huatu:
        plotgraph(plot30playerlist, result, plot30group)
        #####

   # print("final result", result)
   # print("sum:",sum(result))
    return result,phi_1_level_30


def bilevel_explainer_local(X,y):
   # size_constrained_cluster(X)
    clusters = [4, 4, 1, 1, 2, 0, 0, 3, 3, 2, 4, 4, 4, 1, 2, 3, 3, 2, 2, 2, 4, 1, 1, 1, 3, 0, 0, 0, 0, 3]
    local=1

    instanceSize = X.shape[0]
    ResultOfBilevel=[]
    it = IsolationTreeEnsemble(sample_size=256, n_trees=100)
    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    threshold=0.5
    #     print(y, scores, desired_TPR)
    if desired_TPR is not None:
        try:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        # print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
        except:
            print(y, scores, desired_TPR)
            print(type(y), type(scores), type(desired_TPR))
    else:
        threshold = np.percentile(scores, percentile=None)



    X_copyup = X.copy()
    yup=y

    for bigi in range(0, instanceSize):
        X_ = X_copyup[bigi, :]
        y = yup[bigi]
        X_ = X_.reshape((1, 30))
        sha2,phi_1level=calculateFirstLevelMultipleSecondLevel(X_, y, clusters,threshold,it,local)
        print(bigi)
        if(bigi==0):
            ResultOfBilevel=np.array(sha2)
        else:
            ResultOfBilevel=np.append(ResultOfBilevel,np.array(sha2),axis=0)

        print("result:", ResultOfBilevel)

    ResultOfBilevel=ResultOfBilevel.reshape(instanceSize,30)
    str = '/Users/jes/Desktop/学/testresult/ResultOfBilevel' + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S") + '.txt'

    np.savetxt(str, ResultOfBilevel)

    return 1

def bilevel_explainer_globle(X,y):
   # size_constrained_cluster(X)
    clusters = [4, 4, 1, 1, 2, 0, 0, 3, 3, 2, 4, 4, 4, 1, 2, 3, 3, 2, 2, 2, 4, 1, 1, 1, 3, 0, 0, 0, 0, 3]
    local=0

    instanceSize = X.shape[0]
    ResultOfBilevel=[]
    it = IsolationTreeEnsemble(sample_size=256, n_trees=100)
    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    threshold=0.5
    #     print(y, scores, desired_TPR)
    if desired_TPR is not None:
        try:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        # print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
        except:
            print(y, scores, desired_TPR)
            print(type(y), type(scores), type(desired_TPR))
    else:
        threshold = np.percentile(scores, percentile=None)

    X_copyup = X.copy()
    yup=y


    sha2,phi1level=calculateFirstLevelMultipleSecondLevel(X_, y, clusters,threshold,it,local)


   # ResultOfBilevel=np.reshape(-1,30)
    print("final sha2", sha2)

    str = '/Users/jes/Desktop/学/testresult/ResultOfBilevel' + datetime.datetime.now().strftime(
         "%Y-%m-%d %H:%M:%S") + '.txt'
    np.savetxt(str, ResultOfBilevel)

    return sha2,phi1level


def plotgraph(playerlist,Shapvalue,group):
    df = pd.DataFrame({'eventlist':playerlist, 'shapleys': Shapvalue,'group':group})

    df2=df.sort_values(by=['shapleys'], axis=0, ascending=[True])
    colors=['lightsteelblue','g','r','orange','yellow']

    for i in range(len(playerlist)):
        plt.bar(x=0, bottom=i, height=0.9, width=(np.array(df2['shapleys'])[i]), orientation="horizontal", color=np.array(colors)[np.array(df2['group'])[i]])
       # print("i",np.array(df['shapleys'])[i])
    plt.yticks(range(len(df['eventlist'])), np.array(df2['eventlist']))
    plt.show()

if __name__ == '__main__':


    print("test")
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target
    df.loc[df.diagnosis==0,'diagnosis'] = -1
    df.loc[df.diagnosis==1,'diagnosis'] = 0
    df.loc[df.diagnosis==-1,'diagnosis'] = 1
    df.to_csv("cancer.csv", index=False)

    # load JS visualization code to notebook

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    # suss= np.loadtxt('/Users/jes/Desktop/学/testresult/samplingresult.txt')
    # print("loading",suss)


   # suss= np.loadtxt('/Users/jes/Desktop/学/testresult/ResultOfBilevel2020-12-10-04-41-32.txt')
   # print("loading",suss)

    datafile = 'cancer.csv'

    targetcol = 'diagnosis'
    X, y = df.drop(targetcol, axis=1), df[targetcol]
    X_ = np.array(X)

    bilevel_explainer_local(X_,y)
    #sha2,sha1=bilevel_explainer_globle(X,y)


