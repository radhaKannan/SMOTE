# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:27:27 2017

@author: radha
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np
import random

data = pd.read_csv("data.csv")
data = np.matrix(data)
labels = data[:,8]
labels = np.array(labels)
data = data[:,:8]

no_ones = np.count_nonzero(labels)
no_zeros = labels.size-no_ones

data_ones = np.zeros((1,8))
for record,label in zip(data,labels):
    if(label == 1):
        data_ones = np.append(data_ones, record, axis = 0)
data_ones = np.delete(data_ones, 0, 0)

def smoteUnderSampling(samplingAmount, underSample, data_ones, data, labels):
    nbrs = NearestNeighbors(n_neighbors=samplingAmount, algorithm='ball_tree').fit(data_ones)
    distances, indices = nbrs.kneighbors(data_ones)
    for record in indices:
        feature = record[0]
        for index in record:
            if(index != feature):
                mul = random.random()
                new = [fv + (mul * (nn - fv)) for nn, fv in zip(data_ones[index], data_ones[feature])]
                data = np.append(data, new[0], axis = 0)
                labels = np.append(labels, 1)

    no_ones = np.count_nonzero(labels)
    no_zeros = labels.size - no_ones
    max_no_zeros = no_zeros - ((underSample/100) * no_zeros)
    
    while((no_zeros-max_no_zeros) > 0):
        index = random.randint(0,labels.size-1)
        if(labels[index] == 0):
            data = np.delete(data, index, 0)
            labels = np.delete(labels, index)
            no_zeros = no_zeros - 1

    test_data = np.zeros((1,8))
    test_labels = np.zeros(1)
    
    i = 0
    testSet = []
    while i < .1*(data.shape[0]):
        testIndex = random.randint(0,data.shape[0]-1)
        if(testIndex not in testSet):
            testSet.append(testIndex)
            test_data = np.append(test_data, data[testIndex], axis = 0)
            test_labels = np.append(test_labels, labels[testIndex])
            data = np.delete(data, testIndex, 0)
            labels = np.delete(labels, testIndex)
            i = i + 1
    test_data = np.delete(test_data, 0, 0)
    test_labels = np.delete(test_labels, 0)
    
    return data, labels, test_data, test_labels

def classifier(data, labels, test_data, test_labels):
    logistic = linear_model.LogisticRegression()
    LR_Predict = logistic.fit(data, labels).predict_proba(test_data)
    LR_Prob = []
    for record in LR_Predict:
        LR_Prob.append(record[1])
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, LR_Prob, pos_label=1)
    roc_curve_area = roc_auc_score(test_labels, LR_Prob)
    
    np.savetxt("FPR.csv", fpr, delimiter=",")
    np.savetxt("TPR.csv", tpr, delimiter=",")
    print(roc_curve_area)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve area %f' % roc_curve_area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

underSample = [10,15,25,50,75]
for x in range(6,7):
    print('----',(x-1)*100,'% OverSampling ----',underSample[x-2],'% UnderSampling ----')
    data1, labels1, test_data1, test_labels1 = smoteUnderSampling(x, underSample[x-2], data_ones[:], data[:], labels[:])
    classifier(data1, labels1, test_data1, test_labels1)