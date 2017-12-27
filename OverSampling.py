# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:41:37 2017

@author: radha
"""

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

def overSample(samplingAmount, data_ones, data, labels, no_ones):
    max_no_ones = samplingAmount * no_ones
    while (max_no_ones-no_ones) != 0:
        index = random.randint(0,(data_ones.shape[0]-1))
        data = np.append(data, data_ones[index], axis = 0)
        labels = np.append(labels, 1)
        no_ones = no_ones + 1

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

for x in range(2,7):
    print('----',(x-1)*100,'% OverSampling ----')
    data1, labels1, test_data1, test_labels1 = overSample(x, data_ones[:], data[:], labels[:], no_ones)
    classifier(data1, labels1, test_data1, test_labels1)