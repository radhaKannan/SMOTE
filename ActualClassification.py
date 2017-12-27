# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:35:57 2017

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

test_data = np.zeros((1,8))
test_labels = np.zeros(1)

i = 0
testSet = []
while i < (labels.size * .1):
    testIndex = random.randint(0,labels.size-1)
    if(testIndex not in testSet):
        testSet.append(testIndex)
        test_data = np.append(test_data, data[testIndex], axis = 0)
        test_labels = np.append(test_labels, labels[testIndex])
        data = np.delete(data, testIndex, 0)
        labels = np.delete(labels, testIndex)
        i = i + 1
test_data = np.delete(test_data, 0, 0)
test_labels = np.delete(test_labels, 0)

logistic = linear_model.LogisticRegression()
LR_Predict = logistic.fit(data, labels).predict_proba(test_data)
LR_Prob = []
for record in LR_Predict:
    LR_Prob.append(record[1])
fpr, tpr, thresholds = metrics.roc_curve(test_labels, LR_Prob, pos_label=1)
roc_curve_area = roc_auc_score(test_labels, LR_Prob)
print(roc_curve_area)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Actual Dataset')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()