# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:34:59 2017

@author: radha
"""

import pandas as pd

xl = pd.ExcelFile('FPR.xlsx')
fpr = xl.parse('Sheet1')
fprOS1 = fpr[1][0:35]
fprOS2 = fpr[2][0:42]
fprOS3 = fpr[3][0:62]
fprOS4 = fpr[4][0:71]
fprOS5 = fpr[5][0:73]
fprS1 = fpr[1][35:74]
fprS2 = fpr[2][42:79]
fprS3 = fpr[3][62:105]
fprS4 = fpr[4][71:120]
fprS5 = fpr[5][73:128]
fprSUS1 = fpr[1][74:102]
fprSUS2 = fpr[2][79:108]
fprSUS3 = fpr[3][105:137]
fprSUS4 = fpr[4][120:151]
fprSUS5 = fpr[5][128:154]

xl = pd.ExcelFile('TPR.xlsx')
tpr = xl.parse('Sheet1')
tprOS1 = tpr[1][0:35]
tprOS2 = tpr[2][0:42]
tprOS3 = tpr[3][0:62]
tprOS4 = tpr[4][0:71]
tprOS5 = tpr[5][0:73]
tprS1 = tpr[1][35:74]
tprS2 = tpr[2][42:79]
tprS3 = tpr[3][62:105]
tprS4 = tpr[4][71:120]
tprS5 = tpr[5][73:128]
tprSUS1 = tpr[1][74:102]
tprSUS2 = tpr[2][79:108]
tprSUS3 = tpr[3][105:137]
tprSUS4 = tpr[4][120:151]
tprSUS5 = tpr[5][128:154]

xl = pd.ExcelFile('AUC.xlsx')
roc = xl.parse('Sheet1')
rocOS1 = roc[1][0]
rocOS2 = roc[2][0]
rocOS3 = roc[3][0]
rocOS4 = roc[4][0]
rocOS5 = roc[5][0]
rocS1 = roc[1][1]
rocS2 = roc[2][1]
rocS3 = roc[3][1]
rocS4 = roc[4][1]
rocS5 = roc[5][1]
rocSUS1 = roc[1][2]
rocSUS2 = roc[2][2]
rocSUS3 = roc[3][2]
rocSUS4 = roc[4][2]
rocSUS5 = roc[5][2]

import matplotlib.pyplot as plt
plt.figure()
lw = 2

plt.plot(fprOS1, tprOS1, lw=lw, label='100% Random Resampling')
plt.plot(fprS1, tprS1, lw=lw, label='100% SMOTE')
plt.plot(fprSUS1, tprSUS1, lw=lw, label='100% SMOTE; 10% Under Sampling')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()

plt.plot(fprOS2, tprOS2, lw=lw, label='200% Random Resampling')
plt.plot(fprS2, tprS2, lw=lw, label='200% SMOTE')
plt.plot(fprSUS2, tprSUS2, lw=lw, label='200% SMOTE; 15% Under Sampling')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()

plt.plot(fprOS3, tprOS3, lw=lw, label='300% Random Resampling')
plt.plot(fprS3, tprS3, lw=lw, label='300% SMOTE')
plt.plot(fprSUS3, tprSUS3, lw=lw, label='300% SMOTE; 25% Under Sampling')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()

plt.plot(fprOS4, tprOS4, lw=lw, label='400% Random Resampling')
plt.plot(fprS4, tprS4, lw=lw, label='400% SMOTE')
plt.plot(fprSUS4, tprSUS4, lw=lw, label='400% SMOTE; 50% Under Sampling')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()

plt.plot(fprOS5, tprOS5, lw=lw, label='500% Random Resampling')
plt.plot(fprS5, tprS5, lw=lw, label='500% SMOTE')
plt.plot(fprSUS5, tprSUS5, lw=lw, label='500% SMOTE; 75% Under Sampling')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Pima Dataset')
plt.show()
