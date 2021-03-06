# SMOTE

Dealing with imbalanced data-sets: random re-sampling, synthetic oversampling and under-sampling.

Language Used: Python

Version: Python v3.6

Dependencies: roc_auc_score, matplotlib.pyplot, linear_model, metrics, pandas, numpy, random, NearestNeighbors

Executable Files: (a) ActualClassification.py, (b) OverSampling.py, (c) Smote.py, (d) SmoteDownSample.py, (e) Plots.py

Pima: data.csv is the input file for all the .py files except the Plots.py file

Part (a) outputs the ROC curve plot and the AUC score to console. The parts (b, c and d) should be run for one iteration every time and they generate two files. One file contains the False Positive Rate Values and the second file contains the True Positive Rate Values. We need to copy these values to another file and re run the respective .py file to generate the next set of FPR and TPR values. To the Plots.py file, input the aggregated FPR and TPR value files and it will plot the desired ROC curves. The AUC for each iteration will be output to the console during each of its run.