from sklearn.datasets import make_classification
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

import pickle


# https://www.kaggle.com/prashant111/adaboost-classifier-tutorial/notebook
# https://www.kaggle.com/alexisbcook/pipelines


# start_time = time.time()

# Import data
path_data_credit_card = "/Users/greg/Downloads/AR_Master_Thesis/data/creditcard.csv"
path_HCC = "/Users/greg/Downloads/AR_Master_Thesis/data/HCC_preprocessed.csv"
path_ILPD = "/Users/greg/Downloads/AR_Master_Thesis/data/ILPD_preprocessed.csv"

data_credit_card = pd.read_csv(path_data_credit_card)

# iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


X, y = make_classification(n_samples=53, n_features=5,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=5, learning_rate=1, random_state=0)

# Train Adaboost Classifer
model1 = abc.fit(X_train, y_train)

print(len(model1.estimators_))

# Predict the response for test dataset
y_pred = model1.predict(X_test)

# calculate and print model accuracy
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))

# load required classifer


# import Support Vector Classifier

""" 
# import scikit-learn metrics module for accuracy calculation
svc = SVC(probability=True, kernel='linear')


# create adaboost classifer object
abc = AdaBoostClassifier(
    n_estimators=50, base_estimator=svc, learning_rate=1, random_state=0)


# train adaboost classifer
model2 = abc.fit(X_train, y_train)


# predict the response for test dataset
y_pred = model2.predict(X_test)


# calculate and print model accuracy
print("Model Accuracy with SVC Base Estimator:", accuracy_score(y_test, y_pred))
 """
