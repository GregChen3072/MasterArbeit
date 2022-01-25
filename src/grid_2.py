from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import time
import pandas as pd
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

from db_simulator import load_HCC_data, load_ILPD_data
from scoring import make_scores

import pickle
X, y = load_ILPD_data()
parameters = {'n_estimators': [1, 10]}

abc = AdaBoostClassifier(random_state=6)

clf = GridSearchCV(abc, parameters)
clf.fit(X, y)


sorted(clf.cv_results_.keys())
