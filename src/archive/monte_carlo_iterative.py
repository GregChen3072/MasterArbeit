# Load population
import pprint
import pandas as pd

from numpy.lib.function_base import average
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# For computing multiple metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate

from sklearn.ensemble import AdaBoostClassifier

from ref.database import Database

import pickle


def make_database(x, y):
    return Database(x, y)


X, y = load_breast_cancer(return_X_y=True)

# sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.2, random_state=6)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=6)

clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=6)

sss.get_n_splits(X, y)

# Collect results
res = {
    'classifier': 'classifier',
    'acc': 'acc',
    'auc': 'auc',
    'mcc': 'mcc',
    'f-1': 'f-1'
}

# Create 1000 Validation Sets
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Pack up test set
    # Split train set into n sites | split 10 DB pairs | simulate 10 sites with class imbalance

    # Train classical | combiner | iterator using train set
    # Test  classical | combiner | iterator using test  set
    # Get   acc | auc | mcc | f-1   scores for each validation


# https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# scoring = ['precision_macro', 'recall_macro']
scoring = {
    'acc': 'balanced_accuracy',
    'auc': 'roc_auc',
    'mcc': make_scorer(matthews_corrcoef),
    'f-1': 'f1'
}
