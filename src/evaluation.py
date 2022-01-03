# Load population
import pandas as pd

from sklearn.datasets import load_breast_cancer

# For computing multiple metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate

from sklearn.ensemble import AdaBoostClassifier

from ref.database import Database

from pipeline_1_1 import pipeline_1_1
from pipeline_2_1 import pipeline_2_1
from pipeline_3_1 import pipeline_3_1

from pipeline_2_2 import pipeline_2_2_unweighted, pipeline_2_2_weighted
from pipeline_3_2 import pipeline_3_2_unweighted, pipeline_3_2_weighted


import pickle


def make_database(x, y):
    return Database(x, y)


X, y = load_breast_cancer(return_X_y=True)

# sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.2, random_state=6)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)
sss.get_n_splits(X, y)

# Settings
n_estimators = 1000

# Container of results of 1000 repetitions
result_container = None

# Create 1000 Validation Sets
for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Pipeline 1 1 (implementation completed)
    # When all data centralized in one database.
    # pipeline_1_1(X_train, X_test, y_train, y_test)

    # Pipeline 2 1 (implementation completed)
    # pipeline_2_1(X_train, X_test, y_train, y_test)

    # Pipeline 3 1 (implementation completed)
    # pipeline_3_1(X_train, X_test, y_train, y_test)

    # Pipeline 1 2
    # Doesn't exist.
    # None

    # Pipeline 2 2 unweighted
    # pipeline_2_2_unweighted(X_train, X_test, y_train, y_test)

    # Pipeline 3 2 unweighted
    # pipeline_3_2_unweighted(X_train, X_test, y_train, y_test)

    # Pipeline 2 2 weighted
    pipeline_2_2_weighted(X_train, X_test, y_train, y_test)

    # Pipeline 3 2 weighted
    # pipeline_3_2_weighted(X_train, X_test, y_train, y_test)

    # Pack up test set
    # Split train set into n sites | split 10 DB pairs | simulate 10 sites with class imbalance

    # Train classical | combiner | iterator using train set
    # Test  classical | combiner | iterator using test  set
    # Get   acc | auc | mcc | f-1   scores for each validation

# clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=6)
