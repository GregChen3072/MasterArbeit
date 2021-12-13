# Load population
import pprint
import pandas as pd
from statistics import mean

from numpy.lib.function_base import average
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# For computing multiple metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate

from sklearn.ensemble import AdaBoostClassifier

from ref.database import Database
from scoring import make_scores

import pickle


def make_database(x, y):
    return Database(x, y)


# https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# scoring = ['precision_macro', 'recall_macro']
scoring = {
    'acc': 'balanced_accuracy',
    'auc': 'roc_auc',
    'mcc': make_scorer(matthews_corrcoef),
    'f-1': 'f1'
}

X, y = load_breast_cancer(return_X_y=True)

# sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.2, random_state=6)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=6)

clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=6)

# https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# scoring = ['precision_macro', 'recall_macro']
scoring = {
    'acc': 'balanced_accuracy',
    'auc': 'roc_auc',
    'mcc': make_scorer(matthews_corrcoef),
    'f-1': 'f1'
}

scores = cross_validate(
    clf,
    X,
    y,
    cv=sss,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False,  # not necessary
    return_estimator=False
)

pprint.pprint(scores)


'''
sss.get_n_splits(X, y)

list_of_classifiers = list()
list_of_acc = list()
list_of_auc = list()
list_of_mcc = list()
list_of_f_1 = list()


# Create 1000 Validation Sets
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf_classical = clf.fit(X_train, y_train)
    f_1, mcc, auc, acc = make_scores(clf_classical, X_test, y_test)

    list_of_classifiers.append(clf_classical)
    list_of_acc.append(acc)  # 1000 elements
    list_of_auc.append(auc)  # ...
    list_of_mcc.append(mcc)
    list_of_f_1.append(f_1)

print(mean(list_of_acc))
print(mean(list_of_auc))
print(mean(list_of_mcc))
print(mean(list_of_f_1))
'''
