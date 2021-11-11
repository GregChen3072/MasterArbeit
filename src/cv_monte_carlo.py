# Load population
import pprint
from numpy.lib.function_base import average
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# For computing multiple metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate

from sklearn.ensemble import AdaBoostClassifier

import pickle


X, y = load_breast_cancer(return_X_y=True)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=6)

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
    clf, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True, return_estimator=True)

# keys = ['test_acc', 'test_auc', 'test_mcc', 'test_f-1']
# scores_to_print = [scores.get(key) for key in keys]

# print(sorted(scores.keys()))
# print(scores['test_mcc'])
pprint.pprint(scores)
