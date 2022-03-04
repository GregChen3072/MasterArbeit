# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:23:19 2020

@author: johan
"""

import math
import random

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


class Database:
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def get_number_of_batches_of_patients(self, batch_size=1):
        return math.ceil(len(self.y) / batch_size)

    def get_number_of_patients(self):
        return len(self.y)

    def extend_classifier(self, classifier):
        # When the number of samples per site is too small (< 53), only one estimator will be trained.
        # Something like a default setting in AdaBoostClassifier.
        # This problem will not occur if the base sample is big enough meaning each site should have at least 53 data points.
        return classifier.fit(self.x, self.y)

    def extend_bootstrap_fit(self, classifier):
        X = self.x
        y = self.y
        # sss = ShuffleSplit(n_splits=5, test_size=random.uniform(0.7, 0.9), random_state=random.randint(0, 100))
        sss = ShuffleSplit(
            n_splits=1, train_size=0.5, random_state=random.randint(0, 100))
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return classifier.fit(X_train, y_train)

    def get_score(self, classifier):
        classifier.score(self.x, self.y)

    def get_sub_samples():
        pass
