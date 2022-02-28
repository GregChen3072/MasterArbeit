# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:23:19 2020

@author: johan
"""

import math


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

    def get_score(self, classifier):
        classifier.score(self.x, self.y)

    def get_sub_samples():
        pass
