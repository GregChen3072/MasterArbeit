# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:27:58 2020

@author: johan
"""

# from sklearn
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from ref.next_n_size import NextN
import numpy as np
# from sklearn.base import is_regressor
# from sklearn.utils.validation import _check_sample_weight as check_sample_weight
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from ref.helpfunctions import _check_sample_weight


class WarmStartAdaBoostClassifier(AdaBoostClassifier, BaseEstimator):
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         # loss=loss,
                         random_state=random_state)

        self.loss = loss
        self.estimators_ = []
        self.next_start = 0
        # np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_weights_ = np.array([])
        # np.ones(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.array([])
        self.n_estimators = 0
        self.first_fit = True
        # self.random_state = random_state

    def set_beginning(self):
        self.estimators_ = []
        self.next_start = 0
        # np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_weights_ = np.array([])
        # np.ones(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.array([])
        self.n_estimators = 0
        self.first_fit = True
        return self

    def set_next_n(self, n: int = 1):
        if(n == 0):
            raise Exception("n is not alllowed to be 0.")
        #print("Old n: "+str(self.n_estimators))
        #print("New n: "+str(n))
        if self.n_estimators >= n:
            raise Exception("Out of bounds: " + str(n))
        self.next_start = self.n_estimators
        self.n_estimators = n
        if(self.next_start == 0):
            self.estimator_weights_ = np.zeros(
                self.n_estimators, dtype=np.float64)
            self.estimator_errors_ = np.ones(
                self.n_estimators, dtype=np.float64)
        else:
            zeros = np.zeros((self.n_estimators-self.next_start),
                             dtype=np.float64)
            self.estimator_weights_ = np.concatenate(
                (self.estimator_weights_, zeros))
            # except:
            # print(self.estimator_weights_)
            #   print(str(self.n_estimators-self.next_start))
            #   raise Exception("bla")
            ones = np.ones((self.n_estimators-self.next_start),
                           dtype=np.float64)
            self.estimator_errors_ = np.concatenate(
                (self.estimator_errors_, ones))
        return self

    def fit(self, X, y, sample_weight=None):
        # return super().fit(X, y, sample_weight)
        """Build a boosted classifier/regressor from the training set (X, y).
        # Change: But does not delete previous estimator

                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    The training input samples. Sparse matrix can be CSC, CSR, COO,
                    DOK, or LIL. COO, DOK, and LIL are converted to CSR.

                y : array-like of shape (n_samples,)
                    The target values (class labels in classification, real numbers in
                    regression).

                sample_weight : array-like of shape (n_samples,), default=None
                    Sample weights. If None, the sample weights are initialized to
                    1 / n_samples.

                Returns
                -------
                self : object
                """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        # !!! Änderung
        # check_params = {"accept_sparse":['csr', 'csc'],
        #               "ensure_2d":True,
        #               "allow_nd":True,
        #               "dtype":None,
        #               "y_numeric":is_regressor(self)}

        X, y = super()._validate_data(X, y)  # , **check_params)
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # !!! Änderung

        # Clear any previous fit results
        # self.estimators_ = []
        # now defined in __init__
        # self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        # self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initializion of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)

        # !!! Änderung
        # print(range(self.next_start, self.n_estimators))
        for iboost in range(self.next_start, self.n_estimators):  # Änderung
            # print(iboost)
            # for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        # print("fit fertig")
        return self


class Classifier:
    def __init__(self, classifier=WarmStartAdaBoostClassifier(), n_generator: NextN = None) -> None:
        super().__init__()
        #print("N_Generator: "+str(n_generator))
        if n_generator is None:
            self.n_generator = NextN(n_size=50)
            #print("Standard N Generator")
        else:
            self.n_generator = n_generator
            #print("Self Defined N Generator")
        self.classifier = classifier

    def update_classifier(self, classifier):
        self.classifier = classifier

    def get_prepared_classifier(self):
        # print("prepare")
        # if self.classifier is WarmStartAdaBoostClassifier:
        n = self.n_generator.get_next_n()
        # print("With n: "+str(n))
        self.classifier = self.classifier.set_next_n(n=n)
        # elif self.classifier is AdaBoostClassifier:
        #    self.classifier.set_params(n_estimators=self.n_generator.get_next_n())
        return self.classifier

    def finished(self):
        return self.n_generator.finished()
