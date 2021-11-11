# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:30:16 2020

@author: johan
"""


from typing import List
import numpy as np
from numpy import ndarray

from sklearn.ensemble import AdaBoostClassifier
from ref.database import Database


class CombinedAdaBoostClassifier(AdaBoostClassifier):

    def __init__(self, base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 patients_batch_size: int = 1,
                 weight_databases: bool = False):
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         algorithm=algorithm,
                         random_state=random_state)
        self.start_classifier = AdaBoostClassifier(base_estimator,
                                                   n_estimators=n_estimators,
                                                   learning_rate=learning_rate,
                                                   algorithm=algorithm,
                                                   random_state=random_state)
        self.patients_batch_size = patients_batch_size
        self.weight_databases = weight_databases
        self.n_evaluated_batches = 0
        self.list_of_ada_booster = []
        self.list_of_databases_weights = []

    def fit(self, X, y, sample_weight=None):
        return self.add_fit(Database(X, y))

    def add_fit(self, database):
        # database = Database(X, y)
        if len(self.list_of_ada_booster) == 0:
            self.make_fit([database])
        else:
            # booster
            self.list_of_ada_booster.append(
                database.extend_classifier(self.start_classifier))

            # classes already made

            # estimators
            self.estimators_.extend(self.list_of_ada_booster[-1].estimators_)

            # You're actually add & adjusting / updating databases weights after each batch!

            list_of_databases_weights, n_evaluated_batches = self.__add_database_weight(
                list_of_databases_weights=self.list_of_databases_weights,
                database=database,
                n_evaluated_batches=self.n_evaluated_batches,
                batch_size=self.patients_batch_size)
            self.list_of_databases_weights = list_of_databases_weights
            self.n_evaluated_batches = n_evaluated_batches

            # estimators weights
            # make ESTIMATOR WEIGHTS
            list_of_estimator_weights = self.__make_estimator_weights(
                len_list_of_estimators=len(self.estimators_),
                list_of_databases_weights=self.list_of_databases_weights,
                list_of_ada_booster=self.list_of_ada_booster,
                use_weights=self.weight_databases)
            self.estimator_weights_ = list_of_estimator_weights

            # estimator error
            # make ESTIMEATOR ERROR
            estimator_errors = self.__make_estimator_errors(
                list_of_databases_weights=self.list_of_databases_weights,
                list_of_ada_booster=self.list_of_ada_booster,
                use_weights=self.weight_databases)
            self.estimator_errors_ = estimator_errors

    def make_fit(self, list_of_databases):
        if len(list_of_databases) == 0:
            raise Exception("There are no databases.")

        # Make ada booster
        list_of_ada_booster = []
        for database in list_of_databases:
            # if database is not Database:
            #    raise Exception("This is not an Database: " + str(database))
            list_of_ada_booster.append(
                database.extend_classifier(self.start_classifier))
        self.list_of_ada_booster = list_of_ada_booster

        # get CLASSES
        dict_classes = self.__get_classes(list_of_ada_booster)
        self.n_classes_ = dict_classes.get("n_classes_")
        self.classes_ = dict_classes.get("classes_")

        # get ESTIMATORS
        list_of_estimators = self.__get_estimator_list(list_of_ada_booster)
        self.estimators_ = list_of_estimators

        # make DATABASES WEIGHTS
        list_of_databases_weights, number_of_batches = self.__make_database_weights(list_of_databases,
                                                                                    self.patients_batch_size)
        self.list_of_databases_weights = list_of_databases_weights
        self.n_evaluated_batches = number_of_batches

        # make ESTIMATOR WEIGHTS
        list_of_estimator_weights = self.__make_estimator_weights(
            len_list_of_estimators=len(list_of_estimators),
            list_of_databases_weights=list_of_databases_weights,
            list_of_ada_booster=list_of_ada_booster,
            use_weights=self.weight_databases)
        self.estimator_weights_ = list_of_estimator_weights

        # make feature_importances
        # feature_importances = self.__make_feature_importances(
        #        list_of_databases_weights = list_of_databases_weights,
        #        use_weights = self.weight_databases,
        #        list_of_ada_booster = list_of_ada_booster)

        # make ESTIMEATOR ERROR
        estimator_errors = self.__make_estimator_errors(
            list_of_databases_weights=list_of_databases_weights,
            list_of_ada_booster=list_of_ada_booster,
            use_weights=self.weight_databases)
        self.estimator_errors_ = estimator_errors

        # SETTER
        # set_dict = {"estimators_": list_of_estimators,
        #           "estimators_weights_": list_of_estimator_weights,
        #           "feature_importances_": feature_importances,
        #           "n_estimators": self.start_classifier.n_estimators*len(list_of_databases),
        #           "estimator_errors_": estimator_errors,
        #           **dict_classes}
        #self.estimators_ = list_of_estimators
        #self.estimator_weights_ = list_of_estimator_weights
        #self.feature_importances_ = feature_importances
        self.n_estimators = len(self.estimators_)
        #self.n_classes_ = dict_classes.get("n_classes_")
        #self.classes_ = dict_classes.get("classes_")

        # self.current_classifier.set_params(**set_dict)
        # self.set_params(**set_dict)
        return self  # self.current_classifier

    @staticmethod
    def __get_classes(list_of_ada_booster: List[AdaBoostClassifier]):
        # classes = None
        # print(list_of_ada_booster[0].classes_)
        classes = list_of_ada_booster[0].classes_
        # print(classes)
        # for ada_booster in list_of_ada_booster:
        #    if  len(list(set(ada_booster.classes_) & set(classes))) == len(classes):
        #        raise Exception("Different classes.")
        res = {"classes_": classes, "n_classes_": len(classes)}
        return res

    @staticmethod
    def __get_estimator_list(list_of_ada_booster: List[AdaBoostClassifier]):
        """
        This function appends all estimators from all boosters into one list.
        :param list_of_ada_booster: List of AdaBoostClassifier
        :return: List of Estimator
        """
        res = []
        # classes = None
        # classes = set(list_of_ada_booster[0].classes_).sort()
        for ada_booster in list_of_ada_booster:
            # if classes != set(ada_booster.classes_).sort():
            #    raise Exception("Different classes.")
            res.extend(ada_booster.estimators_)
        return res

    @staticmethod
    def __make_database_weights(list_of_databases: List[Database], batch_size: int = 1):
        res = []
        for database in list_of_databases:
            res.append(database.get_number_of_batches_of_patients(
                batch_size=batch_size))
        number_of_batches = sum(res)
        res = np.array(res) / number_of_batches
        return res, number_of_batches

    @staticmethod
    def __add_database_weight(list_of_databases_weights,
                              database: Database,
                              n_evaluated_batches: int,
                              batch_size: int = 1):
        new_batches = database.get_number_of_batches_of_patients(
            batch_size=batch_size)
        # new_batches = 1
        list_of_databases_weights = list_of_databases_weights * n_evaluated_batches
        list_of_databases_weights = np.concatenate(
            (list_of_databases_weights, new_batches), axis=0)
        n_evaluated_batches = n_evaluated_batches + new_batches
        list_of_databases_weights = list_of_databases_weights / n_evaluated_batches
        return list_of_databases_weights, n_evaluated_batches

    @staticmethod
    def __make_estimator_weights(len_list_of_estimators: int,
                                 list_of_databases_weights: ndarray,
                                 list_of_ada_booster: List[AdaBoostClassifier],
                                 use_weights: bool = False):
        if use_weights == False:
            return np.ones(len_list_of_estimators)
        else:
            # self.current_classifier.estimator_weights_ =

            res = np.array([])
            # print(res)
            for i in range(0, len(list_of_ada_booster)):
                # print(i)
                # print(res)
                estimator_weights = list_of_ada_booster[i].estimator_weights_
                # print(estimator_weights)
                database_weight = list_of_databases_weights[i]
                # print(database_weight)
                new_estimator_weights = estimator_weights * database_weight
                # print(new_estimator_weights)
                new_estimator_weights = np.array(new_estimator_weights)
                # print(new_estimator_weights)
                if(len(res) == 0):
                    res = new_estimator_weights
                else:
                    res = np.concatenate((res, new_estimator_weights))
            # print(res)
            return res

    # @staticmethod
    # def __make_feature_importances(
    #                             list_of_databases_weights: ndarray,
    #                             list_of_ada_booster: List[AdaBoostClassifier],
    #                             use_weights:bool = False):
    #    if use_weights == False:
    #        return np.ones(list_of_ada_booster[0].n_classes_)
    #    else:
    #        # self.current_classifier.estimator_weights_ =
    #        res = np.zeros(list_of_ada_booster[0].n_classes_)
    #        for i in range(len(list_of_ada_booster)):
    #            res = np.add(res, list_of_ada_booster[i].feature_importances_/list_of_databases_weights[i])
    #        return res

    @staticmethod
    def __make_estimator_errors(list_of_databases_weights: ndarray,
                                list_of_ada_booster: List[AdaBoostClassifier],
                                use_weights: bool = False):
        if use_weights == False:
            res = np.array([])
            for ada_booster in list_of_ada_booster:
                res = np.concatenate(
                    (res, ada_booster.estimator_errors_), axis=0)
            return res
        else:  # TODO: Nach besseren Variante suchen
            res = np.array([])
            for i in range(0, len(list_of_ada_booster)):
                np.concatenate(
                    (res, (list_of_ada_booster[i].estimator_errors_/list_of_databases_weights[i])))
            return res
