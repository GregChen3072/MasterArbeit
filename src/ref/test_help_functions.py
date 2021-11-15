# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:28:59 2020

@author: johan
"""


from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
from database import Database
from sklearn.ensemble import AdaBoostClassifier
from main import make_iterative_classifier
from main import make_not_iterative_classifier
from numpy import arange
import time


def split_test_train_whole_train_split(data=load_iris(), test_size=0.1, balance_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(
        data.get("data"), data.get("target"), test_size=test_size)
    x_train_split_1, x_train_split_2, y_train_split_1, y_train_split_2 = \
        train_test_split(x_train, y_train, test_size=balance_size)
    return x_train, x_train_split_1, x_train_split_2, x_test, y_train, y_train_split_1, y_train_split_2, y_test


def make_database(x, y):
    return Database(x, y)


# def make_two_databases(x, y, balance_size=0.3):
# hgj


def make_databases_and_test_set_from_data(data=load_iris(), test_size=0.1, balance_size=0.3):
    x_train, x_train_split_1, x_train_split_2, x_test, y_train, y_train_split_1, y_train_split_2, y_test = \
        split_test_train_whole_train_split(data, test_size, balance_size)
    db_whole = make_database(x_train, y_train)
    db_1 = make_database(x_train_split_1, y_train_split_1)
    db_2 = make_database(x_train_split_2, y_train_split_2)
    test = {"X": x_test, "y": y_test}
    res = {"db_whole": db_whole,
           "db_1": db_1,
           "db_2": db_2,
           "test": test}
    # print(res)
    return res


def make_k_databases_for_each_balance_step_and_a_test_set_from_data(data=load_iris(),
                                                                    test_size=0.1,
                                                                    balance_step=0.1,
                                                                    k: int = 10):
    x_train, x_test, y_train, y_test = train_test_split(
        data.get("data"), data.get("target"), test_size=test_size)
    test = {"X": x_test, "y": y_test}
    db_whole = make_database(x_train, y_train)
    db_lists = list()
    balance_list = list()
    for balance_size in arange(balance_step, (0.5 + balance_step), balance_step).tolist():
        for x in range(0, k):
            x_train_split_1, x_train_split_2, y_train_split_1, y_train_split_2 = \
                train_test_split(x_train, y_train, test_size=balance_size)
            db_1 = make_database(x_train_split_1, y_train_split_1)
            db_2 = make_database(x_train_split_2, y_train_split_2)
            db_lists.append([db_1, db_2])
            balance_list.append(balance_size)
    res = {"db_whole": db_whole,
           "db_lists": db_lists,
           "test": test,
           "balance_list": balance_list}
    return res


def test_k_federated_algorithm_for_each_balance_step_iterative(data=load_iris(),
                                                               balance_step: float = 0.1,
                                                               test_size=0.2,
                                                               k: int = 10,
                                                               n_type: str = "bull",
                                                               var_choosing_next_database: str = "worst",
                                                               n_batch_size: int = 10,
                                                               n_estimators: int = 50,
                                                               patients_batch_size: int = 1):
    dic = make_k_databases_for_each_balance_step_and_a_test_set_from_data(data=data,
                                                                          balance_step=balance_step,
                                                                          test_size=test_size,
                                                                          k=k)
    print()
    print("Iterative")
    len_data = len(data.get("data"))
    print("Number of datasets: " + str(len_data))
    len_test = len_data*test_size
    print("Number of datasets in test database: " + str(len_test))

    db_whole = dic.get("db_whole")
    classifier_whole = AdaBoostClassifier()
    classifier_whole.fit(X=db_whole.x, y=db_whole.y)
    score_whole = classifier_whole.score(
        dic.get("test").get("X"), dic.get("test").get("y"))
    print("whole")
    print(score_whole)

    res = list()
    time_list = list()
    print("federated")
    db_lists = dic.get("db_lists")
    balance_list = dic.get("balance_list")
    print("balance, number of datasets, score, time in seconds")
    for i in range(0, len(db_lists)):
        db_list = db_lists[i]
        start = time.time()
        classifier_federated = make_iterative_classifier(databases=db_list,
                                                         n_estimators=n_estimators,
                                                         n_type=n_type,
                                                         n_batch_size=n_batch_size,
                                                         var_choosing_next_database=var_choosing_next_database)  # ,
        # patients_batch_size = patients_batch_size)
        # make_iterative_classifier(databases=db_list,
        #                          n_type=n_type,
        #                          var_choosing_next_database=var_choosing_next_database,
        #                          n_batch_size=n_batch_size,
        #                          n_estimators = 50,
        #                          patients_batch_size = patients_batch_size)
        end = time.time()
        time_list.append(end - start)
        score_federated = classifier_federated.score(
            dic.get("test").get("X"), dic.get("test").get("y"))
        # print(score_federated)
        res.append(score_federated)
        print(str(round(balance_list[i], 1)) + " " + str(round(((len_data-len_test) *
                                                                balance_list[i]), 0)) + " " + str(round(score_federated, 2)) + " " + str(time_list[i]))
    return [score_whole, res, balance_list, time_list]


def test_k_federated_algorithm_for_each_balance_step_not_iterative(data=load_iris(),
                                                                   balance_step: float = 0.1,
                                                                   test_size=0.2,
                                                                   k: int = 10,
                                                                   patients_batch_size: int = 1,
                                                                   weight_databases: bool = True):
    dic = make_k_databases_for_each_balance_step_and_a_test_set_from_data(data=data,
                                                                          balance_step=balance_step,
                                                                          test_size=test_size,
                                                                          k=k)
    print()
    print("Not iterative")
    len_data = len(data.get("data"))
    print("Number of datasets: " + str(len_data))
    len_test = len_data*test_size
    print("Number of datasets in test database: " + str(len_test))

    db_whole = dic.get("db_whole")
    classifier_whole = AdaBoostClassifier()
    classifier_whole.fit(X=db_whole.x, y=db_whole.y)
    score_whole = classifier_whole.score(
        dic.get("test").get("X"), dic.get("test").get("y"))
    print("whole")
    print(score_whole)

    res = list()
    time_list = list()
    print("federated")
    db_lists = dic.get("db_lists")
    balance_list = dic.get("balance_list")
    print("balance, number of datasets, score, time in seconds")
    for i in range(0, len(db_lists)):
        db_list = db_lists[i]
        start = time.time()
        classifier_federated = make_not_iterative_classifier(databases=db_list,
                                                             patients_batch_size=patients_batch_size,
                                                             weight_databases=True)
        end = time.time()
        time_list.append(end - start)
        score_federated = classifier_federated.score(
            dic.get("test").get("X"), dic.get("test").get("y"))
        # print(score_federated)
        res.append(score_federated)
        print(str(round(balance_list[i], 1)) + " " + str(round(((len_data-len_test) *
                                                                balance_list[i]), 0)) + " " + str(round(score_federated, 2)) + " " + str(time_list[i]))
    return [score_whole, res, dic.get("balance_list"), time_list]


def make_k_databases_and_a_test_set_from_data(data=load_iris(), test_size=0.1, balance_size=0.3, k: int = 10):
    x_train, x_test, y_train, y_test = train_test_split(
        data.get("data"), data.get("target"), test_size=test_size)
    test = {"X": x_test, "y": y_test}
    db_whole = make_database(x_train, y_train)
    db_lists = list()
    for x in range(0, k):
        x_train_split_1, x_train_split_2, y_train_split_1, y_train_split_2 = \
            train_test_split(x_train, y_train, test_size=balance_size)
        db_1 = make_database(x_train_split_1, y_train_split_1)
        db_2 = make_database(x_train_split_2, y_train_split_2)
        db_lists.append([db_1, db_2])
    res = {"db_whole": db_whole,
           "db_lists": db_lists,
           "test": test}
    return res


def test_k_federated_algorithm(k: int = 10):
    dic = make_k_databases_and_a_test_set_from_data(data=load_iris(),
                                                    balance_size=0.5,
                                                    test_size=0.2,
                                                    k=k)
    db_whole = dic.get("db_whole")
    classifier_whole = AdaBoostClassifier()
    classifier_whole.fit(X=db_whole.x, y=db_whole.y)
    score_whole = classifier_whole.score(
        dic.get("test").get("X"), dic.get("test").get("y"))
    print("whole")
    print(score_whole)

    res = list()
    print("federated")
    for db_list in dic.get("db_lists"):
        classifier_federated = make_iterative_classifier(databases=db_list,
                                                         n_type="batch",
                                                         var_choosing_next_database="worst",
                                                         n_batch_size=1)
        score_federated = classifier_federated.score(
            dic.get("test").get("X"), dic.get("test").get("y"))
        # print(score_federated)
        res.append(score_federated)
    return [score_whole, res]


def test_one_federated_algorithm():
    dic = make_databases_and_test_set_from_data(data=load_iris(),
                                                balance_size=0.5,
                                                test_size=0.2)
    db_whole = dic.get("db_whole")
    classifier_whole = AdaBoostClassifier()
    classifier_whole.fit(X=db_whole.x, y=db_whole.y)
    db_list = [dic.get("db_1"), dic.get("db_2")]
    classifier_federated = make_iterative_classifier(databases=db_list,
                                                     n_type="random",
                                                     var_choosing_next_database="smallest",
                                                     n_batch_size=10)
    score_whole = classifier_whole.score(
        dic.get("test").get("X"), dic.get("test").get("y"))
    score_federated = classifier_federated.score(
        dic.get("test").get("X"), dic.get("test").get("y"))
    print("whole")
    print(score_whole)
    print("federated")
    print(score_federated)
