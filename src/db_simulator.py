from scipy import rand
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

from ref.database import Database
from numpy import arange
import time


def load_iris_data():
    iris = load_iris()
    X = iris.get("data")
    y = iris.get("target")
    return X, y


def load_HCC_data():
    path = "/Users/greg/Downloads/AR_Master_Thesis/data/HCC_preprocessed.csv"
    data = pd.read_csv(path)

    # Remove the 1st column because it is an index
    X = data.iloc[:, 2:-1].to_numpy()
    y = data.iloc[:, 1].to_numpy()

    return X, y, "HCC"


def load_ILPD_data():
    path = "/Users/greg/Downloads/AR_Master_Thesis/data/ILPD_preprocessed.csv"
    data = pd.read_csv(path)

    # Remove the 1st column because it is an index
    X = data.iloc[:, 1:-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    return X, y, "ILPD"


def load_WDBC_data():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, "WDBC"


def make_database(x, y):
    return Database(x, y)


def simulate_1_database_with_all_data_centralized():
    # df = load_breast_cancer(as_frame=True).frame
    binar = load_breast_cancer()
    X = binar.get("data")
    y = binar.get("target")
    return X, y


def simulate_n_databases_with_equal_sample_size(X_train, y_train, list_of_n):
    # List of db_list
    # Define an empty list which will be filled with dataframes as list elements.
    # Each list element represents a database at a certain site / hospital.
    '''
        Calling this function will simulate n databases with equal sample sizes based on the sklearn dataset for binary classification. 
        Input: DataFrame object; n
        Ouput: List of DataFrame objects. 

        Set n = {1, 2, 5, 10, 20, 50, 100}. 
        #samples in each site = population size / n
        Range of n = [2, 100] (specified in the drafted paper. )
    '''
    '''
        Structure: 
        db_list = [
            [db_obj_1], 
            [db_obj_1, db_obj_2], 
            [db_obj_1, db_obj_2, db_obj_3], 
            ...            
        ]
    '''
    list_of_n_dbs = list()

    # list_of_n = [1, 2, 5, 10, 20, 50, 100]
    for n in list_of_n:
        # Convert arrays into df and get ready for sampling without replacement.
        df_train_set = pd.DataFrame(data=X_train)
        df_train_set['target'] = y_train

        # Set number of databases.
        n_dbs = list()

        # sample size = 1 / n
        n_samples_per_db = int(len(y_train) / n)

        # Construct a list of DBs containing n times DB
        # Divide the population in n parts.
        # n = {1, 2, 3, ..., n}
        for n in range(0, n):
            db_n_df = df_train_set.sample(
                n=n_samples_per_db, replace=False, random_state=6)

            X_train_of_db_i = db_n_df.drop(columns=['target']).to_numpy()
            y_train_of_db_i = db_n_df['target'].to_numpy()

            db_n = make_database(X_train_of_db_i, y_train_of_db_i)
            # Here constructing db_list filled with single Database objects.
            # Reason: CombinedAdaBoostClassifier.add_fit() is designed to accept this data structure.
            n_dbs.append(db_n)
            # Sampling without replacement
            df_train_set.drop(db_n_df.index)

        list_of_n_dbs.append(n_dbs)

    prepared_data = list_of_n_dbs

    return prepared_data


def simulate_db_size_imbalance(x_train, y_train, balance_step: float = 0.05, k: int = 1):
    '''
        Interval: 5%
        5%  vs 95%
        10% vs 90%
        ...
        45% vs 55%
        50% vs 50%
        DB * 10
    '''

    db_pairs = list()
    balance_list = list()

    # balance_sizes = arange(balance_step, round((0.5 + balance_step), 2), balance_step).tolist()
    balance_sizes = [0.05, 0.10, 0.15, 0.20,
                     0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for balance_degree in balance_sizes:
        for x in range(0, k):
            x_train_split_1, x_train_split_2, y_train_split_1, y_train_split_2 = train_test_split(
                x_train, y_train, test_size=round(balance_degree, 2), random_state=6)
            db_1 = make_database(x_train_split_1, y_train_split_1)
            db_2 = make_database(x_train_split_2, y_train_split_2)
            db_pairs.append([db_1, db_2])
            balance_list.append(balance_degree)

    res = {
        "db_pairs": db_pairs,
        "balance_list": balance_list
    }

    return res
