from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def load_iris_data():
    iris = load_iris()
    X = iris.get("data")
    y = iris.get("target")
    return X, y


def load_data_binary_classification():
    '''
        Load sklearn dataset for binary classification problems. 
        Return the entire dataset as pandas dataframe. 
    '''
    df = load_breast_cancer(as_frame=True).frame
    # X = iris.get("data")
    # y = iris.get("target")
    return df


def simulate_n_databases_with_equal_sample_size(n=5):
    '''
        Calling this function will simulate n databases with equal sample sizes based on the sklearn dataset for binary classification. 
        Set n = {1, 2, 5, 10, 20, 50, 100}. 
        #samples in each site = population size / n
        Range of n = [2, 100] (specified in the drafted paper. )
    '''
    # Load data of binary classification as pandas dataframe.
    df = load_data_binary_classification()

    # Set number of databases.
    n_db = n
    # Define an empty list which will be filled with dataframes as list elements.
    # Each list element represents a database at a certain site / hospital.
    db_list = []
    n_samples_per_db = int(len(df) / n_db)

    # Divide the population in n parts.
    for i in range(0, n_db):
        db_i = df.sample(n=n_samples_per_db, replace=False, random_state=1)
        db_list.append(db_i)
        # Sampling without replacement
        df.drop(db_i.index)

    # Check class balance after sampling
    for i, db in enumerate(db_list):
        # Print ratio p / n for each db.
        print(
            f'The P/N Ratio of DB {i+1}: ',
            db.target.value_counts()[1] / db.target.value_counts()[0]
        )

    return db_list


def simulate_n_samples_per_site():
    '''
        Interval: 5%
        5%  vs 95%
        10% vs 90%
        ...
        45% vs 55%
        50% vs 50%
    '''
    pass


def simulate_class_imbalance():
    '''
        Interval: 10%
        P 10% vs N 90%
        ...
        P 90% vs N 10%
    '''
    pass


def run_proto(n_estimators, test_size):
    iris = load_iris()
    X = iris.get("data")
    y = iris.get("target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=n_estimators,
                             learning_rate=1, random_state=0)

    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    # calculate and print model accuracy
    print("AdaBoost Classifier Model Accuracy:",
          accuracy_score(y_test, y_pred))
