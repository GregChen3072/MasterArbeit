# Load data
from sklearn.datasets import load_breast_cancer
from db_simulator import load_credit_card_fraud_data, load_HCC_data, load_ILPD_data

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

# Utils
import pandas as pd
import csv
import pickle

import time

timer_start = time.time()


def make_database(x, y):
    return Database(x, y)


# X, y = load_breast_cancer(return_X_y=True)
# X, y = load_credit_card_fraud_data()
X, y = load_HCC_data()
# X, y = load_ILPD_data()

E = 500  # Number of all estimators to be collected (from all sites all rounds)
n_splits = 100  # 100

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=6)
sss.get_n_splits(X, y)

# Settings Classifier
# n_estimators = 100

# Container of results of 1000 repetitions
results_1_1 = list()
results_2_1 = list()
results_3_1 = list()

results_2_2_unweighted = list()
results_3_2_unweighted = list()
results_2_2_weighted = list()
results_3_2_weighted = list()

sss_counter = 0

# Create 100 Validation Sets
for train_index, test_index in sss.split(X, y):

    sss_counter = sss_counter + 1
    # print(f'Ongoing Shuffle Split: {sss_counter}')

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ''' Part I '''

    # Pipeline 1 1 (implementation completed)
    # When all data centralized in one database.
    """ res_1_1 = pipeline_1_1(X_train, X_test, y_train, y_test, s=sss_counter, E=n_estimators)
    results_1_1.append(res_1_1) """

    # Pipeline 2 1 (implementation completed)
    """ N = [1, 2, 5, 10, 20, 50, 100]
    res_2_1 = pipeline_2_1(X_train, X_test, y_train, y_test,
                          s=sss_counter, N=N, E=E)
    results_2_1.append(res_2_1) """

    # Pipeline 3 1 (implementation completed)
    """ N = [1, 2, 5, 10, 20, 50, 100]
    # E = 500
    res_3_1 = pipeline_3_1(X_train, X_test, y_train, y_test,
                           s=sss_counter, N=N, E=E, r=1)
    results_3_1.append(res_3_1) """

    ''' Part II '''

    # Pipeline 2 2 unweighted (implementation completed)
    # pipeline_2_2_unweighted(X_train, X_test, y_train, y_test, sss_counter)

    # Pipeline 3 2 unweighted (implementation completed)
    # pipeline_3_2_unweighted(X_train, X_test, y_train, y_test, sss_counter)

    # Pipeline 2 2 weighted (implementation completed)
    # pipeline_2_2_weighted(X_train, X_test, y_train, y_test, sss_counter)

    # Pipeline 3 2 weighted (implementation completed)
    '''
    res_3_2_weighted = pipeline_3_2_weighted(
        X_train,
        X_test,
        y_train,
        y_test,
        sss_counter
    )
    results_3_2_weighted.append(res_3_2_weighted)
    '''

print('Processing results...')

# Saving results for 1 1
""" df_1_1 = pd.DataFrame(
    results_1_1,
    columns=['s', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

df_1_1.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_1_1_ILPD.csv',
              index=False, header=True)

print('Results saved for pipeline 1 1. ') """

# Saving results for 2 1
""" res_2_1_flat = [res for sublist in results_2_1 for res in sublist]
df_2_1 = pd.DataFrame(
    res_2_1_flat,
    columns=['s', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

df_2_1.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_2_1_ILPD.csv',
              index=False, header=True)

print('Results saved for pipeline 2 1. ') """

# Saving results for 3 1
""" res_3_1_flat = [res for sublist in results_3_1 for res in sublist]
df_3_1 = pd.DataFrame(
    res_3_1_flat,
    columns=['s', 'r', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)
df_3_1.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_3_1_HCC.csv',
              index=False, header=True)
print('Results saved for pipeline 2 1. ') """

# Saving results for 2 2 unweighted


# Saving results for 2 2 weighted


# Saving results for 3 2 unweighted


# Saving results for 3 2 weighted
'''res_3_2_weighted_flat = [
    res for sublist in results_3_2_weighted for res in sublist]

df_3_2_weighted = pd.DataFrame(
    res_3_2_weighted_flat,
    columns=['Shuffle Round', 'Degree Imbalance', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score', 'Training Time']
)

print(df_3_2_weighted)

df_3_2_weighted.to_csv(
    '/Users/greg/Downloads/test_visual.csv', index=False, header=False)

print('Results saved for pipeline 3 2. ')'''


print('Experiments completed! ')
timer_stop = time.time()

print('Time comsuption in seconds ' + str(timer_stop - timer_start))
