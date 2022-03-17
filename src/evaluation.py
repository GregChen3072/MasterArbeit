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
from pipeline_1_2 import pipeline_1_2
from pipeline_1_3 import pipeline_1_3

from pipeline_2_2 import pipeline_2_2_unweighted, pipeline_2_2_weighted
from pipeline_2_3 import pipeline_2_3_unweighted, pipeline_2_3_weighted

# Utils
import pandas as pd

import time

timer_start = time.time()


def make_database(x, y):
    return Database(x, y)


X, y = load_breast_cancer(return_X_y=True)
# X, y = load_credit_card_fraud_data()
# X, y = load_HCC_data()
# X, y = load_ILPD_data()

E = 500  # Number of all estimators to be collected (from all sites all rounds)
# n_estimators = 500 set as default for binary inter-site imbalance
n_splits = 100  # 100

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=6)
sss.get_n_splits(X, y)

# Settings Classifier
# n_estimators = 100

# Container of results of 1000 repetitions
results_1_1 = list()
results_1_2 = list()
results_1_3 = list()

results_2_2_unweighted = list()
results_2_3_unweighted = list()
results_2_2_weighted = list()
results_2_3_weighted = list()

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
    res_1_1 = pipeline_1_1(X_train, X_test, y_train,
                           y_test, s=sss_counter, E=E)
    results_1_1.append(res_1_1)

    # Pipeline 1 2 (implementation completed)
    N = [1, 2, 5, 10, 20, 50, 100]
    res_1_2 = pipeline_1_2(X_train, X_test, y_train, y_test,
                           s=sss_counter, N=N, E=E)
    results_1_2.append(res_1_2)

    # Pipeline 1 3 (implementation completed)
    N = [1, 2, 5, 10, 20, 50, 100]
    # E = 500
    res_1_3 = pipeline_1_3(X_train, X_test, y_train, y_test,
                           s=sss_counter, N=N, E=E, r=1)
    results_1_3.append(res_1_3)

    ''' Part II '''

    # Pipeline 2 2 unweighted (implementation completed)
    """ res_2_2_unweighted = pipeline_2_2_unweighted(
        X_train, X_test, y_train, y_test, s=sss_counter, E=E)
    results_2_2_unweighted.append(res_2_2_unweighted) """

    # Pipeline 2 2 weighted (implementation completed)
    """ res_2_2_weighted = pipeline_2_2_weighted(
        X_train, X_test, y_train, y_test, s=sss_counter, E=E)
    results_2_2_weighted.append(res_2_2_weighted) """

    # Pipeline 2 3 unweighted (implementation completed)
    """ res_2_3_unweighted = pipeline_2_3_unweighted(
        X_train, X_test, y_train, y_test, s=sss_counter, E=E)
    results_2_3_unweighted.append(res_2_3_unweighted) """

    # Pipeline 2 3 weighted (implementation completed)
    """ res_2_3_weighted = pipeline_2_3_weighted(
        X_train, X_test, y_train, y_test, s=sss_counter, E=E)
    results_2_3_weighted.append(res_2_3_weighted) """


print('Processing results...')

# Saving results for 1 1
df_1_1 = pd.DataFrame(
    results_1_1,
    columns=['s', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

df_1_1.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_1_1.csv',
              index=False, header=True)
df_1_1.to_csv('/Users/greg/Documents/thesis/output/vis_BRCA_1_1.csv',
              index=False, header=True)

print('Results saved for pipeline 1 1. ')

# Saving results for 1 2
res_1_2_flat = [res for sublist in results_1_2 for res in sublist]
df_1_2 = pd.DataFrame(
    res_1_2_flat,
    columns=['s', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

df_1_2.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_1_2.csv',
              index=False, header=True)
df_1_2.to_csv('/Users/greg/Documents/thesis/output/vis_BRCA_1_2.csv',
              index=False, header=True)

print('Results saved for pipeline 1 2. ')

# Saving results for 1 3
res_1_3_flat = [res for sublist in results_1_3 for res in sublist]
df_1_3 = pd.DataFrame(
    res_1_3_flat,
    columns=['s', 'r', 'n', 'e', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)
df_1_3.to_csv('/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_1_3.csv',
              index=False, header=True)
df_1_3.to_csv('/Users/greg/Documents/thesis/output/vis_BRCA_1_3.csv',
              index=False, header=True)

print('Results saved for pipeline 1 2. ')

# Saving results for 2 2 unweighted
""" res_2_2_unweighted_flat = [
    res for sublist in results_2_2_unweighted for res in sublist]

df_2_2_unweighted = pd.DataFrame(
    res_2_2_unweighted_flat,
    columns=['s', 'Degree Imbalance', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

print(df_2_2_unweighted)

df_2_2_unweighted.to_csv(
    '/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_2_2_unweighted.csv', index=False, header=True)   
df_2_2_unweighted.to_csv(
    '/Users/greg/Documents/thesis/output/vis_BRCA_2_2_unweighted.csv', index=False, header=True)

print('Results saved for pipeline 2 2 unweighted. ') """

# Saving results for 2 2 weighted
""" res_2_2_weighted_flat = [
    res for sublist in results_2_2_weighted for res in sublist]

df_2_2_weighted = pd.DataFrame(
    res_2_2_weighted_flat,
    columns=['s', 'Degree Imbalance', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

print(df_2_2_weighted)

df_2_2_weighted.to_csv(
    '/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_2_2_weighted.csv', index=False, header=True)
df_2_2_weighted.to_csv(
    '/Users/greg/Documents/thesis/output/vis_BRCA_2_2_weighted.csv', index=False, header=True)
print('Results saved for pipeline 2 2 weighted. ') """

# Saving results for 2 3 unweighted
""" res_2_3_unweighted_flat = [
    res for sublist in results_2_3_unweighted for res in sublist]

df_2_3_unweighted = pd.DataFrame(
    res_2_3_unweighted_flat,
    columns=['s', 'Degree Imbalance', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

print(df_2_3_unweighted)

df_2_3_unweighted.to_csv(
    '/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_2_3_unweighted.csv', index=False, header=True)
df_2_3_unweighted.to_csv(
    '/Users/greg/Documents/thesis/output/vis_BRCA_2_3_unweighted.csv', index=False, header=True)

print('Results saved for pipeline 2 3 unweighted. ') """

# Saving results for 2 3 weighted
""" res_2_3_weighted_flat = [
    res for sublist in results_2_3_weighted for res in sublist]

df_2_3_weighted = pd.DataFrame(
    res_2_3_weighted_flat,
    columns=['s', 'Degree Imbalance', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)

print(df_2_3_weighted)

df_2_3_weighted.to_csv(
    '/Users/greg/Downloads/AR_Master_Thesis/output/vis_BRCA_2_3_weighted.csv', index=False, header=True)
df_2_3_weighted.to_csv(
    '/Users/greg/Documents/thesis/output/vis_BRCA_2_3_weighted.csv', index=False, header=True)
print('Results saved for pipeline 2 3 weighted. ') """

print('Experiments completed! ')
timer_stop = time.time()

print('Time comsuption in seconds ' + str(timer_stop - timer_start))
