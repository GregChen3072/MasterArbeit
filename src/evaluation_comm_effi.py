# Load data
from sklearn.datasets import load_breast_cancer
from db_simulator import load_credit_card_fraud_data, load_HCC_data, load_ILPD_data

# Load objects
from sklearn.model_selection import StratifiedShuffleSplit
from ref.database import Database

from pipeline_1_3 import pipeline_1_3_comm_effi

# Utils
import pandas as pd
import time

timer_start = time.time()


def make_database(x, y):
    return Database(x, y)


# X, y = load_breast_cancer(return_X_y=True)
X, y = load_HCC_data()
# X, y = load_credit_card_fraud_data()
# X, y = load_ILPD_data()

# Settings Evaluation
n_splits = 100

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=6)
sss.get_n_splits(X, y)

res_comm_effi = list()

sss_counter = 0

# Print title
print()
print("Communication Efficiency Experiment for N Sites Iterative")
print()

# Create 1000 Validation Sets
for train_index, test_index in sss.split(X, y):

    sss_counter = sss_counter + 1
    print(f'Ongoing Shuffle Split: {sss_counter}')

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ''' Part III '''
    # for r in range(1, 6):
    for r in range(20, 21):

        # for e in [1, 2, 5, 10]:
        for e in [1, 2, 5]:
            # Pipeline 3 1
            # Communication Efficiency Test
            # #e = 1 (number of estimators collected per site per round)
            # #s = [10, 20, 50] (numbers of sites)
            results = pipeline_1_3_comm_effi(X_train, X_test, y_train,
                                             y_test, e=e, r=r, s=sss_counter)
            # Collect 100 * 5 * 4 * 4
            res_comm_effi.append(results)


print('Processing results...')


res_comm_effi_flat = [res for sublist in res_comm_effi for res in sublist]


df_res_comm_effi = pd.DataFrame(
    res_comm_effi_flat,
    columns=['s', 'r', 'v', 'n', 'e', 'f_1', 'mcc',
             'auc', 'acc']
)

df_res_comm_effi.to_csv(
    '/Users/greg/Downloads/AR_Master_Thesis/output/vis_HCC_comm_effi.csv', index=False, header=True)


print('Results saved for Test Comm Effi. ')
print('Experiments completed! ')
timer_stop = time.time()

print('Time comsuption in seconds ' + str(timer_stop - timer_start))
