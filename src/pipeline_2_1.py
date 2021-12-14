# Load population
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# User defined functions
from db_simulator import simulate_n_databases_with_equal_sample_size

# Data preparation
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from scoring import make_scores

# Reference
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time


def pipeline_2_1(X_train, X_test, y_train, y_test):

    # Settings
    list_of_n = [1, 2, 5, 10, 20, 50, 100]
    n_estimators = 1000

    # Simulate n DBs for n = [1, 2, 5, 10, 20, 50, 100]
    list_of_n_dbs = simulate_n_databases_with_equal_sample_size(
        X_train, X_test, y_train, y_test,
        list_of_n=list_of_n,
    )

    # Initialize
    res = list()
    timer_list = list()

    # Count of data points for global train and test sets
    # len_data = len(data.get("data"))
    # len_test = len_data*test_size

    # Initialize
    res_f_1 = list()
    res_mcc = list()
    res_auc = list()
    res_acc = list()  # Score Containers

    # Granularity of sites
    # list_of_n = list_of_n

    # Print title
    print("N Sites Combined")
    print()

    print("n Databases\tF-1 Score\t\tMCC Score\tAUC Score\tACC Score\tDuration in Seconds")

    for i_n_dbs in range(0, len(list_of_n_dbs)):
        # i_n_dbs = i_n_dbs  # n sites / DBs for a given n

        # Instantiate classifier
        classifier_fed_aggregated = CombinedAdaBoostClassifier(
            n_estimators=int(n_estimators/list_of_n[i_n_dbs]),
            learning_rate=1.,
            algorithm='SAMME.R',
            random_state=6,
            patients_batch_size=1,
            # For n sites evaluation, all sites have equal weights, no need to weigh by the sizes of db
            weight_databases=False
        )

        # Collect all estimators / sending estimators to the central
        timer_max = 0

        for db in list_of_n_dbs[i_n_dbs]:
            # Start timer
            timer_start = time.time()

            # Feeding one db object
            classifier_fed_aggregated.add_fit(db)

            # Stop timer
            timer_stop = time.time()

            timer_actual = timer_stop - timer_start

            # The communication efficiency of a combined model depends on the slowest site.
            if timer_max <= timer_actual:
                timer_max = timer_actual

        timer_list.append(timer_max)

        f_1, mcc, auc, acc = make_scores(
            classifier_fed_aggregated, X_test, y_test)
        res_f_1.append(f_1)
        res_mcc.append(mcc)
        res_auc.append(auc)
        res_acc.append(acc)

        print(
            str(list_of_n[i_n_dbs]) +
            "\t\t" +
            str(f_1) +
            "\t" +
            str(round(mcc, 2)) +
            "\t\t" +
            str(round(auc, 2)) +
            "\t\t" +
            str(round(acc, 2)) +
            "\t\t" +
            str(timer_list[i_n_dbs])
        )
