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


def pipeline_2_1(X_train, X_test, y_train, y_test, s, N, E):

    # Settings
    # ns = ns
    # E = E

    # Simulate n DBs for n = {1, 2, 5, 10, 20, 50, 100}
    list_of_n_dbs = simulate_n_databases_with_equal_sample_size(
        X_train, y_train, list_of_n=N)

    # Initialize
    results = list()

    # Count of data points for global train and test sets
    # len_data = len(data.get("data"))
    # len_test = len_data*test_size

    # Granularity of sites
    # list_of_n = list_of_n

    print("s\tn\te\tF-1\t\tMCC Score\tAUC Score\tACC Score")

    for i_n_dbs in range(0, len(list_of_n_dbs)):
        # i_n_dbs = i_n_dbs  # n sites / DBs for a given n
        n = N[i_n_dbs]
        # print(len(list_of_n_dbs[i_n_dbs]))
        e = int(E/n)
        # print("e = " + str(e))
        # Instantiate classifier
        classifier_fed_aggregated = CombinedAdaBoostClassifier(
            n_estimators=e,
            learning_rate=1.,
            algorithm='SAMME.R',
            random_state=6,
            patients_batch_size=1,
            # For n sites evaluation, all sites have equal weights, no need to weigh by the sizes of db
            weight_databases=False
        )
        # print(classifier_fed_aggregated.n_estimators)
        # Collect all estimators / sending estimators to the central
        # timer_max = 0

        # For each db in this experiment (in which the number of sites = {1, 2, 5, 10, 20, 50, 100})
        # for db in list_of_n_dbs[i_n_dbs]:
        #     classifier_fed_aggregated.add_fit(db)

        classifier_fed_aggregated.make_fit(list_of_n_dbs[i_n_dbs])

        f_1, mcc, auc, acc = make_scores(
            classifier_fed_aggregated, X_test, y_test)

        results.append([s, n, e,
                        f_1, mcc, auc, acc])

        print(
            str(s) +
            "\t" +
            str(n) +
            "\t" +
            str(e) +
            "\t" +
            str(round(f_1, 3)) +
            "\t\t" +
            str(round(mcc, 3)) +
            "\t\t" +
            str(round(auc, 3)) +
            "\t\t" +
            str(round(acc, 3))
        )

    return results
