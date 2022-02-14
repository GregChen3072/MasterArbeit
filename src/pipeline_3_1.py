# User defined functions
from db_simulator import simulate_n_databases_with_equal_sample_size

# Model Evaluation
from scoring import make_scores

# Reference
from ref.next_n_size import NextN
from ref.next_n_size import NextDataSets
from ref.classifier import WarmStartAdaBoostClassifier
from ref.classifier import Classifier
from ref.main import make_iterative_classifier

# Utils
import time


def pipeline_3_1(X_train, X_test, y_train, y_test):

    # Settings
    list_of_n = [1, 2, 5, 10, 20, 50, 100]
    n_estimators = 500
    # n_db = 10
    n_iteration = 5

    n_type = "batch"
    var_choosing_next_database = "iterate"
    patients_batch_size = 1  # Spielt keine Rolle

    # Simulate n DBs for n = [1, 2, 5, 10, 20, 50, 100]
    list_of_n_dbs = simulate_n_databases_with_equal_sample_size(
        X_train, X_test, y_train, y_test,
        list_of_n=list_of_n,
    )

    # Initialize
    res = list()
    timer_list = list()  # Timers

    res_f_1 = list()
    res_mcc = list()
    res_auc = list()
    res_acc = list()  # Score Containers

    # Granularity of sites
    # list_of_n = list_of_n

    # Print title
    print("N Sites Iterative")
    print()

    print("n Databases\tF-1 Score\t\tMCC Score\tAUC Score\tACC Score\tDuration in Seconds")

    for i_n_dbs in range(0, len(list_of_n_dbs)):

        n_db = list_of_n[i_n_dbs]
        n_batch_size = int(n_estimators / (n_db * n_iteration))
        ''' n_estimators = n_batch_size * n_db * n_iteration '''
        # 20 estimators / 5 dbs / 1 iterations = collect 4 estimators at each db for each round
        # 20 estimators / 5 dbs / 2 iterations = collect 2 estimators at each db for each round

        n_dbs = list_of_n_dbs[i_n_dbs]

        timer_start = time.time()

        classifier_iterative = make_iterative_classifier(
            databases=n_dbs,  # list
            # Default: classifier=WarmStartAdaBoostClassifier()
            n_estimators=n_estimators,
            n_type=n_type,
            n_batch_size=n_batch_size,
            var_choosing_next_database=var_choosing_next_database
            # Default: patients_batch_size: int = 1
        )

        # Stop timer
        timer_stop = time.time()
        timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))

        f_1, mcc, auc, acc = make_scores(classifier_iterative, X_test, y_test)
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

    print()
    print()


def pipeline_3_1_comm_effi(X_train, X_test, y_train, y_test, e, r, s):

    # Settings
    list_of_n = [1, 2, 5, 10, 20, 50]
    n_iteration = r  # 0 to 100
    n_batch_size = e  # 1 or 2 or 5 or 10
    # n_estimators E = e * n * r

    n_type = "batch"
    var_choosing_next_database = "iterate"
    patients_batch_size = 1  # Spielt keine Rolle

    # Simulate n DBs for n = [1, 2, 5, 10, 20, 50]
    list_of_n_dbs = simulate_n_databases_with_equal_sample_size(
        X_train, X_test, y_train, y_test,
        list_of_n=list_of_n,
    )

    # Initialize
    results = list()
    timer_list = list()  # Timers

    res_f_1 = list()
    res_mcc = list()
    res_auc = list()
    res_acc = list()  # Score Containers

    # Granularity of sites
    # list_of_n = list_of_n

    # Print title
    print("Communication Efficiency Experiment for N Sites Iterative")
    print()

    print("s\tn\tr\te\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i_n_dbs in range(0, len(list_of_n_dbs)):

        # n_db = list_of_n[i_n_dbs]
        # n_batch_size = int(n_estimators / (n_db * n_iteration))
        n_estimators = n_batch_size * list_of_n[i_n_dbs] * n_iteration
        # 20 estimators / 5 dbs / 1 iterations = collect 4 estimators at each db for each round
        # 20 estimators / 5 dbs / 2 iterations = collect 2 estimators at each db for each round

        n_dbs = list_of_n_dbs[i_n_dbs]

        # timer_start = time.time()

        classifier_iterative = make_iterative_classifier(
            databases=n_dbs,  # list
            # Default: classifier=WarmStartAdaBoostClassifier()
            n_estimators=n_estimators,
            n_type=n_type,
            n_batch_size=n_batch_size,
            var_choosing_next_database=var_choosing_next_database
            # Default: patients_batch_size: int = 1
        )

        # Stop timer
        # timer_stop = time.time()
        # timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))

        f_1, mcc, auc, acc = make_scores(classifier_iterative, X_test, y_test)
        res_f_1.append(f_1)
        res_mcc.append(mcc)
        res_auc.append(auc)
        res_acc.append(acc)

        print(
            str(s) +
            "\t" +
            str(list_of_n[i_n_dbs]) +
            "\t" +
            str(r) +
            "\t" +
            str(e) +
            "\t" +
            str(round(f_1, 5)) +
            "\t\t" +
            str(round(mcc, 5)) +
            "\t\t" +
            str(round(auc, 5)) +
            "\t\t" +
            str(round(acc, 5))
        )

        results.append([s, list_of_n[i_n_dbs], r, e,
                        f_1, mcc, auc, acc])

    return results
