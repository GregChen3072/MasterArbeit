# Load population
from scipy.sparse.construct import random
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

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


'''
        # Instantiate classifier
        warm_start = WarmStartAdaBoostClassifier()
        warm_start = warm_start.set_beginning()

        # n_estimators should always be given.
        #classifier_fed_iterative = Classifier(warm_start)

        n_generator = NextN(
            n_size=n_estimators,
            n_data_sets=len(list_of_n_dbs[n_dbs]),
            n_type=n_type,
            batch_size=n_batch_size
        )

        classifier_iterator = Classifier(
            classifier=warm_start,
            n_generator=n_generator
        )


        database_chooser = NextDataSets(data_sets=list_of_n_dbs[n_dbs],
                                        next_type=var_choosing_next_database,
                                        accuracy_batch_size=patients_batch_size)

        # Initialize iterative model;
        # classifier_fed_iterative will collect knowledge from site to site iteratively;
        classifier_fed_iterative = None

        n_visits = int(1)

        while classifier_iterator.finished() is False:
            # erhöht die Anzahl der Entscheidungsbäume
            # .get_prepared_classifier() returns a WarmStartABC object.
            classifier_fed_iterative = classifier_iterator.get_prepared_classifier()
            index = database_chooser.get_next_index(classifier_fed_iterative)
            # print("Index: "+str(index))
            # print("Length Databases: "+str(len(databases)))
            current_database = 2[index]  # wählt die nächste Datenbank
            # erweitert auf der ausgewählten Datenbank den Klassifizieren
            classifier_fed_iterative = current_database.extend_classifier(
                classifier_fed_iterative)
            classifier_iterator.update_classifier(
                classifier_fed_iterative)  # updatet den Klassifizierer
            # print("Round finished.")
            if n_visits % n_db == 0:
                print(f'Round {int(n_visits / n_db)} is complete! ')
            n_visits += 1

        # Stop timer
        timer_stop = time.time()
        duration = timer_stop - timer_start

        # Initialize
        res = list()
        timer_list = list()

        # Stop timer
        timer_stop = time.time()
        timer_list.append(timer_stop - timer_start)

        f_1, mcc, auc, acc = make_scores(
            classifier_fed_aggregated, X_test, y_test)
        res_f_1.append(f_1)
        res_mcc.append(mcc)
        res_auc.append(auc)
        res_acc.append(acc)

        print(
            str(list_of_n[n_dbs]) +
            "\t\t" +
            str(f_1) +
            "\t" +
            str(round(mcc, 2)) +
            "\t\t" +
            str(round(auc, 2)) +
            "\t\t" +
            str(round(acc, 2)) +
            "\t\t" +
            str(timer_list[n_dbs])
        )
'''
