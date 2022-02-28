# User defined functions
from db_simulator import simulate_db_size_imbalance

# Model Evaluation
from scoring import make_scores

# Reference
from ref.next_n_size import NextN
from ref.next_n_size import NextDataSets
from ref.classifier import WarmStartAdaBoostClassifier
from ref.classifier import Classifier
from ref.main import make_iterative_classifier, make_weighted_iterative_classifier

# Utils
import time


def pipeline_3_2_unweighted(X_train, X_test, y_train, y_test, s, E):
    # Settings I
    n_estimators = E
    n_db = 2
    n_iteration = 5

    n_batch_size = int(n_estimators / (n_db * n_iteration))
    ''' n_estimators = n_batch_size * n_db * n_iteration '''

    # 20 estimators / 5 dbs / 1 rounds = 4 estimators at each db for each round
    # 20 estimators / 5 dbs / 2 rounds = 2 estimators at each db for each round

    # Settings II
    n_type = "batch"
    var_choosing_next_database = "iterate"

    # Simulate n DB pairs with decreasing sample size imbalance
    prepared_data = simulate_db_size_imbalance(
        X_train, y_train, balance_step=0.05, k=1
    )

    # Print title
    print()
    print("Federation Iterative not Weighted")

    # Initialize
    results = list()
    # timer_list = list()  # Timers

    db_pairs = prepared_data.get("db_pairs")  # DB Pairs
    # Degrees of balance for each DB pair
    balance_list = prepared_data.get("balance_list")

    # print()
    # print("Settings")
    # print(f'Number of estimators: {n_estimators}')
    # print(f'Number of DBs: {n_db}')
    # print(f'Number of rounds: {n_iteration}')
    # print(
    #     f'Number of estimators per DB per iteration: {n_batch_size}'
    # )
    # print()
    # print("Progress: ")

    n_visits = int(1)

    print("s\tDegree Imbalance\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i in range(0, len(db_pairs)):

        db_pair = db_pairs[i]

        # timer_start = time.time()

        classifier_iterative = make_iterative_classifier(
            databases=db_pair,
            n_estimators=n_estimators,
            n_type=n_type,
            n_batch_size=n_batch_size,
            var_choosing_next_database=var_choosing_next_database
        )

        # Stop timer
        # timer_stop = time.time()
        # timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))

        f_1, mcc, auc, acc = make_scores(classifier_iterative, X_test, y_test)

        print(
            str(s) + "\t" +
            str(round(balance_list[i], 3)) +
            "\t\t\t" +
            str(round(f_1, 3)) +
            "\t\t" +
            str(round(mcc, 3)) +
            "\t\t" +
            str(round(auc, 3)) +
            "\t\t" +
            str(round(acc, 3))
        )

        results.append([s, balance_list[i],
                        f_1, mcc, auc, acc])

    return results


def pipeline_3_2_weighted(X_train, X_test, y_train, y_test, s, E):
    # Settings I
    n_estimators = E
    n_db = 2
    n_iteration = 5

    n_batch_size = int(n_estimators / (n_db * n_iteration))
    ''' n_estimators = n_batch_size * n_db * n_iteration '''

    # 20 estimators / 5 dbs / 1 rounds = 4 estimators at each db for each round
    # 20 estimators / 5 dbs / 2 rounds = 2 estimators at each db for each round

    # Settings II
    # n_type = "batch"
    n_type = "proportional"
    var_choosing_next_database = "iterate"

    # Simulate n DB pairs with decreasing sample size imbalance
    prepared_data = simulate_db_size_imbalance(
        X_train, y_train, balance_step=0.05, k=1
    )

    # Print title
    print()
    print("Federation Iterative not Weighted")

    # Initialize
    results = list()

    # timer_list = list()  # Timers

    db_pairs = prepared_data.get("db_pairs")  # DB Pairs
    # Degrees of balance for each DB pair
    balance_list = prepared_data.get("balance_list")

    """ print()
    print("Settings")
    print(f'Number of estimators: {n_estimators}')
    print(f'Number of DBs: {n_db}')
    print(f'Number of rounds: {n_iteration}')
    print(
        f'Number of estimators per DB per iteration: {n_batch_size}'
    )
    print()

    print("Progress: ") """

    n_visits = int(1)

    print("s\tDegree Imbalance\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i in range(0, len(db_pairs)):

        db_pair = db_pairs[i]
        degree_of_balance = round(balance_list[i], 2)

        # timer_start = time.time()

        classifier_iterative = make_weighted_iterative_classifier(
            databases=db_pair,
            proportion=degree_of_balance,
            n_iteration=n_iteration,
            n_estimators=n_estimators,
            n_type=n_type,
            n_batch_size=n_batch_size,
            var_choosing_next_database=var_choosing_next_database
        )

        # Stop timer
        # timer_stop = time.time()
        # timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))

        f_1, mcc, auc, acc = make_scores(classifier_iterative, X_test, y_test)

        print(
            str(s) + "\t" +
            str(round(balance_list[i], 2)) +
            "\t\t\t" +
            str(round(f_1, 3)) +
            "\t\t" +
            str(round(mcc, 3)) +
            "\t\t" +
            str(round(auc, 3)) +
            "\t\t" +
            str(round(acc, 3))
        )

        results.append([s, balance_list[i],
                        f_1, mcc, auc, acc])

    return results
