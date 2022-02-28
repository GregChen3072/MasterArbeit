# User defined functions
from numpy.lib.function_base import blackman
from db_simulator import simulate_db_size_imbalance

# Model Evaluation
from scoring import make_scores

# Reference
from ref.main import make_iterative_classifier
from ref.main import make_not_iterative_classifier
from ref.database import Database
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time


def pipeline_2_2_unweighted(X_train, X_test, y_train, y_test, s, E):

    n_estimators = E

    # Simulate n DB pairs with decreasing sample size imbalance
    prepared_data = simulate_db_size_imbalance(
        X_train, y_train, balance_step=0.05, k=1
    )

    # Print title
    print()
    print("Federation Non-iterative not Weighted")

    # Initialize
    results = list()
    # timer_list = list()  # Timers

    db_pairs = prepared_data.get("db_pairs")  # DB Pairs

    # Degrees of balance for each DB pair
    balance_list = prepared_data.get("balance_list")

    print("s\tDegree Imbalance\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i in range(0, len(db_pairs)):
        db_pair = db_pairs[i]

        # timer_start = time.time()

        classifier = CombinedAdaBoostClassifier(
            base_estimator=None,
            learning_rate=1.,
            n_estimators=int(n_estimators/2),
            algorithm='SAMME.R',
            random_state=6,
            patients_batch_size=1,
            weight_databases=False
        )

        classifier_combined = classifier.make_fit(
            db_pair)

        # timer_stop = time.time()
        # timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))
        f_1, mcc, auc, acc = make_scores(classifier_combined, X_test, y_test)

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


def pipeline_2_2_weighted(X_train, X_test, y_train, y_test, s, E):

    n_estimators = E

    # Simulate n DB pairs with decreasing sample size imbalance
    prepared_data = simulate_db_size_imbalance(
        X_train, y_train, balance_step=0.05, k=1
    )

    # Print title
    print()
    print("Federation Non-iterative not Weighted")

    # Initialize
    results = list()

    # timer_list = list()  # Timers

    db_pairs = prepared_data.get("db_pairs")  # DB Pairs

    # Degrees of balance for each DB pair
    balance_list = prepared_data.get("balance_list")

    print("s\tDegree Imbalance\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i in range(0, len(db_pairs)):
        db_pair = db_pairs[i]
        # current_db_size_balance = balance_list[i]

        # timer_start = time.time()

        classifier = CombinedAdaBoostClassifier(
            base_estimator=None,
            learning_rate=1.,
            n_estimators=int(n_estimators),
            algorithm='SAMME.R',
            random_state=6,
            patients_batch_size=1,
            weight_databases=False
        )

        classifier_combined = classifier.make_fit_w(
            db_pair)

        # timer_stop = time.time()
        # timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))
        f_1, mcc, auc, acc = make_scores(classifier_combined, X_test, y_test)

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
