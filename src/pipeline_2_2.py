# User defined functions
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


def pipeline_2_2_unweighted(X_train, X_test, y_train, y_test):

    n_estimators = 100

    # Simulate n DB pairs with decreasing sample size imbalance
    prepared_data = simulate_db_size_imbalance(
        X_train, X_test, y_train, y_test,
        balance_step=0.05,
        k=1
    )

    # Print title
    print()
    print("Federation Non-iterative not Weighted")

    # Initialize
    # res = list()
    res_f_1 = list()
    res_mcc = list()
    res_auc = list()
    res_acc = list()  # Score Containers

    timer_list = list()  # Timers

    db_pairs = prepared_data.get("db_pairs")  # DB Pairs

    # Degrees of balance for each DB pair
    balance_list = prepared_data.get("balance_list")

    print("Degree Imbalance\tF-1 Score\t\tMCC Score\tAUC Score\tACC Score\tDuration in Seconds")

    for i in range(0, len(db_pairs)):
        db_pair = db_pairs[i]

        timer_start = time.time()

        classifier = CombinedAdaBoostClassifier(
            base_estimator=None,
            learning_rate=1.,
            n_estimators=int(n_estimators/2),
            algorithm='SAMME.R',
            random_state=6,
            patients_batch_size=1,
            weight_databases=False
        )

        classifier_combined = classifier.make_fit(db_pair)

        timer_stop = time.time()
        timer_list.append(timer_stop - timer_start)
        # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))
        f_1, mcc, auc, acc = make_scores(classifier_combined, X_test, y_test)
        res_f_1.append(f_1)
        res_mcc.append(mcc)
        res_auc.append(auc)
        res_acc.append(acc)

        print(
            str(round(balance_list[i], 2)) +
            "\t\t\t" +
            str(f_1) +
            "\t" +
            str(round(mcc, 2)) +
            "\t\t" +
            str(round(auc, 2)) +
            "\t\t" +
            str(round(acc, 2)) +
            "\t\t" +
            str(timer_list[i])
        )

    # return [score_whole, res, dic.get("balance_list"), time_list]
