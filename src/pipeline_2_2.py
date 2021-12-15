# Load population
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# User defined functions
from db_simulator import simulate_db_size_imbalance

# Data preparation
import pandas as pd
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from scoring import make_scores

# Reference
from ref.main import make_iterative_classifier
from ref.main import make_not_iterative_classifier
from ref.database import Database
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time


def pipeline_2_2(X_train, X_test, y_train, y_test):
    pass


data = load_breast_cancer()

# Settings
test_size = 0.2


# Simulate n DB pairs with decreasing sample size imbalance
prepared_data = simulate_db_size_imbalance(
    data=load_breast_cancer(),
    test_size=test_size,
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

    # Setting: weighing or not (weight_databases=False / True)
    classifier_combined = make_not_iterative_classifier(databases=db_pair,
                                                        patients_batch_size=1,
                                                        weight_databases=False)

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
