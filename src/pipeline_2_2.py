# Load population
from numpy.lib.function_base import kaiser
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
from evaluation import evaluate

# Reference
from ref.main import make_iterative_classifier
from ref.main import make_not_iterative_classifier
from ref.database import Database
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time


data = load_breast_cancer()

# Settings
n_db = 5
test_size = 0.2
#

# Simulate n DB pairs with decreasing sample size imbalance
prepared_data = simulate_db_size_imbalance(
    data=load_breast_cancer(), test_size=0.2, balance_step=0.05, k=1)

len_data = len(data.get("data"))
len_test = len_data*test_size

# Extract Test Set (20%)
X_test = prepared_data.get("test").get("X")
y_test = prepared_data.get("test").get("y")

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
    db_list = db_pairs[i]

    timer_start = time.time()

    classifier_combined = make_not_iterative_classifier(databases=db_list,
                                                        patients_batch_size=1,
                                                        weight_databases=True)
    timer_stop = time.time()
    timer_list.append(timer_stop - timer_start)
    # score_federated = classifier_combined.score(prepared_data.get("test").get("X"), prepared_data.get("test").get("y"))
    f_1, mcc, auc, acc = evaluate(classifier_combined, X_test, y_test)
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
