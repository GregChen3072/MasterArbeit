# Load population
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# User defined functions
from db_simulator import simulate_n_databases_with_equal_sample_size

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from evaluation import evaluate

# Reference
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time

data = load_breast_cancer()

# Settings
list_of_n = [1, 2, 5, 10, 20, 50, 100]
# n_db = 5
test_size = 0.2
n_estimators = 20

# Simulate n DBs for n = [1, 2, 5, 10, 20, 50, 100]
prepared_data = simulate_n_databases_with_equal_sample_size(
    data=data,
    list_of_n=list_of_n,
    test_size=test_size
)

# Initialize
res = list()
timer_list = list()

# Count of data points for global train and test sets
# len_data = len(data.get("data"))
# len_test = len_data*test_size

# Extract test set from prepared data
X_test = prepared_data.get("test_set").get("X_test")
y_test = prepared_data.get("test_set").get("y_test")

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

# n DBs for n = [1, 2, 5, 10, 20, 50, 100]
list_of_n_dbs = prepared_data.get("list_of_n_dbs")

# Granularity of sites
# list_of_n = list_of_n

print("n Databases\tF-1 Score\t\tMCC Score\tAUC Score\tACC Score\tDuration in Seconds")

# Print title
print()
print("Federation Non-iterative not Weighted")

for n_dbs in range(0, len(list_of_n_dbs)):
    # n_dbs = n_dbs  # n sites / DBs for a given n

    # Start timer
    timer_start = time.time()

    # Instantiate classifier
    classifier_fed_aggregated = CombinedAdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=1.,
        algorithm='SAMME.R',
        random_state=6,
        patients_batch_size=1,
        weight_databases=False
    )

    # Collect all estimators
    for db in list_of_n_dbs[n_dbs]:
        classifier_fed_aggregated.add_fit(db)

    # Stop timer
    timer_stop = time.time()
    duration = timer_stop - timer_start

    f_1, mcc, auc, acc = evaluate(classifier_fed_aggregated, X_test, y_test)
    res_f_1.append(f_1)
    res_mcc.append(mcc)
    res_auc.append(auc)
    res_acc.append(acc)

    print(
        str(list_of_n[n_dbs]) +
        "\t\t\t" +
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
