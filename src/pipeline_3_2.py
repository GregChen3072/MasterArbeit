# Load population
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# User defined functions
from db_simulator import simulate_db_size_imbalance

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

# Load population
data = load_breast_cancer()

# Settings I
n_estimators = 100
n_db = 10
n_iteration = 5

n_batch_size = int(n_estimators / (n_db * n_iteration))
''' n_estimators = n_batch_size * n_db * n_iteration '''

# 20 estimators / 5 dbs / 1 rounds = 4 estimators at each db for each round
# 20 estimators / 5 dbs / 2 rounds = 2 estimators at each db for each round
# n_iteration = 2

# Settings II
test_size = 0.2
n_type = "batch"
var_choosing_next_database = "iterate"
#n_estimators = 50
patients_batch_size = 10  # Spielt keine Rolle


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
print("Federation Iterative not Weighted")

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

# Instantiate classifier
warm_start = WarmStartAdaBoostClassifier()
warm_start = warm_start.set_beginning()

classifier_fed_iterative = Classifier(warm_start)

n_generator = NextN(
    n_size=n_estimators,
    n_data_sets=len(db_pairs),
    n_type=n_type,
    batch_size=n_batch_size
)

classifier_iterator = Classifier(
    classifier=warm_start,
    n_generator=n_generator
)

database_chooser = NextDataSets(data_sets=db_pairs,
                                next_type=var_choosing_next_database,
                                accuracy_batch_size=patients_batch_size)


# Initialize iterative model;
# classifier_fed_iterative will collect knowledge from site to site iteratively;
classifier_fed_iterative = None


print()
print("Settings")
print(f'Number of DBs: {n_db}')
print(f'Number of estimators: {n_estimators}')
print(
    f'Number of estimators per DB per iteration: {n_batch_size}'
)
print(f'Number of rounds: {n_iteration}')
print()

print("Progress: ")

n_visits = int(1)

print("Degree Imbalance\tF-1 Score\t\tMCC Score\tAUC Score\tACC Score\tDuration in Seconds")

for i in range(0, len(db_pairs)):
    db_pair = db_pairs[i]

    timer_start = time.time()

    classifier_iterative = make_iterative_classifier(
        databases=db_pair,
        n_estimators=n_estimators,
        n_type=n_type,
        n_batch_size=n_batch_size,
        var_choosing_next_database=var_choosing_next_database
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
