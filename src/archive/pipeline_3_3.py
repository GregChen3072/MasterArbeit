# Load population
from scipy.sparse.construct import random
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# User defined functions
from db_simulator import simulate_n_databases_with_equal_sample_size

# Data preparation
import pandas as pd
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from scoring import make_scores

# Reference
from ref.next_n_size import NextN
from ref.next_n_size import NextDataSets
from ref.classifier import WarmStartAdaBoostClassifier
from ref.classifier import Classifier

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


# Simulate n DBs with equal sample size
prepared_data = simulate_n_databases_with_equal_sample_size(
    data=data, n_db=n_db, test_size=test_size)

# print(type(prepared_data.get('db_list')[0]))

########
print()
print("Non-Federated Model")
len_data = len(data.get("data"))
print("Number of data points: " + str(len_data))
len_test = int(len_data * test_size)
print("Number of data points in test database: " + str(len_test))

# db_central is the "Database" object representing the database with data centralization.
db_central = prepared_data.get("db_central")

# Train a central model using centralized data.
classifier_central = AdaBoostClassifier()
classifier_central.fit(X=db_central.x, y=db_central.y)

# Output the performance measure of the central model.
score_central = classifier_central.score(prepared_data.get("test_set").get(
    "X_test"), prepared_data.get("test_set").get("y_test"))

print()
print("Non-Federated Model Score")
print(str(round(score_central, 5)))
########

# Print title
print()
print("Federation Iterative")

# Initialize
res = list()
timer_list = list()
db_list = prepared_data.get("db_list")

# Start timer
timer_start = time.time()


# db_list input format: [db yin, db yang]
# classifier_fed_iterative = make_not_iterative_classifier(
#    databases=db_list,
#    patients_batch_size=patients_batch_size,
#    weight_databases=True
# )


# Instantiate classifier
warm_start = WarmStartAdaBoostClassifier()
warm_start = warm_start.set_beginning()

classifier_fed_iterative = Classifier(warm_start)

n_generator = NextN(
    n_size=n_estimators,
    n_data_sets=len(db_list),
    n_type=n_type,
    batch_size=n_batch_size
)

classifier_iterator = Classifier(
    classifier=warm_start,
    n_generator=n_generator
)

database_chooser = NextDataSets(data_sets=db_list,
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

while classifier_iterator.finished() is False:
    # erhöht die Anzahl der Entscheidungsbäume
    # .get_prepared_classifier() returns a WarmStartABC object.
    classifier_fed_iterative = classifier_iterator.get_prepared_classifier()
    index = database_chooser.get_next_index(classifier_fed_iterative)
    # print("Index: "+str(index))
    # print("Length Databases: "+str(len(databases)))
    current_database = db_list[index]  # wählt die nächste Datenbank
    # erweitert auf der ausgewählten Datenbank den Klassifizieren
    classifier_fed_iterative = current_database.extend_classifier(
        classifier_fed_iterative)
    classifier_iterator.update_classifier(
        classifier_fed_iterative)  # updatet den Klassifizierer
    # print("Round finished.")
    if n_visits % n_db == 0:
        print(f'Round {int(n_visits / n_db)} is complete! ')
    n_visits += 1
# print("classifier finished.")

# Stop timer
timer_stop = time.time()
duration = timer_stop - timer_start

# Basic Scoring
'''
score_federated = classifier_fed_iterative.score(
    X=prepared_data.get("test_set").get("X_test"),
    y=prepared_data.get("test_set").get("y_test")
)

print()
print("n_db\tscore\tduration in seconds")
print(
    str(len(db_list)) +
    "\t" +
    str(round(score_federated, 5)) +
    "\t" +
    str(duration)
)
print()
'''

print(f'Training Time: {duration} seconds. ')
print()

X_test = prepared_data.get("test_set").get("X_test")
y_test = prepared_data.get("test_set").get("y_test")

f_1, mcc, auc, acc = make_scores(classifier_fed_iterative, X_test, y_test)

print(f'F-1 Score: {f_1}')
print(f'MCC Score: {mcc}')
print(f'AUC Score: {auc}')
print(f'ACC Score: {acc}')
