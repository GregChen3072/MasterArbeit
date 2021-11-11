# Load population
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
from evaluation import evaluate

# Reference
from ref.database import Database
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time

data = load_breast_cancer()

# Settings
n_db = 5
test_size = 0.2
#

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
print("Federation Non-iterative not Weighted")

# Initialize
res = list()
timer_list = list()
db_list = prepared_data.get("db_list")

# Start timer
# print("n_db\tscore\tduration in seconds")
timer_start = time.time()

# Instantiate classifier
classifier_fed_aggregated = CombinedAdaBoostClassifier(
    learning_rate=1.,
    n_estimators=20,
    algorithm='SAMME.R',
    random_state=6,
    patients_batch_size=1,
    weight_databases=False
)


# Collect all estimators
for i in range(0, len(db_list)):
    # Data Structure of "db_list": [[Database], [Database], ..., [Database]]
    # Data Structure of "db": [Database]
    db = db_list[i]
    classifier_fed_aggregated.add_fit(db)

# Stop timer
timer_stop = time.time()
duration = timer_stop - timer_start

'''
# Basic scoring
score_federated = classifier_fed_aggregated.score(
    X=prepared_data.get("test_set").get("X_test"),
    y=prepared_data.get("test_set").get("y_test")
)

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

f_1, mcc, auc, acc = evaluate(classifier_fed_aggregated, X_test, y_test)

print(f'F-1 Score: {f_1}')
print(f'MCC Score: {mcc}')
print(f'AUC Score: {auc}')
print(f'ACC Score: {acc}')
