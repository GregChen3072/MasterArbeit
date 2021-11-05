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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score

# Reference
from ref.database import Database
from ref.main import make_non_iterative_classifier
from ref.combiner import CombinedAdaBoostClassifier

# Utils
import time

data = load_breast_cancer()

n_db = 5
test_size = 0.2

prepared_data = simulate_n_databases_with_equal_sample_size(
    data=data, n_db=n_db, test_size=test_size)

# print(type(prepared_data.get('db_list')[0]))

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


res = list()
timer_list = list()

print()
print("Federation Non-iterative not Weighted")


db_list = prepared_data.get("db_list")

print("n_db\tscore\tduration in seconds")
timer_start = time.time()

classifier_federated = CombinedAdaBoostClassifier(
    learning_rate=1.,
    n_estimators=20,
    algorithm='SAMME.R',
    random_state=6,
    patients_batch_size=1,
    weight_databases=False
)

for i in range(0, len(db_list)):
    # Data Structure of "db_list": [[Database], [Database], ..., [Database]]
    # Data Structure of "db": [Database]
    db = db_list[i]
    classifier_federated.add_fit(db)

timer_stop = time.time()
duration = timer_stop - timer_start

score_federated = classifier_federated.score(
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
