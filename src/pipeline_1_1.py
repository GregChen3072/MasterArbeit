# User defined functions
from db_simulator import simulate_1_database_with_all_data_centralized

# Data preparation
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from evaluation import evaluate

# Utils
import time


X, y = simulate_1_database_with_all_data_centralized()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=6)

timer_start = time.time()

abc = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=6)

classifier_central = abc.fit(X_train, y_train)

# Stop timer
timer_stop = time.time()
duration = timer_stop - timer_start


y_pred = classifier_central.predict(X_test)

print(f'Training Time: {duration} seconds. ')
print()

# X_test = prepared_data.get("test_set").get("X_test")
# y_test = prepared_data.get("test_set").get("y_test")

f_1, mcc, auc, acc = evaluate(classifier_central, X_test, y_test)

print(f'F-1 Score: {f_1}')
print(f'MCC Score: {mcc}')
print(f'AUC Score: {auc}')
print(f'ACC Score: {acc}')
