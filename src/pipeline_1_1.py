# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from scoring import make_scores

# Utils
import time


def pipeline_1_1(X_train, X_test, y_train, y_test):
    timer_start = time.time()

    abc = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=6)

    classifier_central = abc.fit(X_train, y_train)

    # Stop timer
    timer_stop = time.time()
    duration = timer_stop - timer_start

    # y_pred = classifier_central.predict(X_test)

    print(f'Training Time: {duration} seconds. ')
    print()

    f_1, mcc, auc, acc = make_scores(classifier_central, X_test, y_test)

    print(f'F-1 Score: {f_1}')
    print(f'MCC Score: {mcc}')
    print(f'AUC Score: {auc}')
    print(f'ACC Score: {acc}')
