import time
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

from db_simulator import load_HCC_data, load_ILPD_data
from scoring import make_scores

import pickle
import sys


# https://www.kaggle.com/prashant111/adaboost-classifier-tutorial/notebook
# https://www.kaggle.com/alexisbcook/pipelines


# start_time = time.time()

# iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
X, y = load_ILPD_data()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=6)

res = list()

print("n_estimators\tsize\tF-1 Score\tMCC Score\tAUC Score\tACC Score")
for n in range(1, 101):
    # print(n)
    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=n, random_state=6)

    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    # Use of sys.getsizeof() can be done to find the storage size of a particular object that occupies some space in the memory.
    # This function returns the size of the object in bytes.
    # Note: Only the memory consumption directly attributed to the object is accounted for, not the memory consumption of objects it refers to.

    siz = sys.getsizeof(model)

    # Predict the response for test dataset
    # y_pred = model.predict(X_test)

    f_1, mcc, auc, acc = make_scores(model, X_test, y_test)
    # calculate and print model accuracy
    # print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))
    print(
        str(n) +
        "\t\t" +
        str(siz) +
        "\t" +
        str(round(f_1, 3)) +
        "\t\t" +
        str(round(mcc, 3)) +
        "\t\t" +
        str(round(auc, 3)) +
        "\t\t" +
        str(round(acc, 3))
    )

    res.append([n, siz, f_1, mcc, auc, acc])

'''
df_res = pd.DataFrame(
    res,
    columns=['N Estimators', 'Model Size', 'F-1 Score', 'MCC Score',
             'AUC Score', 'ACC Score']
)


df_res.to_csv('/Users/greg/Downloads/ilpd_grid_search.csv', index=False, header=True)
df_res.to_excel('/Users/greg/Downloads/ilpd_grid_search.xlsx', index=False, header=True)
'''
