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
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

import pickle

start_time = time.time()

# Custom function transformers in piplines
# Creating a ready-to-deploy pipeline featuring data transformation
# Define a feature extractor to flag values abvoe average and store in col 0 of Z.


def a_udf_called_more_than_average(X, multiplier=1.0):
    Z = X.copy()
    Z[:, 1] = Z[:, 0] > multiplier*np.mean(Z[:, 1])
    return Z


def basic_data_split(dataset, features, target, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.loc[:, features],
        dataset.loc[:, target],
        test_size=test_size
    )
    return X_train, X_test, y_train, y_test


def training_model(X_train, y_train):
    _value = []
    _model = []
    _best_features = []
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    F5 = []
    F6 = []
    F7 = []

    subsets_sum = [F1] + [F2] + [F3] + [F4] + [F5] + [F6] + [F7]
    print(subsets_sum)

    # 12 random states randomly choosen for the outer-Monte Carlo CV
    for i in [32, 41, 45, 52, 65, 72, 96, 97, 112, 114, 128, 142]:
        print('\n\nRandom State: ', i)
        model = AdaBoostClassifier(random_state=i, n_estimators=200)

        # Split the dataset into 2 stratified parts, 80% for outer training set
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X_train,
            y_train,
            train_size=0.8,
            random_state=i,
            stratify=y_train
        )

        # Choose k-Fold cross-validation techinique for the inner loop
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        # Set temporary variables
        best_subset = []
        best_auc = -np.inf

        # Loop over the features combinations
        for subset in subsets_sum:
            score = cross_val_score(
                model,
                X=X1_train[subset],
                y=y1_train,
                cv=inner_cv.split(X1_train[subset], y1_train),
                scoring='accuracy'
            )
            if score.mean() > best_auc:
                best_auc = score.mean()
                best_subset = subset
        # Train the model on the Outer training set with the selected feature combination
        model = model.fit(X1_train[best_subset], y1_train)

        # Calculate the predicted labels with the model on the Outer test set with the selected feature combination.
        y1_pred = model.predict(X1_test[best_subset])

        # Calculate the accuracy between predicted and true labels
        acc = accuracy_score(y1_test, y1_pred)
        print('Selected features: ', best_subset, '; Outer Test ACC: ', acc)

        _best_features.append(best_subset)
        _value.append(acc)
        _model.append(model)

    print()
    print(_value)
    print()
    print('Maximum Accuracy Index: ', np.argmax(_value))

    idx = np.argmax(_value)
    print("\nBest model parameters with random_state: ")
    print(_model[idx])
    print("\nBest feature combination: ")
    print(_best_features[idx])
    print("\nBest accuracy from Monte Carlo CV: ")
    print(_value[idx])

    return (_model[idx], _best_features[idx])


def evaluate_model(model, features, X_train, y_train, X_test, y_test):
    print()
    print(model.get_params(deep=True))

    # Evaluate the performance of the trained model
    pred_Class = model.predict(X_test[features])
    acc = accuracy_score(y_test, pred_Class)
    classReport = classification_report(y_test, pred_Class)
    confMatrix = confusion_matrix(y_test, pred_Class)
    kappa_score = cohen_kappa_score(y_test, pred_Class)

    print()
    print('Evaluation of the trained model: ')
    print()
    print('Accuracy: ', acc)
    print()
    print('Kappa Score: ', kappa_score)
    print()
    print('Confusion Matrix: \n', confMatrix)
    print()
    print('Classification Report: \n', classReport)

    pred_proba = model.predict_proba(X_test[features])

    # Add more plots here using scikit-plot
    # ROC curves
    skplt.metrics.plot_roc(y_test, pred_proba, figsize=(9, 6))
    plt.show()

    # Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_test, pred_Class, figsize=(9, 6))
    plt.show()

    # Precision Recall Curve
    skplt.metrics.plot_precision_recall(
        y_test,
        pred_proba,
        title='Precision-Recall Curve',
        plot_micro=True,
        classes_to_plot=None,
        ax=None,
        figsize=(9, 6),
        cmap='nipy_spectral',
        title_fontsize='large',
        text_fontsize='medium'
    )
    plt.show()

    skplt.estimators.plot_learning_curve(
        model, X_train[features], y_train, figsize=(9, 6))
    plt.show()

    return model


# model = evaluate_model(model, feature_names, X_train, y_train, X_test, y_test)

print()
print("Required Time %s seconds: " % (time.time() - start_time))


pipe = Pipeline(
    [
        ('ft', FunctionTransformer(a_udf_called_more_than_average)),
        ('ada', AdaBoostClassifier(random_state=2))
    ]
)
params = dict(
    ft__multiplier=[1, 2, 3],
    ada__n_estimators=[50, 100]
)

grid_search = GridSearchCV(pipe, param_grid=params)

X = None
y = None
X_train, y_train, X_test, y_test = train_test_split(X, y)
#clf = RandomForestClassifier().fit(X_train, y_train)
gs = grid_search.fit(X_train, y_train)

with open('pipe.pkl', 'wb') as file:
    pickle.dump(gs, file=file)

with open('pipe.pkl', 'rb') as file:
    gs_transported = pickle.load(file)

gs_transported.predict(X_test)

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(gs_transported, X, y, cv=cv)
