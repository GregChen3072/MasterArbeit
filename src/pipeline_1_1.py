# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from scoring import make_scores

# Utils
import time


def pipeline_1_1(X_train, X_test, y_train, y_test, E, s):

    n = 1
    # results = list()

    #print("s\tn\te\tF_1 Score\tMCC Score\tAUC Score\tACC Score")

    abc = AdaBoostClassifier(n_estimators=E, random_state=6)

    classifier_central = abc.fit(X_train, y_train)

    f_1, mcc, auc, acc = make_scores(classifier_central, X_test, y_test)

    print(
        str(s) + "\t" +
        str(n) + "\t" +
        str(E) + "\t" +
        str(1) + "\t" +
        str(round(f_1, 3)) +
        "\t\t" +
        str(round(mcc, 3)) +
        "\t\t" +
        str(round(auc, 3)) +
        "\t\t" +
        str(round(acc, 3))
    )

    # results.append([s, n, e, f_1, mcc, auc, acc])

    return [s, n, E, f_1, mcc, auc, acc]
