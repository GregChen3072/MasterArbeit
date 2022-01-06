# Model Evaluation
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score


def make_scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f_1 = f1_score(y_test, y_pred, average='binary')
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred, average=None)
    acc = balanced_accuracy_score(y_test, y_pred)

    return f_1, mcc, auc, acc
