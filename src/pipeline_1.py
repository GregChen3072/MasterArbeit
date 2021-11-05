# User defined functions
from db_simulator import simulate_1_database_with_all_data_centralized

# Data preparation
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import AdaBoostClassifier

# Model Evaluation
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, f1_score


X, y = simulate_1_database_with_all_data_centralized()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=6)

abc = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=1)

model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("AdaBoost Classifier Model F-1 Score:",
      f1_score(y_test, y_pred, average=None))
print("AdaBoost Classifier Model MCC Score:",
      matthews_corrcoef(y_test, y_pred))
print("AdaBoost Classifier Model AUC Score:",
      roc_auc_score(y_test, y_pred, average=None))
print("AdaBoost Classifier Model ACC Score:",
      balanced_accuracy_score(y_test, y_pred))
