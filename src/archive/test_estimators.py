from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.get("data")
y = data.get("target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1)
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1, random_state=0)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = model.predict(X_test)
# calculate and print model accuracy
print("AdaBoost Classifier Model Accuracy:",
      accuracy_score(y_test, y_pred))
