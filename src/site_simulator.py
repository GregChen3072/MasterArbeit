from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


def load_data():
    iris = load_iris()
    X = iris.get("data")
    y = iris.get("target")
    return X, y


def simulate_n_sites(n=100):
    '''
        Set n = {1, 2, 5, 10, 20, 50, 100}. 
        #samples in each site = population size / n
        Range of n = [2, 100] (specified in the drafted paper. )
    '''

    pass


def simulate_n_samples_per_site():
    '''
        Interval: 5%
        5%  vs 95%
        10% vs 90%
        ...
        45% vs 55%
        50% vs 50%
    '''
    pass


def simulate_class_imbalance():
    '''
        Interval: 10%
        P 10% vs N 90%
        ...
        P 90% vs N 10%
    '''
    pass


def run_proto(n_estimators, test_size):
    iris = load_iris()
    X = iris.get("data")
    y = iris.get("target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=n_estimators,
                             learning_rate=1, random_state=0)

    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    # calculate and print model accuracy
    print("AdaBoost Classifier Model Accuracy:",
          accuracy_score(y_test, y_pred))
