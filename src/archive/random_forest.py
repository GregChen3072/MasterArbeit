from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from numpy import arange

rfc = RandomForestClassifier()
abc = AdaBoostClassifier()

balance_step = 0.05

for i in arange(balance_step, (0.5 + balance_step), balance_step).tolist():
    print(round(i, 2))
