# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:26:52 2020

@author: johan
"""

#from test_help_functions import test_one_federated_algorithm
#from test_help_functions import test_k_federated_algorithm
from test_help_functions import test_k_federated_algorithm_for_each_balance_step_iterative
from test_help_functions import test_k_federated_algorithm_for_each_balance_step_not_iterative
# from test.plot import make_plot
# import numpy as np

# test_one_federated_algorithm()

#res = test_k_federated_algorithm(k=30)
# print(res)
# make_plot(np.array(res[1]))

# test size: Prozentualer Anteil vom Gesamten Datensatz, der zum letzendlichen Test verwendet wird
# balance_step:
# k: Anzahl der Klassifierer pro balance step
# n_type: Typ von Auswahlkriterium von n, wobei n die Anzahl der Entscheidungsbäume, die in einem Iterartionsschritt hinzukommen
#       Varianten {"random", "batch", "single", "communist", "bear", "bull"}
#               random: wählt zufällig eine Zahl zwischen 1 und den noh freien Entscheidungsbäumen
#               batch: wählt immer n_batch_size
#               single: überlässt einer Datenbank alles
#               communist: Wie batch nur, dass n_batch_size = n_estimator/(number of databases)
# n_batch_size: Falls n_type = "batch", dann lässt der Algorithmus immer n_batch_size Entscheidungsbäume trainieren
# var_choosing_next_database: Auswahlkriterium, nach welcher die nächste Datenbank gewählt wird
#       Varianten: {"iterate", "random", "biggest", "smallest", "best", "worst"}
#               iterate: nimmt die übergebene Reihenfolge und iteriert durch
#               random: wählt jedes mal eine zufällige Databank
#               biggest: sortiert absteigend nach der Größe und iteriert durch
#               smallest: sortiert aufsteigend nach der Größe und iteriert durch
#               worst: Alle Datenbanken werden aufgefordert den Score des derzeitigen Klassifizier über ihre Daten anzugeben.
#                           Sie werden danach sortiert. Die Datenbank mit dem kleinsten Score wird für den Iterationsschritt gewählt.
#                           Diese Berechnung wird bei jedem Iterationsschritt vorgenommen.
#               best: Wie worst, nur dass der größte Score gewählt wird
# n_estimators: Wähle die Anzahl der Entscheidungsbäume, die trainiert werden sollen
# patients_batch_size: spielt hier gerade keine Rolle
res = test_k_federated_algorithm_for_each_balance_step_iterative(balance_step=0.1,
                                                                 test_size=0.2,
                                                                 k=2,
                                                                 n_type="batch",
                                                                 var_choosing_next_database="iterate",
                                                                 n_batch_size=10,
                                                                 n_estimators=50,
                                                                 patients_batch_size=10)
# print(res)

# balance_step, test_size, k, patients_batch_size wie oben
# weight_databases: Ob die Krankenhäuser zusätzlich nach Größe gewichtet werden.
res = test_k_federated_algorithm_for_each_balance_step_not_iterative(balance_step=0.1,
                                                                     test_size=0.2,
                                                                     k=2,
                                                                     patients_batch_size=10,
                                                                     weight_databases=False)
# print(res)
