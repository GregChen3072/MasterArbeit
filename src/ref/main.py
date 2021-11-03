# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:21:06 2020

@author: johan
"""


from classifier import Classifier
# from sklearn.ensemble import AdaBoostClassifier
from next_n_size import NextN
from next_n_size import NextDataSets
from combiner import CombinedAdaBoostClassifier
from classifier import WarmStartAdaBoostClassifier

def make_iterative_classifier(databases: list,
                              classifier=WarmStartAdaBoostClassifier(),
                              n_estimators: int=None,
                              n_type: str="random",
                              n_batch_size: int=1,
                              var_choosing_next_database: str="worst",
                              patients_batch_size: int = 1):
    #print("n_type: "+str(n_type))
    #print("var_choosing_next_database: "+str(var_choosing_next_database))
    classifier = classifier.set_beginning()
    if n_estimators is None:
        classifier_iterator = Classifier(classifier)
    else:
        n_generator = NextN(n_size=n_estimators,
                            n_data_sets=len(databases),
                            n_type=n_type,
                            batch_size=n_batch_size)
        # print("Make: N Generator: "+str(n_generator))
        classifier_iterator = Classifier(classifier = classifier,
                                         n_generator = n_generator)
    database_chooser = NextDataSets(data_sets=databases, 
                                    next_type=var_choosing_next_database, 
                                    accuracy_batch_size = patients_batch_size)

    # i = 0

    while classifier_iterator.finished() is False:
        # print(i)
        # i = i+1
        # erhöht die Anzahl der Entscheidungsbäume
        classifier = classifier_iterator.get_prepared_classifier()  
        index = database_chooser.get_next_index(classifier)
        #print("Index: "+str(index))
        #print("Length Databases: "+str(len(databases)))
        current_database = databases[index]  # wählt die nächste Datenbank
        classifier = current_database.extend_classifier(
            classifier)  # erweitert auf der ausgewählten Datenbank den Klassifizieren
        classifier_iterator.update_classifier(classifier)  # updatet den Klassifizierer
        # print("Round finished.")
    # print("classifier finished.")
    return classifier

def make_not_iterative_classifier(databases: list,
                              base_estimator=None,
                              n_estimators=50,
                              learning_rate=1.,
                              algorithm='SAMME.R',
                              random_state=None,
                              patients_batch_size: int = 1,
                              weight_databases:bool = False):
    
    classifier = CombinedAdaBoostClassifier(base_estimator = base_estimator,
                              learning_rate = learning_rate,
                              n_estimators = n_estimators,
                              algorithm = algorithm,
                              random_state = random_state,
                              patients_batch_size=patients_batch_size,
                              weight_databases=weight_databases)
    
    return classifier.make_fit(databases)