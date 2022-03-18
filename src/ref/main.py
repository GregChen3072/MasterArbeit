# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:21:06 2020

@author: johan
"""


from ref.classifier import Classifier
# from sklearn.ensemble import AdaBoostClassifier
from ref.next_n_size import NextN
from ref.next_n_size import NextDataSets
from ref.combiner import CombinedAdaBoostClassifier
from ref.classifier import WarmStartAdaBoostClassifier
from ref.database import Database


def make_non_iterative_classifier(database: Database,
                                  base_estimator=None,
                                  n_estimators=50,
                                  learning_rate=1.,
                                  algorithm='SAMME.R',
                                  random_state=None,
                                  patients_batch_size: int = 1,
                                  weight_databases: bool = False):
    ''' For one-shot aggregation '''
    classifier = CombinedAdaBoostClassifier(base_estimator=base_estimator,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            algorithm=algorithm,
                                            random_state=random_state,
                                            patients_batch_size=patients_batch_size,
                                            weight_databases=weight_databases)
    # return classifier.make_fit(databases)
    return classifier.add_fit(database)


def make_iterative_classifier_for_n_sites(databases: list,
                                          classifier=WarmStartAdaBoostClassifier(),
                                          n_estimators: int = None,
                                          n_type: str = "batch",
                                          n_batch_size: int = 1,
                                          var_choosing_next_database: str = "iterate",
                                          patients_batch_size: int = 1
                                          ):
    ''' For sequential iteration, pipeline 1 3 '''
    classifier = classifier.set_beginning()
    if n_estimators is None:
        classifier_iterator = Classifier(classifier)
    else:
        n_generator = NextN(n_size=n_estimators,
                            n_data_sets=len(databases),
                            n_type=n_type,
                            batch_size=n_batch_size)
        # print("Make: N Generator: "+str(n_generator))
        classifier_iterator = Classifier(classifier=classifier,
                                         n_generator=n_generator)
    database_chooser = NextDataSets(data_sets=databases,
                                    next_type=var_choosing_next_database,
                                    accuracy_batch_size=patients_batch_size)

    while classifier_iterator.finished() is False:
        # This while loop: One iteration indicates one visit to the next DB in turn.

        # erhöht die Anzahl der Entscheidungsbäume
        classifier = classifier_iterator.get_prepared_classifier()
        index = database_chooser.get_next_index(classifier)
        #print("Index: "+str(index))
        #print("Length Databases: "+str(len(databases)))

        # wählt die nächste Datenbank
        current_database = databases[index]

        # erweitert auf der ausgewählten Datenbank den Klassifizieren
        classifier = current_database.extend_classifier(classifier)

        # updatet den Klassifizierer
        classifier_iterator.update_classifier(classifier)

        # print("Round finished.")
    # print("classifier finished.")
    return classifier


def make_unweighted_iterative_classifier(databases: list,
                                         classifier=WarmStartAdaBoostClassifier(),
                                         n_estimators: int = None,
                                         n_type: str = "batch",
                                         n_batch_size: int = 1,
                                         var_choosing_next_database: str = "iterate",
                                         patients_batch_size: int = 1
                                         ):
    ''' For unweighted sequential iteration. n_type="batch" determines the unweighted learning process '''
    classifier = classifier.set_beginning()

    if n_estimators is None:
        classifier_iterator = Classifier(classifier)

    else:
        n_generator = NextN(n_size=n_estimators,
                            n_data_sets=len(databases),
                            n_type=n_type,
                            batch_size=n_batch_size)

        classifier_iterator = Classifier(classifier=classifier,
                                         n_generator=n_generator)

    database_chooser = NextDataSets(data_sets=databases,
                                    next_type=var_choosing_next_database,
                                    accuracy_batch_size=patients_batch_size)

    while classifier_iterator.finished() is False:
        # This while loop: One while loop indicates one visit to the next DB in turn.

        # Initialize new weak learners
        classifier = classifier_iterator.get_prepared_classifier()
        index = database_chooser.get_next_index(classifier)

        # Visit the next site
        current_database = databases[index]

        # Extend the classifier on the site that it is visiting.
        classifier = current_database.extend_classifier(classifier)

        # Update the classifier
        classifier_iterator.update_classifier(classifier)

    return classifier


def make_weighted_iterative_classifier(databases: list,
                                       proportion: float = 0.5,
                                       n_iteration: int = 5,
                                       classifier=WarmStartAdaBoostClassifier(),
                                       n_estimators: int = None,
                                       n_type: str = "proportional",
                                       n_batch_size: int = 1,
                                       var_choosing_next_database: str = "iterate",
                                       patients_batch_size: int = 1
                                       ):
    ''' For weighted sequential iteration. n_type="proportional" determines the weighted learning process '''
    classifier = classifier.set_beginning()

    if n_estimators is None:
        classifier_iterator = Classifier(classifier)

    else:
        n_generator = NextN(n_size=n_estimators,
                            n_data_sets=len(databases),
                            n_type=n_type,
                            batch_size=n_batch_size,
                            proportion=proportion,
                            n_iteration=n_iteration)

        classifier_iterator = Classifier(classifier=classifier,
                                         n_generator=n_generator)

    database_chooser = NextDataSets(data_sets=databases,
                                    next_type=var_choosing_next_database,
                                    accuracy_batch_size=patients_batch_size)

    while classifier_iterator.finished() is False:
        # erhöht die Anzahl der Entscheidungsbäume
        classifier = classifier_iterator.get_prepared_classifier()

        index = database_chooser.get_next_index(classifier)

        current_database = databases[index]  # wählt die nächste Datenbank

        # Extend the classifier on the site that it is visiting
        classifier = current_database.extend_classifier(
            classifier)  # erweitert auf der ausgewählten Datenbank den Klassifizieren

        classifier_iterator.update_classifier(
            classifier)  # updatet den Klassifizierer

    return classifier
