# User defined functions
from db_simulator import simulate_n_databases_with_equal_sample_size

# Model Evaluation
from scoring import make_scores

# Reference
from ref.database import Database
from ref.next_n_size import NextN
from ref.next_n_size import NextDataSets
from ref.classifier import WarmStartAdaBoostClassifier
from ref.classifier import Classifier
# from sklearn.ensemble import AdaBoostClassifier

from ref.main import make_iterative_classifier


def pipeline_1_3(X_train, X_test, y_train, y_test, s, N, E, r=1):

    # Settings
    # E = E
    n_iteration = r  # Default: 1

    n_type = "batch"
    var_choosing_next_database = "iterate"
    patients_batch_size = 1  # Spielt keine Rolle

    # Simulate n DBs for n = [1, 2, 5, 10, 20, 50, 100]
    n_instances_of_dispersion = simulate_n_databases_with_equal_sample_size(
        X_train, y_train, list_of_n=N
    )

    # Initialize
    results = list()

    # Granularity of sites
    # degrees_of_data_dispersion = degrees_of_data_dispersion

    print("s\tr\tn\te\tF-1 Score\tMCC Score\tAUC Score\tACC Score")

    for i in range(0, len(n_instances_of_dispersion)):

        n = N[i]
        e = int(E / (n * n_iteration))
        ''' n_estimators = n_batch_size * n * n_iteration '''
        # 20 estimators / 5 dbs / 1 iterations = collect 4 estimators at each db for each round
        # 20 estimators / 5 dbs / 2 iterations = collect 2 estimators at each db for each round

        n_dbs = n_instances_of_dispersion[i]

        classifier_iterative = make_iterative_classifier(
            databases=n_dbs,  # list
            # Default: classifier=WarmStartAdaBoostClassifier()
            n_estimators=E,
            n_type=n_type,
            n_batch_size=e,
            var_choosing_next_database=var_choosing_next_database
            # Default: patients_batch_size: int = 1
        )

        f_1, mcc, auc, acc = make_scores(classifier_iterative, X_test, y_test)
        print(
            str(s) +
            "\t" +
            str(r) +
            "\t" +
            str(n) +
            "\t" +
            str(e) +
            "\t" +
            str(round(f_1, 3)) +
            "\t\t" +
            str(round(mcc, 3)) +
            "\t\t" +
            str(round(auc, 3)) +
            "\t\t" +
            str(round(acc, 3))
        )
        results.append([s, r, n, e, f_1, mcc, auc, acc])

    return results


def pipeline_1_3_comm_effi(X_train, X_test, y_train, y_test, e, r, s):

    # Settings
    # degrees_of_data_dispersion = [1, 2, 5, 10]
    degrees_of_data_dispersion = [1, 2, 5, 10]
    n_rounds = r  # Wertebereich [1:5]
    n_estimators_per_site_per_round = e  # e = {1, 2, 5, 10}
    # n_estimators E = e * n * r

    n_type = "batch"
    var_choosing_next_database = "iterate"
    patients_batch_size = 1  # Spielt keine Rolle

    # Simulate n DBs for n = [1, 2, 5, 10, 20]
    n_instances_of_dispersion = simulate_n_databases_with_equal_sample_size(
        X_train, y_train, list_of_n=degrees_of_data_dispersion
    )

    # Initialize
    results = list()

    for i in range(0, len(n_instances_of_dispersion)):

        n = degrees_of_data_dispersion[i]  # i. e. number of sites / dbs
        n_estimators = n_estimators_per_site_per_round * n * n_rounds
        # 20 estimators / 5 dbs / 1 iterations = collect 4 estimators at each db for each round
        # 20 estimators / 5 dbs / 2 iterations = collect 2 estimators at each db for each round

        n_dbs = n_instances_of_dispersion[i]

        classifier = WarmStartAdaBoostClassifier()
        classifier = classifier.set_beginning()

        if n_estimators is None:
            classifier_iterator = Classifier(classifier)
        else:
            n_generator = NextN(n_size=n_estimators,
                                n_data_sets=len(n_dbs),
                                n_type=n_type,
                                batch_size=n_estimators_per_site_per_round)

            classifier_iterator = Classifier(classifier=classifier,
                                             n_generator=n_generator)

        database_chooser = NextDataSets(data_sets=n_dbs,
                                        next_type=var_choosing_next_database,
                                        accuracy_batch_size=patients_batch_size)

        # v stands for number of visits (different sites)
        v = 0
        while classifier_iterator.finished() is False:
            # This while loop: One iteration indicates one visit to the next DB in turn.
            # print(i)
            v = v + 1
            # erhöht die Anzahl der Entscheidungsbäume
            classifier = classifier_iterator.get_prepared_classifier()
            index = database_chooser.get_next_index(classifier)
            #print("Index: "+str(index))
            #print("Length Databases: "+str(len(databases)))
            current_database = n_dbs[index]  # wählt die nächste Datenbank

            # erweitert auf der ausgewählten Datenbank den Klassifizieren
            # classifier = current_database.extend_classifier(classifier)
            classifier = current_database.extend_bootstrap_fit(classifier)

            classifier_iterator.update_classifier(
                classifier)  # updatet den Klassifizierer
            f_1, mcc, auc, acc = make_scores(
                classifier_iterator.classifier, X_test, y_test)
            print(
                str(s) + "\t" +
                str(r) + "\t" +
                str(v) + "\t" +
                str(n) + "\t" +
                str(e) + "\t" +
                str(round(f_1, 3)) +
                "\t\t" +
                str(round(mcc, 3)) +
                "\t\t" +
                str(round(auc, 3)) +
                "\t\t" +
                str(round(acc, 3))
            )

            results.append([s, r, v, n, e,
                            f_1, mcc, auc, acc])

    return results
