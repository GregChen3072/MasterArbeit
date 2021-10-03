import inspect
from sklearn.utils import all_estimators

'''
    Ok, we have a systematic method to find out all the base learners supported by AdaBoostClassifier. 
    Compatible base learner's fit method needs to support sample_weight, 
    which can be obtained by running following code:
'''

for name, clf in all_estimators(type_filter='classifier'):
    if 'sample_weight' in inspect.getfullargspec(clf().fit)[0]:
        print(name)

'''
    This results in following output: 
        AdaBoostClassifier, 
        BernoulliNB, 
        DecisionTreeClassifier, 
        ExtraTreeClassifier, 
        ExtraTreesClassifier, 
        MultinomialNB, 
        NuSVC, 
        Perceptron, 
        RandomForestClassifier, 
        RidgeClassifierCV, 
        SGDClassifier, 
        SVC. 

    If the classifier doesn't implement predict_proba, you will have to set AdaBoostClassifier parameter algorithm = 'SAMME'.
'''
