# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:22:14 2019

@author: Zach Nguyen
"""

# Define a function to create best model with grid search

import os, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Define a function to create best model with grid search

def create_classifier(train_set, train_label, model, parameter, cv, metric, model_path, preprocess = None):
    """
    Create the best grid searched model with given parameters.
    """
    if preprocess == None:
        steps = [('model', model)]
    else:
        steps = [('pre', preprocess),
                 ('model', model)]
    pipeline = Pipeline(steps)
    parameters = parameter

    clf = GridSearchCV(pipeline, 
                       parameters, 
                       cv = cv, 
                       n_jobs = -1, 
                       scoring = metric, 
                       verbose = 10)
    
    clf.fit(train_set, train_label)
    
    pickle.dump(clf, model_path)
    
    return clf

def evaluate_classifier(clf, test_set, test_label):
    """
    Evaluate model with f1macro_score
    """
    f1macro_score = f1_score(test_label, clf.predict(test_set), average = 'macro')
    cm = confusion_matrix(test_label, clf.predict(test_set))
    return f1macro_score, cm

def get_model_results(models, model_dir, test_set, test_label):
    """
    Get the result table from a list of model
    """
    f1 = []
    for model in models:
        clf = pickle.load(open(os.path.join(model_dir, model), 'rb'))
        f1.append(evaluate_classifier(clf, test_set, test_label)[0])
    
    results = pd.DataFrame({'model': models, 'f1_macro': f1})
    return results