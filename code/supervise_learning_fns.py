#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:16:57 2017

@author: kayacik
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


#%%
# Run kfold CV experiments
# X: numpy array of shape (numrows, numcols) 
# y: numpy array of shape (numrows, ) NOT (numrows, 1)
#   reshape it y = np.reshape(y, (numrows, ) ) before passing to this function
# k: number of folds k>1        
def run_kfold_experiments(X, y, k):
    fold_results = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        rf_results = run_randomforest_classification(X_train, y_train, X_test, y_test)
        fold_results.append( rf_results )
    return fold_results

def run_randomforest_classification(X_train, y_train, X_test, y_test):
    results = [] # outputs go here
    n_estimators_values = [50, 100] # TODO Make these a function parameter
    max_features_values = [10, 100]
    min_samples_split_values = [2, 5]
    for prm_n_estimators in n_estimators_values:
        for prm_max_features in max_features_values:
            for prm_min_samples_split in min_samples_split_values:
                config = (prm_n_estimators, prm_max_features, prm_min_samples_split)
                print "[INFO] Training a random forrest with (n_estimators =", prm_n_estimators, \
                                                            ", max_features =", prm_max_features, \
                                                            ", min_samples_split =", prm_min_samples_split, ")"
                cf = RandomForestClassifier(n_estimators = prm_n_estimators, 
                                            max_features=prm_max_features, 
                                            max_depth=None, 
                                            min_samples_split=prm_min_samples_split)
                cf = cf.fit( X_train, y_train )
                tp, fp, tn, fn, precision, recall, f1score = 0
                # The below are classification related, disable for now
                #y_pred = cf.predict(X_test)
                #apply_threshold(y_pred)
                #(tp, fp, tn, fn) = evaluate_classifier(y_pred, y_test)
                #(precision, recall, f1score) = compute_prf1(tp, fp, tn, fn)
                results.append( [cf, config, tp, fp, tn, fn, precision, recall, f1score] )
    return results

def run_randomforest_regression(X_train, y_train, X_test, y_test):
    results = [] # outputs go here
    n_estimators_values = [50, 100] # TODO Make these a function parameter
    max_features_values = [2,5,10]
    min_samples_split_values = [2, 5]
    for prm_n_estimators in n_estimators_values:
        for prm_max_features in max_features_values:
            for prm_min_samples_split in min_samples_split_values:
                config = (prm_n_estimators, prm_max_features, prm_min_samples_split)
                print "[INFO] Training a random forrest with (n_estimators =", prm_n_estimators, \
                                                            ", max_features =", prm_max_features, \
                                                            ", min_samples_split =", prm_min_samples_split, ")"
                cf = RandomForestRegressor(n_estimators = prm_n_estimators, 
                                            max_features=prm_max_features, 
                                            max_depth=None, 
                                            min_samples_split=prm_min_samples_split)
                cf = cf.fit( X_train, y_train )
                results.append( [cf, config] )
    return results

def cost_function_regression(y_pred, y_test):
    (m, ) = y_pred.shape
    return 0.5 * 1/m * np.sum( (y_pred - y_test) * (y_pred - y_test) )

def mean_squared_error(y_pred, y_test):
    (m, ) = y_pred.shape
    return np.sum( (y_pred - y_test) * (y_pred - y_test) ) / m