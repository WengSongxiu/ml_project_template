# -*- coding:utf-8 -*-
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def parameter_optimization(model, x_train, y_train):
    parameters = get_parameters("lightgbm")
    estimate = GridSearchCV(
        model,
        parameters,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1)
    estimate = estimate.fit(x_train, y_train)
    print("Best score: %f using %s" % (estimate.best_score_, estimate.best_params_))


def get_parameters(model_type):
    parameter = {}
    if model_type == 'lightgbm':
        parameter = {'learning_rate': [0.1, 0.3, 0.6],
                     'feature_fraction': [0.5, 0.8, 1],
                     'num_leaves': [16, 32, 64],
                     'max_depth': [-1, 3, 5, 8]}
    return parameter


def feature_importance(model, type_list):
    for t in type_list:
        ax1 = plot_importance(model, importance_type=t)
        ax1.set_title(t)
        plt.show()


def model_evaluate(y_test, y_predict):
    print('The accuracy of the LGB Classifier is(test):',
          metrics.accuracy_score(y_test, y_predict))
