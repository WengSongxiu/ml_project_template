# -*- coding:utf-8 -*-

from lightgbm.sklearn import LGBMClassifier
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor


def decision_tree_classifier():
    model = DecisionTreeClassifier(criterion="gini",
                                   splitter="best",
                                   max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.,
                                   max_features=None,
                                   random_state=None,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.,
                                   min_impurity_split=None,
                                   class_weight=None,)
    model = DecisionTreeClassifier()
    return model


def decision_tree_regressor():
    model = DecisionTreeRegressor(criterion="mse",
                                  splitter="best",
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.,
                                  max_features=None,
                                  random_state=None,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.,
                                  min_impurity_split=None,
                                  presort=False)
    return model


def random_forest_classifier():
    model = RandomForestClassifier()
    return model


def catboost_classifier():
    model = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               learning_rate=0.01,
                               iterations=100,
                               random_seed=42,
                               od_type="Iter",
                               depth=10,
                               early_stopping_rounds=500
                               )
    return model


def catboost_regressor():
    model = CatBoostRegressor(iterations=700,
                              learning_rate=0.02,
                              depth=12,
                              eval_metric='RMSE',
                              random_seed=23,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=75,
                              od_wait=100)
    return model


def lightgbm_classifier():

    model = LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='auc',
        num_leaves=15,
        learning_rate=0.03,
        max_bin=55,
        bagging_fraction=0.7,
        bagging_freq=5,
        bagging_seed=9,
        feature_fraction=0.9,
        feature_fraction_seed=9,
        min_data_in_leaf=6,
        min_sum_hessian_in_leaf=11,
        lambda_l1=1,
        lambda_l2=1,
        verbosity=0,
        num_threads=1,
        is_unbalance=True,
        num_iterations=400)

    return model


def xgboost_classifier():
    model = XGBClassifier(
        metric='huber',
        learning_rate=0.05,
        max_bin=55,
        bagging_fraction=0.8,
        bagging_freq=5,
        bagging_seed=9,
        feature_fraction=0.7,
        feature_fraction_seed=9,
        min_data_in_leaf=6,
        min_sum_hessian_in_leaf=11,)
    return model
