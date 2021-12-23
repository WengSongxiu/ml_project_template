import catboost
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def catboost(x_train, y_train, n, param_grid):
    """
    catboost回归模型

    :param x_train: {array-like of shape (n_samples, n_features)} 输入训练数据集
    :param y_train: {array-like of shape (n_samples, n_output) \
                   or (n_samples,), default=None} 输入训练标签
    :param n: {float} 交叉验证集的比例
    :param param_grid: {dict} 参数字典
    :return: 模型
    """
    clf = catboost.CatBoostRegressor()
    model = GridSearchCV(clf, param_grid, cv=n)
    model.fit(x_train, y_train)
    return model


def decision_tree(x_train, y_train, n, param_grid):
    """
    决策树回归模型

    :param x_train: {array-like of shape (n_samples, n_features)} 输入训练数据集
    :param y_train: array-like of shape (n_samples, n_output) \
                   or (n_samples,), default=None 输入训练标签
    :param n:{float} 交叉验证集比例
    :param param_grid: {dict} 参数字典
    :return: 模型
    """
    clf = DecisionTreeRegressor()
    model = GridSearchCV(clf, param_grid, cv=n)
    model.fit(x_train, y_train)
    return model


def lightgbm(x_train, y_train, n, param_grid):
    """
    Lightbgm回归模型

    :param x_train: {array-like of shape (n_samples, n_features)} 输入训练数据集
    :param y_train: {array-like of shape (n_samples, n_output) \
                   or (n_samples,), default=None} 输入测试数据集
    :param n:{float} 交叉验证集比例
    :param param_grid: {dict} 参数字典
    :return: 模型
    """
    clf = lgb.LGBMRegressor()
    model = GridSearchCV(clf, param_grid, cv=n)
    model.fit(x_train, y_train)
    return model


def xgboost(x_train, y_train, n, param_grid):
    """
    xgboost回归模型

    :param x_train: {array-like of shape (n_samples, n_features)} 输入训练数据集
    :param y_train: {array-like of shape (n_samples, n_output) \
                   or (n_samples,), default=None} 输入测试数据集
    :param n:{float} 交叉验证集比例
    :param param_grid: {dict} 参数字典
    :return: 模型
    """
    clf = xgb.XGBRegressor()
    model = GridSearchCV(clf, param_grid, cv=n)
    model.fit(x_train, y_train)
    return model


def random_forest(x_train, y_train, n, param_grid):
    """
    随机森林回归模型

    :param x_train: {array-like of shape (n_samples, n_features)} 输入训练数据集
    :param y_train: {array-like of shape (n_samples, n_output) \
                   or (n_samples,), default=None} 输入测试数据集
    :param n:{float} 交叉验证集比例
    :param param_grid: {dict} 参数字典
    :return: 模型
    """
    clf = RandomForestRegressor()
    model = GridSearchCV(clf, param_grid, cv=n)
    model.fit(x_train, y_train)
    return model
