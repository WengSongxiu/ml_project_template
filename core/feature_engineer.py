from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


def feature_engineer(feature, target):
    category_feature = feature.select_dtypes(include=['object']).columns
    numerical_feature = feature.select_dtypes(exclude=['object']).columns
    drop_cols = [
        'gameId',
        'redFirstBlood',
        'redKills',
        'redDeaths',
        'redGoldDiff',
        'redExperienceDiff',
        'blueCSPerMin',
        'blueGoldPerMin',
        'redCSPerMin',
        'redGoldPerMin',
        'redAvgLevel',
        'blueAvgLevel']
    feature.drop(drop_cols, axis=1, inplace=True)
    return feature, target


def pcamodel(x, n):
    """
    使用decomposition库的PCA类选择特征
    主成分分析法，返回降维后的数据
    参数n_components为主成分数目

    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param n: {int}主成分数目
    :return: 降维后的值
    """
    pca = PCA(n_components=n)
    selector = pca.fit_transform(x)
    return selector


def ldamodel(x, y, n):
    """
    使用discriminant_analysis库的L类选择特征
    线性判别分析法，返回降维后的数据
    参数n_components为降维后的维数

    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param y: {array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None} 目标值（无监督转换）
    :param n: {int} 降维后的维数
    :return: 降维后的值
    """
    pca = LinearDiscriminantAnalysis(n_components=n)
    selector = pca.fit_transform(x, y)
    return selector


def variance_filter(x, n):
    """
    特征、标签之间热力图

    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param n: {int} 阈值设置，大于n的保留特征，否则删除
    :return: 过滤后的值
    """
    selector = VarianceThreshold(threshold=n)
    x_var0 = selector.fit_transform(x)
    return x_var0


def kbest_filter(x, y, n):
    """
    # 卡方过滤，专门针对离散型标签（分类问题）的相关性过滤
    # 卡方检验类feature_selection.chi2计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。再结合feature_selection.SelectKBest
    # 这个可以输入”评分标准“来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签，与我们分类目的无关的特征
    特征、标签之间热力图

    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param y: {str} 输入标签
    :param n: {int} 选取n个分数最高的特征的类
    :return: 过滤后的值
    """
    x_fschi = SelectKBest(chi2, k=n).fit_transform(x, y)
    return x_fschi


def wrapper_filter(x, y, n):
    """
    递归特征消除法，返回特征选择后的数据
    参数estimator为基模型
    参数n_features_to_select为选择的特征个数
    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param y: {str} 输入标签
    :param n: {int} 选择的特征个数
    :return: 过滤后的值
    """
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=n)
    wrapper = selector.fit_transform(x, y)
    return wrapper


def embedded_filter(x, y, n):
    """
    使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
    使用feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型
    :param x: {array-like of shape (n_samples, n_features)} 输入特征数据
    :param y: {str} 输入标签
    :param n: {float} 逻辑回归C值
    :return: 过滤后的值
    """
    selector = SelectFromModel(LogisticRegression(penalty="l2", C=n))
    embedded = selector.fit_transform(x, y)
    return embedded


def gbdt_filter(x, y):
    """
    树模型中GBDT可用来作为基模型进行特征选择
    使用feature_selection库的SelectFromModel类结合GBDT模型
    :param x: array-like of shape (n_samples, n_features) 输入特征数据
    :param y: {str} 输入标签
    :return: 过滤后的值
    """
    selector = SelectFromModel(GradientBoostingClassifier())
    gbdt = selector.fit_transform(x, y)
    return gbdt


def cutbins(x, bins, right, labels, retbins, precision, include_lowest):
    """
    本模块用于特征的分箱

    :param x: {array} 必须为一维，待切割的原形式
    :param bins: {int, sequence of scalars, or IntervalIndex}
    如果bins是一个整数，它定义了x宽度范围内的等宽面元数量，但是在这种情况下，x的范围在每个边上被延长1%，以保证包括x的最小值或最大值。
    如果bin是序列，它定义了允许非均匀in宽度的bin边缘。在这种情况下没有x的范围的扩展。
    :param right: {bool} 是否是左开右闭区间
    :param labels: 用作结果箱的标签。必须与结果箱相同长度。如果FALSE，只返回整数指标面元。
    :param retbins: {bool} 是否返回箱
    :param precision: {int} 返回面元的小数点几位
    :param include_lowest: {bool} 第一个区间的左端点是否包含
    :return: 返回切箱后的数据
    """
    cats = pd.cut(x, bins, right, labels, retbins, precision, include_lowest)
    return cats


def auto_cutbins(data, n, labels, retbins, precision, duplicates):
    """
    本模块用于自己划分组 -无监督 使用的是qcut方法

    :param data: {1d ndarray or Series} 输入数据
    :param n: {int or list-like of float} 按n分位数进行切割
    :param labels: {array or False, default None}
          用作结果箱的标签。 长度必须与结果箱一样。 如果为 False，则只返回整数指标箱。 如果为 True，则报错
    :param retbins: {bool, optional} 是否返回（箱，标签）。
    :param precision: {int, optional} 存储和显示 bin 标签的精度
    :param duplicates: {default 'raise', 'drop'}, 可选
            如果 bin 边缘不唯一，则引发 ValueError 或删除非唯一值
    :return: 返回切箱后的数据
    """
    s = pd.Series(data)
    a = pd.qcut(s, n, labels, retbins, precision, duplicates)
    cats = pd.value_counts(a)
    return cats
