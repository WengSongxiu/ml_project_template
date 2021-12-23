# -*- coding: utf-8 -*-
"""
xgb祖传参数
"""

#  线性回归预测得分，比赛：CCF 云移杯 景点评论得分预测（1-5）
linear_params = {
    'objective': 'reg:linear',
    'eta': 0.1,
    'max_depth': 9,
    'eval_metric': 'rmse',
    'seed': 0,
    'silent': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
}

#  逻辑斯蒂回归预测得分，预测0-1之间的值，天池工业质量大赛
logistic_params = {
    'objective': 'reg:logistic',
    'eta': 0.01,
    'seed': 27,
    'missing': -1,
    'silent': 0,
    'tree_method': 'exact',
}

# 2017 天池商铺定位（多分类问题）
num_class = 100  # 比如100分类
softmax_params = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': 9,
    'eval_metric': 'merror',
    'seed': 0,
    'missing': -999,
    'num_class': num_class,
    'silent': 1
}
