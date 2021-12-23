# -*- coding: utf-8 -*-
"""
lgb祖传参数
"""

#  回归预测得分，比赛：2018年甜橙杯 药物分子蛋白质活性预测
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.8,
    'num_leaves': 600,
    'colsample_bytree': 0.85,
    'feature_fraction': 0.83,
    'min_data': 70,
    'min_hessian': 1,
    'verbose': -1,
    'max_depth': 28,
}

# 多分类：根据用户行为预测用户的未来服务需求
num_class = 11  # 比如11分类
params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'sub_feature': 0.8,
    'num_leaves': 100,
    'colsample_bytree': 0.8,
    'verbose': -1,
    'num_class': 11,
}
