# -*- coding:utf-8 -*-

# 导入依赖
from sklearn.model_selection import train_test_split
from core.feature_engineer import *
from core.classification import *
from core.tool import *

# 读取数据
data = pd.read_csv('../data/input/high_diamond_ranked_10min.csv')
target_name = 'blueWins'
target = data[target_name]
feature = data.drop([target_name], axis=1)

# 特征工程
feature, target = feature_engineer(feature, target)

# 训练预测
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.2, random_state=2020)

model = decision_tree_classifier()

model = model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# 模型评估
print(y_predict)
model_evaluate(y_test, y_predict)

# 特征重要性评分
# feature_importance(model, ['gain', 'split'])

# 参数调优
# parameter_optimization(model, x_train, y_train)