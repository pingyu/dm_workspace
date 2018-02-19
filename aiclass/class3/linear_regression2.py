# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing

#载入波士顿房价数据集
boston = load_boston()

#查看数据集结构
print(boston.feature_names)
#print(boston.DESCR)
print(boston.data.shape)
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print(bos.head())

###训练线性回归模型
lm = LinearRegression()
#X=boston.data
#min_max_scaler=preprocessing.MinMaxScaler()
#X=min_max_scaler.fit_transform(boston.data)


###特征归一化
X=preprocessing.scale(boston.data)
Y=boston.target

#切分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.4)

#训练模型
lm.fit(X_train, Y_train)

#打印特征权重及截距项
print('coefficients: ',  lm.coef_)
print('intercept: ' ,  lm.intercept_)

############################训练集误差评估#####################################
#模型预估
Y_predict_train=lm.predict(X_train)
print('predict price: ', Y_predict_train[0:5])
print('real price: ', Y_train[0:5])

#计算训练集均方误差
mse_train = np.mean((Y_train - Y_predict_train) ** 2)
print(mse_train)

#可视化预估误差
fig, ax = plt.subplots()
ax.scatter(Y_train, Y_predict_train, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

############################测试集误差评估#####################################
#模型预估
Y_predict_test=lm.predict(X_test)
print('predict price: ', Y_predict_test[0:5])
print('real price: ', Y_test[0:5])

#计算训练集均方误差
mse_test = np.mean((Y_test - Y_predict_test) ** 2)
print(mse_test)

#可视化预估误差
fig, ax = plt.subplots()
ax.scatter(Y_train, Y_predict_train, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


