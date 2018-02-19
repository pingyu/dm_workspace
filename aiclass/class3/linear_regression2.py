# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing

#���벨ʿ�ٷ������ݼ�
boston = load_boston()

#�鿴���ݼ��ṹ
print(boston.feature_names)
#print(boston.DESCR)
print(boston.data.shape)
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
print(bos.head())

###ѵ�����Իع�ģ��
lm = LinearRegression()
#X=boston.data
#min_max_scaler=preprocessing.MinMaxScaler()
#X=min_max_scaler.fit_transform(boston.data)


###������һ��
X=preprocessing.scale(boston.data)
Y=boston.target

#�з�ѵ�����Ͳ��Լ�
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.4)

#ѵ��ģ��
lm.fit(X_train, Y_train)

#��ӡ����Ȩ�ؼ��ؾ���
print('coefficients: ',  lm.coef_)
print('intercept: ' ,  lm.intercept_)

############################ѵ�����������#####################################
#ģ��Ԥ��
Y_predict_train=lm.predict(X_train)
print('predict price: ', Y_predict_train[0:5])
print('real price: ', Y_train[0:5])

#����ѵ�����������
mse_train = np.mean((Y_train - Y_predict_train) ** 2)
print(mse_train)

#���ӻ�Ԥ�����
fig, ax = plt.subplots()
ax.scatter(Y_train, Y_predict_train, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

############################���Լ��������#####################################
#ģ��Ԥ��
Y_predict_test=lm.predict(X_test)
print('predict price: ', Y_predict_test[0:5])
print('real price: ', Y_test[0:5])

#����ѵ�����������
mse_test = np.mean((Y_test - Y_predict_test) ** 2)
print(mse_test)

#���ӻ�Ԥ�����
fig, ax = plt.subplots()
ax.scatter(Y_train, Y_predict_train, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


