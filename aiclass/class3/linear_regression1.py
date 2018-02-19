# coding:utf-8
# author:vike
# time: 2018/01/20

import numpy as np
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linear_regression_demo(n = 30):

    #�����������һ��������y = k * x + b �����ݼ�
    k = random.random()
    b = random.random() * 1
    x = np.linspace(0,n,n)
    y = [ item * k +(random.random() - 0.5) * k * 5 + b for item in x]

    #����һԪ���Իع�
    model = LinearRegression()
    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
    yy = model.predict(np.reshape(x,[len(x),1]))
	
    #��ͼ
    plt.figure()
    kk = model.coef_[0][0] # ���Ԥ��ģ�͵Ĳ���
    bb = model.intercept_[0] #���Ԥ��ģ�͵Ľؾ�
    plt.title('Vike\'s Scikit-Learn Notes : Linear Regression Demo \n True: y='+str(k)[0:4]+'x +'+str(b)[0:4]+'  Predicted:y='+str(kk)[0:4]+'x +'+str(bb)[0:4] );
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False) # ��ʾ����
    plt.plot(x,y,'r.') # ��ͼ
    plt.plot(x,yy,'g-') # ��ͼ
    plt.show() # ��ʾͼ��

linear_regression_demo()