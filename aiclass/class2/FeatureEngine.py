# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:38:47 2018

@author: murrayzhang
"""

from sklearn.datasets import load_iris
 
#导入IRIS数据集
iris = load_iris()
#特征矩阵
print(iris.data)
#目标向量
print(iris.target)

data = iris.data[:10]
print(data)
target = iris.target[:10]
print(target)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,Binarizer


print("----------标准差----------")
#标准化，返回值为标准化后的数据
print(StandardScaler().fit_transform(data))


print("----------区间缩放法----------")
#区间缩放，返回值为缩放到[0, 1]区间的数据
print(MinMaxScaler().fit_transform(data))


print("----------归一化-----------")
#归一化，返回值为归一化后的数据
print(Normalizer().fit_transform(data))


print("----------二值化-----------")
#二值化，阈值设置为3，返回值为二值化后的数据
print(Binarizer(threshold=3).fit_transform(data))


print("----------缺失值-----------")
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
print(vstack((array([nan, nan, nan, nan]), data)))
print(Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), data))))


print("接下来是特征选择")

print("----------方差选择法----------")
from sklearn.feature_selection import VarianceThreshold
#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
print(VarianceThreshold(threshold=0.01).fit_transform(data))


print("----------卡方检验选择法-----------")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#选择K个最好的特征，返回选择特征后的数据
print(SelectKBest(chi2, k=2).fit_transform(data, target))






