# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 19:13:31 2018

@author: murrayzhang
"""


import numpy as np
from sklearn import datasets
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


digits_data = datasets.load_digits()

X = digits_data.data
Y = digits_data.target

#特征降纬
pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)
print(pca_X)

#plt.scatter(pca_X[:,0],pca_X[:,1],marker='o',c=Y)
#plt.show()

#ax=plt.subplot(111,projection='3d')
#ax.scatter(X[:,0],X[:,1],X[:,2],marker='o',c=Y)
#plt.show()

#使用降维后的数据进行分析
X_train, X_test, Y_train, Y_test = train_test_split(pca_X, Y,test_size=0.25)


print("----------- KNN with PCA: -----------")
#KNN results with PCA
nbrs = neighbors.KNeighborsClassifier(n_neighbors=4)
grid = GridSearchCV(estimator=nbrs,param_grid=dict(n_neighbors=np.array([1,2,3,4,5,6])))
grid.fit(pca_X,Y)
#print(grid.best_score_)
#print(grid.best_estimator_.n_neighbors)
nbrs2 = neighbors.KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors)
nbrs2.fit(X_train,Y_train)
predict_Y = nbrs2.predict(X_test)
print(metrics.classification_report(Y_test, predict_Y))



#使用降维前的原始数据进行分析
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)


print("----------- KNN without PCA: -----------")
#KNN results without PCA
grid.fit(X,Y)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
nbrs2 = neighbors.KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors)
nbrs2.fit(X_train,Y_train)
predict_Y = nbrs2.predict(X_test)
print(metrics.classification_report(Y_test, predict_Y))


print("------------Decision Tree: ----------")
#Decision Tree
from sklearn import datasets,tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
predict_Y = clf.predict(X_test)
print(metrics.classification_report(Y_test,predict_Y))


print("------------Random Forest: ----------")
#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=5)
clf = clf.fit(X_train, Y_train)
predict_Y = clf.predict(X_test)
print(metrics.classification_report(Y_test,predict_Y))


print("------------SVM: ----------")
#SVM
from sklearn import svm
clf = svm.LinearSVC()
clf = clf.fit(X_train, Y_train)
predict_Y = clf.predict(X_test)
print(metrics.classification_report(Y_test,predict_Y))


print("------------Nerual Network: ----------")
#Nerual Network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
clf = clf.fit(X_train, Y_train)
predict_Y = clf.predict(X_test)
print(metrics.classification_report(Y_test,predict_Y))



