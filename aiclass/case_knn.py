# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris_data = datasets.load_iris()

X = iris_data.data
y = iris_data.target


ax=plt.subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],marker='o',c=y)
plt.show()

pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

ax=plt.subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],marker='o',c=y)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = KNeighborsClassifier(n_neighbors=2)
grid = GridSearchCV(estimator=model, param_grid=dict(n_neighbors=np.array([1,2,3,4,5,6])))


grid.fit(X_train, y_train)
print grid.best_score_
print grid.best_estimator_.n_neighbors

model_best = KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors)
model.fit(X_train, y_train)

predict_Y = model.predict(X_test)

report = metrics.classification_report(y_test, predict_Y)
print report
