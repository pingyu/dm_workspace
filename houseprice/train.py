# coding='utf-8'

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib

import matplotlib.pyplot as plt

import feature
reload(feature)


def loss_func(ground_truth, predictions):
    return np.sqrt(np.mean(np.square( np.log(ground_truth) - np.log(predictions) )))

scorer = metrics.make_scorer(loss_func, greater_is_better=False)

def train_linear(X_train, X_test, y_train, y_test):
    reg = linear_model.LinearRegression(normalize=True)
    reg.fit(X_train, y_train)

    print reg
    print 'LinearRegression, r2 score:', reg.score(X_test, y_test)

    return reg

def select_ridge(X, y):
    def rmse_cv(model):
      rmse = -cross_val_score(model, X, y, scoring=scorer, cv=5)
      return rmse
    
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100, 300]
    cv_ridge = [rmse_cv(linear_model.Ridge(alpha=alpha)).mean() for alpha in alphas]

    cv_df = Series(cv_ridge, index=alphas)
    print cv_df
    cv_df.plot()
    plt.xlabel('alpha')
    plt.ylabel('rmse')
    plt.show()

    BEST_ALPHA = 50
    return BEST_ALPHA


def train_ridge(X, y, alpha):
    model = linear_model.Ridge(alpha=alpha)
    model.fit(X, y)

    print model
    print 'Ridge, score:', model.score(X, y)
    
    return model

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv', index_col='Id')

    print '##### df cleaned #####'
    df = feature.clean_df(df)
    print df
    df.describe()

    y_all = df['SalePrice']

    print '##### correlation #####'
    corr = df.corrwith(y_all)
    corr_ordered = corr.sort_values(ascending=False)
    print corr_ordered[:20]
    print corr_ordered[-20:]

    print '##### feature #####'
    X_all = feature.feature_df(df)
    print X_all
    X_all.describe()

    print '##### sample ready #####'
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=23)

    ###########################################
    print '##### train linear #####'
    model = train_linear(X_train, X_test, y_train, y_test)

    print '##### evalue #####'
    y_pred = model.predict(X_test)
    score = loss_func(y_test, y_pred)
    print 'score:', score

    ############################################
    print '##### train ridge #####'
    select_ridge(X_all, y_all)

    #joblib.dump(model, 'model.pkl')

