# coding='utf-8'

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import feature
reload(feature)


def loss_func(ground_truth, predictions):
    return np.sqrt(np.mean(np.square( np.log(ground_truth) - np.log(predictions) )))

def train_linear(X_train, X_test, y_train, y_test):
    reg = linear_model.LinearRegression(normalize=True)
    reg.fit(X_train, y_train)

    print reg
    print '##### r2 score #####'
    print reg.score(X_test, y_test)

    return reg



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

    #score = metrics.make_scorer(loss_func, greater_is_better=True)

    print '##### train #####'
    model = train_linear(X_train, X_test, y_train, y_test)

    print '##### evalue #####'
    y_pred = model.predict(X_test)
    score = loss_func(y_test, y_pred)
    print 'score:', score

    joblib.dump(model, 'model.pkl')

