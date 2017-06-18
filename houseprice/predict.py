#coding='utf-8'

from pandas import DataFrame
import pandas as pd
from sklearn.externals import joblib

import feature
reload(feature)

if __name__ == '__main__':
    df_pred = pd.read_csv('data/test.csv', index_col='Id')
    df_pred = feature.clean_df(df_pred)
    df_pred = feature.feature_df(df_pred)
    print df_pred
    df_pred.describe()

    model = joblib.load('model.pkl')
    df_pred['SalePrice'] = model.predict(df_pred)

    df_pred.to_csv('result.csv', columns=['SalePrice'])
