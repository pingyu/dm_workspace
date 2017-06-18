#coding='utf-8'

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

YEAR_NOW = 2014

FEATURE_COLUMNS = [
  'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
  'BsmtQual_Ex', 'TotRmsAbvGrd', 'KitchenQual_Ex',

  'ExterQual_TA', 'DurationBuilt', 'KitchenQual_TA', 'DurationRemod',
]

def clean_df(df):
    # get dummies
    dummy_columns = [
      'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
      'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
      'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
      'ExterQual', 'ExterCond', 'Foundation', 
      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
      'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
      'KitchenQual', 'Functional', 'FireplaceQu',
      'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
      'PoolQC', 'Fence', 'MiscFeature',
      'SaleType', 'SaleCondition'
    ]
    df = pd.get_dummies(df, columns=dummy_columns)


    # YearBuilt
    df['DurationBuilt'] = np.log(YEAR_NOW - df['YearBuilt'])
    # YearRemodAdd
    df['DurationRemod'] = np.log(YEAR_NOW - df['YearRemodAdd'])

    # GarageCars
    df.loc[ df['GarageCars'].isnull(), 'GarageCars' ] = 0
    # TotalBsmtSF
    df.loc[ df['TotalBsmtSF'].isnull(), 'TotalBsmtSF' ] = 0

    # GarageYrBlt

    # MoSold
    # YrSold


    # SalePrice

    return df


def feature_df(df):
    return df[FEATURE_COLUMNS]

