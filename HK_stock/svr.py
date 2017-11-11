from utilities import utilities as util
from sklearn import svm

import math as m
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

np.random.seed(7)

input_file = 'HK_stock.csv'
# output_file = 'svr_predicted.csv'
output_file = 'svr_predicted.csv'

# Read csv file
data = pd.read_csv(input_file)

# Select factors
factors = ['cash_flow_yield', 'free_cash_flow_yield', 'cfroic', 'cash_flow_to_total_ssets',
           'sales_yield', 'sales_yield_fy1', 'sales_yield_fy2', 'sales_yield_mean',
           'sales_to_ev', 'asset_turnover', 'asset_turnover_12m_change', 'rec_1m_change',
           'rec_3m_change', 'earning_mom_1m_mean_fy1_fy2', 'rsi_9day', 'bbg_mom_12m_vol_adj',
           'market_cap']

data = util.FactorsSelection(data, factors)

# Generate target label
data = util.TargetLabelCaculation(data)

# Eliminate the stocks which market cap < 800M
# data = util.MarketCapElimination(data, 800)
data = util.MarketCapElimination(data, 800, mode='by_ticker', date='2016-01-31')

# Split dataset to train/test
df_train, df_test = util.TrainTestSpliting(data, 2010)

# Perform Min Max scaling on each stock
# df_train, df_test = util.MinMaxNormalization(df_train, df_test)
df_train, df_test = util.MinMaxNormalization_p(df_train, df_test)

# Drop the row which target is NA
df_train.dropna(subset=['target'], inplace=True)
# df_test.dropna(subset=['target'], inplace=True)
# Fill the target of test set to 0.5 (for convenient)
df_test['target'].fillna(0.5, inplace=True)

# Prepare np array for train/test
train_X, train_y = df_train.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_train['target'].as_matrix()
test_X, test_y = df_test.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_test['target'].as_matrix()

# Build the model
SVMRegressor = svm.SVR(C=1, tol=0.00001, verbose=True)
SVMRegressor.fit(train_X, train_y)

predicted_result = SVMRegressor.predict(test_X)

# Cal AE, MSE
abs_error = 0

for i, j in zip(test_y, predicted_result):
    abs_error += np.absolute(i - j)

print "Absolute Error: ", abs_error
print "MSE :", mean_squared_error(test_y, predicted_result)

# Output the predicted value
util.WriteResultFile(df_test, predicted_result, output_file)
