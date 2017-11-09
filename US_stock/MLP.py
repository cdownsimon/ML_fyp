from utilities import utilities as util

import math as m
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers
from keras.constraints import maxnorm

from sklearn.metrics import mean_squared_error

np.random.seed(7)

input_file = 'US_stock.csv'
output_file = 'MLP_predicted.csv'

# Read csv file
data = pd.read_csv(input_file)

# Select factors
factors = ['cash_flow_yield', 'free_cash_flow_yield', 'cfroic', 'cash_flow_to_total_ssets',
           'sales_yield', 'sales_yield_fy1', 'sales_yield_fy2', 'sales_yield_mean',
           'sales_to_ev', 'asset_turnover', 'asset_turnover_12m_change', 'rec_1m_change',
           'rec_3m_change', 'earning_mom_1m_mean_fy1_fy2', 'rsi_9day', 'bbg_mom_12m_vol_adj',
           'market_cap']

data = util.FactorsSelection(data, factors)

# print('Original: {}', format(data.ticker.unique().shape[0]))
# tic = data[(data['date'] == '2016-12-31') & (data['market_cap'] > 100)]['ticker']
# data = data[data['ticker'].isin(tic)]
# print('After: {}', format(data.ticker.unique().shape[0]))

# Generate target label
data = util.TargetLabelCaculation(data)

# Eliminate the stocks which market cap < 100M
# data = util.MarketCapElimination(data, 100)

# Split dataset to train/test
df_train, df_test = util.TrainTestSpliting(data, 2010)

# Perform Min Max scaling on each stock
df_train, df_test = util.MinMaxNormalization(df_train, df_test)

# Drop the row which target is NA
df_train.dropna(subset=['target'], inplace=True)
# df_test.dropna(subset=['target'], inplace=True)
# Fill the target of test set to 0.5 (for convenient)
df_test['target'].fillna(0.5, inplace=True)

# Prepare np array for train/test
train_X, train_y = df_train.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_train['target'].as_matrix()
test_X, test_y = df_test.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_test['target'].as_matrix()

# Build the model
# batch_size = 200
batch_size = len(train_X) / 1000
model = Sequential()
model.add(Dense(300, input_shape=(train_X.shape[1],)))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(150))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
optimizer = optimizers.Adam()
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.fit(train_X, train_y, epochs=500, batch_size=batch_size, verbose=1, shuffle=True)

# Predict the target of test set
predicted_result = model.predict(test_X)
predicted_result = predicted_result[:, 0]

# Cal AE, MSE
abs_error = 0

for i, j in zip(test_y, predicted_result):
    abs_error += np.absolute(i - j)

print "Absolute Error: ", abs_error
print "MSE :", mean_squared_error(test_y, predicted_result)

# Output the predicted value
util.WriteResultFile(df_test, predicted_result, output_file)
