from utilities import utilities as util

import math as m
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers
from keras.constraints import maxnorm
from keras.layers.core import Lambda
from keras import backend as K

from sklearn.metrics import mean_squared_error

np.random.seed(7)

input_file = 'result.csv'
output_file = 'MLP_MCDropout_predicted.csv'

# Read csv file
data = pd.read_csv(input_file)

# Select factors
factors = ['cash_flow_yield', 'free_cash_flow_yield', 'cfroic', 'cash_flow_to_total_ssets',
           'sales_yield', 'sales_yield_fy1', 'sales_yield_fy2', 'sales_yield_mean',
           'sales_to_ev', 'asset_turnover', 'asset_turnover_12m_change', 'rec_1m_change',
           'rec_3m_change', 'earning_mom_1m_mean_fy1_fy2', 'rsi_9day', 'bbg_mom_12m_vol_ad',
           'market_cap']

data = util.FactorsSelection(data, factors)

# Generate target label
data = util.TargetLabelCaculation(data)

# Eliminate the stocks which market cap < 100M
data = util.MarketCapElimination(data, 100)

# Split dataset to train/test
df_train, df_test = util.TrainTestSpliting(data, 2010)

# Perform Min Max scaling on each stock
df_train, df_test = util.MinMaxNormalization(df_train, df_test)

# Drop the row which target is NA
df_train.dropna(subset=['target'], inplace=True)
df_test.dropna(subset=['target'], inplace=True)

# Prepare np array for train/test
train_X, train_y = df_train.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_train['target'].as_matrix()
test_X, test_y = df_test.drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1).as_matrix(), df_test['target'].as_matrix()

# Build the model
model = Sequential()
#model.add(Lambda(lambda x: K.dropout(x, level=0.2), input_shape=(train_X.shape[1],)))
model.add(Dense(300, input_shape=(train_X.shape[1],)))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
model.add(Dense(150))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = optimizers.adam()
model.compile(loss='binary_crossentropy', optimizer=adam)
model.fit(train_X, train_y, epochs=150, batch_size=len(train_X) / 100, verbose=1, shuffle=True)

# Sampling using the dropout NN
print("Sampling...")
train_result = model.predict(train_X)
for _ in tqdm(range(500), desc="Train Set"):
    train_result = np.append(train_result, model.predict(train_X), axis=1)

test_result = model.predict(test_X)
for _ in tqdm(range(500), desc="Test Set"):
    test_result = np.append(test_result, model.predict(test_X), axis=1)

# Calculate the mean and std of the samples
mean_train_result = np.mean(train_result, axis=1)
train_std = np.std(train_result, axis=1)

mean_test_result = np.mean(test_result, axis=1)
test_std = np.std(test_result, axis=1)

predicted_result = mean_test_result

# Cal AE, MSE
abs_error = 0

for i, j in zip(test_y, predicted_result):
    abs_error += np.absolute(i - j)

print "Absolute Error: ", abs_error
print "MSE :", mean_squared_error(test_y, predicted_result)

# Output the predicted value
util.WriteResultFile(df_test, predicted_result, output_file)
