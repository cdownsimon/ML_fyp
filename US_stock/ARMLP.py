from utilities import utilities as util

import math as m
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.models import Sequential
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers
from keras.constraints import maxnorm
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras import backend as K

from sklearn.metrics import mean_squared_error

np.random.seed(7)

input_file = 'result.csv'
output_file = 'ARMLP_predicted.csv'

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
input1 = Input(shape=(train_X.shape[1],))
# Non-linear Feature Extraction
hidden1 = Dense(300)(input1)
h_a1 = Activation('tanh')(hidden1)
dropout1 = Lambda(lambda x: K.dropout(x, level=0.5))(h_a1)
hidden2 = Dense(150)(dropout1)
h_a2 = Activation('tanh')(hidden2)
# linear Feature Extraction
hidden3 = Dense(300)(input1)
h_a3 = Activation('linear')(hidden3)
dropout2 = Lambda(lambda x: K.dropout(x, level=0.5))(h_a3)
hidden4 = Dense(150)(dropout2)
h_a4 = Activation('linear')(hidden4)
# Merge
merge = concatenate([h_a2, h_a4])
dropout3 = Lambda(lambda x: K.dropout(x, level=0.5))(merge)
# Ouput
output = Dense(1, activation='sigmoid')(dropout3)
model = Model(input1, output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='ARMLP.png')
# Training
adam = optimizers.adam()
model.compile(loss='binary_crossentropy', optimizer=adam)
model.fit(train_X, train_y, epochs=500, batch_size=len(train_X) / 100, verbose=1, shuffle=True)


# Predict the target of test set
#predicted_result = model.predict(test_X)
#predicted_result = predicted_result[:, 0]

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
