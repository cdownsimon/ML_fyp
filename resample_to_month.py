#Read csv file
import pandas as pd

df = pd.read_csv('ML_factor_table_file.csv', parse_dates=['date'])

#Seperate unique tickers and dates
tickers = df.ticker.unique()
dates = df.date.unique()

#Seperate the dataset by ticker to a dictionary
tic = df.groupby(['ticker'])

data_by_ticker = {}

for t,d in tic:
    data_by_ticker[t] = d

#Resample the data to monthly and concat to one DataFrame
for t in data_by_ticker:
    data_by_ticker[t] = data_by_ticker[t].set_index(['date']).resample('M').ffill()

whole_df = pd.concat([data_by_ticker[t] for t in data_by_ticker])

#Save the result df to file
whole_df.to_csv("./ML_factor_table_file_monthly.csv")