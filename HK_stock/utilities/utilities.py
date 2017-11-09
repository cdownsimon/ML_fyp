from datetime import datetime
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm


def MarketCapElimination(df, market_cap=None):
    if not market_cap:
        raise Exception("MarketCapElimination: market_cap must not be None.")
    if not type(market_cap) == int and not type(market_cap) == float:
        raise Exception("MarketCapElimination: market_cap must be a number.")

    return df[df.market_cap >= market_cap]


def FactorsSelection(df, factors_list=None):
    if not factors_list:
        print("FactorsSelection: factors_list is None, include all the factors by default.")

    if not factors_list or factors_list == 'all':
        return df

    factors_list = ['date', 'ticker', 'last_price'] + factors_list

    try:
        df = df[factors_list]
    except KeyError as e:
        raise e

    return df


def TrainTestSpliting(df, split_year=None):
    if not split_year:
        raise Exception("TrainTestSpliting: split_year must be specified.")
    if not type(split_year) == int:
        raise Exception("TrainTestSpliting: split_year must be a integer.")

    split_year = datetime(year=split_year, month=1, day=1)
    if split_year < datetime.strptime(df.date.min(), '%Y-%m-%d') or split_year >= datetime.strptime(df.date.max(), '%Y-%m-%d'):
        raise Exception("TrainTestSpliting: split_year must be greater than {} and less than {}.".format(df.date.min().year, df.date.max().year))

    df_train, df_test = df[df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') < split_year)], df[df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d') >= split_year)]

    return (df_train, df_test)


def TargetLabelCaculation(df, p=None, period=None):
    if not p:
        print("TargetLabelCaculation: p is None, set p = 10% by default.")
        p = 10
    if not period:
        print("TargetLabelCaculation: period is None, set period = 1 month by default.")
        period = 1

    df_by_ticker = {}
    for t, d in df.groupby(['ticker']):
        df_by_ticker[t] = d.sort_values(['date'])
        df_by_ticker[t]['next_return'] = (df_by_ticker[t]['last_price'].shift(-period) - df_by_ticker[t]['last_price']) / df_by_ticker[t]['last_price'] * 100

    df = pd.concat([df_by_ticker[t] for t in df_by_ticker])
    df.dropna(axis=0, how='any', inplace=True)

    df_by_date = {}
    for date, d in df.groupby(['date']):
        size = d.shape[0]
        sort_next_return = d.sort_values(['next_return'], ascending=False)
        sort_next_return.reset_index(drop=True, inplace=True)
        """
        top = sort_next_return.head(size*p//100)
        top.loc[:,'target'] = 1
        bottom = sort_next_return.tail(size*p//100)
        bottom.loc[:,'target'] = 0
        df_by_date[date] = pd.concat([top,bottom])
        """
        top = int(size * p / 100.)
        tail = size - int(size * p / 100.)

        sort_next_return.loc[:, 'target'] = float('nan')
        sort_next_return.loc[:top, 'target'] = 1
        sort_next_return.loc[tail:, 'target'] = 0

        df_by_date[date] = sort_next_return

    df = pd.concat([df_by_date[d] for d in df_by_date])

    return df


def MinMaxNormalization(df_train, df_test, min_max=None):
    if not min_max:
        print("MinMaxNormalization: min_max is None, set min = 0 and max = 1 by default.")
        min_max = (0, 1)

    df_train_key, df_test_key = df_train['ticker'].unique(), df_test['ticker'].unique()
    df_train_by_ticker, df_test_by_ticker = {}, {}
    for t, d in df_train.groupby(['ticker']):
        df_train_by_ticker[t] = d
    for t, d in df_test.groupby(['ticker']):
        df_test_by_ticker[t] = d

    cnt = 0
    for t in df_test_key:
        if t not in df_train_key:
            print('MinMaxNormalization: ticker {} only appear in test set,eliminated.'.format(t))
            df_test_by_ticker.pop(t, None)
            cnt += 1
    print('MinMaxNormalization: {} tickers are elimnated.'.format(cnt))

    print('MinMaxNormalization: Processing...')
    df_train_normalized_list, df_test_normalized_list = [], []
    for t in tqdm(df_train_by_ticker):
        try:
            df_train_tmp_dict, df_test_tmp_dict = {}, {}
            df_train_tmp_dict['date'] = df_train_by_ticker[t]['date'].tolist(),
            df_train_tmp_dict['ticker'] = df_train_by_ticker[t]['ticker'].tolist()
            df_train_tmp_dict['last_price'] = df_train_by_ticker[t]['last_price'].tolist()
            df_train_tmp_dict['next_return'] = df_train_by_ticker[t]['next_return'].tolist()
            df_train_tmp_dict['target'] = df_train_by_ticker[t]['target'].tolist()
        except KeyError as KE:
            raise KE

        df_train_by_ticker[t].drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1, inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler()
        np_train_scaled = min_max_scaler.fit_transform(df_train_by_ticker[t])
        df_train_normalized = pd.DataFrame(np_train_scaled, columns=df_train_by_ticker[t].columns)

        for i in df_train_tmp_dict:
            # To prevent TypeError
            if i == 'date':
                if len(df_train_tmp_dict[i]) == 1:
                    df_train_normalized.loc[:, i] = df_train_tmp_dict[i][0]
                else:
                    df_train_normalized.loc[:, i] = df_train_tmp_dict[i]
            else:
                df_train_normalized.loc[:, i] = df_train_tmp_dict[i]

        df_train_normalized_list.append(df_train_normalized)

        try:
            df_test_tmp_dict['date'] = df_test_by_ticker[t]['date'].tolist()
            df_test_tmp_dict['ticker'] = df_test_by_ticker[t]['ticker'].tolist()
            df_test_tmp_dict['last_price'] = df_test_by_ticker[t]['last_price'].tolist()
            df_test_tmp_dict['next_return'] = df_test_by_ticker[t]['next_return'].tolist()
            df_test_tmp_dict['target'] = df_test_by_ticker[t]['target'].tolist()
        except KeyError as KE:
            if t in df_test_by_ticker:
                raise KE
        if t not in df_test_by_ticker:
            continue

        df_test_by_ticker[t].drop(['date', 'ticker', 'last_price', 'next_return', 'target'], axis=1, inplace=True)

        np_test_scaled = min_max_scaler.transform(df_test_by_ticker[t])
        df_test_normalized = pd.DataFrame(np_test_scaled, columns=df_test_by_ticker[t].columns)

        for i in df_test_tmp_dict:
            # To prevent TypeError
            if i == 'date':
                if len(df_test_tmp_dict[i]) == 1:
                    df_test_normalized.loc[:, i] = df_test_tmp_dict[i][0]
                else:
                    df_test_normalized.loc[:, i] = df_test_tmp_dict[i]
            else:
                df_test_normalized.loc[:, i] = df_test_tmp_dict[i]

        df_test_normalized_list.append(df_test_normalized)

    df_train_min_max_normalized, df_test_min_max_normalized = pd.concat(df_train_normalized_list), pd.concat(df_test_normalized_list)

    df_train_min_max_normalized.sort_values(['date'], inplace=True)
    df_test_min_max_normalized.sort_values(['date'], inplace=True)

    return (df_train_min_max_normalized, df_test_min_max_normalized)


def EliminateToSeasonal(df, months=None):
    if not months:
        print('EliminateToSeasonal: months is None, choose 1,4,7,10 by default.')
        months = [1, 4, 7, 10]

    return df[df.date.apply(lambda x:x.month in months)]


def WriteResultFile(df, predicted_value, f):
    result = pd.DataFrame(df[['date', 'ticker', 'next_return']].as_matrix(), columns=['Date', 'Ticker', 'Next Return'], copy=True)
    result.loc[:, 'Predicted'] = predicted_value
    print('Writing Result to file {}...'.format(f))
    result.to_csv(f, index=False)
    print('Done.')
