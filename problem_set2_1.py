"""
Program: problem set 2
Author: cai
Date: 2021-09-10
"""
import math
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_csv_path(directory):
    """
    :param directory: the path of directory which contains all csv files
    :return: a list of strings that contains all csv file paths
    """
    for root, dirs, files in os.walk(directory):
        file_name = files
        result = []
        for i in file_name:
            result.append(directory + '/' + i)
    return result


def clean_data(path):
    """
    :param path: the path (a string) of a historical price data (csv file) from
                January 1st 2010 for the following ETFs on yahoo finance
    :return: a dataframe which is checked and cleaned (no splits and anomalies)
    """
    data = pd.read_csv(path, na_values='null')
    data.Date = pd.to_datetime(data.Date)
    data = data.set_index('Date')
    data.fillna(method='ffill')
    return data


def daily_returns(df):
    """
    :param df: a pd.dataframe containing the yahoo finance data after cleaning
    :return: a pd.dataframe only containing daily returns
    """
    df2 = df.copy()
    df2['daily return'] = df2['Adj Close'] / df2['Adj Close'].shift() - 1
    df2.iloc[0, -1] = df2.iloc[0, -4] / df2.iloc[0, 0] - 1
    result = pd.DataFrame()
    result['daily return'] = df2['daily return']
    return result


def monthly_returns(df):
    """
    :param df: the dataframe only containing daily returns
    :return: the dataframe only containing monthly returns
    """
    df2 = df.copy()
    df2['daily return'] += 1
    result = df2.resample('M').prod()
    month_time = result.index.strftime('%Y-%m')
    result.index = month_time
    result.columns = ['monthly return']
    result['monthly return'] -= 1
    return result



def annualized_return(df):
    """
    :param df: a pd.dataframe only containing daily return
    :return: annualized return
    """
    df['daily return'] += 1
    df['cum return'] = df['daily return'].cumprod()
    result = df['cum return'][-1] ** (252 / len(df.index)) - 1
    return result


def standard_deviation(df):
    """
    :param df: a pd.dataframe only containing daily return
    :return: annualized standard deviation of daily return
    """
    result = df['daily return'].std() * (252 ** 0.5)
    return result


def covariance_matrix_of_daily_returns(path):
    """
    :param path: a list of strings that contains all csv file paths
    :return: the covariance matrix of daily returns
    """
    temp = pd.DataFrame()
    temp2 = []
    for i in path:
        df = clean_data(i)
        df2 = daily_returns(df)
        temp = pd.concat([df2, temp], axis=1)
        temp2.append(i[-7:-4])
    temp.columns = temp2
    result = temp.corr()
    return result


def covariance_matrix_of_monthly_returns(path):
    """
    :param path: a list of strings that contains all csv file paths
    :return: the covariance matrix of monthly returns
    """
    temp = pd.DataFrame()
    temp2 = []
    for i in path:
        df = clean_data(i)
        df2 = daily_returns(df)
        df3 = monthly_returns(df2)
        temp = pd.concat([df3, temp], axis=1)
        temp2.append(i[-7:-4])
    temp.columns = temp2
    result = temp.cov()
    return result


def rolling_correlation(days, market_return, sector_return):
    """
    :param days: rolling number
    :param market: market daily return(SPY)
    :param sector: sector daily return
    :return: the rolling correlation
    """
    rolling_result = np.array([])
    for i in range(len(market_return.index) - days):
        temp = pd.DataFrame()
        temp['market_return'] = market_return[i: i + days]
        temp['sector_return'] = sector_return[i: i + days]
        result = temp.corr()
        rolling_result = np.append(rolling_result, [result.iloc[0, 1]])
    return rolling_result



def entire_period_CAPM(market_return, sector_return):
    """
    :param market: market daily return
    :param sector: scetor daily return
    :return: linear regression coef
    """
    x = np.array(market_return['daily return'])
    x = np.reshape(x, newshape=(len(market_return['daily return']), 1))
    y = np.array(sector_return['daily return'])
    y = np.reshape(y, newshape=(len(sector_return['daily return']), 1))
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return regr.coef_


def rolling_CAPM(days, market_return, sector_return):
    """
    :param days: rolling number
    :param market: market daily return
    :param sector: scetor daily return
    :return: rolling linear regression coef
    """
    rolling_result = np.array([])
    for i in range(len(market_return.index) - days):
        temp1 = pd.DataFrame()
        temp2 = pd.DataFrame()
        temp1 = market_return[i: i + days]
        temp2 = sector_return[i: i + days]
        result = entire_period_CAPM(temp1, temp2)
        rolling_result = np.append(rolling_result, [result])
    return rolling_result


def auto_correlation(daily_return):
    """
    :param daily_return: a dataframe only containing daily return of an ETF
    :return: the alpha of auto-correlation AR(1)
    """
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(daily_return)
    if pvalue <= 0.1:
        arma = ARIMA(daily_return.to_numpy(), order=(1, 0, 0)).fit()
        result = arma.params
        return result[1]
    return 'non-stationary'




