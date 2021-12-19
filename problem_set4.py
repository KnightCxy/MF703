"""
Program:
Author: cai
Date: 2021-09-30
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn import datasets, linear_model
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import math
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)


def get_data(ticker):
    """
    :param ticker: ETF abbr
    :return: get ETF data from yahoo finance and calculate the daily returns of an ETF (from 1/1/2010 to 1/1/2021)
    """
    df = pdr.get_data_yahoo(ticker, '1/1/2010', '1/1/2021')
    df.fillna(method='ffill')
    # result = pd.DataFrame()
    # result[ticker] = df['Adj Close']
    return df


def daily_returns(df):
    """
    :param df: a pd.dataframe containing the yahoo finance data after cleaning
    :return: a pd.dataframe only containing daily returns
    """
    df2 = df.copy()
    df2['daily return'] = np.log(df2['Adj Close'] / df2['Adj Close'].shift())
    df2.iloc[0, -1] = np.log(df2.iloc[0, -4] / df2.iloc[0, 0])
    result = pd.DataFrame()
    result['daily return'] = df2['daily return']
    return result


def auto_correlation(df):
    """
    :param df:
    :return: the alpha of auto-correlation AR(1)
    """
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(df)
    if pvalue <= 0.1:
        arma = ARIMA(df.to_numpy(), order=(1, 0, 0)).fit()
        result = arma.params
    return result[1]
    return 'non-stationary'


def rolling_correlation(days, df):
    """
    :param days: rolling number
    :param df: data containing SPY and VIX
    :return: rolling correaltions (an array)
    """
    rolling_result = np.array([])
    for i in range(len(df.index) - days):
        temp= df[i: i + days]
        result = temp.corr()
        rolling_result = np.append(rolling_result, [result.iloc[0, 1]])
    final_result = pd.DataFrame()
    final_result['rolling corr'] = rolling_result
    final_result.index = df.index[days:]
    return final_result

def calc_Euro_put(s0, exercise_price, maturity, sigma, rf):
    """
    :param s0: initial underlying price
    :param exercise_price: exercise price
    :param maturity: options maturity
    :param sigma: volatility
    :param rf: risk free rate
    :return: the price of an Euro put in BSM model
    """
    d1 = (math.log(s0 / exercise_price) + maturity * (rf + sigma ** 2 / 2)) \
         * (1 / (sigma * math.sqrt(maturity)))
    d2 = d1 - sigma * (math.sqrt(maturity))
    price = exercise_price * math.exp(- rf * maturity) * stats.norm.cdf(- d2, loc=0, scale=1) \
            - s0 * stats.norm.cdf(- d1, loc=0, scale=1)
    return price


def calc_Euro_call(s0, exercise_price, maturity, sigma, rf):
    """
    :param s0: initial underlying price
    :param exercise_price: exercise price
    :param maturity: options maturity
    :param sigma: volatility
    :param rf: risk free rate
    :return: the price of an Euro call in BSM model
    """
    d1 = (math.log(s0 / exercise_price) + maturity * (rf + sigma ** 2 / 2)) \
         * (1 / (sigma * math.sqrt(maturity)))
    d2 = d1 - sigma * (math.sqrt(maturity))
    price = s0 * stats.norm.cdf(d1, loc=0, scale=1) - exercise_price * math.exp(- rf * maturity) \
            * stats.norm.cdf(d2, loc=0, scale=1)
    return price



if __name__ == "__main__":

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(a):', '\n')
    tickers = ['SPY', '^VIX']
    SPY = get_data('SPY')
    print(SPY)
    VIX = get_data('^VIX')
    print(VIX)
    SPY2 = daily_returns(SPY)
    VIX2 = daily_returns(VIX)
    df = pd.concat([SPY2, VIX2], axis=1)
    df.columns = ['SPY', 'VIX']
    print(df)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(b):', '\n')
    print('autocorrelation for SPY:')
    print(auto_correlation(df['SPY']))
    print('autocorrelation for VIX:')
    print(auto_correlation(df['VIX']))

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(c):', '\n')
    print('the correlation of the S&P and its implied volatility on a daily basis:')
    print(stats.pearsonr(df['SPY'], df['VIX']))
    print('the correlation is significant.')
    df2 = df.resample('M').mean()
    month_time = df2.index.strftime('%Y-%m')
    df2.index = month_time
    print('\n', 'the correlation of the S&P and its implied volatility on a monthly basis:')
    print(stats.pearsonr(df2['SPY'], df2['VIX']))
    print('the correlation is significant.')

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(d):', '\n')
    rollingcorr = rolling_correlation(90, df)
    print(rollingcorr)
    print('long run average is:')
    print(np.mean(rollingcorr['rolling corr']))
    print('the date when the correlation deviate the most from its long-run average is:')
    print(rollingcorr[rollingcorr['rolling corr'] == np.max(rollingcorr['rolling corr'])].index)
    plt.plot(rollingcorr)
    plt.xlabel('time span')
    plt.ylabel('correlations')
    plt.title('90days rolling correlations of SPY and VIX')
    plt.axhline(y=np.mean(rollingcorr['rolling corr']), color='r', label='long-run average')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(e):', '\n')
    realized_vol = df['SPY'].rolling(90).std()
    realized_vol = realized_vol[90:] * math.sqrt(252) * 100
    # print(realized_vol)
    implied_vol = VIX['Adj Close']
    implied_vol = implied_vol[90:]
    # print(implied_vol)
    vol_premium = pd.DataFrame()
    vol_premium['premium'] = implied_vol - realized_vol
    print(vol_premium)
    print('the date when the premium is highest:')
    print(vol_premium[vol_premium['premium'] == np.max(vol_premium['premium'])].index)
    print('the date when the premium is lowest:')
    print(vol_premium[vol_premium['premium'] == np.min(vol_premium['premium'])].index)
    plt.plot(vol_premium)
    plt.axhline(y=0, color='r')
    plt.grid(linestyle='--')
    plt.xlabel('time span')
    plt.ylabel('implied vol - realized vol')
    plt.title('The preminum between implied vol and realized vol')
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(f):', '\n')
    option_price = pd.DataFrame(columns=['Euro call', 'Euro put'], index=df.index)
    for i in range(len(df.index)):
        s0 = SPY.iloc[i, - 1]
        sigma = VIX.iloc[i, - 1] / 100
        temp = calc_Euro_put(s0, s0, 1 / 12, sigma, 0)
        temp2 = calc_Euro_call(s0, s0, 1 / 12, sigma, 0)
        option_price.iloc[i, 0] = temp2
        option_price.iloc[i, 1] = temp
    option_price['long straddle'] = option_price['Euro call'] + option_price['Euro put']
    print(option_price)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(g):', '\n')
    payoff = []
    for i in range(2769 - 21):
        temp = max(SPY.iloc[i + 21, 0] - SPY.iloc[i, 0], 0)
        temp2 = max(SPY.iloc[i, 0] - SPY.iloc[i + 21, 0], 0)
        payoff.append(temp + temp2)
    payoff_and_PL = pd.DataFrame()
    payoff_and_PL.index = option_price.index[21:]
    option_price = option_price[0: 2769 - 21]
    payoff_and_PL['payoff'] = payoff
    payoff_and_PL['P&L'] = payoff_and_PL['payoff'] - np.array(option_price['long straddle'])
    print(payoff_and_PL)
    plt.plot(payoff_and_PL, linewidth=0.5)
    plt.axhline(y=np.mean(payoff_and_PL['P&L']), label='P&L average', color='r', linewidth=1)
    plt.legend(['payoff', 'P&L', 'P&L average'], loc='upper left')
    plt.grid(linestyle='--')
    plt.title('payoff and P&L of long straddle')
    plt.show()
    print(np.mean(payoff_and_PL['P&L']))

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(h):', '\n')
    payoff_and_PL['vol premium'] = implied_vol - realized_vol
    payoff_and_PL = payoff_and_PL['2010-05-13':]
    print(payoff_and_PL)
    plt.scatter(payoff_and_PL['vol premium'],payoff_and_PL['P&L'], s=1)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    plt.grid(linestyle='--')
    plt.xlabel('implied vol - realized vol')
    plt.ylabel('profit or loss of long straddle')
    plt.title('relationship between P&L and vol premium')
    plt.show()








