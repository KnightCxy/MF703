"""
Program: assignment3.1
Author: cai
Date: 2021-09-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn import datasets, linear_model
from scipy import stats
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)



def read_and_clean_data():
    """
    :return: the F-F factors daily data after cleaning (dataframe)
    """
    path = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment3/F-F_Research_Data_Factors_daily.CSV'
    df = pd.read_csv(path, skiprows=4, skipfooter=2, index_col=0, engine='python', dtype=float)
    if np.any(df.isnull()):
        df.fillna(method='ffill')
    return df


def covariance_matrix_of_factos_returns(df):
    """
    :param df: the F-F factors daily data after cleaning (dataframe)
    :return: the covariance matrix of three factors returns in entire period
    """
    df2 = df.copy()
    df2 = df2.drop(labels='RF',axis=1)
    result = df2.cov()
    return result


def correlation_matrix_of_factos_returns(df):
    """
    :param df: the F-F factors daily data after cleaning (dataframe)
    :return: the correlation matrix of three factors returns in entire period
    """
    df2 = df.copy()
    df2 = df2.drop(labels='RF',axis=1)
    result = df2.corr()
    return result

def rolling_factors_correlation(days, df):
    """
    :param days: rolling number
    :param df: the F-F factors daily data after cleaning (dataframe)
    :return: rolling correlations of F-F factors returns
    """
    df2 = df.copy()
    df2 = df2.drop(labels='RF', axis=1)
    rolling_result = pd.DataFrame()
    MS = np.array([])
    MH = np.array([])
    SH = np.array([])
    for i in range(len(df2.index) - days):
        temp = pd.DataFrame()
        temp = df2[i: i + days]
        result = temp.corr()
        MS = np.append(MS, [result.iloc[0, 1]])
        MH = np.append(MH, [result.iloc[0, 2]])
        SH = np.append(SH, [result.iloc[1, 2]])
    rolling_result['Mkt-RF,SMB'] = MS
    rolling_result['Mkt-RF,HML'] = MH
    rolling_result['SMB,HML'] = SH
    rolling_result.index = df2.index[days:]
    return rolling_result


def testing_normality(df):
    """
    :param df: a dataframe
    :return: the normality test results for each column (kstest)
    """
    df2 = df.copy()
    for i in df2.columns:
        print('normaltest result for', i, ':')
        temp = stats.normaltest(df2[i])
        print(temp)
        if temp.pvalue <= 0.05:
            print(f'{i} is not a normal distribution.')
        else:
            print(f'{i} is a normal distribution.')
    return ''


def get_ETF_daily_return(ticker):
    """
    :param ticker: ETF abbr
    :return: get ETF data from yahoo finance and calculate the daily returns of an ETF (from 1/1/2010 to 1/1/2021)
    """
    df = pdr.get_data_yahoo(ticker, '1/1/2010', '1/1/2021')
    df.fillna(method='ffill')
    df['daily return'] = df['Adj Close'] / df['Adj Close'].shift() - 1
    df.iloc[0, -1] = df.iloc[0, 3] / df.iloc[0, 2] - 1
    result = pd.DataFrame()
    result['daily return'] = df['daily return'] * 100
    return result


def entire_period_three_factors_model(ETF, factors):
    """
    :param ETF: a sector ETF's daily return
    :param factors: the F-F three factors daily data after cleaning (dataframe)
    :return: Fama French three-factor model in the entire period (2010/01/01-2021/01/01)
    """
    factors2 = pd.DataFrame()
    factors2['Mkt-RF'] = factors['Mkt-RF']
    factors2['SMB'] = factors['SMB']
    factors2['HML'] = factors['HML']
    x = factors2.values
    x = sm.add_constant(x)
    y = ETF.values
    regr = sm.OLS(y, x)
    results = regr.fit()
    return results.params, results.resid


def rolling_three_factors_model(ETF, factors, days):
    """
    :param ETF: a sector ETF's daily return
    :param factors: the F-F three factors daily data after cleaning (dataframe)
    :param days: rolling number
    :return: the rolling Fama French three-factor model in the entire period (2010/01/01-2021/01/01)
    """
    rolling_result = pd.DataFrame(columns=['β1(Mkt)', 'β2(SMB)', 'β3(HML)'])
    for i in range(len(factors.index) - days):
        temp1 = factors[i: i + days]
        temp2 = ETF[i: i + days]
        temp3 = entire_period_three_factors_model(temp2, temp1)
        temp4 = temp3[0][1:]
        rolling_result.loc[i] = temp4
    rolling_result.index = ETF.index[days:]
    return rolling_result


if __name__ == "__main__":
    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(a):', '\n')
    factors = read_and_clean_data()
    print(factors)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(b):', '\n')
    factors_returns_cov = covariance_matrix_of_factos_returns(factors)
    print(factors_returns_cov)
    facctors_returns_corr = correlation_matrix_of_factos_returns(factors)
    print(facctors_returns_corr)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(c):', '\n')
    factors_sub = factors.loc['20100101': '20210101']
    factors_sub.index = pd.to_datetime(factors_sub.index, format='%Y%m%d')
    rolling_corr = rolling_factors_correlation(90, factors_sub)
    print(rolling_corr)
    plt.plot(rolling_corr)
    plt.title('rolling correlations between factors')
    plt.xlabel('time span')
    plt.ylabel('correlations')
    plt.legend(rolling_corr.columns)
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(d):', '\n')
    print(testing_normality(factors))

    print('-------------------------------------------------------------------------', '\n')
    print('Answer 1(e):', '\n')
    sector_ETF = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    print('the entire period β for each ETF:')
    entire_period_beta = pd.DataFrame()
    entire_period_resid = pd.DataFrame()
    for i in sector_ETF:
        ETF_daily_return = get_ETF_daily_return(i)
        temp = entire_period_three_factors_model(ETF_daily_return, factors_sub)
        entire_period_beta[i] = temp[0][1:]
        entire_period_resid[i] = temp[1]
    entire_period_beta.index = ['β1(Mkt)', 'β2(SMB)', 'β3(HML)']
    entire_period_resid.index = factors_sub.index
    print(entire_period_beta)
    print('\n', 'the rolling 90-day β for each ETF:')
    for i in sector_ETF:
        ETF_daily_return = get_ETF_daily_return(i)
        rolling_result = rolling_three_factors_model(ETF_daily_return, factors_sub, 90)
        plt.plot(rolling_result)
        plt.title(f'{i} - F-F three-factor model')
        plt.xlabel('time span')
        plt.ylabel('beta')
        plt.legend(rolling_result.columns)
        plt.show()

    print('-------------------------------------------------------------------------', '\n')
    print('Answer 1(f):', '\n')
    mean_and_var_of_resid = pd.DataFrame()
    for i in entire_period_resid.columns:
        temp1 = entire_period_resid[i].mean()
        temp2 = entire_period_resid[i].var()
        temp3 = [temp1, temp2]
        mean_and_var_of_resid[i] = temp3
        plt.hist(entire_period_resid[i], bins=40)
        plt.xlabel('residuals')
        plt.ylabel('frequency')
        plt.title(f'{i}: daily residuals')
        plt.show()
    mean_and_var_of_resid.index = ['mean', 'variance']
    print('All ETFs daily residuals:')
    print(entire_period_resid)
    print('\n', 'The mean and std of each ETFs residuals:')
    print(mean_and_var_of_resid)
    print('\n', 'Test the normality:')
    print(testing_normality(entire_period_resid))








