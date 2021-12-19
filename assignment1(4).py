"""
Program: assignment1(4)
Author: xuyu cai
Date: 2021-09-03
"""

import pandas as pd
import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)


def read_data(path):
    """
    :param path: the path of your yahoo finance data(string)
    :return: a pd.dataframe
    """
    df = pd.read_csv(path, index_col='Date')
    return df


def annualized_returns(df):
    """
    :param df: a pd.dataframe containing the yahoo finance data
    :return: the annualized returns
    """
    df2 = df.copy()
    df2['daily return'] = df2['Adj Close'] / df2['Adj Close'].shift()
    df2.iloc[0, -1] = df2.iloc[0, -3] / df2.iloc[0, 0]
    df2['cum return'] = df2['daily return'].cumprod()
    result = df2['cum return'][-1] ** (252 / len(df2.index)) - 1
    return result


def annualized_volatility(df):
    """
    :param df: a pd.dataframe containing the yahoo finance data
    :return: annualized volatility
    """
    df2 = df.copy()
    df2['daily return'] = df2['Adj Close'] / df2['Adj Close'].shift() - 1
    df2.iloc[0, -1] = df2.iloc[0, -3] / df2.iloc[0, 0] - 1
    result = df2['daily return'].std() * (252 ** 0.5)
    return result


temp = ['/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/SPY.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/DBC.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/HYG.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/EEM.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/EFA.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/AGG.csv',
        '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment1/IAGG.csv']
final_result = pd.DataFrame(columns=['annualized_returns', 'annualized_volatility'], index=['SPY', 'DBC', 'HYG', 'EEM', 'EAFE', 'AGG', 'IAGG'])
daily_return_set = pd.DataFrame()
cum_return = pd.DataFrame()
for i in range(len(temp)):
    df = read_data(temp[i])
    final_result.iloc[i, 0] = annualized_returns(df)
    final_result.iloc[i, 1] = annualized_volatility(df)
    df['daily return'] = df['Adj Close'] / df['Adj Close'].shift() - 1
    df.iloc[0, -1] = df.iloc[0, -3] / df.iloc[0, 0] - 1
    daily_return_set[i] = df['daily return']
    cum_return[i] = (df['daily return'] + 1).cumprod() - 1
temp2 = cum_return.sum(axis=1) / 7
cum_return['portfolio'] = temp2
daily_return_set.columns = ['SPY', 'DBC', 'HYG', 'EEM', 'EAFE', 'AGG', 'IAGG']
cum_return.columns = ['SPY', 'DBC', 'HYG', 'EEM', 'EAFE', 'AGG', 'IAGG', 'portfolio']
print(final_result)  # answer4(b)

correlation_matrix = daily_return_set.corr()
print(correlation_matrix)  # answer4(c)

figure = cum_return.plot()
plt.show()
print(figure)  # answer4(d)






