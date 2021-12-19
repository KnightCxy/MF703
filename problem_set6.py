"""
Program:
Author: cai
Date: 2021-12-03
"""
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

def get_data(ticker):
    """
    :param ticker: ETF abbr
    :return: get ETF data from yahoo finance and calculate the daily returns of an ETF (from 1/1/2010 to 11/30/2021)
    """
    df = pdr.get_data_yahoo(ticker, '1/1/2010', '11/30/2021')
    df.fillna(method='ffill')
    # result = pd.DataFrame()
    # result[ticker] = df['Adj Close']
    return df



if __name__ == "__main__":

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(a):', '\n')
    ticker = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    XLB = get_data(ticker[-2])
    print(XLB)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(b):', '\n')
    daily_return = pd.DataFrame()
    for i in range(len(ticker)):
        df = get_data(ticker[i])
        daily_return[ticker[i]] = np.log(df['Adj Close'] / df['Adj Close'].shift())
    # daily_return = daily_return.transpose()
    # daily_return.drop(labels='2010-01-04', axis=1, inplace=True)
    # print(daily_return)
    daily_return_cov = daily_return.cov()
    print(daily_return_cov)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(c):', '\n')
    eigenvalue, vector = np.linalg.eig(daily_return_cov)
    eigenvalue = np.real(eigenvalue)
    eigenvalue = eigenvalue[::1]
    print('eigenvalues are as follow:')
    print(eigenvalue)
    print(f'positive: {len(eigenvalue[eigenvalue > 0])}')
    print(f'negative: {len(eigenvalue[eigenvalue < 0])}')
    print(f'zero: {len(eigenvalue[eigenvalue == 0])}')
    plt.plot(eigenvalue)
    plt.title("the eigenvalues")
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(d):', '\n')
    random_cov = np.random.standard_normal(size=(9, 9))
    print(random_cov)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 1(e):', '\n')
    eigenvalue2, vector2 = np.linalg.eig(random_cov)
    eigenvalue2 = np.real(eigenvalue2)
    eigenvalue2 = eigenvalue2[::1]
    print('eigenvalues are as follow:')
    print(eigenvalue2)
    print(f'positive: {len(eigenvalue2[eigenvalue2 > 0])}')
    print(f'negative: {len(eigenvalue2[eigenvalue2 < 0])}')
    print(f'zero: {len(eigenvalue2[eigenvalue2 == 0])}')
    plt.plot(eigenvalue2)
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(a):', '\n')
    daily_return.drop(labels='2010-01-04', axis=0, inplace=True)
    print(daily_return)
    annualized_return = daily_return.sum()
    annualized_return = (annualized_return / 2998) * 252
    print(annualized_return)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(b):', '\n')
    temp = np.array(daily_return_cov)
    temp2 = np.linalg.inv(temp)
    weight = 0.5 * np.dot(temp2, annualized_return)
    print(weight)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(c):', '\n')
    sigma = [0.005, 0.01, 0.05, 0.1]
    for i in sigma:
        annualized_return2 = annualized_return + i * np.random.normal()
        print(annualized_return2)
        weight2 = 0.5 * np.dot(temp2, annualized_return2)
        plt.plot(weight2)
    plt.plot(weight)
    plt.legend(['sigma=0.005', 'sigma=0.01', 'sigma=0.05', 'sigma=0.1', 'sigma=0'])
    plt.show()

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(d):', '\n')
    temp3 = []
    for i in range(9):
        temp3.append(temp[i][i])
    # print(temp3)
    diag_matrix = np.diag(temp3)
    # print(diag_matrix)
    theta = 1
    sum_matrix = (1 - theta) * temp + theta * diag_matrix
    print(sum_matrix)

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(e):', '\n')
    eigenvalue3, vector3 = np.linalg.eig(sum_matrix)
    eigenvalue3 = np.real(eigenvalue3)
    eigenvalue3 = eigenvalue3[::1]
    print('eigenvalues are as follow:')
    print(eigenvalue3)
    print(f'positive: {len(eigenvalue3[eigenvalue3 > 0])}')
    print(f'negative: {len(eigenvalue3[eigenvalue3 < 0])}')
    print(f'zero: {len(eigenvalue3[eigenvalue3 == 0])}')

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(f):', '\n')
    theta = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in theta:

        sum_matrix = (1 - i) * temp + i * diag_matrix
        eigenvalue4, vector4 = np.linalg.eig(sum_matrix)
        eigenvalue4 = np.real(eigenvalue4)
        print(f'delta:{i}')
        print('eigenvalues are as follow:')
        print(eigenvalue4)
        print(f'positive: {len(eigenvalue4[eigenvalue4 > 0])}')
        print(f'negative: {len(eigenvalue4[eigenvalue4 < 0])}')
        print(f'zero: {len(eigenvalue4[eigenvalue4 == 0])}')

    print('-----------------------------------------------------------------------', '\n')
    print('Answer 2(f):', '\n')

    sigma = [0.005, 0.01, 0.05, 0.1]
    for i in sigma:
        annualized_return = annualized_return + i * np.random.normal()
        for j in theta:
            sum_matrix = (1 - j) * temp + j * diag_matrix
            temp2 = np.linalg.inv(sum_matrix)
            weight2 = 0.5 * np.dot(temp2, annualized_return)
            plt.plot(weight2)


        plt.legend(['delta=0', 'delta=0.2', 'delta=0.4', 'delta=0.6', 'delta=0.8', 'delta=1'])
        plt.title(f'sigma={i}')
        plt.show()




