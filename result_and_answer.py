"""
Program: result and answer
Author: cai
Date: 2021-09-14

Please only run the code within the question, if you dont want to run it, just commenting it.
The answer or comment of a question is below the code.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from problem_set2_1 import *
from problem_set2_2 import *
pd.set_option('display.width', 1000)


# 1(a) Download historical price data from January 1st 2010 for the following ETFs on yahoo finance.
# Clean/check the data for splits and other anomalies.

# directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'
# path = get_csv_path(directory)
# path = path[1:]
# result = pd.DataFrame()
# for i in range(len(path)):
#     data = clean_data(path[i])
#     result = pd.concat([result, data])
# print(result)  # print all data after cleaning


# 1(b) Calculate the annualized return and standard deviation of each ETF

# directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'  # the path of the directory containing csv files
# path = get_csv_path(directory)
# daily_return = pd.DataFrame()
# std = {}
# column = []
# path = path[1:]
# for i in path:
#     data = clean_data(i)
#     temp_daily_return = daily_returns(data)
#     temp_std = standard_deviation(temp_daily_return)
#     daily_return = pd.concat([daily_return, temp_daily_return], axis=1)
#     column.append(i[-7:-4])
#     std[i[-7:-4]] = temp_std
# daily_return.columns = column
# print(daily_return)  # print daily return of each ETF
# print(std)  # print annualized std of daily return


# 1(c) Calculate the covariance matrix of daily and monthly returns.
# Comment on the differences in correlations at different frequencies.

# directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'
# path = get_csv_path(directory)
# path = path[1:]
# daily_return_cov = covariance_matrix_of_daily_returns(path)
# monthly_return_cov = covariance_matrix_of_monthly_returns(path)
# print(daily_return_cov)  # print daily return covariance
# print(monthly_return_cov)  # print monthly return covariance

# Comment: covariances are all greater than 0; monthly return covariance is higher than daily return covariance, which
# means that different sectors will be more likely to move towards to the same direction in lower frequent time span.
# In other words, the correlation will be positively greater in lower frequent time span


# 1(d) Calculate a rolling 90-day correlation of each sector ETF with the S&P index.
# Do the correlations appear to be stable over time? What seems to cause them to vary the most?

# directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'
# path = get_csv_path(directory)
# path = path[1:]
# for i in path:
#     if 'SPY.csv' in i:
#         market = i
#         path.remove(i)
# martket_data = clean_data(market)
# market_return = daily_returns(martket_data)
# days = 90
# final_result = pd.DataFrame()
# for i in path:
#     sector_data = clean_data(i)
#     sector_return = daily_returns(sector_data)
#     rolling_corr = rolling_correlation(days, market_return, sector_return)
#     final_result[i[-7:-4]] = rolling_corr
# final_result.index = market_return.index[days:]
# print(final_result)  # print the rolling 90-day correlations of each sector ETF with the S&P index
# plt.plot(final_result)  # the figure of rolling 90-day correlations
# plt.legend(final_result.columns)
# plt.ylabel('rolling correlation with SPY')
# plt.show()

# Comment: correlations are not stable over time. some industries have countercyclicality hence they may behave low
# positive correlation or negative correlation with markets. When the market has a strong momentum to rise or fall, the
# correlation between market and all industries will increase.


# 1(e) Compute the β for the entire historical period and also compute rolling 90-day β’s.
# Are the β’s that you calculated consistent over the entire period? How do they compare to the rolling correlations?

final_result = pd.DataFrame()
directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'
path = get_csv_path(directory)
path = path[1:]
for i in path:
    if 'SPY.csv' in i:
        market = i
        path.remove(i)
martket_data = clean_data(market)
market_return = daily_returns(martket_data)
days = 90
for i in path:
    sector_data = clean_data(i)
    sector_return = daily_returns(sector_data)
    rolling_beta = rolling_CAPM(days, market_return, sector_return)
    final_result[i[-7:-4]] = rolling_beta
final_result.index = market_return.index[days:]
print(final_result)  # print rolling 90-day β
plt.plot(final_result)  # the figure
plt.legend(final_result.columns)
plt.ylabel('rolling 90-day beta (CAPM)')
plt.xlabel('time span')
plt.show()

# Comment: Betas are not stable over time. When beta is between 0-1, it means that this sector
# is less sensitive to the market during this period; when beta is greater than 1, it means this sector is
# more sensitive to market return. When beta is less than 0, it means it moves to opposite direction from market.
# Correlation reflects the degree to which linear relationship exists between market and sectors. Beta measures
# the volatility of sector return when market fluctuates.


# 1(f) Compute the auto-correlation of each ETF by regressing each ETFs current days return against
# its previous days return. Is there any evidence of auto-correlation in this ETF universe? Present
# the α’s for each ETF and comment on your results.

# autocorr = []
# column = []
# final_result = pd.DataFrame()
# directory = '/Users/cai/python_program/MF703_Prog_for_MathFin/assignment2/data'
# path = get_csv_path(directory)
# path = path[1:]
# for i in path:
#     data = clean_data(i)
#     daily_return = daily_returns(data)
#     alpha = auto_correlation(daily_return)
#     column.append(i[-7:-4])
#     autocorr.append(alpha)
# final_result.index = column
# final_result['autocorr'] = autocorr
# print(final_result)  # print autocorrelation of each ETF
# plt.rcdefaults()  # the figure
# fig, ax = plt.subplots()
# y_pos = np.arange(len(column))
# ax.barh(y_pos, autocorr)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(column)
# ax.invert_yaxis()
# ax.invert_xaxis()
# ax.set_xlabel('Auto correlation')
# plt.show()

# Comment: All ETFs have negative aotocorrelations, which means that the last daily return have a negative influence
# on next daily return


# 2(a) Generate a series of normally distributed random numbers and use these to generate
# simulated paths for the underlying asset. What is the mean and variance of the terminal
# value of these paths? Does it appear to be consistent with the underlying dynamics?

# s0 = 100
# r = 0
# sigma = 0.25
# mu = 0
# try_number = 3000
# option = Option(1, 100, 1 / 252)
# terminal_value = np.array([])
# for i in range(try_number):
#     path = option.generate_simulated_path(s0, r, sigma, mu)
#     terminal_value = np.append(terminal_value, path[-1])
#     plt.plot(path)
# mean = np.mean(terminal_value)
# std = np.std(terminal_value)
# print(terminal_value)  # print all terminal values via simulation
# print(f"mean: {mean:.8f}, " + f"std: {std:.8f}")  # print the mean and std of all terminal values
# plt.show()  # print the figure containing all paths
# plt.hist(terminal_value, bins=50)
# plt.xlabel('terminal value')
# plt.ylabel('frequency')
# plt.show()  # print the distribution of all terminal values

# Comment: it seems like it's consistent with the underlying dynamics


# 2(b) Calculate the payoff of a European put option with strike 100 along all simulated
# paths. Make a histogram of the payoffs for the European option. What is the mean
# and standard deviation of the payoffs?

# s0 = 100
# r = 0
# sigma = 0.25
# mu = 0
# try_numbers = 2000
# option1 = EuropeanPutOption(1, 100, 1 / 252)
# payoff = option1.find_payoff(s0, r, sigma, mu, try_numbers)
# mean = np.mean(payoff)
# std = np.std(payoff)
# print(payoff)  # print payoffs via simulation
# print(f"mean: {mean:.8f}, " + f"std: {std:.8f}")  # print the mean and std of payoffs
# plt.hist(payoff)  # the figure
# plt.xlabel('payoff')
# plt.ylabel('frequency')
# plt.show()


# 2(c) Calculate a simulation approximation to the price of a European put option by taking
# the average discounted payoff across all paths
# 2(d) Compare the price of the European put option obtained via simulation to the price
# you obtain using the Black-Scholes formula. Comment on any difference between the two prices.

# s0 = 100
# r = 0
# sigma = 0.25
# mu = 0
# rf = 0  # discount rate or risk-free rate
# try_numbers = 3000
# option1 = EuropeanPutOption(1, 100, 1 / 252)
# price1 = option1.find_price_approximation(s0, r, sigma, mu, rf, try_numbers)
# print(f"option price via simulation: {price1:.8f}")  # price via average discounted payoff across all paths
# price2 = option1.find_BSmodel_price(s0, sigma, rf)
# print(f"option price via BS model: {price2:.8f}")  # price via BS model
# print(f"the difference is: {abs(price1 - price2):.8f}")  # the difference

# Comment: There is a small difference between two prices


# 2(e) Calculate the payoff of a fixed strike lookback put option with stike 100 along all
# simulated path. (HINT: The option holder should exercise at the minimum price
# along each simulated path). Calculate the simulation price for the lookback option by
# averaging the discounted payoffs.

# s0 = 100
# r = 0
# rf = 0
# sigma = 0.25
# mu = 0
# try_numbers = 1000
# option2 = LookbackPutOption(1, 100, 1 / 252)
# payoff = option2.find_payoff(s0, r, sigma, mu, try_numbers)
# print(payoff)
# plt.hist(payoff)
# plt.xlabel('payoff')
# plt.ylabel('frequency')
# price = option2.find_price_approximation(s0, r, sigma, mu, rf, try_numbers)
# print(f"option price via simulation: {price:.8f}")
# plt.show()

# 2(f) Calculate the premium that the buyer is charged for the extra optionality embedded in
# the lookback. When would this premium be highest? Lowest? Can it ever be negative?
# 2(g) Try a few different values of σ and comment on what happens to the price of the
# European, the Lookback option and the spread/premium between the two.

# try different sigma: (it may take some time to run)

# s0 = 100
# r = 0
# rf = 0
# sigma = np.arange(0, 1, 0.05)
# mu = 0
# try_numbers = 2000
# option1 = EuropeanPutOption(1, 100, 1 / 252)
# option2 = LookbackPutOption(1, 100, 1 / 252)
# dif = []
# price1 = []
# price2 = []
# for i in sigma:
#     temp1 = option1.find_price_approximation(s0, r, i, mu, rf, try_numbers)
#     price1.append(temp1)
#     temp2 = option2.find_price_approximation(s0, r, i, mu, rf, try_numbers)
#     price2.append(temp2)
#     dif.append(temp2 - temp1)
# plt.plot(sigma, price1)  # print the Euro put price, Look back put price and spread
# plt.plot(sigma, price2)
# plt.plot(sigma, dif)
# plt.xlabel('different sigma')
# plt.legend(['Euro put price', 'Look back put price', 'spread / premium'])
# plt.show()

# Comment: the spread will never be negative because lookback put has more flexibility than Euro put.
# the higher sigma, the higher spread

# try different r (underlying asset return): (it may take some time to run...)

# s0 = 100
# r = np.arange(0, 1, 0.05)
# rf = 0
# sigma = 0.25
# mu = 0
# try_numbers = 2000
# option1 = EuropeanPutOption(1, 100, 1 / 252)
# option2 = LookbackPutOption(1, 100, 1 / 252)
# dif = []
# price1 = []
# price2 = []
# for i in r:
#     temp1 = option1.find_price_approximation(s0, i, sigma, mu, rf, try_numbers)
#     price1.append(temp1)
#     temp2 = option2.find_price_approximation(s0, i, sigma, mu, rf, try_numbers)
#     price2.append(temp2)
#     dif.append(temp2 - temp1)
# plt.plot(r, price1)
# plt.plot(r, price2)
# plt.plot(r, dif)
# plt.xlabel('different underlying asset return')
# plt.legend(['Euro put price', 'Look back put price', 'spread / premium'])
# plt.show()







