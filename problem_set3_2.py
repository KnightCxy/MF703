"""
Program:
Author: cai
Date: 2021-09-25
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import norm
from scipy import stats


class Option:
    def __init__(self, maturity, exercise_price, s0, sigma, dt=1 / 252, r=0,  rf=0):
        """
        :param maturity: time to maturity (years)
        :param exercise_price: exercise price of the option
        :param dt: single time period (1 / trade days per year or a relatively small number, because
        Brownian motion increment dWt is normally distributed with variance dt. The time interval
        in Brownian motion simulation is very small, even continuous)
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param rf: risk-free rate (discount rate)
        """
        self.maturity = maturity
        self.exercise_price = exercise_price
        self.dt = dt
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.rf = rf

    def __repr__(self):
        pass

    def generate_path_BM(self):
        """
        :return: the path of underlying asset price in Bachelier model (dSt = r dt + σ dWt).
        """
        temp = int(self.maturity / self.dt)
        dWt = np.random.normal(loc=0, scale=math.sqrt(self.dt), size=temp)
        path = np.array([self.s0])
        for i in range(temp):
            dSt = self.r * self.dt + self.sigma * dWt[i]
            St = path[i] + dSt
            path = np.append(path, St)
        return path

    def generate_path_BSM(self):
        """
        :return: the path of underlying asset price in Black-Scholes model (dSt = rSt dt + σSt dWt).
        """
        temp = int(self.maturity / self.dt)
        dWt = np.random.normal(loc=0, scale=math.sqrt(self.dt), size=temp)
        path = np.array([self.s0])
        for i in range(temp):
            dSt = self.r * path[i] * self.dt + self.sigma * path[i] * dWt[i]
            St = path[i] + dSt
            path = np.append(path, St)
        return path

    def find_terminal_value_distribution_BM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: terminal_value_distribution in Bachelier model
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_path_BM(self)
            temp2 = path[-1]
            temp.append(temp2)
        terminal_price = np.array(temp)
        plt.hist(terminal_price, bins=40)
        plt.xlabel('terminal value')
        plt.ylabel('frequency')
        plt.title('terminal value distribution')
        plt.show()
        return terminal_price

class EuropeanCallOption(Option):
    def find_payoff_BM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: an array of the payoffs for an European put option
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_path_BM(self)
            terminal_value = path[-1]
            if terminal_value > self.exercise_price:
                temp.append(terminal_value - self.exercise_price)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff


    def find_price_approximation_BM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: the price estimation of an European option via simulation
        """
        payoff = EuropeanPutOption.find_payoff_BM(self, try_numbers)
        price = np.mean(payoff) * math.exp(- self.rf * self.maturity)
        return price

    def find_payoff_BSM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: an array of the payoffs for an European put option
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_path_BSM(self)
            terminal_value = path[-1]
            if terminal_value > self.exercise_price:
                temp.append(terminal_value - self.exercise_price)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff

    def find_price_approximation_BSM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: the price estimation of an European option via simulation
        """
        payoff = EuropeanPutOption.find_payoff_BSM(self, try_numbers)
        price = np.mean(payoff) * math.exp(- self.rf * self.maturity)
        return price

    def find_BSM_price(self):
        """
        :return: the calculation of BSmodel_price of an European option
        """
        d1 = (math.log(self.s0 / self.exercise_price) + self.maturity * (self.rf + self.sigma ** 2 / 2)) \
             * (1 / (self.sigma * math.sqrt(self.maturity)))
        d2 = d1 - self.sigma * (math.sqrt(self.maturity))
        price = self.exercise_price * math.exp(- self.rf * self.maturity) * norm.cdf(- d2, loc=0, scale=1) \
                - self.s0 * norm.cdf(- d1)
        return price



    # def find_Bacheliermodel_price(self):
    #     """
    #     :return: the calculation of BSmodel_price of an European option
    #     """


class LookbackPutOption(Option):
    def find_payoff_BM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: an array of payoffs of a lookback put option via simulation according to Bachelier model path
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_path_BM(self)
            terminal_value = np.min(path)
            if terminal_value < self.exercise_price:
                temp.append(self.exercise_price - terminal_value)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff


    def find_price_approximation_BM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: the price estimation of a lookback put option via simulation
        """
        payoff = LookbackPutOption.find_payoff_BM(self, try_numbers)
        price = np.mean(payoff) * math.exp(- self.rf * self.maturity)
        return price


    def find_payoff_BSM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: an array of payoffs of a lookback put option via simulation
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_path_BSM(self)
            terminal_value = np.min(path)
            if terminal_value < self.exercise_price:
                temp.append(self.exercise_price - terminal_value)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff


    def find_price_approximation_BSM(self, try_numbers):
        """
        :param try_numbers: simulation times
        :return: the price estimation of a lookback put option via simulation
        """
        payoff = LookbackPutOption.find_payoff_BSM(self, try_numbers)
        price = np.mean(payoff) * math.exp(- self.rf * self.maturity)
        return price

    def find_delta(self, epc, try_numbers):
        """
        :param epc: a number
        :return: the delta
        """
        self.s0 = self.s0 + epc
        temp = LookbackPutOption.find_price_approximation_BM(self, try_numbers)
        self.s0 = self.s0 - 2 * epc
        temp2 = LookbackPutOption.find_price_approximation_BM(self, try_numbers)
        delta = (temp - temp2) / (2 * epc)
        return delta


if __name__ == "__main__":
    # print('-----------------------------------------------------------------------', '\n')
    # print('Answer 2(a):', '\n')
    option1 = LookbackPutOption(0.5, 120, 100, 25)
    option2 = LookbackPutOption(0.5, 120, 100, 0.25)
    try_numbers = 1500
    list1 = []
    list2 = []
    for i in range(try_numbers):
        # path = option2.generate_path_BSM()
        # plt.plot(path)
        path2 = option1.generate_path_BM()
        list1.append(path2[-1])
    # plt.hist(list1, bins=50)
    # print('The answer is shown in figure')
    # plt.ylabel('underlying asset price Bachlier')
    # plt.title('simulated paths for the underlying asset')
    plt.show()
    for i in range(try_numbers):
        path = option2.generate_path_BSM()
        list2.append(path[-1])
    temp = pd.DataFrame()
    temp['BM'] = list1
    temp['BSM'] = list2
    temp.hist(bins=50)
        # path2 = option1.generate_path_BM()
        # plt.plot(path2)
    print('The answer is shown in figure')
    plt.ylabel('underlying asset price BlackScholes')
    plt.title('simulated paths for the underlying asset')
    plt.show()




    # print('-----------------------------------------------------------------------', '\n')
    # print('Answer 2(b):', '\n')
    # terminal_value = option1.find_terminal_value_distribution_BM(try_numbers)
    # print('normaltest result for terminal_value:')
    # test_result = stats.normaltest(terminal_value)
    # print(test_result)
    # if test_result.pvalue <= 0.05:
    #     print('terminal_value is not a normal distribution.')
    # else:
    #     print('terminal_value is a normal distribution.')
    #
    # print('-----------------------------------------------------------------------', '\n')
    # print('Answer 2(c):', '\n')
    BM_price_approximation = option1.find_price_approximation_BM(try_numbers)
    BSM_price_approximation = option2.find_price_approximation_BSM(try_numbers)
    print(f'The price using Bachelier model path is: {BM_price_approximation}')
    print(f'The price using Black-Scholes model path is: {BSM_price_approximation}')
    print(f'The difference is: {BM_price_approximation - BSM_price_approximation}')
    #
    #
    # print('-----------------------------------------------------------------------', '\n')
    # print('Answer 2(d):', '\n')
    # epc = np.arange(0.001, 5, 0.05)
    # delta = []
    # for i in epc:
    #     temp = option2.find_delta(i, try_numbers)
    #     delta.append(temp)
    # plt.plot(epc, delta)
    # plt.xlabel('different epsilon')
    # plt.ylabel('delta')
    # plt.show()



