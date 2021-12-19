"""
Program:
Author: cai
Date: 2021-09-11
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


class Option:
    def __init__(self, maturity, exercise_price, dt):
        """
        :param maturity: time to maturity (years)
        :param exercise_price: exercise price of the option
        :param dt: single time period (1 / trade days per year or a relatively small number, because
        Brownian motion increment dWt is normally distributed with variance dt. The time interval
        in Brownian motion simulation is very small, even continuous)
        """
        self.maturity = maturity
        self.exercise_price = exercise_price
        self.dt = dt


    def __repr__(self):
        pass

    def generate_simulated_path(self, s0, r, sigma, mu):
        """
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param mu: Brownian motion increment dWt is normally distributed with mean mu
        :return: the simulated path (an array) for the underlying asset
        """
        temp = int(self.maturity / self.dt)
        dWt = np.random.normal(loc=mu, scale=math.sqrt(self.dt), size=temp)
        path = np.array([s0])
        for i in range(temp):
            dSt = r * path[i] * self.dt + sigma * path[i] * dWt[i]
            St = path[i] + dSt
            path = np.append(path, St)
        return path


    def find_payoff(self):
        pass


class EuropeanPutOption(Option):
    def find_payoff(self, s0, r, sigma, mu, try_numbers):
        """
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param mu: Brownian motion increment dWt is normally distributed with mean mu
        :param try_numbers: simulation times
        :return: an array of the payoffs for an European put option
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_simulated_path(self, s0, r, sigma, mu)
            terminal_value = path[-1]
            if terminal_value < self.exercise_price:
                temp.append(self.exercise_price - terminal_value)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff


    def find_price_approximation(self, s0, r, sigma, mu, r2, try_numbers):
        """
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param mu: Brownian motion increment dWt is normally distributed with mean mu
        :param r2: discount rate (risk free rate)
        :param try_numbers: simulation times
        :return: the price estimation of an European option via simulation
        """
        payoff = EuropeanPutOption.find_payoff(self, s0, r, sigma, mu, try_numbers)
        price = np.mean(payoff) * math.exp(- r2 * self.maturity)
        return price


    def find_BSmodel_price(self, s0, sigma, rf):
        """
        :param s0: initial price of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param rf: discount rate (risk free rate)
        :return: the calculation of BSmodel_price of an European option
        """
        d1 = (math.log(s0 / self.exercise_price) + self.maturity * (rf + sigma ** 2 / 2)) \
             * (1 / (sigma * math.sqrt(self.maturity)))
        d2 = d1 - sigma * (math.sqrt(self.maturity))
        price = self.exercise_price * math.exp(- rf * self.maturity) * norm.cdf(- d2, loc=0, scale=1) \
                - s0 * norm.cdf(- d1)
        return price


class LookbackPutOption(Option):
    def find_payoff(self, s0, r, sigma, mu, try_numbers):
        """
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param mu: Brownian motion increment dWt is normally distributed with mean mu
        :param try_numbers: simulation times
        :return: an array of payoffs of a lookback put option via simulation
        """
        temp = []
        for i in range(try_numbers):
            path = Option.generate_simulated_path(self, s0, r, sigma, mu)
            terminal_value = np.min(path)
            if terminal_value < self.exercise_price:
                temp.append(self.exercise_price - terminal_value)
            else:
                temp.append(0)
        payoff = np.array(temp)
        return payoff


    def find_price_approximation(self, s0, r, sigma, mu, r2, try_numbers):
        """
        :param s0: initial price of underlying asset
        :param r: interest rate of underlying asset
        :param sigma: standard deviation of underlying asset price
        :param mu: Brownian motion increment dWt is normally distributed with mean mu
        :param r2: discount rate
        :param try_numbers: simulation times
        :return: the price estimation of a lookback put option via simulation
        """
        payoff = LookbackPutOption.find_payoff(self, s0, r, sigma, mu, try_numbers)
        price = np.mean(payoff) * math.exp(- r2 * self.maturity)
        return price






