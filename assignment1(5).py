"""
Program:
Author: cai
Date: 2021-09-03
"""
import numpy as np
import matplotlib.pyplot as plt


def short_call_option(price):
    """
    :param price: exercise price
    :return: payoff plot
    """
    x = np.arange(0, 2 * price, 0.01)
    y = [0 for i in x if i <= price]
    temp = [price - i for i in x if i > price]
    y += temp
    # plt.plot(x, y, label='short call option')
    # plt.xlabel('asset price')
    # plt.ylabel('payoff')
    # plt.show()
    return y


def long_underlying_asset(price):
    """
    :param price: cost
    :return: payoff
    """
    x = np.arange(0, 2 * price, 0.01)
    y = [i - price for i in x]
    # plt.plot(x, y, label='long asset')
    # plt.xlabel('asset price')
    # plt.ylabel('payoff')
    # plt.show()
    return y


def short_put_option(price):
    """
    :param price: exercise price
    :return: payoff plot
    """
    x = np.arange(0, 2 * price, 0.01)
    y = [i - price for i in x if i <= price]
    temp = [0 for i in x if i > price]
    y += temp
    # plt.plot(x, y)
    # plt.xlabel('asset price')
    # plt.ylabel('payoff')
    # plt.show()
    return y


def synthetic_position(price):
    """
    :param price: exercise price and cost
    :return: the synthetic_position payoff (short call + long asset)
    """
    temp1 = short_call_option(price)
    temp2 = long_underlying_asset(price)
    x = np.arange(0, 2 * price, 0.01)
    y = [temp1[i] + temp2[i] for i in range(len(temp1))]
    # plt.plot(x, y)
    # plt.xlabel('asset price')
    # plt.ylabel('payoff')
    # plt.show()
    return y


# short_put_option(30)
# synthetic_position(30)
def plot(price):
    """
    :param price:
    :return:
    """
    x = np.arange(0, 2 * price, 0.01)
    y1 = short_put_option(price)
    y2 = short_call_option(price)
    y3 = long_underlying_asset(price)
    y4 = synthetic_position(price)
    plt.plot(x, y1, label='short put')
    plt.xlabel('asset price')
    plt.ylabel('payoff')
    plt.legend()
    plt.show()
    plt.plot(x, y2, label='short call')
    plt.plot(x, y3, label='long asset')
    plt.plot(x, y4, label='synthetic position')
    plt.xlabel('asset price')
    plt.ylabel('payoff')
    plt.legend()
    plt.show()


plot(30)




