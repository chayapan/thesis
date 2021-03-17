'''
See DataGeneratorFunction.ipynb in 4.0

# Function 1: linear function - ReLU -ish
# Function 2: linear function upward
# Function 3: linear function downward
# Function 4: exponential growth function upward
# Function 5: monotonically decreasing
# Function 6: concave up
# Function 7: concave down
# Function 8: s-curve
# Function 9: s-curve opposite

'''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gd2df(x,y):
    """Convert generated data to data frame."""

    df = pd.DataFrame(data=[x,y]).T
    df.rename(columns={0:'x',1:'y'}, inplace=True)
    df.index = df.index.values + 1 # Start index at 1.
    return df

def plot_line(x,y, xlim=(-10,1500), ylim=(-100,500)):
    # x-axis is time, y-axis is amplitude.
    plt.plot(x,y) # x-axis is time, y-axis is amplitude.
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1]) # Set x and y limits so plot show same scale.

# Time index: 1...1200 steps
T_1 = 1
T_N = 1200

# Function 1: linear function
# Test: x,y = dgf1(); plot_line(x,y)
def dgf1(a=0.5, b=100, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = a * x + b
    return x,y

# Function 2: linear function upward
# Test: x,y = dgf2(); plot_line(x,y)
def dgf2(a=0.1, b=100, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = a * x + b
    return x,y

# Function 3: linear function downward
# Test: x,y = dgf3(); plot_line(x,y)
def dgf3(a=-0.1, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = a * x + b
    return x,y

# Function 4: exponential growth function upward
# Test: x,y = dgf4(); plot_line(x,y)
def dgf4(a=1.004, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = a ** x + b
    return x,y

# Function 5: monotonically decreasing
# Test: x,y = dgf5(); plot_line(x,y)
def dgf5(a=1.004, b=150,nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = (a ** x) * -1  + b
    return x,y

# Function 6: concave up
# Test: x,y = dgf6(); plot_line(x,y)
def dgf6(a=1.5, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = x ** a / 360 + b
    return x,y

# Function 7: concave down
# Test: x,y = dgf7(); plot_line(x,y)
def dgf7(a=1.5, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = -1 * (x ** a / 360) + b
    return x,y

# Function 8: s-curve
# Test: x,y = dgf8(); plot_line(x,y)
#
# S(x) = (1/(1+exp(-kx))^a is the simple form of the equation, where the minimum value is 0 and the maximum value is 1, k and a both >0 and control the shape.
# https://blog.arkieva.com/basics-on-s-curves/
# https://en.wikipedia.org/wiki/Sigmoid_function
def dgf8(a=1.5, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = 1 / (1 + np.exp(-x) * 500) + b
    return x,y

# Function 9: s-curve opposite
# Test: x,y = dgf9(); plot_line(x,y)
def dgf9(a=1.5, b=150, nsample=1000):
    # b is  y threshold.
    x = np.linspace(T_1, T_N, nsample)
    y = -1 / (1 + np.exp(-x) * 500) + b
    return x,y

def make_dataset():
    f = [dgf1, dgf2, dgf3, dgf4, dgf5, dgf6, dgf7, dgf8, dgf9]
    b = 50 # b parameter set the starting height.

    dataset = {} # stores generated data
    for i in range(50):
        g = random.choice(f)
        x,y = g(b=b)
        x = x + random.choice([100,50,30,40,50,-50,20])
        dataset[i] = (g.__name__, x, y)
        plot_line(x,y, xlim=(-5,1200), ylim=(-100,300))
    return dataset
