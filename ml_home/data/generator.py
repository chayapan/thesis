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

# Function 10: linear
# Function 11: exponential

'''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_gbm_series(start_price=20, days=250, mu=0.001, sigma=0.01):
    """Geometric brownian motion code from
     https://stackoverflow.com/questions/13202799/python-code-geometric-brownian-motion-whats-wrong
    """
    T = days # Days to simulate
    # mu = 0.001
    # sigma = 0.01
    S0 = start_price
    dt = 1
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return t, S # time index and price value

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

def add_noise(x):
    noise = np.random.normal(loc=0, scale=3.0, size=len(x))
    # noise = np.random.standard_normal(size=len(x))
    # See geometric brownian model
    # https://stackoverflow.com/questions/13202799/python-code-geometric-brownian-motion-whats-wrong
    
    # noise
    # pd.Series(noise).plot.hist(title='Noise: Size=%s Normal distribution' % SIM_DAYS, bins=50)
    x_noisy = x+noise
    return x, x_noisy
    
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


# Shift line to the left or right
# https://math.stackexchange.com/questions/1803012/how-to-shift-a-line-in-a-graph-regardless-of-slope
    
# Function 10: linear function
# Test: x,y = dgf10(); plot_line(x,y)
def dgf10(a=0, c=0, m=0.4, b=0, h=0, v=0, d=0, days=250, with_noise=False):
    # b is  y-axis start.
    x = np.linspace(T_1, T_1 + days, days)
    # Decompose the two axises https://math.stackexchange.com/questions/1803012/how-to-shift-a-line-in-a-graph-regardless-of-slope
    # and apply the shift separately.
    y = m * ( x - c ) + b + a
    # h-param is value to shift along horizontal (x-axis)
    y = y + v    
    # v-param is value to shift along vertical (y-axis)
    x = x + h
    
    if with_noise:
        y, y_noisy = add_noise(y)
        y = y_noisy # overwrite
    
    return x,y

def dgf11(a=1.8, b=150, h=0, v=0, d=0, days=250, with_noise=False):
    # b is  y threshold.
    x = np.linspace(T_1, T_1 + days, days)
    y = x ** a / 360 + b
    # h-param is value to shift along horizontal (x-axis)
    y = y + v    
    # v-param is value to shift along vertical (y-axis)
    x = x + h
    
    if with_noise:
        y, y_noisy = add_noise(y)
        y = y_noisy # overwrite
        
    return x,y