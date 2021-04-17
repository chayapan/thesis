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
     
     This function is replaced by dgf_gbm_path function below.
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

def dgf_sine(t, A, f=1/100., phi=0, Z=0.):
    """t, A. Default f to 100 Hz.
    
See https://en.wikipedia.org/wiki/Sine_wave

A = 4 # amplitude, the peak deviation of the function from zero
f = 1/100. # ordinary frequency, the number of oscillations (cycles) that occur each second of time.
omega = 2 * np.pi * f # angular frequency, the rate of change of the function argument. Radians per second.
phi = 2 # phase, specifies in radians. the cycle the oscillation is at t=0

y = A * np.sin(omega * t + phi)
    
Example 1:

t = np.linspace(0, 50*np.pi, 1001) # discrete time axis of 1000 increments

sin1 = dgf_sine(t,2, f=1/100.)
sin2 = dgf_sine(t,3, f=1/50.)
sin3 = dgf_sine(t,2, f=1/50., phi=2)

plt.plot(sin1, label='1')
plt.plot(sin2, label='2')
plt.plot(sin3, label='3')
plt.legend()
plt.title('Generated Sine wave')
    
Example 2:

t = np.linspace(0, 50*np.pi, 1001) # discrete time axis of 1000 increments
sin1 = dgf_sine(t,A=1,Z=5.0)
sin2 = dgf_sine(t,A=1,Z=0.0)
plt.plot(sin1, label='3')
plt.plot(sin2, label='4')
plt.legend()
plt.title('Two sine waves with different amplitude level.')
    """
    omega = 2 * np.pi * f
    y = A * np.sin(omega * t + phi)
    # shift y-axis by Z
    y += np.ones(t.size) * Z
    return y

def dgf_lin_path(t, m=0., Z=0.):
    """ ## TODO: fix error so the 'm' parameter work.
    
m = # Slope
Z = # y-intercept

Example 1:

t = np.linspace(0, 50*np.pi, 1001) # discrete time axis of 1000 increments

line1 = np.ones(t.size)
line2 = dgf_line(t,5)
sin1 = dgf_line(t,A=1,Z=5.0)
sin2 = dgf_line(t,A=1,Z=0.0)

plt.plot(line1, label='1')
plt.plot(line2, label='2')
plt.plot(sin1, label='3')
plt.plot(sin2, label='4')
plt.legend()
plt.title("Two lines and two sine waves.")
    """
    y = np.zeros(t.size)
    y += Z # shift y-axis
    
    # line4 = line3 + np.arange(line3.size) / line3 * 0.01
    incr = np.arange(t.size) / y  
    y = y + m # increment by slope  ## FIXME
    return y


def dgf_hpr_path(t, s, R):
    """Holding-period Return path plot.
    
Example 1:

t = np.linspace(0, 50*np.pi, 1001) # discrete time axis of 1000 increments

plus15 = my_hpr_(t, 100, 0.15) # Return 15% from 100 principal
neg15 = my_hpr_(t, 100, -0.15) # Return -15% from 100 principal
    """    
    # R = 0.15 # 15 % return for the holding period.
    # s = 100 # start value, e = end value
    e = s + s * R
    g = s * R # gain/loss # 15.0
    c = g/t.size # gain/loss average per time unit
    a = np.ones(t.size) * c # accomulating...
    b = np.zeros(t.size) # base line, the starting amount
    b += a.cumsum()
    b += s # add principal amount
    return b # series of increasing value corresponding to the holding period return

def dgf_gbm_path(t, s0=20, mu=0.01, sigma=0.01):
    """See above make_gbm_series for original code.
    TODO: how to control mu and sigma?
    """
    days = t.size - 1
    T = days # Days to simulate
    # mu = 0.001
    mu = mu / 250  # say return per day annualized
    # sigma = 0.01
    sigma = sigma / 250 # say variance per day annualized
    S0 = s0
    dt = 1
    N = round(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S # time index and price value

dgf12 = dgf_sine  # Alias to Sine wave generator
dgf13 = dgf_lin_path  # Alias to straigth line generator
dgf14 = dgf_hpr_path # Alias to Holding-period Return path plot
dgf15 = dgf_gbm_path # Alias to Geometric Brownian Motion path plot