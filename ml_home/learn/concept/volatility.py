'''
1.3 Eikon_Volatility_StDev_5d.ipynb
2.0 Feature Engineering/Volatility.ipynb

Usage:

df['rolling5d_stddev'] = df['Price'].rolling(5).std() # Short cut available from pandas
df['stddev_ndays5'] = df['Price'].rolling(5).apply(stddev_ndays, (5,)) # Manual window function
df['stddev_5d'] = df['Price'].rolling(5).apply(stddev_5d)

'''
import numpy as np

# Calcualtes price volatility (std.dev.) for 5 day window.
# 1. Find mean. 2. Find deviation. 3. Squared deviation.
# 4. Sum squared devations. 5. Divide by number of ovservations. This is variance.
# 6. Take square root.
stddev_5d = lambda x: np.sqrt((np.square(x[0]-x.mean()) + np.square(x[1]-x.mean()) + np.square(x[2]-x.mean()) + np.square(x[3]-x.mean()) + np.square(x[4]-x.mean())) / 5)

def stddev_ndays(x, n=5):
    """Calcualte standard deviation which is the proxy for volatility.
       Improve for clarity over stddev_5d one-liner above. """
    mean = x.mean() # Mean of the window.
    total_squared_deviation = 0.0 # For collecting squared deviation.
    for i in range(n):
        obs = x[i]
        deviation = obs - mean
        sqrd_dev = np.square(deviation)
        total_squared_deviation += sqrd_dev
    return np.sqrt(total_squared_deviation / n) # Take square root of variance and return st.dev. value
