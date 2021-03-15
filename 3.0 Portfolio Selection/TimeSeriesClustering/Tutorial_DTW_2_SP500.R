# 
# 
# https://medium.com/@panda061325/stock-clustering-with-time-series-clustering-in-r-63fe1fabe1b6

library(dtwclust) # DTW clustering
library(quantmod)
stock = getSymbols("AAPL",src='yahoo',from = '2015-07-01', auto.assign = F)

# Figure 1. As we can see, it is hard to detect which companies worth our money. However, generally there is a rising trend for S&P 500 companies in the recent three years. The coefficient for the regression line is 0.002
# normalized_price not shown.

# He said the normalized_price is the list or time-series. Should match the input specs described in the dtwclust paper.
#
# Look at Tutorial_DTW_1.R for analysis of DTW in detailed.
# There is also something about step pattern...

normalized_price <- BBmisc::normalize(stock, method="standardize")

# DTW distance
# The input of normalized_price is a list of normalized time-series stock price
dtw_cluster = tsclust(normalized_price, type="partitional",k=5,
                      distance="dtw_basic",centroid = "pam",seed=1234,trace=T,
                      args = tsclust_args(dist = list(window.size = 5)))
# Euclidean distance
eu_cluster = tsclust(normalized_price, type="partitional",k=5,
                     distance="Euclidean",centroid = "pam",seed=1234,trace=T)


# Figure 2: clustering with DTW distance for 5, 10 and 20 clusters