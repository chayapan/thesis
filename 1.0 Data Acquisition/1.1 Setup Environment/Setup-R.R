# This script set up R environment with required packages
#
#
#

#
# apt-get install libcurl4-openssl-dev 
install.packages('curl') # Requires libcurl4

# quantmod package requires curl and TTR
install.packages('quantmod')

install.packages('dtwclust')
install.packages('flexclust')
install.packages('TSdist')
install.packages('TSclust')

#
# apt-get install libxml2-dev libudunits2-dev 
install.packages('eurostat') # for DTW emission data time-series clustering analysis


#for
#
# normalized_price <- BBmisc::normalize(stock, method="standardize")
install.packages('BBmisc')