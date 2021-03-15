# Dynamic Time Warp Tutorial 1 
# https://rstudio-pubs-static.s3.amazonaws.com/474160_0761428e8683401e941a92a4177254a4.html#5_k
#
#
# This tutorial seems to be created using RNotebook format.

a1 <- c(7,9,6,9,12,6,4,6,8)
a2 <- c(5,6,4,3,9,5,6,8,9)

xrange <- range(1:9)
yrange <- range(c(a1,a2))

plot(xrange, yrange, type="n", xlab="time",
     ylab="value", xaxp  = c(0,10,10), yaxp  = c(3,12,9)) 
lines(a1, col='blue', type='l')
lines(a2, col='magenta', type='l')


dtw(a1,a2)$index1

dtw(a1,a2)$index2


plot(dtw(a1,a2), xlab="a1 - blue", ylab="a2 - magenta", xaxp  = c(0,10,10), yaxp = c(0,10,10))

dtw(a1,a2)$index2

plot(dtw(a1,a2), xlab="a1 - blue", ylab="a2 - magenta", xaxp  = c(0,10,10), yaxp = c(0,10,10))

plot(dtw(a1,a2, keep=TRUE), xlab="a1 - blue", ylab="a2 - magenta", xaxp  = c(0,10,10), yaxp = c(0,10,10), type="threeway")




plot(dtw(a1,a2, keep=TRUE), xlab="a1 - blue", ylab="a2 - magenta", xaxp  = c(0,10,10), yaxp = c(0,10,10), type="threeway")



plot(dtw(a1,a2, keep=TRUE), xaxp  = c(0,10,10), yaxp = c(0,10,10), type="twoway", col=c('blue', 'magenta'))

dtw(a1, a2)$stepPattern


dtw_matrix = matrix(rep(c(0),81), nrow=9, ncol=9, byrow = TRUE)
dtw_matrix

dtw_matrix[1,1] = sqrt(a1[1]^2 + a2[1]^2)
dtw_matrix



for (i in 2:9){
  dtw_matrix[i,1] = sqrt((a1[i] - a2[1])^2) + dtw_matrix[i-1,1]
}
dtw_matrix



for (j in 2:9){
  dtw_matrix[1,j] = sqrt((a1[1] - a2[j])^2) + dtw_matrix[1,j-1]
}
dtw_matrix



for (i in 2:9){
  for (j in 2:9){
    dtw_matrix[i,j] = sqrt((a1[i] - a2[j])^2) + min(dtw_matrix[i,j-1], dtw_matrix[i-1,j], dtw_matrix[i-1,j-1] + sqrt((a1[i] - a2[j])^2))
  }
}
dtw_matrix


path = c(9,9) # starting with furthest place in matrix
i = 9
j = 9
while(i>1 & j>1){
  if (j == 1) {
    j = j - 1
  } else if (i == 1) {
    i = i - 1
  } else if (dtw_matrix[i,j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1])){
    j = j - 1
  } else if (dtw_matrix[i-1,j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1])){
    i = i - 1
    j = j - 1
  } else {
    i = i - 1
  }
  path = rbind(path, c(i,j))
}
path = rbind(path, c(1,1))


plot(dtw(a1,a2))
points(path[,1], path[,2], type="l")

# OK, here we have the time-warp index plot. The mapping between two sequences.

# 4 Using the dtwclust package for time series clustering
# 4.1 dtwclust::tsclust

# The data:
# the analysis on data collected from Eurostat on emissions of greenhouse gasses per capita Eurostat (2014).


# 5 k-medoids
# First I am normalising the chosen data and performing clustering using PAM and DTW distance on a range of 2-17 clusters.
emissions.norm <- BBmisc::normalize(emissions, method="standardize")
clust.pam <- tsclust(emissions.norm, type="partitional", k=2L:17L, distance="dtw", centroid="pam")
