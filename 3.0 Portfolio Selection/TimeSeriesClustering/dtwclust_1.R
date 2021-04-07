# call to dtwclust, specifying the value of k indicates the number of desired
# clusters, so that the cutree function is called internally. Additionally, the shape extraction
# function is provided in the centroid argument so that, once the k clusters are obtained,
# their prototypes are extracted. Therefore, the data is z -normalized by means of the zscore
# function. The seed is provided because of the randomness in shape extraction when choosing
# a reference series
# https://github.com/asardaes/dtwclust

# install.packages('curl') 
# install.packages('quantmod')
# install.packages("proxy")
# install.packages("dtw")
# install.packages("dtwclust")

library(dtwclust)

# hc_sbd <- dtwclust(CharTraj, type = "h", k = 20L,
#                    method = "average", preproc = zscore,
#                    distance = "sbd", centroid = shape_extraction,
#                    seed = 899, control = list(trace = TRUE))


hc <- tsclust(CharTraj, type = "hierarchical", k = 20L, 
              distance = "sbd", trace = TRUE,
              control = hierarchical_control(method = "average"))
#> 
#> Calculating distance matrix...
#> Performing hierarchical clustering...
#> Extracting centroids...
#> 
#>  Elapsed time is 0.147 seconds.
plot(hc)