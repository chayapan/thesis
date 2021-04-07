# 
# This example takes the autocorrelation-based distance included in the TSclust package (Montero and Vilar 2014) and registers it with proxy so that it can be used either directly, or with
# dtwclust.
#
# Add ACF distance function
require(TSclust)
proxy::pr_DB$set_entry(FUN = diss.ACF, names=c("ACFD"),
                       loop = TRUE, type = "metric", distance = TRUE,
                       description = "Autocorrelation-based distance")

# Taking just a subset of the data
# Note that subsetting with single brackets preserves the list format
proxy::dist(CharTraj[3:8], method = "ACFD", upper = TRUE)