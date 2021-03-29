#
# TSclust $ 5.1 An sxample with a synthetic dataset

# install.packages("TSclust")

# Load data set and the correct labels.
library("TSclust")
data("synthetic.tseries")
true_cluster <- rep(1:6, each = 3)

# Compute dissimilarity matrix
IP.dis <- diss(synthetic.tseries, "INT.PER")

# Hierarchical cluster 
IP.hclus <- cutree(hclust(IP.dis), k = 6)

cluster.evaluation(true_cluster, IP.hclus)

library("cluster")
IP.pamclus <- pam(IP.dis, k = 6)$clustering
cluster.evaluation(true_cluster, IP.pamclus)

# Use ACF function to create dissimilarity matrix.
ACF.dis <- diss(synthetic.tseries, "ACF", p = 0.05)

ACF.hclus <- cutree(hclust(ACF.dis), k = 6)
