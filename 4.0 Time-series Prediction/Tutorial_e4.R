install.packages('astsa')
library(astsa)
require("doParallel")
# Create parallel workers
workers <- makeCluster(2L)
# Preload dtwclust in each worker; not necessary but useful
invisible(clusterEvalQ(workers, library("dtwclust")))
# Register the backend; this step MUST be done
registerDoParallel(workers)
# Backend detected automatically
pc_par <- tsclust(CharTraj[1L:20L], k = 4L,
                  distance = "dtw_basic", centroid = "dba",
                  window.size = 15L, seed = 938,
                  control = partitional_control(nrep = 2L))
# Stop parallel workers
stopCluster(workers)
# Go back to sequential computation
registerDoSEQ()