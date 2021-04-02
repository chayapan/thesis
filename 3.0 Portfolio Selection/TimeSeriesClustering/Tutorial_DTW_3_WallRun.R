#
# Wall-run
# https://damien-datasci-blog.netlify.app/post/time-series-clustering-with-dynamic-time-warp/

# install.packages("ggdendro")
# install.packages("gplots")

library(dplyr) # data wrangling
library(ggplot2) # grammar of graphics
library(gridExtra) # merge plots
library(ggdendro) # dendrograms
library(gplots) # heatmap
library(tseries) # bootstrap
library(TSclust) # cluster time series
library(dtwclust) # cluster time series with dynamic time warping



#### SINE WAVE ####

# classic run
noise <- runif(420) # random noise
x <- seq(1,420) # 42km with a measure every 100m
pace_min <- 5 # min/km (corresponds to fast run)

ts_sim_classic_run <- (sin(x/10)+x/100+noise+pace_min) %>%
  as.ts(.)

ts.plot(ts_sim_classic_run, xlab = "Distance [x100m]", ylab = "Differential pace [min/km]", main = "Example of classic run", ylim=c(0,25))

# wall run
noise <- runif(210) # random noise
x <- seq(1,210) # 21km with a measure every 100m 
pace_min <- 5 # min/km (corresponds to fast run)
pace_wall <- 20 # min/km (corresponds to very slow run) 
ts_sim_part1 <- sin(x/5)+x/50+noise+pace_min
ts_sim_part2 <- sin(x/5)+noise+pace_wall

ts_sim_wall_run <- c(ts_sim_part1,ts_sim_part2) %>%
  as.ts(.)

ts.plot(ts_sim_wall_run, xlab = "Distance [x100m]", ylab = "Differential pace [min/km]", main = "Example of wall run", ylim=c(0,25))



#### ARIMA with an auto regressive model (AR) ####

pace_min <- 5 # min/km (corresponds to fast run)
pace_wall <- 20 # min/km (corresponds to very slow run) 

# classic run
ts_sim_classic_run <- abs(arima.sim(n = 420, mean = 0.001, model = list(order = c(1,0,0), ar = 0.9))) + pace_min

ts.plot(ts_sim_classic_run, xlab = "Distance [x100m]", ylab = "Differential pace [min/km]", main = "Example of classic run", ylim=c(0,25))

# wall run
ts_sim_part1 <- abs(arima.sim(n = 210, model = list(order = c(1,0,0), ar = 0.9))) + pace_min
ts_sim_part2 <- ts(arima.sim(n = 210, model = list(order = c(1,0,0), ar = 0.9)) + pace_wall, start = 211,end =420)

ts_sim_wall_run <- ts.union(ts_sim_part1,ts_sim_part2)
ts_sim_wall_run<- pmin(ts_sim_wall_run[,1], ts_sim_wall_run[,2], na.rm = TRUE)

ts.plot(ts_sim_wall_run, xlab = "Distance [x100m]", ylab = "Differential pace [min/km]", main = "Example of wall run", ylim=c(0,25))


#### Bootstrap -  two groups of 5 individuals for each run type  ####

ts_sim_boot_classic <- ts_sim_classic_run %>%
  tseries::tsbootstrap(., nb=5, b=200, type = "block") %>%
  as.data.frame(.) %>%
  dplyr::rename_all(funs(c(paste0("classic_",.))))

ts_sim_boot_wall <- ts_sim_wall_run %>%
  tseries::tsbootstrap(., nb=5, b=350, type = "block") %>%
  as.data.frame(.) %>%
  dplyr::rename_all(funs(c(paste0("wall_",.))))

ts_sim_df <- cbind(ts_sim_boot_classic,ts_sim_boot_wall)


#### Heatmap with DTW ####


dtw_dist <- function(x){dist(x, method="DTW")}

ts_sim_df %>%
  as.matrix() %>%
  gplots::heatmap.2 (
    # dendrogram control
    distfun = dtw_dist,
    hclustfun = hclust,
    dendrogram = "column",
    Rowv = FALSE,
    labRow = FALSE
  )



#### implement TSclust and dtwclust packages.
## for DTW analysis

### Using TSclust
# cluster analysis
dist_ts <- TSclust::diss(SERIES = t(ts_sim_df), METHOD = "DTWARP") # note the dataframe must be transposed
hc <- stats::hclust(dist_ts, method="complete") # meathod can be also "average" or diana (for DIvisive ANAlysis Clustering)
# k for cluster which is 2 in our case (classic vs. wall)
hclus <- stats::cutree(hc, k = 2) %>% # hclus <- cluster::pam(dist_ts, k = 2)$clustering has a similar result
  as.data.frame(.) %>%
  dplyr::rename(.,cluster_group = .) %>%
  tibble::rownames_to_column("type_col")

hcdata <- ggdendro::dendro_data(hc)
names_order <- hcdata$labels$label
# Use the folloing to remove labels from dendogram so not doubling up - but good for checking hcdata$labels$label <- ""

p1 <- hcdata %>%
  ggdendro::ggdendrogram(., rotate=TRUE, leaf_labels=FALSE)

p2 <- ts_sim_df %>%
  dplyr::mutate(index = 1:420) %>%
  tidyr::gather(key = type_col,value = value, -index) %>%
  dplyr::full_join(., hclus, by = "type_col") %>% 
  mutate(type_col = factor(type_col, levels = rev(as.character(names_order)))) %>% 
  ggplot(aes(x = index, y = value, colour = cluster_group)) +
  geom_line() +
  facet_wrap(~type_col, ncol = 1, strip.position="left") + 
  guides(color=FALSE) +
  theme_bw() + 
  theme(strip.background = element_blank(), strip.text = element_blank())

gp1<-ggplotGrob(p1)
gp2<-ggplotGrob(p2) 

grid.arrange(gp2, gp1, ncol=2, widths=c(4,2))

### Using dtwclust
## The main asset of dtwclust is the possibility to customize the DTW clustering. 


cluster_dtw_h2 <- dtwclust::tsclust(t(ts_sim_df), 
                                    type = "h", 
                                    k = 2,  
                                    distance = "dtw", 
                                    control = hierarchical_control(method = "complete"),
                                    preproc = NULL, 
                                    args = tsclust_args(dist = list(window.size = 5L)))

hclus <- stats::cutree(cluster_dtw_h2, k = 2) %>% # hclus <- cluster::pam(dist_ts, k = 2)$clustering has a similar result
  as.data.frame(.) %>%
  dplyr::rename(.,cluster_group = .) %>%
  tibble::rownames_to_column("type_col")

hcdata <- ggdendro::dendro_data(cluster_dtw_h2)
names_order <- hcdata$labels$label
# Use the folloing to remove labels from dendogram so not doubling up - but good for checking hcdata$labels$label <- ""

p1 <- hcdata %>%
  ggdendro::ggdendrogram(., rotate=TRUE, leaf_labels=FALSE)

p2 <- ts_sim_df %>%
  dplyr::mutate(index = 1:420) %>%
  tidyr::gather(key = type_col,value = value, -index) %>%
  dplyr::full_join(., hclus, by = "type_col") %>% 
  mutate(type_col = factor(type_col, levels = rev(as.character(names_order)))) %>% 
  ggplot(aes(x = index, y = value, colour = cluster_group)) +
  geom_line() +
  facet_wrap(~type_col, ncol = 1, strip.position="left") + 
  guides(color=FALSE) +
  theme_bw() + 
  theme(strip.background = element_blank(), strip.text = element_blank())

gp1<-ggplotGrob(p1)
gp2<-ggplotGrob(p2) 

grid.arrange(gp2, gp1, ncol=2, widths=c(4,2))