library(Rcpp)
library(ggplot2)
library(ggforce)
library(tikzDevice)

sourceCpp("weighted_anchor_circles.cpp")
sourceCpp("centered_weighted_circles.cpp")

# Small test case
#
# U1 <- rbind(
#   c(0,0),
#   c(-2,0),
#   c(2,0),
#   c(0,-2.5)
# )
# U2 <- rbind(
#   c(0,0),
#   c(-1,0),
#   c(2,0)
# )
# 
# weighted_anchor_circle_at(U1,c(0,0),3)$radius #should be 1.600781
# weighted_anchor_circle_at(U2,c(0,0),3)$radius #should be 1.5

# Using synthetic enterprise data from sdcSpatial package
U <- sdcSpatial::enterprises@coords
# Using synthetic dwellings data from sdcSpatial package
U <- as.matrix(sdcSpatial::dwellings[,c(1,2)])

plot_df <- as.data.frame(U)

# Selected locations from enterprise population to plot their circles explicitly
# Chosen by hand
target_points <- which(plot_df$x==71528 & plot_df$y==440373)
#target_points <- c(target_points, which(plot_df$x==68805 & plot_df$y==440349))
#target_points <- c(target_points, which(plot_df$x==69807 & plot_df$y==440436))
target_points <- c(target_points, which(plot_df$x==80051 & plot_df$y==445414))
#target_points <- c(target_points, which(plot_df$x==71949 & plot_df$y==443804))

target_points <- c(target_points, which(plot_df$x==74232 & plot_df$y==444356))

target_points <- c(target_points, which(plot_df$x==81270 & plot_df$y==441407))
#target_points <- c(target_points, which(plot_df$x==75601 & plot_df$y==444956))
target_points <- c(target_points, which(plot_df$x==75597 & plot_df$y==448228))
target_points <- c(target_points, which(plot_df$x==70753 & plot_df$y==446872))

# # Using synthetic dwellings data from sdcSpatial package
# U <- sdcSpatial::dwellings
# plot_df <- data.frame(x=U$x,y=U$y)
# U$unemployed <- NULL
# U$consumption <- NULL
# U <- as.matrix(U)
# 
# # Selected locations from dwellings population to plot their circles explicitly
# # Chosen by hand
# target_points <- which(plot_df$x==150448 & plot_df$y==468293)
# target_points <- c(target_points,which(plot_df$x==153447 & plot_df$y==464368))
# target_points <- c(target_points,which(plot_df$x==157875 & plot_df$y==461919))

# Start calculations
# First set parameter k for "spatial k-anonymity"
k <- 10

# Calculate weighted minimal circles for all locations in (multiset) U
# delta = infty
microbenchmark::microbenchmark(PPdata <- weighted_anchor_circles(U,k), times=5, unit = "s")
# delta = 0
microbenchmark::microbenchmark(PPdata <- centered_weighted_circles(U,k), times=5, unit="s")


# Calculate risk for all locations in (multiset) U
PPdata$risk <- PPdata$radius/PPdata$weight

# Plot information for selected points and their circles
point_data <-  data.frame(x0=PPdata[target_points,]$px,
                          y0=PPdata[target_points,]$py)

circle_data <- data.frame(x0=PPdata[target_points,]$cx,
                          y0=PPdata[target_points,]$cy,
                          r=PPdata[target_points,]$radius)

tikz(file = "kAnonymRegionInf.tex", width=5, height=2.5, pointsize=10)
# Plot risk for all points and add some selected circles
ggplot(PPdata) +
  geom_point(aes(px, py, color = risk), size=0.5) +
  geom_circle(data=circle_data, aes(x0 = x0, y0 = y0, r = r), linewidth=0.5, color="blue", fill="blue", alpha = 0.1) +
  geom_point(data=point_data, aes(x = x0, y = y0), color = "black", size=1, shape=4, stroke=1.1) +
  scale_color_viridis_c(option="magma",direction=-1) +
  #scale_color_gradient(low = "#ffddbb", high="red") +
  #xlim(157500,159000) +
  ylim(438800,449000) +
  coord_fixed() +
  theme_minimal() + #labs(title=title)+
  theme(axis.title = element_blank(), axis.text = element_blank())
dev.off()
 