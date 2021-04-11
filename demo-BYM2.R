pkgs <- c("sf", "spdep", "INLA", "rstan")
lapply(pkgs, require, character.only = TRUE)
rstan_options(javascript=FALSE)
source("icar-functions.R")

## the commented-out code will set you up to use the same data as in the README document
## get a shapefile
## url <- "https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_state_20m.zip"
## get_shp(url, "states")
## states <- st_read("states")

## prep data for ICAR function in Stan
##C <- spdep::nb2mat(spdep::poly2nb(states, queen = TRUE), style = "B", zero.policy = TRUE)
##icar.data <- prep_icar_data(C)

# load data: from Spatial Epi package, a data.frame and a shapefile
library(SpatialEpi)
data(scotland)
df <- scotland$data
sp <- scotland$spatial.polygon

## prep data for ICAR function in Stan
C <- spdep::nb2mat(spdep::poly2nb(sp, queen = TRUE), style = "B", zero.policy = TRUE)
icar.data <- prep_icar_data(C)

## notice that the scale_factor is just ones. 
icar.data$inv_sqrt_scale_factor

## calculate the scale factor for each of k connected group of nodes, using scale_c function
k <- icar.data$k
scale_factor <- vector(mode = "numeric", length = k)
for (j in 1:k) {
  g.idx <- which(icar.data$comp_id == j) 
  if (length(g.idx) == 1) {
    scale_factor[j] <- 1
    next
  }    
  Cg <- C[g.idx, g.idx] 
  scale_factor[j] <- scale_c(Cg) 
}

## update the data list for Stan
icar.data$inv_sqrt_scale_factor <- 1 / sqrt( scale_factor )

## see the new values
print(icar.data$inv_sqrt_scale_factor)

## add outcome data to the list
dl <- list(    
    n = nrow(df),
    y = df$cases,
    offset = log(df$expected),
    prior_only = 0
)

dl <- c(dl, icar.data)

## compile the model
BYM2 <- stan_model("BYM2.stan")

## sample
fit <- sampling(BYM2, data = dl, chains = 4, cores = 4)

## view some results from the joint prior probability
plot(fit, pars = "phi_tilde")
plot(fit, pars = "convolution")
plot(fit, pars = "spatial_scale", plotfun = "hist")
plot(fit, pars = "rho", plotfun = "hist")

## degree of spatial autocorrelation (SA) in the convolution term
convolution <- as.matrix(fit, pars = "convolution")
sa <- apply(convolution, 1, mc, w=C)
hist(sa)

## simple map of the posterior mean of the convolution term 
spx <- st_as_sf(sp)
spx$convolution <- apply(convolution, 2, mean)
plot(spx[,"convolution"])
