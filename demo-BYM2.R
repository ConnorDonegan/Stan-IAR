pkgs <- c("sf", "spdep", "INLA", "rstan")
lapply(pkgs, require, character.only = TRUE)
rstan_options(javascript=FALSE)
source("icar-functions.R")

## get a shapefile
url <- "https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_state_20m.zip"
get_shp(url, "states")
states <- st_read("states")

## prep data for ICAR function in Stan
C <- spdep::nb2mat(spdep::poly2nb(states, queen = TRUE), style = "B", zero.policy = TRUE)
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

## and add in some (fake) outcome data with offset on log scale
n <- nrow(C)
dl <- list(n = n, 
           y = rep(1, n), ## just a placeholder
           offset = rep(log(100), n), ## just a placeholder
           prior_only = 1 ## telling Stan to ignore the outcome data (y, offset)
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

## prior degree of spatial autocorrelation (SA) in the convolution term
convolution <- as.matrix(fit, pars = "convolution")
sa <- apply(convolution, 1, mc, w=C)
hist(sa)
