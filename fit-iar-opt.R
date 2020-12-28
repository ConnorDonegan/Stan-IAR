library(rstan)
library(sf)
library(spdep)
library(INLA)
library(ggplot2)
rstan_options(auto_write = TRUE)
model = stan_model("iar-opt.stan")
source("https://raw.githubusercontent.com/ConnorDonegan/geostan/main/R/convenience-functions.R")
c_scale <- function(C) { # create scale factor for BYM2 model
    # C is N by N connectivity matrix
    N <- nrow(C)
    adj.matrix <- Matrix::Matrix(C, sparse = TRUE)
    Q =  Diagonal(N, rowSums(adj.matrix)) - adj.matrix
    Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
    Q_inv = INLA::inla.qinv(Q_pert, constr=list(A = matrix(1,1,N),e=0))
    scale_factor <- exp(mean(log(diag(Q_inv))))
    return (scale_factor)
}

# load data
states <- st_read("states")
n <- nrow(states)
C <- shape2mat(states)

# gather edge list and group index
dl.iar <- prep_iar_data(C)

# calculate scale_factors
scale_factor <- vector(mode = "numeric", length = n)
for (j in 1:dl.iar$k) {
    group.j.idx <- which(dl.iar$comp.id == j)
    if (length(group.j.idx) == 1) {
        scale_factor[group.j.idx] <- 1
        next
    }    
    Cg <- C[group.j.idx, group.j.idx] 
    scale.j <- c_scale(Cg)
    scale_factor[group.j.idx] <- scale.j
}
dl.iar$scale_factor <- scale_factor

# combine with other data
dl <- list(n = n,
           prior_only = TRUE,
           y = rep(1, n),
           offset = rep(1, n))
dl <- c(dl, dl.iar)

## draw samples from the model (prior only)
# 1 = IAR
dl$type <- 1
fit <- sampling(model, data = dl, control = list(max_treedepth = 13),
                chains = 4,
                cores = 4
                )
# 2 = BYM
dl$type <- 2
fit2 <- sampling(model, data = dl, control = list(max_treedepth = 13),
                chains = 4,
                cores = 4
                 )

# 3 = BYM2
dl$type <- 3
fit3 <- sampling(model, data = dl, control = list(max_treedepth = 13),
                 chains = 4,
                 cores = 4
                 )

