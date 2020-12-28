library(rstan)
library(sf)
library(spdep)
library(INLA)
library(ggplot2)
rstan_options(auto_write = TRUE)
model = stan_model("iar-opt.stan")
source("https://raw.githubusercontent.com/ConnorDonegan/geostan/main/R/convenience-functions.R")
source("https://raw.githubusercontent.com/ConnorDonegan/helpful/master/get-shp.R")
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

#' @param C connectivity matrix
#' @param scale_factor n-length vector with the scale factor for each observation's respective group; user provided else fixed to \code{rep(1, n)}
#' 
#' @importFrom spdep poly2nb n.comp.nb
#' 
#' @return list of data to add to Stan data list:
#' \describe{
#' \item{k}{number of groups}
#' \item{group_size}{number of nodes per group}
#' \item{n_edges}{number of connections between nodes (unique pairs only)}
#' \item{node1}{first node}
#' \item{node2}{second node---node1[1], node2[1] form a connected pair}
#' \item{group_idx}{indices for each observation belonging each group, ordered by group.}
#' }
prep_iar_data <- function(C, scale_factor = NULL) {
    n <- nrow(C)
    if (inherits(scale_factor, "NULL")) scale_factor <- rep(1, n)
    E <- edges(C)
    G <- list(np = nrow(C), # confrom to spdep graph structure
              from = E$node1,
              to = E$node2,
              nedges = nrow(E)
              )
    class(G) <- "Graph"
    nb2 <- spdep::n.comp.nb(spdep::graph2nb(G))
    k = nb2$nc
    group_idx = NULL
    for (j in 1:k) group_idx <- c(group_idx, which(nb2$comp.id == j))
    group_size <- NULL
    for (j in 1:k) group_size <- c(group_size, sum(nb2$comp.id == j))
    l <- list(
        k = k,
        group_size = array(group_size, dim = k),
        n_edges = nrow(E),
        node1 = E$node1,
        node2 = E$node2,
        group_idx = array(group_idx, dim = n),
        scale_factor = scale_factor,
        comp.id = nb2$comp.id
    )
    return (l)
}

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
dl$type <- 1
fit <- sampling(model, data = dl, control = list(max_treedepth = 13),
                chains = 4,
                cores = 4
                )

dl$type <- 2
fit2 <- sampling(model, data = dl, control = list(max_treedepth = 13),
                chains = 4,
                cores = 4
                 )

dl$type <- 3
fit3 <- sampling(model, data = dl, control = list(max_treedepth = 13),
                 chains = 4,
                 cores = 4
                 )



colnames(as.matrix(fit))
colnames(as.matrix(fit3))

drop.idx <- which(states$NAME %in% c("Alaska", "Hawaii", "Puerto Rico"))
cont <- states[-drop.idx,]

sp1 <- as.matrix(fit, pars = "spatial")[,-drop.idx]
sp2 <- as.matrix(fit2, pars = "spatial")[,-drop.idx]
sp3 <- as.matrix(fit3, pars = "spatial")[,-drop.idx]

ggplot(cont) +
    geom_sf(aes(fill = sp1[10,])) +
    scale_fill_gradient2()

## fit to data
dl <- list(n = n,
           prior_only = TRUE,
           y = rpois(n = n, lambda = 25),
           offset = rep(10e3, n)
           )
dl <- c(dl, dl.iar)
dl$type <- 3

fitb <- sampling(model, data = dl, control = list(max_treedepth = 13),
                 chains = 4,
                 cores = 4
                 )

phi <- as.matrix(fitb, pars = "sure")[,-drop.idx]

phi.mean <- apply(phi, 2, mean)

ggplot(cont) +
    geom_sf(aes(fill = phi.mean)) +
    scale_fill_gradient2()

plot(fitb, pars = "rho")


q <- apply(phi, 2, quantile, probs= c(0.025, 0.975))

hist(q[2,])
b


library(devtools)
load_all("~/dev/geostan")

C = shape2mat(sentencing)
fit = stan_car(sents ~ offset(log(expected_sents)),
                data = sentencing,
                C = C,
                chains = 3,
                cores = 3
                )

fit
