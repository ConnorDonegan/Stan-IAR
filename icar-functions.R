#' Download and unzip a shapefile into your working directory
#' 
#' @param url URL to a shapefile. If you're downloading a shapefil manually, right click the "Download" button and copy the URL to your clipboard.
#' @param folder name of the folder to unzip the file into (character string).
#' 
#' @return a shapefile (or whatever you provided a URL to) in your working directory; also prints the contents of the folder with full file paths.
#' 
#' @author Connor Donegan
#' 
get_shp <- function(url, folder = "shape") {
  tmp.dir <- tempfile(fileext = ".zip")
  download.file(url, destfile = tmp.dir)
  unzip(tmp.dir, exdir = folder)
  list.files(folder, full.names = TRUE)
}


#' connect_regions
#' 
#' Given an nb object and the names of two areal regions, update the nb
#' object so that the two regions are connected.

#' The nb object is a list of n integer vectors.  It also has attribute
#' region.id which is a character vector with n unique values (like the
#' row.names of a data.frame object); n is the number of spatial entities.
#' Component i of this list contains the integer identifiers its neighbours
#' as a sorted vector with no duplication and values in 1:n;  if i has no
#' neighbours, the component is a vector of length 1 with value 0L.
#' see:  https://cran.r-project.org/web/packages/spdep/vignettes/nb_igraph.html
#'
#' param nb: nb object over areal regions
#' param name1:  name of region 1.
#' param name2:  name of region 1.
#' returns: updated nb object
#' 
#' @author Mitzi Morris
#' 
connect_regions <- function(nb, name1, name2) {
  if (name1 == name2) {
    cat("Cannot connect region to itself: ", name1)
    return(nb)
  }
  id1 <- which(attr(nb, "region.id") == name1)
  if (length(id1) == 0) {
    cat("Unknown region: ", name1)
    return(nb)
  }
  id2 <- which(attr(nb, "region.id") == name2)
  if (length(id2) == 0) {
    print("Unknown region: ", name2)
    return(nb);
  }
  if (nb[[id1]][1] == 0)  # singleton
    nb[[id1]] <- c(as.integer(id2))
  else
    nb[[id1]] <- unique(sort(c(nb[[id1]], as.integer(id2))))
  
  if (nb[[id2]][1] == 0)  # singleton
    nb[[id2]] <- c(as.integer(id1))
  else
    nb[[id2]] <- unique(sort(c(nb[[id2]], as.integer(id1))))
  nb
}


#' convert connectivity matrix to unique pairs of connected nodes (graph structure)
#' 
#' @param w a connectivity matrix
#' 
#' @return a data.frame with three columns: node1 and node2 (the indices of connected nodes) and their weight (the element w[i,j]).
#'   Only unique pairs of connected nodes are included---that is, each pair `[i,j]` is listed once, 
#'    with all i < j. This means that if `[i, j]` is included, then `[j, i]` is not also listed.
#'    
#' @author Connor Donegan
#' 
edges <- function (w) {
  lw <- apply(w, 1, function(r) {
    which(r != 0)
  })
  all.edges <- lapply(1:length(lw), function(i) {
    nbs <- lw[[i]]
    if (length(nbs)) 
      data.frame(node1 = i, node2 = nbs, weight = w[i, nbs])
  })
  all.edges <- do.call("rbind", all.edges)
  edges <- all.edges[which(all.edges$node1 < all.edges$node2), ]
  return(edges)
}


#' compute scaling factor for adjacency matrix
#' accounts for differences in spatial connectivity 
#' 
#' @param C connectivity matrix
#' 
#' Requires the following packages: 
#' 
#' library(Matrix)
#' library(INLA);
#' library(spdep)
#' library(igraph)
#' 
#' @author Mitzi Morris
#' 
scale_c <- function(C) {
  #' compute geometric mean of a vector
  geometric_mean <- function(x) exp(mean(log(x))) 
  
  N = dim(C)[1]
  
  # Create ICAR precision matrix  (diag - C): this is singular
  # function Diagonal creates a square matrix with given diagonal
  Q =  Diagonal(N, rowSums(C)) - C
  
  # Add a small jitter to the diagonal for numerical stability (optional but recommended)
  Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
  
  # Function inla.qinv provides efficient way to calculate the elements of the
  # the inverse corresponding to the non-zero elements of Q
  Q_inv = inla.qinv(Q_pert, constr=list(A = matrix(1,1,N),e=0))
  
  # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
  scaling_factor <- geometric_mean(Matrix::diag(Q_inv)) 
  return(scaling_factor) 
}

#' prepare Stan data for ICAR model given a connectivity matrix
#' 
#' @param C a connectivity matrix
#' @param scale_factor optional vector of scale factors for each connected portion of the graph structure. 
#'   Generally, you will ignore this and update the scale factor manually.
#'   
#' @return a list with all that is needed for the Stan ICAR prior. If you do not provide inv_sqrt_scale_factor, 
#'   it will be set to a vector of 1s.
#'   
#' @author Connor Donegan
#' 
prep_icar_data <- function (C, inv_sqrt_scale_factor = NULL) {
  n <- nrow(C)
  E <- edges(C)
  G <- list(np = nrow(C), from = E$node1, to = E$node2, nedges = nrow(E))
  class(G) <- "Graph"
  nb2 <- spdep::n.comp.nb(spdep::graph2nb(G))
  k = nb2$nc
  if (inherits(inv_sqrt_scale_factor, "NULL")) inv_sqrt_scale_factor <- array(rep(1, k), dim = k)
  group_idx = NULL
  for (j in 1:k) group_idx <- c(group_idx, which(nb2$comp.id == j))
  group_size <- NULL
  for (j in 1:k) group_size <- c(group_size, sum(nb2$comp.id == j))
  # intercept per connected component of size > 1, if multiple.
  m <- sum(group_size > 1) - 1
  if (m) {
    GS <- group_size
    ID <- nb2$comp.id
    change.to.one <- which(GS == 1)
    ID[which(ID == change.to.one)] <- 1
    A = model.matrix(~ factor(ID))
    A <- as.matrix(A[,-1])
  } else {
    A <- model.matrix(~ 0, data.frame(C))
  }
  l <- list(k = k, 
            group_size = array(group_size, dim = k), 
            n_edges = nrow(E), 
            node1 = E$node1, 
            node2 = E$node2, 
            group_idx = array(group_idx, dim = n), 
            m = m,
            A = A,
            inv_sqrt_scale_factor = inv_sqrt_scale_factor, 
            comp_id = nb2$comp.id)
  return(l)
}

#' Moran Coefficient
#' 
#' @param x vector of numeric values 
#' @param w connectivity matrix 
#' 
#' @author Connor Donegan
mc <- function (x, w, digits = 3, warn = TRUE) {
  if (missing(x) | missing(w)) 
    stop("Must provide data x (length n vector) and n x n spatial weights matrix (w).")
  if (any(rowSums(w) == 0)) {
    zero.idx <- which(rowSums(w) == 0)
    if (warn) 
      message(length(zero.idx), " observations with no neighbors found. They will be dropped from the data.")
    x <- x[-zero.idx]
    w <- w[-zero.idx, -zero.idx]
  }
  xbar <- mean(x)
  z <- x - xbar
  ztilde <- as.numeric(w %*% z)
  A <- sum(rowSums(w))
  n <- length(x)
  mc <- as.numeric(n/A * (z %*% ztilde)/(z %*% z))
  return(round(mc, digits = digits))
}
