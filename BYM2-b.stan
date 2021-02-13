functions {
  /**
   * Return the log probability density of the specified vector of
   * coefficients under the ICAR model with unit variance, where
   * adjacency is determined by the adjacency array and the spatial
   * structure is a disconnected graph which has at least one
   * connected component.  The spatial structure is described by
   * an array of component sizes and a corresponding 2-D array
   * where each row contains the indices of the nodes in that
   * component.  The adjacency array contains two parallel arrays
   * of adjacent element indexes (i.e. edges in the component graph).
   *
   * For example, a series of four coefficients phi[1:4] for a
   * disconnected graph containing 1 singleton would have
   * adjacency array {{1, 2}, {2, 3}}, signaling that coefficient 1
   * is adjacent to coefficient 2, and 2 is adjacent to 3,
   * component size array {3, 1}, and (zero-padded) component members
   * array of arrays { { 1, 2, 3, 0}, {4, 0, 0, 0} }.
   *
   * Each connected component has a soft sum-to-zero constraint.
   * Singleton components don't contribute to the ICAR model.
   *
   * @param phi vector of varying effects
   * @param adjacency parallel arrays of indexes of adjacent elements of phi
   * @param comp_size array of sizes of components in spatial structure graph
   * @param comp_members array of arrays of per-components coefficients.
   *
   * @return ICAR log probability density
   *
   * @reject if the the adjacency matrix does not have two rows
   * @reject if size mismatch between comp_size and comp_members
   * @reject if size mismatch between phi and dimension 2 of comp_members
   *
   * author: Mitzi Morris
   */
  real standard_icar_disconnected_lpdf(vector phi,
				       int[ , ] adjacency,
				       int[ ] comp_size,
				       int[ , ] comp_members) {
    if (size(adjacency) != 2)
      reject("require 2 rows for adjacency array;",
             " found rows = ", size(adjacency));
    if (size(comp_size) != size(comp_members))
      reject("require ", size(comp_size), " rows for members matrix;",
             " found rows = ", size(comp_members));
    if (size(phi) != dims(comp_members)[2])
      reject("require ", size(phi), " columns for members matrix;",
             " found columns = ", dims(comp_members)[2]);

    real total = 0;
    for (n in 1:size(comp_size)) {
      if (comp_size[n] > 1)
	total += -0.5 * dot_self(phi[adjacency[1, comp_members[n, 1:comp_size[n]]]] -
				 phi[adjacency[2, comp_members[n, 1:comp_size[n]]]])
	  + normal_lpdf(sum(phi[comp_members[n, 1:comp_size[n]]]) | 0, 0.001 * comp_size[n]);
    }
    return total;
  }
}
data {
  // spatial structure
  int<lower = 0> I;  // number of nodes
  int<lower = 0> J;  // number of edges
  int<lower = 1, upper = I> edges[2, J];  // node[1, j] adjacent to node[2, j]

  int<lower=0, upper=I> K;  // number of components in spatial graph
  int<lower=0, upper=K> K_size[K];   // component sizes
  int<lower=0, upper=I> K_members[K, I];  // rows contain per-component areal region indices

  vector<lower=0>[K] tau_sp; // per-component scaling factor, 0 for singletons
}
parameters {
  // spatial effects
  real<lower = 0> sigma_sp;  // scale of spatial effects
  real<lower = 0, upper = 1> rho_sp;  // proportion of spatial effect that's spatially smoothed
  vector[I] theta_sp;  // standardized heterogeneous spatial effects
  vector[I] phi_sp;  // standardized spatially smoothed spatial effects

}
transformed parameters {
  vector[I] gamma;
  // each component has its own spatial smoothing
  for (k in 1:K) {
    if (K_size[k] == 1) {
      gamma[K_members[k,1]] = theta_sp[K_members[k,1]] * sigma_sp;
    } else {
      gamma[K_members[k, 1:K_size[k]]] = 
	    (sqrt(1 - rho_sp) * theta_sp[K_members[k, 1:K_size[k]]]
	     + (sqrt(rho_sp) * sqrt(1 / tau_sp[k])
		* phi_sp[K_members[k, 1:K_size[k]]]) * sigma_sp);
    }
  }
}
model {
  // spatial hyperpriors and priors
  sigma_sp ~ normal(0, 1);
  rho_sp ~ beta(0.5, 0.5);
  theta_sp ~ normal(0, 1);
  phi_sp ~ standard_icar_disconnected(edges, K_size, K_members);
}
