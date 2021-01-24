// The BYM model //
functions {
#include icar-functions.stan
}
data {
  int n;    // no. observations
  int<lower=1> k; // no. of groups
  int group_size[k]; // observational units per group
  int group_idx[n]; // index of observations, ordered by group
  int<lower=1> n_edges; 
  int<lower=1, upper=n> node1[n_edges];
  int<lower=1, upper=n> node2[n_edges];
  int<lower=1, upper=k> comp_id[n]; 
  int<lower=0, upper=1> prior_only;
  int y[n];
  vector[n] offset;
}

transformed data {
  int<lower=0,upper=1> has_theta=1;
}

parameters {
  real alpha;
  vector[n] phi_tilde;
  real<lower=0> spatial_scale;
  vector[n] theta_tilde;
  real<lower=0> theta_scale;
}

transformed parameters {
  vector[n] convolution;
  vector[n] phi = phi_tilde * spatial_scale;
  vector[n] theta = theta_tilde * theta_scale;
  convolution = convolve_bym(phi, theta, n, k, group_size, group_idx);
}

model {
   phi_tilde ~ icar_normal(node1, node2, k, group_size, group_idx, has_theta);
   theta_tilde ~ std_normal();
   spatial_scale ~ std_normal();
   theta_scale ~ std_normal();
   alpha ~ std_normal();
   if (!prior_only) y ~ poisson_log(offset + alpha + convolution);
}

