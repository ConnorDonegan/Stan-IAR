// The BYM model //
functions {
#include icar-functions.stan
}
data {
  int n;    // no. observations
  int<lower=1> k; // no. of groups
  int group_size[k]; // observational units per group
  int group_idx[n]; // index of observations, ordered by group
  int<lower=0> m; // no of components requiring additional intercepts
  matrix[n, m] A; // dummy variables for any extra graph component intercepts
  int<lower=1> n_edges; 
  int<lower=1, upper=n> node1[n_edges];
  int<lower=1, upper=n> node2[n_edges];
  int<lower=1, upper=k> comp_id[n]; 
  vector[k] inv_sqrt_scale_factor; // can be a vector of ones, as a placeholder
  int<lower=0, upper=1> prior_only;
  int y[n];
  vector[n] offset; // e.g., log of population at risk
}

transformed data {
  int<lower=0,upper=1> has_theta=1;
}

parameters {
  real alpha;
  vector[m] alpha_phi;
  vector[n] phi_tilde;
  real<lower=0> spatial_scale;
  vector[n] theta_tilde;
  real<lower=0> theta_scale;
}

transformed parameters {
  vector[n] phi = make_phi(phi_tilde, spatial_scale, inv_sqrt_scale_factor, n, k, group_size, group_idx);
  vector[n] theta = theta_tilde * theta_scale;
  vector[n] convolution = convolve_bym(phi, theta, n, k, group_size, group_idx);
  vector[n] eta = offset + alpha + convolution;
  if (m) eta += A * alpha_phi;
}

model {
   phi_tilde ~ icar_normal(spatial_scale, node1, node2, k, group_size, group_idx, has_theta);
   theta_tilde ~ std_normal();
   spatial_scale ~ std_normal(); // the standard normal priors on the *_scale parameters if designed for Poisson models (log rates)
   theta_scale ~ std_normal();
   alpha ~ normal(0, 10); // this should be adjusted for each problem. It's the mean log rate.
   if (m) alpha_phi ~ normal(0, 2);
   if (!prior_only) y ~ poisson_log(eta);
}

