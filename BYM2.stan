// The BYM2 model //
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
  vector[k] inv_sqrt_scale_factor; // ICAR scale factor, with singletons represented by 1
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
  vector[n] theta_tilde;  
  real<lower=0> spatial_scale;
  real<lower=0,upper=1> rho;
}

transformed parameters {
  vector[n] convolution;
  vector[n] eta;
  convolution = convolve_bym2(phi_tilde, theta_tilde, spatial_scale, n, k, group_size, group_idx, rho, inv_sqrt_scale_factor);
  eta = offset + alpha + convolution;
}

model {
   phi_tilde ~ icar_normal(spatial_scale, node1, node2, k, group_size, group_idx, has_theta);
   theta_tilde ~ std_normal();
   spatial_scale ~ std_normal();
   rho ~ beta(1,1);
   alpha ~ std_normal();
   if (!prior_only) y ~ poisson_log(eta);
}

generated quantities {
  vector[n] resid;
  for (i in 1:n) resid[i] = exp(eta[i]) - y[i];
}

