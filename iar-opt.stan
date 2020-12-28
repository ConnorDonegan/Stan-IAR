// IAR, BYM, and BYM2 spatial Poisson models
data {
  int n;    // no. observations
  int k;    // no. groups
  int group_size[k]; // group sizes---to extract data by group
  int<lower=0> n_edges; // total number of edges
  int<lower=0> node1[n_edges];
  int<lower=0> node2[n_edges];
  int group_idx[n]; // the first k=group_size[1] elements contain the k indices for group 1: phi[group_idx[1:group_size[1]]] is group 1
  vector[n] scale_factor; // will repeat same value for all observations by group;
  int<lower=1,upper=3> type; // 1=iar, 2=bym, 3=bym2
  int<lower=0, upper=1> prior_only;
  int y[n];
  vector[n] offset;
}

transformed data {
  vector[n] log_off = log(offset);
}

parameters {
  vector[n] phi;
  real<lower=0> spatial_scale;
  vector[type > 1 ? n : 0] theta;
  real<lower=0> theta_scale[type == 2 ? 1 : 0];
  real logit_rho[type == 3 ? 1 : 0];
  real alpha;
}

transformed parameters {
  vector[n] spatial;
  vector[n] f;
  real<lower=0, upper=1> rho[type == 3 ? 1 : 0];
  if (type == 1) spatial = phi * spatial_scale;
  if (type == 2) spatial = phi * spatial_scale + theta * theta_scale[1];
  if (type == 3) {
    rho[1] = inv_logit(logit_rho[1]);
    spatial = ( sqrt(rho[1] * inv(scale_factor)) .* phi + sqrt(1 - rho[1]) * theta ) * spatial_scale;
  }
  f = log_off + alpha + spatial;
}

model {
  int pos;
  pos = 1;
  if (type == 1) {
    for (j in 1:k) {
      if (group_size[j] > 1) {
        sum(phi[segment(group_idx, pos, group_size[j])]) ~ normal(0, 0.001 * group_size[j]);
	} else {
	phi[segment(group_idx, pos, group_size[j])] ~ std_normal();
        }
      pos = pos + group_size[j];
    }
  }
  if (type > 1) {
    for (j in 1:k) {
      sum(phi[segment(group_idx, pos, group_size[j])]) ~ normal(0, 0.001 * group_size[j]);
      pos = pos + group_size[j];
    }
   theta ~ std_normal();
   if (type == 2) theta_scale[1] ~ std_normal();
   if (type == 3) logit_rho[1] ~ std_normal();
   }
   target += -0.5 * dot_self(phi[node1] - phi[node2]);
   spatial_scale ~ std_normal();
  // regression/process model stuff:
   alpha ~ std_normal();
   if (!prior_only) y ~ poisson_log(f);
}

generated quantities {
  vector[n] ssre; // spatial structure
  vector[type > 1 ? n : 0] sure; // unstructure partail-pooling term
  for (i in 1:n) {
    if (type == 1) ssre[i] = phi[i] * spatial_scale;
    if (type == 2) {
      ssre[i] = phi[i] * spatial_scale;
      sure[i] = theta[i] * theta_scale[1];
    }
    if (type == 3) {
      ssre[i] = spatial_scale * sqrt(rho[1] * inv(scale_factor[i])) * phi[i];
      sure[i] = spatial_scale * sqrt(1 - rho[1]) * theta[i];
    }   
}
}

