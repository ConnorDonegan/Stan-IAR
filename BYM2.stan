// BYM2 model with possibly disconnected group structure and islands
data {
  int n;    // no. observations
  int k;    // no. groups
  int group_size[k]; // group sizes---to extract data by group
  int<lower=0> n_edges; // total number of edges
  int<lower=0> node1[n_edges];
  int<lower=0> node2[n_edges];
  int group_idx[n]; // the first k=group_size[1] elements contain the k indices for group 1: phi[group_idx[1:group_size[1]]] is group 1
  vector[n] scale_factor; // will repeat same value for all observations by group;
  int<lower=0, upper=1> prior_only;
  int y[n];
  vector[n] offset;
}

transformed data {
  vector[n] log_off = log(offset);
}

parameters {
  real alpha;
  real<lower=0> sigma_re;
  vector[n] phi;
  vector[n] theta;
  real logit_rho;
}

transformed parameters {
  vector[n] convolved_re;
  real<lower=0, upper=1> rho;
  rho = inv_logit(logit_rho);
  convolved_re = sqrt(rho * inv(scale_factor)) .* phi + sqrt(1 - rho) * theta;      
}

model {
  int pos;
    pos = 1;
    for (j in 1:k) {
      sum(phi[segment(group_idx, pos, group_size[j])]) ~ normal(0, 0.001 * group_size[j]);
      pos = pos + group_size[j];
    }
   target += -0.5 * dot_self(phi[node1] - phi[node2]);
   sigma_re ~ std_normal();
   theta ~ std_normal();
   logit_rho ~ std_normal();
   alpha ~ std_normal();
   if (!prior_only) y ~ poisson_log(log_off + alpha + convolved_re * sigma_re);
}
