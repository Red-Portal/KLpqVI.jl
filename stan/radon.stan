data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] x;
  vector[N] y;
  real<lower=0,upper=1> t;
} 
parameters {
  vector[85] a1;
  vector[85] a2;
  real mu_a1;
  real mu_a2;
  real<lower=2.22e-16> sigma_a1;
  real<lower=2.22e-16> sigma_a2;
  real<lower=2.22e-16> sigma_y;
}
transformed parameters {
  vector[N] y_hat;
  real loglikelihood;

  for (i in 1:N)
    y_hat[i] = a1[county[i]] + a2[county[i]] * x[i];

  loglikelihood = normal_lpdf(y | y_hat, sigma_y);
} 
model {
  sigma_a1 ~ Gamma(1, 0.02)
  sigma_a2 ~ Gamma(1, 0.02)
  sigma_y  ~ Gamma(1, 0.02)

  mu_a1 ~ normal(0, 1);
  a1 ~ normal(mu_a1, sigma_a1);
  mu_a2 ~ normal(0, 1);
  a2 ~ normal(0.1 * mu_a2, sigma_a2);

  target += t*loglikelihood;
}
