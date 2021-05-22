
data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
  real<lower=0,upper=1> beta;
}
parameters {
  real mu;                     // mean log volatility
  real<lower=-1+1e-10,upper=1-1e-10> phi;  // persistence of volatility
  real<lower=1e-10> sigma;         // white noise shock scale
  vector[T] h_std;             // std log volatility at time t
}
transformed parameters {
  real loglikelihood;
  {
    vector[T] h;
    h = h_std * sigma;  // now h ~ normal(0, sigma)
    h[1] /= sqrt(1 - phi * phi);  // rescale h[1]
    h += mu;
    for (t in 2:T)
      h[t] += phi * (h[t-1] - mu);
    loglikelihood = normal_lpdf(y | 0, exp(h / 2));
  }
}
model {
  phi ~ uniform(-1, 1);
  sigma ~ cauchy(0, 5);
  mu ~ cauchy(0, 10);  
  h_std ~ std_normal();

  target += beta*loglikelihood;
}
