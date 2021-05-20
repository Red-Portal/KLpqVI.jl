
library(bridgesampling)
library(rstan)

cores <- 4
options(mc.cores = cores)

data("radon.data")
rd_model <- stan_model("radon.stan",
                       model_name="radon")
source("radon.data.R")

rd_stanfit <- sampling(rd_model,
                       data = list(N = N,
                                   county = county,
                                   x = x,
                                   y = y),
                       iter = 8192, warmup = 4096, chains = 20,
                       cores = cores, seed = 1)
rd_bridge <- bridge_sampler(rd_stanfit, method = "warp3",
                            repetitions = 128, cores = cores)

sv_model <- stan_model("stochastic_volatility.stan",
                       model_name="stochastic_volatility")

rd_model <- stan_model("stochastic_volatility.stan",
                       model_name="sv")
source("stochastic_volatility.data.R")

rd_stanfit <- sampling(rd_model,
                       data = list(y = y,
                                   T = T),
                       control = list(adapt_delta=0.95,
                                      max_treedepth=12),
                       iter = 4096,
                       warmup = 2048,
                       chains = 20,
                       cores = cores,
                       seed = 1)
sv_bridge <- bridge_sampler(rd_stanfit, method = "warp3",
                            repetitions = 20, cores = cores)
