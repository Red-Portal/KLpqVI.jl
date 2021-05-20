
library(bridgesampling)
library(rstan)

cores <- 20
options(mc.cores = cores)

sv_model <- stan_model("stochastic_volatility.stan",
                       model_name="stochastic_volatility")

rd_model <- stan_model("radon.stan",
                       model_name="radon")

stanfit <- sampling(sv_model,
                    data = list(Y = ier, T = nrow(ier),
                                m = ncol(ier), k = k),
                    iter = 4096, warmup = 4096, chains = 4,
                    cores = cores, seed = 1)
bridge <- bridge_sampler(stanfit, method = "warp3",
                         repetitions = 10, cores = cores)
summary(bridge)
logml(bridge)
