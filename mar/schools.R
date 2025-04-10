library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
schools_dat <- list(J = 8,
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

fit <- stan(file = 'schools.stan', data = schools_dat)

print(fit)
plot(fit)