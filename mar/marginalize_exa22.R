library(devtools)
load_all('../')
fit1 <- brm(bf(time ~ age + (age+1|disease*sex) + (1|patient), sigma ~ 1|disease*sex),
            iter = 20000, data = kidney, family = "gaussian",
            prior = c(set_prior("cauchy(0,2)", class = "sd"), set_prior("normal(0,1)", class="sd", group="patient"), set_prior("normal(0,1)", class="sd", group="disease")),
            marginalize = NULL)
summary(fit1)
sink('stan_true_sigma_tau.stan')
stancode(fit1)
sink()
