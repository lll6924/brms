library(devtools)
load_all('../')
fit1 <- brm(time ~ age + (age+1|disease*sex) + (1|patient),
            data = kidney, family = "gaussian",   prior = set_prior("cauchy(0,2)", class = "sd"), marginalize = 'disease:sex')
summary(fit1)
stancode(fit1)
