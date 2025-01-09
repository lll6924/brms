functions {
   #include marginalization.stan
}
data {
  int<lower=0> J;         // number of schools
  array[J] real y;              // estimated treatment effects
  array[J] real<lower=0> sigma; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
generated quantities {
  array[8] int c = {1, 2, 3, 1, 2, 3, 1, 2};
  vector[3] tr;
  tr = bincount(theta, c);
  array[2,2,2] real d = {{{2, 1},{1, 2}}, {{3, 1},{1, 3}}};
  array[2,2,2] real inv_d = batch_inverse(d);
  real logdet_d = batch_log_determinant(d);
}