  /* multi-normal log-PDF for time-series covariance structures
   * in Cholesky parameterization and assuming homogoneous variances
   * and known standard errors
   * Args:
   *   y: response vector
   *   mu: mean parameter vector
   *   sigma: residual standard deviation
   *   se2: square of user defined standard errors
   *   chol_cor: cholesky factor of the correlation matrix
   *   nobs: number of observations in each group
   *   begin: the first observation in each group
   *   end: the last observation in each group
   * Returns:
   *   sum of the log-PDF values of all observations
   */
  real normal_time_hom_se_lpdf(vector y, vector mu, real sigma, data vector se2,
                               matrix chol_cor, int[] nobs, int[] begin, int[] end) {
    int I = size(nobs);
    vector[I] lp;
    for (i in 1:I) {
      matrix[nobs[i], nobs[i]] L;
      L = sigma * chol_cor[1:nobs[i], 1:nobs[i]];
      // need to add 'se' to the covariance matrix itself
      L = multiply_lower_tri_self_transpose(L);
      L += diag_matrix(se2[begin[i]:end[i]]);
      L = cholesky_decompose(L);
      lp[i] = multi_normal_cholesky_lpdf(
        y[begin[i]:end[i]] | mu[begin[i]:end[i]], L
      );
    }
    return sum(lp);
  }
  /* multi-normal log-PDF for time-series covariance structures
   * in Cholesky parameterization and assuming heterogenous variances
   * and known standard errors
   * Deviating Args:
   *   sigma: residual standard deviation vector
   * Returns:
   *   sum of the log-PDF values of all observations
   */
  real normal_time_het_se_lpdf(vector y, vector mu, vector sigma, data vector se2,
                               matrix chol_cor, int[] nobs, int[] begin, int[] end) {
    int I = size(nobs);
    vector[I] lp;
    for (i in 1:I) {
      matrix[nobs[i], nobs[i]] L;
      L = diag_pre_multiply(sigma[begin[i]:end[i]], chol_cor[1:nobs[i], 1:nobs[i]]);
      // need to add 'se' to the covariance matrix itself
      L = multiply_lower_tri_self_transpose(L);
      L += diag_matrix(se2[begin[i]:end[i]]);
      L = cholesky_decompose(L);
      lp[i] = multi_normal_cholesky_lpdf(
        y[begin[i]:end[i]] | mu[begin[i]:end[i]], L
      );
    }
    return sum(lp);
  }
  /* multi-normal log-PDF for time-series covariance structures
   * in Cholesky parameterization and assuming homogoneous variances
   * and known standard errors
   * allows for flexible correlation matrix subsets
   * Deviating Args:
   *   Jtime: array of time indices per group
   * Returns:
   *   sum of the log-PDF values of all observations
   */
  real normal_time_hom_se_flex_lpdf(vector y, vector mu, real sigma, data vector se2,
                                    matrix chol_cor, int[] nobs, int[] begin,
                                    int[] end, int[,] Jtime) {
    int I = size(nobs);
    vector[I] lp;
    matrix[rows(chol_cor), cols(chol_cor)] Cor;
    Cor = multiply_lower_tri_self_transpose(chol_cor);
    for (i in 1:I) {
      int iobs[nobs[i]] = Jtime[i, 1:nobs[i]];
      matrix[nobs[i], nobs[i]] Cov = sigma^2 * Cor[iobs, iobs];
      Cov += diag_matrix(se2[begin[i]:end[i]]);
      lp[i] = multi_normal_lpdf(y[begin[i]:end[i]] | mu[begin[i]:end[i]], Cov);
    }
    return sum(lp);
  }
  /* multi-normal log-PDF for time-series covariance structures
   * in Cholesky parameterization and assuming heterogenous variances
   * and known standard errors
   * allows for flexible correlation matrix subsets
   * Deviating Args:
   *   sigma: residual standard deviation vector
   *   Jtime: array of time indices per group
   * Returns:
   *   sum of the log-PDF values of all observations
   */
  real normal_time_het_se_flex_lpdf(vector y, vector mu, vector sigma, data vector se2,
                                    matrix chol_cor, int[] nobs, int[] begin,
                                    int[] end, int[,] Jtime) {
    int I = size(nobs);
    vector[I] lp;
    matrix[rows(chol_cor), cols(chol_cor)] Cor;
    Cor = multiply_lower_tri_self_transpose(chol_cor);
    for (i in 1:I) {
      int iobs[nobs[i]] = Jtime[i, 1:nobs[i]];
      matrix[nobs[i], nobs[i]] Cov = quad_form_diag(Cor[iobs, iobs], sigma[begin[i]:end[i]]);
      Cov += diag_matrix(se2[begin[i]:end[i]]);
      lp[i] = multi_normal_lpdf(y[begin[i]:end[i]] | mu[begin[i]:end[i]], Cov);
    }
    return sum(lp);
  }
