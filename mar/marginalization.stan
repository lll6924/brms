    vector bincount(vector elements, array[] int J) {
        vector[max(J)] res = rep_vector(0, max(J));
        for (n in 1:size(J)) {
          res[J[n]] = res[J[n]] + elements[n];
        }
        return res;
    }
    array[,,] real batch_inverse(array[,,] real ar){
        array[size(ar),size(ar[1]),size(ar[1])] real res;
        for (n in 1:size(ar)) {
            res[n] = to_array_2d(inverse(to_matrix(ar[n])));
        }
        return res;
    }
    real batch_log_determinant(array[,,] real ar){
        real res = 0;
        for (n in 1:size(ar)) {
            res = res + log_determinant(to_matrix(ar[n]));
        }
        return res;
    }
    real normal_id_glm_marginalized_lpdf(vector Y, matrix Xc, vector mu, vector b, real sigma, array[] int J, int N, int M, vector sd, vector Z){
        int n_obs = size(Y);
        vector[n_obs] z = Y - mu - Xc * b;
        real s_u = sd[1] * sd[1];
        real s_y = sigma * sigma;
        vector[N] F = 1 / s_u + bincount(rep_vector(1, n_obs), J) / s_y;
        vector[N] x = bincount(z/s_y, J);
        real a = sum(log(F)) + log(s_u) * N + log(s_y) * n_obs;
        real c = sum(z .* z / s_y) - sum(x .* x ./ F);
        return - (a + c) / 2;
    }
    array[] vector normal_id_glm_marginalized_recover_rng(vector Y, matrix Xc, vector mu, vector b, real sigma, array[] int J, int N, int M, vector sd, vector Z){
        int n_obs = size(Y);
        vector[n_obs] z = Y - mu - Xc * b;
        real s_u = sd[1] * sd[1];
        real s_y = sigma * sigma;
        vector[N] G = bincount(rep_vector(1, n_obs), J) / s_y;
        vector[N] F = 1 / s_u + G;
        vector[N] x = bincount(z/s_y, J);
        vector[N] Mu = s_u * (1 - G ./ F) .* x;
        vector[N] L = s_u * sqrt(1/s_u - G + G ./ F .* G);
        array[M] vector[N] ans;
        for (n in 1:N) {
            ans[1,n] = normal_rng(Mu[n], L[n]);
        }
        return ans;
    }