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
        vector[N] F = 1 / s_u + bincount(Z .* Z, J) / s_y;
        vector[N] x = bincount(z.* Z /s_y, J);
        real a = sum(log(F)) + log(s_u) * N + log(s_y) * n_obs;
        real c = sum(z .* z / s_y) - sum(x .* x ./ F);
        return - (a + c) / 2;
        //matrix[1,1] L = rep_matrix(1,1,1);
        //array[1] vector[size(Y)] ZZ;
        //ZZ[1] = Z;
        //return multi_normal_id_glm_marginalized_lpdf(Y| Xc, mu, b, sigma, J, N, M, sd, L, ZZ);
    }

    real multi_normal_id_glm_marginalized_lpdf(vector Y, matrix Xc, vector mu, vector b, real sigma, array[] int J, int N, int M, vector sd, matrix L, array[] vector Z){
        int n_obs = size(Y);
        vector[n_obs] z = Y - mu - Xc * b;
        matrix[M,M] s_u = multiply_lower_tri_self_transpose(diag_pre_multiply(sd, L));
        matrix[M,M] inv_s_u = inverse(s_u);
        real s_y = sigma * sigma;
        array[N,M,M] real F;
        for (m1 in 1:M){
            for (m2 in 1:M){
                F[:,m1,m2] = to_array_1d(bincount(Z[m1] .* Z[m2], J) / s_y);
            }
        }
        for (n in 1:N){
            F[n] = to_array_2d(to_matrix(F[n]) + inv_s_u);
        }
        array[N,M,M] real F_inv = batch_inverse(F);
        array[N,M] real x;
        for (m in 1:M){
            x[:,m] = to_array_1d(bincount(z .* Z[m] / s_y, J));
        }
        real a = batch_log_determinant(F) + log_determinant(s_u) * N + log(s_y) * n_obs;
        real c = sum(z .* z / s_y);
        for (n in 1:N){
            c = c - quad_form(to_matrix(F_inv[n]), to_vector(x[n]));
        }
        return - (a + c) / 2;
    }

    array[] vector normal_id_glm_marginalized_recover_rng(vector Y, matrix Xc, vector mu, vector b, real sigma, array[] int J, int N, int M, vector sd, vector Z){
        int n_obs = size(Y);
        vector[n_obs] z = Y - mu - Xc * b;
        real s_u = sd[1] * sd[1];
        real s_y = sigma * sigma;
        vector[N] G = bincount(Z .* Z, J) / s_y;
        vector[N] F = 1 / s_u + G;
        vector[N] x = bincount(z .* Z /s_y, J);
        vector[N] Mu = s_u * (1 - G ./ F) .* x;
        vector[N] L = sqrt(1/F);
        array[M] vector[N] ans;
        for (n in 1:N) {
            ans[1,n] = normal_rng(Mu[n], L[n]);
        }
        return ans;
    }

    matrix multi_normal_id_glm_marginalized_recover_rng(vector Y, matrix Xc, vector mu, vector b, real sigma, array[] int J, int N, int M, vector sd, matrix L, array[] vector Z){
        int n_obs = size(Y);
        vector[n_obs] z = Y - mu - Xc * b;
        matrix[M,M] s_u = multiply_lower_tri_self_transpose(diag_pre_multiply(sd, L));
        matrix[M,M] inv_s_u = inverse(s_u);
        real s_y = sigma * sigma;
        array[N,M,M] real G;
        for (m1 in 1:M){
            for (m2 in 1:M){
                G[:,m1,m2] = to_array_1d(bincount(Z[m1] .* Z[m2], J)/ s_y);
            }
        }
        array[N,M,M] real F;
        for (n in 1:N){
            F[n] = to_array_2d(to_matrix(G[n]) + inv_s_u);
        }
        array[N,M,M] real F_inv = batch_inverse(F);
        array[N,M] real x;
        for (m in 1:M){
            x[:,m] = to_array_1d(bincount(z .* Z[m] / s_y, J));
        }
        matrix[M,N] ans;
        for (n in 1:N){
            ans[:,n] = multi_normal_rng(s_u * (identity_matrix(M) - to_matrix(G[n]) * to_matrix(F_inv[n])) * to_vector(x[n]), to_matrix(F_inv[n]));
        }
        return ans;
    }