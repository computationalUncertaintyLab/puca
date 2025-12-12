#mcandrew

class puca( object ):

    def __init__(self,target_y=None, past_y = None, X = None):
        self.X__input         = X
        self.target_y__input  = target_y
        self.past_y__input    = past_y
        
        self.organize_data()

    def smooth_gaussian_anchored(self,x, sigma=2.0):
        import numpy as np
        """
        Heavy 1D smoothing with a Gaussian kernel.
        - Uses reflect padding to avoid edge artifacts.
        - Forces first/last value of the smoothed series to equal the original.
        """
        x = np.asarray(x, float)
        radius = int(3 * sigma)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()

        # reflect-pad to avoid weird edge behavior
        x_pad = np.pad(x, pad_width=radius, mode="reflect")

        # smooth
        y_full = np.convolve(x_pad, kernel, mode="same")
        y = y_full[radius:-radius]

        # anchor endpoints
        y[0] = x[0]
        y[-1] = x[-1]
        return y
       
    def organize_data(self):
        import pandas as pd
        import numpy as np

        # if d is not None:
        #     d_wide = pd.pivot_table( index = ["MMWRWK"], columns = ["season"], values = ["value"], data = self.d )
        # else:
        #     pass

        # try:
        #     d_wide = d_wide.loc[ list(np.arange(40,53+1)) + list(np.arange(1,20+1))  ]
        # except KeyError:
        #     d_wide = d_wide.loc[ list(np.arange(40,52+1)) + list(np.arange(1,20+1))  ]
            
        #self.d_wide = d_wide

        #--fill in any odd banks
        #d_numeric = d_wide#.to_numpy() 

        #--we should store the centers and sclaes from past copies of y.
        #--this will be used in prediction and forecasting.
        d_numeric = self.past_y__input
        
        centers   = np.nanmean(d_numeric,axis=0)
        scales    = np.nanstd( d_numeric,axis=0).reshape(1,-1)

        self.centers = centers
        self.scales = scales

        #--now we should center and scale both past y and X (is X exists) to help us predict target_y
        if self.X__input is not None:
            d_numeric = np.hstack([self.past_y__input, self.X__input])
            
        centers   = np.nanmean(d_numeric,axis=0)
        scales    = np.nanstd( d_numeric,axis=0).reshape(1,-1)

        centers   = np.mean(centers)
        scales    = np.mean(scales)
        d_scaled  = (d_numeric - centers) / scales
        
        xdata             = pd.DataFrame(d_scaled)
        xdata.iloc[0,:]   = np.nan_to_num(xdata.iloc[0,:] , nan= np.nanmean(xdata.iloc[0,:]) )
        xdata.iloc[-1,:]  = np.nan_to_num(xdata.iloc[-1,:], nan= np.nanmean(xdata.iloc[-1,:]) )
        xdata             = xdata.interpolate().to_numpy()

        y,X = self.target_y__input, xdata

        smooth_X = []
        for column in X.T:
           smooth_X.append(self.smooth_gaussian_anchored(column,sigma=2))
        smooth_X = np.array(smooth_X).T

        self.y=y
        self.X=smooth_X #<--this includes both past_y and X. However, we should make a distinction and store those

        self.past_y_scaled = smooth_X[:, :self.past_y__input.shape[-1] ]

        return y,smooth_X

    def model(self,y,X,past_y,LMAX,forecast=False, centers = None, scales = None, anchor = None, obs=None,test=None):
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp
 
        T,S     = X.shape
        ncopies = past_y.shape[-1]

        #--smooth X
        #lam = numpyro.sample("smooth_strength", dist.Beta(2., 2.).expand([S]))
        #smooth_X = smooth_gaussian_anchored(X, lam)
        
        #--L
        L = numpyro.sample( "L", dist.Normal(0,1).expand([T,LMAX]) )

        G = L.T @ L                  # (l, l)
        off_diag = G - jnp.diag(jnp.diag(G))
        lambda_ortho = numpyro.sample( "lambda_ortho", dist.Exponential(1./2) )
        numpyro.factor("ortho_penalty", -lambda_ortho * jnp.sum(off_diag**2))

        #--W
        tau_W     = numpyro.sample("tau_W"    , dist.HalfNormal(1./2) )
        lambdas_W = numpyro.sample("lambdas_W", dist.HalfCauchy(1./2).expand([LMAX,1]) )

        # --- W: (LMAX, S) with first LMAX columns lower-triangular diag=1 ---
        # 1) Strict lower triangular in first LMAX×LMAX blocLMAX
        n_free_tri = LMAX * (LMAX - 1) // 2
        z_tri = numpyro.sample("z_W_tri", dist.Normal(0, 1).expand([n_free_tri]))

        W_block = jnp.eye(LMAX)  # (LMAX, LMAX) with ones on diag
        tri_idx = jnp.tril_indices(LMAX, k=-1)
        W_block = W_block.at[tri_idx].set(z_tri)

        # 2) Remaining columns (LMAX × (S-LMAX)) fully free
        if S > LMAX:
            n_free_rest = LMAX * (S - LMAX)
            z_rest = numpyro.sample(
                "z_W_rest",
                dist.Normal(0, 1).expand([n_free_rest])
            )
            W_rest = z_rest.reshape(LMAX, S - LMAX)
            z_W = jnp.concatenate([W_block, W_rest], axis=1)  # (LMAX, S)
        else:
            z_W = W_block  # S == LMAX

        W   = numpyro.deterministic("W_raw",z_W*tau_W*lambdas_W)

        Xhat = numpyro.deterministic( "Xhat", (L @ W) )

        present_X = ~jnp.isnan(X)
        X_filled = jnp.where(present_X, X, 0.0)

        sigma_X = numpyro.sample("sigma_X", dist.HalfNormal(1./2))
        with numpyro.handlers.mask(mask=present_X):
            numpyro.sample("obs_X", dist.Normal( Xhat, sigma_X ), obs = X_filled )

        #--y----------------------------------------------------------------------

        beta_scale = numpyro.sample("beta_scale", dist.HalfNormal(0.5))
        z_beta     = numpyro.sample("z_beta", dist.Normal(0, 1).expand([LMAX, 1]))
        betas      = numpyro.deterministic("betas", z_beta * beta_scale)

        yhat_scaled  = numpyro.deterministic( "y_scaled",   (L @ betas).reshape(-1,) )

        err_sigma = numpyro.sample("err_sigma", dist.HalfNormal(1./10))
        sigma_y = numpyro.sample("sigma_y", dist.HalfCauchy(1))

        def rw_cov_jax(T, sigma2):
            idx = jnp.arange(1, T + 1)
            M = jnp.minimum(idx[:, None], idx[None, :])   # shape (T, T)
            return sigma2 * M 

        Cobs = rw_cov_jax(len(obs) ,err_sigma)
        Cobs = Cobs + (sigma_y) * jnp.eye(len(obs))

        #--multi task learning approach for cetner and scales
        w = numpyro.sample("w", dist.Dirichlet((1./ncopies)*jnp.ones(ncopies)))

        center = jnp.sum(centers*w)
        scale  = jnp.sum(scales*w)

        y_data_scaled = (y[obs] - center)/scale

        numpyro.sample("obs_y", dist.MultivariateNormal( yhat_scaled[obs], Cobs ), obs = y_data_scaled[obs])

        #--anchor
        if anchor is not None:
            numpyro.sample("anchor", dist.Normal( anchor[0],anchor[1] ), obs = yhat_scaled[-1])

        if forecast:
            idx_obs  = obs
            idx_test = test

            C = rw_cov_jax(T ,err_sigma)
            
            KOO = C[idx_obs][:, idx_obs]
            KOO = KOO + jnp.eye(len(KOO))*sigma_y
            
            KTT = C[idx_test][:, idx_test]
            KOT = C[idx_obs][:, idx_test]

            r_obs = y_data_scaled[idx_obs] - yhat_scaled[idx_obs] #<-y is scaled here 

            # conditional residual mean and covariance
            alpha = jnp.linalg.solve(KOO, r_obs)
            mean_resid_star = KOT.T @ alpha
            cov_resid_star  = KTT - KOT.T @ jnp.linalg.solve(KOO, KOT)

            # full predictive mean at test points (scaled)
            mean_star = yhat_scaled[idx_test] + mean_resid_star

            yhat_scaled_pred = numpyro.sample("yhat_scaled_pred",dist.MultivariateNormal(mean_star, cov_resid_star))

            numpyro.deterministic("forecast_scaled",jnp.concatenate([y_data_scaled[idx_obs], yhat_scaled_pred], axis=0))

            natural_y_predictions = yhat_scaled_pred*scale + center

            numpyro.deterministic("forecast_natural", jnp.clip( jnp.concatenate([y[idx_obs], natural_y_predictions ], axis=0), 0, None))

    def fit(self, estimate_lmax=True,fixed_factors=None, use_anchor=False):
        import jax
        import numpy as np
        from numpyro.infer import MCMC, NUTS
        jax.clear_caches()

        y,X    = self.y, self.X
        past_y = self.past_y_scaled

        if estimate_lmax:
            try:
                u,s,vt = np.linalg.svd(X)
                splain = np.cumsum(s**2)/np.sum(s**2)

                estimated_factors = ( np.min(np.argwhere(splain>.95)) + 1) + 1 #--one additional factor
            except:
                estimated_factors = 3
        else:
            estimated_factors = fixed_factors
        self.estimated_factors = estimated_factors

        if use_anchor:
            anchor = ( np.nanmean(past_y[-1,:]), np.nanstd(past_y[-1,:]) )
        else:
            anchor=None
        self.anchor = anchor

        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_warmup=3000, num_samples=10000, num_chains=1)
        mcmc.run(jax.random.PRNGKey(20200320)
                 ,y                     = y
                 ,X                     = X
                 ,past_y                = past_y
                 ,LMAX                  = estimated_factors
                 ,forecast              = False
                 ,centers               = self.centers.reshape(-1,)
                 ,scales                = self.scales[0].reshape(-1,)
                 , obs                  = np.where(~np.isnan(y)) [0]
                 , test                 = np.where( np.isnan(y)) [0]
                 ,anchor = anchor)

        mcmc.print_summary()

        post_samples      = mcmc.get_samples()
        self.post_samples = post_samples

        return self

    def forecast(self):
        from numpyro.infer import Predictive
        import jax
        import numpy as np 
        
        y,X = self.y, self.X
        past_y = self.past_y_scaled

        predictive   = Predictive(self.model, posterior_samples=self.post_samples)
        pred_samples = predictive(jax.random.PRNGKey(20200320)
                                  , y=y
                                  , X=X
                                  , past_y                = past_y
                                  , LMAX=self.estimated_factors
                                  , forecast              = True
                                  , centers               = self.centers.reshape(-1,)
                                  , scales                = self.scales[0].reshape(-1,)
                                  , anchor                = self.anchor
                                  , obs                   = np.where(~np.isnan(y)) [0]
                                  , test                  = np.where( np.isnan(y)) [0])
        self.forecasts = pred_samples
        return pred_samples

if __name__ == "__main__":

    pass
