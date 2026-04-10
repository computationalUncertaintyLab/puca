

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

numpyro.enable_validation(True)

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta

# Substeps per observation interval (e.g. week). dt = 1/n_substeps; total time still T.
class puca( object ):
    
    euler_substeps = 1

    def __init__(self
                 , y                 = None
                 , Y                 = None
                 , X                 = None
                 ,anchor             = None):

        self.X__input          = X
        self.y__input          = y
        self.Y__input          = Y
        self.anchor            = anchor  
        
        self.organize_data()

    def organize_data(self):
        def smooth_gaussian_anchored_nan_safe(x, sigma=1.0, keep_nan_positions=True):
            x = np.asarray(x, float)
            is_1d = (x.ndim == 1)
            if is_1d:
                x = x.reshape(-1, 1)

            radius = int(3 * sigma)
            t = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (t / sigma) ** 2)
            kernel /= kernel.sum()

            mask = np.isfinite(x).astype(float)
            x0   = np.where(np.isfinite(x), x, 0.0)

            # pad both signal and mask the same way
            x_pad = np.pad(x0,   ((radius, radius), (0, 0)), mode="reflect")
            m_pad = np.pad(mask, ((radius, radius), (0, 0)), mode="reflect")

            y = np.zeros_like(x0)
            for i in range(x.shape[1]):
                num_full = np.convolve(x_pad[:, i], kernel, mode="same")
                den_full = np.convolve(m_pad[:, i], kernel, mode="same")

                num = num_full[radius:-radius]
                den = den_full[radius:-radius]

                yi = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
                y[:, i] = yi

            # anchor endpoints only if they exist (finite)
            for i in range(x.shape[1]):
                if np.isfinite(x[0, i]):  y[0, i]  = x[0, i]
                if np.isfinite(x[-1, i]): y[-1, i] = x[-1, i]

            if keep_nan_positions:
                y = np.where(np.isfinite(x), y, np.nan)

            return y.ravel() if is_1d else y

        def smooth_y_data(y,Y):
            all_y             = np.array([])
            smooth_ys         = [] 
            y_means, y_scales = [] , []

            for n,(past,current) in enumerate(zip(Y.T,y)):
                means  = np.mean(past,axis=0)
                scales =  np.std(past,axis=0)

                y_means.append(means)
                y_scales.append(scales)

                smooth_y       =  smooth_gaussian_anchored_nan_safe(x=past,sigma=1)
                smooth_ys.append(smooth_y)

                if n==0:
                    all_y = np.array([smooth_y])
                else:
                    all_y = np.vstack([all_y,smooth_y])
            return all_y.T

        def center_scale(y):
            global_mu  = np.nanmean( np.nanmean(y,0))
            global_std = np.nanmean( np.nanstd(y,0))
            y = (y - global_mu) / global_std

            return y, (global_mu, global_std)

        #--ROUTINE STARTED
        y           =  self.y__input[0] #<-- for now we assume one target only
        Y           =  self.Y__input[0]
        X           =  None  #self.X__input

        #--This block standardizes Y data to z-scores, collects the mean and sd, and smooths past Ys.
        smoothed_ys                                             = smooth_y_data( y,Y )

        #--remove the current season from this computation
        centered_smoothed_ys, (self.global_mu, self.global_std) = center_scale( smoothed_ys )

        #register( centered_smoothed_ys )

        #--record the last observations from the current season y
        tobs =  int( min(np.argwhere(np.isnan(y))) )
        self.tobs = tobs

        #--record number of copies of y from past seasons
        copies      = [y.shape[-1] for y in Y]
        self.copies = copies
            
        #--STORE items
        self.T        = Y.shape[0]  #<--assumption here is Y must be same row for all items in list

        self.y      = y                     #<--Target
        self.Y      = centered_smoothed_ys  #<--remove the current season
        self.X      = X                         #<--covariate information 

        return y, Y, X

    @staticmethod
    def model( y          = None
              ,X          = None
              ,anchor     = None 
              ,global_mu  = None
              ,global_std = None
              ,forecast   = False):
        eps = 10**-6

        T, S  = X.shape
        S     = S+1 #<--adding in the y season

        #W     = B.shape[-1]
        L    = 1

        
        def sir_euler_incidence(
            S0,
            I0,
            R0,
            beta,
            gamma,
            T,
            n_substeps,
        ):
            """
            Euler integration of an SIR model with cumulative infections C.

            Uses n_substeps per observation interval (dt = 1/n_substeps) so the
            same calendar length T is covered with finer steps. Cumulative C is
            then read off at the original integer times t = 0,1,...,T (indices
            0, n_substeps, 2*n_substeps, ...); incidence per interval is the
            difference (equivalent to summing sub-step dC, since C is integrated
            linearly in the Euler steps).

            Returns
            -------
            inc : length T
                New infections over each original time interval [t, t+1).
            """
            N = S0 + I0 + R0
            C0 = 0.0
            n_steps = T * n_substeps
            dt = 1.0 / float(n_substeps)
            beta = jnp.repeat(beta, n_steps)

            def step_fn(state, array):
                S, I, R, C = state
                beta_t     = array

                new_inf = beta_t * S * I / N
                new_rec = gamma * I

                S_next = S - dt * new_inf
                I_next = I + dt * (new_inf - new_rec)
                R_next = R + dt * new_rec
                C_next = C + dt * new_inf

                return (S_next, I_next, R_next, C_next), C_next

            init_state = (S0, I0, R0, C0)
            _, C_path = jax.lax.scan(step_fn, init_state, xs=beta)

            C_path        = jnp.concatenate([jnp.array([C0]), C_path])
            #idx           = jnp.arange(T + 1, dtype=jnp.int32) * int(n_substeps)
            #C_at_integers = C_path[idx]
            #inc           = jnp.diff(C_at_integers)
            inc           = jnp.diff(C_path)
            return inc

        #T,S = y.shape

        # ------------------------------------------------------------
        # Hierarchical repo: repo_s[s] ~ lognormal(log(repo_global), repo_scale_local)
        # ------------------------------------------------------------
        logit_I0 = numpyro.sample("logit_I0", dist.Normal(-4.0, 1.0))
        I0       = jax.nn.sigmoid(logit_I0)
        S0       = 1.0 - I0


        #--For A
        repo_global = numpyro.sample("repo_global", dist.Gamma(2, 1.0))
        repo_scale_local = numpyro.sample("repo_scale_local",dist.HalfNormal(0.10))
        with numpyro.plate("season_repo", S):
            z_repo = numpyro.sample("z_repo", dist.Normal(0.0, 1.0))
        repo_s = jnp.exp(jnp.log(repo_global) + repo_scale_local * z_repo)
        numpyro.deterministic("repo_s", repo_s)

        gamma = numpyro.sample("gamma", dist.Gamma(6, 6))
        beta_s = repo_s * gamma

        inca = jax.vmap(
            lambda b: sir_euler_incidence(
                S0=S0,
                I0=I0,
                R0=0.0,
                beta=b,
                gamma=gamma,
                T=T,
                n_substeps=puca.euler_substeps), in_axes=0)(beta_s)
        numpyro.deterministic("inca", inca)

        #--For B
        repo_globalB = numpyro.sample("repo_globalB", dist.Gamma(2, 1.0))
        repo_scale_localB = numpyro.sample("repo_scale_localB",dist.HalfNormal(0.1))
        with numpyro.plate("season_repo", S):
            z_repoB = numpyro.sample("z_repoB", dist.Normal(0.0, 1.0))
        repo_sB = jnp.exp(jnp.log(repo_globalB) + repo_scale_localB * z_repoB)
        numpyro.deterministic("repo_sB", repo_sB)

        gammaB = numpyro.sample("gammaB", dist.Gamma(3, 3))
        beta_sB = repo_sB * gammaB

        incb = jax.vmap(
            lambda b: sir_euler_incidence(
                S0=S0,
                I0=I0,
                R0=0.0,
                beta=b,
                gamma=gammaB,
                T=T,
                n_substeps=puca.euler_substeps), in_axes=0)(beta_sB)
        numpyro.deterministic("incb", incb)

        #--time warping
        peaks       = jnp.nanargmax(X, axis=0)
        peak_deltas = (T - 2 * peaks)/2. #<-- pretty sure this should be (T-2*peak)/2

        peakb_deltas = anchor

        #print(jnp.nanmean(peak_deltas))
        #print(jnp.nanmean(peakb_deltas))
        

        delta_std = numpyro.sample("delta_std"    , dist.HalfNormal(1.0))
        delta_z   = numpyro.sample("delta_fit_z"  , dist.Normal(0.0, 1.0))
        delta     = jnp.mean(peak_deltas) + delta_z * delta_std
        deltas    = numpyro.deterministic("deltas", jnp.append(peak_deltas, delta))
        

        delta_stdB = numpyro.sample("delta_stdB"     , dist.HalfNormal(1.0))
        delta_zB   = numpyro.sample("delta_fit_zB"   , dist.Normal(0.0, 1.0))
        deltaB     = jnp.nanmean(peakb_deltas) + delta_zB * delta_stdB

        #deltaB = numpyro.sample("deltaB", dist.Uniform(-0.5*T,0.5*T))
        
        deltasB    = numpyro.deterministic("deltasB" , jnp.append(peakb_deltas,  deltaB))


        calendar_time = jnp.arange(0, T)
        original_taus = (2 * calendar_time + T + 0) / (2 * T)

        #--For A
        ha = (2 * calendar_time[:, None] + T + deltas[None, :]) / (2 * T)
        ha = numpyro.deterministic("ha", ha)

        main_trenda = jax.vmap( lambda hh, inc_row: jnp.interp(hh, original_taus, inc_row),in_axes=(1, 0),)(ha, inca)
        main_trenda = main_trenda.T
        numpyro.deterministic("main_trenda", main_trenda)

        #--For B
        hb = (2 * calendar_time[:, None] + T + deltasB[None, :]) / (2 * T)
        hb = numpyro.deterministic("hb", hb)
        
        main_trendb = jax.vmap( lambda hh, inc_row: jnp.interp(hh, original_taus, inc_row),in_axes=(1, 0),)(hb, incb)
        main_trendb = main_trendb.T
        numpyro.deterministic("main_trendb", main_trendb)


        mu_log_a  = numpyro.sample("mu_log_a", dist.Normal(0.0, 1.0))
        tau_log_a = numpyro.sample("tau_log_a", dist.HalfNormal(0.5))

        mu_log_b  = numpyro.sample("mu_log_b", dist.Normal(0.0, 1.0))
        tau_log_b = numpyro.sample("tau_log_b", dist.HalfNormal(0.5))
       
        mu_int      = numpyro.sample("mu_int", dist.Normal(0.0, 0.1))
        tau_int     = numpyro.sample("tau_int", dist.HalfNormal(0.5))
        
        with numpyro.plate("season_ab", S):
            #--For A
            z_log_a = numpyro.sample("z_log_a", dist.Normal(0.0, 1.0))
            log_a_s = mu_log_a + tau_log_a * z_log_a
            a_s     = numpyro.deterministic("a_s", jnp.exp(log_a_s))

            #--For B
            z_log_b = numpyro.sample("z_log_b", dist.Normal(0.0, 1.0))
            log_b_s = (mu_log_b-1) + tau_log_b * z_log_b
            b_s     = numpyro.deterministic("b_s", jnp.exp(log_b_s))

            #--intercept
            z_int     = numpyro.sample("int_b"       , dist.Normal(0.0, 1.0))
            int_s     = numpyro.deterministic("int_s", mu_int + tau_int * z_int)

        mu = numpyro.deterministic("mu", (a_s[None, :] * main_trenda) +  (b_s[None, :] * main_trendb)  + int_s[None, :])

        trend = mu

        # Hierarchical observation noise (per season)
        #mu_log_sigma = numpyro.sample("mu_log_sigma", dist.Normal(-2.0, jnp.sqrt(2)/2))
        mu_log_sigma = numpyro.sample("mu_log_sigma", dist.HalfNormal(1))
        with numpyro.plate("season", S):
            tau_log_sigma = numpyro.sample("tau_log_sigma", dist.HalfNormal( 0.1/2 ))
            #tau_log_sigma = numpyro.sample("tau_log_sigma", dist.StudentT(df=3,loc=0,scale=1./10))
            #sigma         = (mu_log_sigma + jnp.abs(tau_log_sigma))
            sigma         = (mu_log_sigma + tau_log_sigma)
            #z_sigma = numpyro.sample("z_sigma", dist.Normal(0,1))
            #sigma   = numpyro.deterministic("sigma_s", jnp.exp(mu_log_sigma + tau_log_sigma*z_sigma))

        #sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        #--X likelihood
        numpyro.sample( "llx", dist.Normal(trend[:,:-1], sigma[:-1]), obs = X.reshape(T,S-1) )

        #--y likelihood
        with numpyro.handlers.mask(mask=jnp.isfinite(y.reshape(T,))):
            numpyro.sample( "lly", dist.Normal(trend[:,-1].reshape(T,), sigma[-1]), obs = y.reshape(T,) )

        if forecast:
            forecast = numpyro.sample("forecast", dist.Normal(trend[:,-1], sigma[-1]))
            numpyro.deterministic("y_pred", forecast * global_std + global_mu)

    def fit(self
            , M                          = 0
            , estimated_num_components_y = None):

        y, Y, X     = self.y, self.Y, self.X
        self.M      = M

        #--MCMC start
        dense_blocks = [
            ("repo_global", "gamma", "logit_I0", "repo_scale_local"),
        ]

        from patsy import dmatrix
        nuts_kernel = NUTS(self.model
                           , init_strategy = init_to_median(num_samples=100)
                           , dense_mass = dense_blocks
                           ,  find_heuristic_step_size=True)     
        kernel      = nuts_kernel 
        mcmc        = MCMC(kernel
                    , num_warmup     = 1000
                    , num_samples    = 1000
                    , num_chains     = 1
                    , jit_model_args = False)

        mcmc.run(jax.random.PRNGKey(20200320)
                              ,X            = Y
                              ,y            = (y - self.global_mu) / self.global_std
                              ,anchor =  self.anchor
                              ,global_mu    = self.global_mu
                              ,global_std   = self.global_std
                              ,forecast     = None 
                              ,extra_fields = ("diverging", "num_steps", "accept_prob", "energy","adapt_state.step_size"))

        self.mcmc = mcmc
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.posterior_samples = samples

        return self

    def forecast(self):

        #--MCMC START
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred"] )
        #--MCMC END

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,X            =  self.Y
                              ,y            = (self.y - self.global_mu) / self.global_std
                              ,anchor =  self.anchor
                              ,global_mu    = self.global_mu
                              ,global_std   = self.global_std
                              ,forecast     = True
                                  )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        yhat_draws = yhat_draws.squeeze()

        forecasts = yhat_draws
        
        self.pred_samples = pred_samples
        self.forecast     = forecasts
        return forecasts


if __name__ == "__main__":

    pass
