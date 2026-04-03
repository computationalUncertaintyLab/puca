

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

numpyro.enable_validation(True)

from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta

class puca( object ):

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
            dt=1.0
        ):
            """
            Euler integration of an SIR model with cumulative infections C.

            States:
                dS/dt = -beta * S * I / N
                dI/dt =  beta * S * I / N - gamma * I
                dR/dt =  gamma * I
                dC/dt =  beta * S * I / N

            Returns:
                jnp.diff(C), i.e. incident infections over each Euler step.

            Parameters
            ----------
            S0, I0, R0 : float
                Initial compartment values.
            beta : float
                Transmission rate.
            gamma : float
                Recovery rate.
            T : int
                Number of Euler steps.
            dt : float
                Step size.
            """
            N = S0 + I0 + R0
            C0 = 0.0
            beta = jnp.repeat(beta,T)

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
            _, C_path = jax.lax.scan(step_fn, init_state, xs=beta )#, length=T)

            # prepend C0 so diff gives length T
            C_path = jnp.concatenate([jnp.array([C0]), C_path])

            return jnp.diff(C_path)

        #T,S = y.shape

        M=1
        tau_grid = jnp.linspace( 0,2, 200) # The "center" here is 1

        # ------------------------------------------------------------
        # hierarchical season effects
        # ------------------------------------------------------------
        mu_log_a  = numpyro.sample("mu_log_a" , dist.Normal(0.0, 1.0))
        mu_log_b  = numpyro.sample("mu_log_b" , dist.Normal(0.0, 1.0))
        tau_log_a = numpyro.sample("tau_log_a", dist.HalfNormal(0.5 ))
        tau_log_b = numpyro.sample("tau_log_b", dist.HalfNormal(0.5 ))

        mu_b      = numpyro.sample("mu_b"     , dist.Normal(0.0, 0.1))
        tau_b     = numpyro.sample("tau_b"    , dist.HalfNormal(0.5 ))

        #main_var               = numpyro.sample("main_var", dist.Dirichlet(5*jnp.array([0.90,0.08,0.02])))
        spacing                = jnp.sqrt(tau_grid[1]) #<--this is only bc i start at zero and take same size increments.

        logit_I0 = numpyro.sample("logit_I0", dist.Normal(-4.0, 1.0))
        I0       = jax.nn.sigmoid(logit_I0)
        S0       = 1.0 - I0

        repo_a_global       = numpyro.sample("repo_a_global", dist.Gamma(2,1))
        repo_b_global       = numpyro.sample("repo_b_global", dist.Gamma(2,1))
        repo_a_scale_local  = numpyro.sample("repo_a_scale_local",dist.HalfNormal(1./2))
        repo_b_scale_local  = numpyro.sample("repo_b_scale_local",dist.HalfNormal(1./2))
        with numpyro.plate("season", S):
            z_repo_a   = numpyro.sample( "z_repo_a", dist.Normal(0,1)  )
            z_repo_b   = numpyro.sample( "z_repo_b", dist.Normal(0,1)  )
        repo_a = jnp.exp(jnp.log(repo_a_global) + repo_a_scale_local*z_repo_a)
        repo_b = jnp.exp(jnp.log(repo_b_global) + repo_b_scale_local*z_repo_b)
        
        gamma             = numpyro.sample("gamma",dist.Gamma(2,1))

        beta_a            = repo_a*gamma
        beta_b            = repo_b*gamma

       
        inc_a = jax.vmap( lambda beta: sir_euler_incidence(S0=S0, I0=I0, R0 = 0., beta=beta, gamma = gamma, T = T, dt=1) , in_axes=0 )(beta_a)
        inc_b = jax.vmap( lambda beta: sir_euler_incidence(S0=S0, I0=I0, R0 = 0., beta=beta, gamma = gamma, T = T, dt=1) , in_axes=0 )(beta_b)

        print(inc_a.shape)
        numpyro.deterministic("inc_a",inc_a)
        numpyro.deterministic("inc_b",inc_b)

        peaks       = jnp.nanargmax( X, axis=0 )
        peak_deltas = (T+1) - 2*peaks
        
        delta_a  = numpyro.sample("delta_a", dist.Uniform( -T, T ) )
        delta_b  = numpyro.sample("delta_b", dist.Uniform( -T, T ) )
        
        deltas_a = numpyro.deterministic( "deltas_a", jnp.append( peak_deltas, delta_a))
        deltas_b = numpyro.deterministic( "deltas_b", jnp.append( peak_deltas, delta_b))

        numpyro.sample("delta_a_fit", dist.Normal( jnp.mean(peak_deltas), jnp.std(peak_deltas) ), obs = delta_a )
        numpyro.sample("delta_b_fit", dist.Normal( jnp.mean(peak_deltas), jnp.std(peak_deltas) ), obs = delta_b )

        calendar_time = jnp.arange(0,T)
        original_taus = (2*calendar_time + T + 0)/(2*T)
        
        h_a             = (2*calendar_time[:,None]+T+deltas_a[None,:] )/(2*T)  #--maps from t to tau and is TXS
        h_b             = (2*calendar_time[:,None]+T+deltas_b[None,:] )/(2*T)  #--maps from t to tau and is TXS
        h_a             = numpyro.deterministic( "h_a", h_a )
        h_b             = numpyro.deterministic( "h_b", h_b )
        
        main_trend_a = jax.vmap(lambda h,inc: jnp.interp( h, original_taus , inc  ), in_axes = (1,0))( h_a, inc_a )
        main_trend_b = jax.vmap(lambda h,inc: jnp.interp( h, original_taus , inc  ), in_axes = (1,0))( h_b, inc_b )
        
        main_trend_a = main_trend_a.T
        main_trend_b = main_trend_b.T
        numpyro.deterministic("main_trend_a", main_trend_a)
        numpyro.deterministic("main_trend_b", main_trend_b)

        with numpyro.plate("season", S):
            z_log_a_a = numpyro.sample("z_log_a_a", dist.Normal(0.0, 1.0))
            z_log_a_b = numpyro.sample("z_log_a_b", dist.Normal(0.0, 1.0))
            log_a_a   = mu_log_a + tau_log_a * z_log_a_a
            log_a_b   = mu_log_b + tau_log_b * z_log_a_b
            
            a_a       = numpyro.deterministic("a_a", jnp.exp(log_a_a))
            a_b       = numpyro.deterministic("a_b", jnp.exp(log_a_b))

            z_b     = numpyro.sample("z_b"     , dist.Normal(0.0, 1.0))
            b       = numpyro.deterministic("b", mu_b + tau_b * z_b)

        mu      = numpyro.deterministic( "mu", (a_a[None,:]*main_trend_a + a_b[None,:]*main_trend_b + b[None,:]) )

        #--main trend
        trend = mu 

        #--Sigma resdiaul is hierarchical
        mu_log_sigma = numpyro.sample("mu_log_sigma", dist.Normal(-2.0, 0.5))
        tau_log_sigma = numpyro.sample("tau_log_sigma", dist.HalfNormal(0.3))
        with numpyro.plate("season", S):
            z_sigma = numpyro.sample("z_sigma", dist.Normal(0,1))
            sigma = numpyro.deterministic("sigma_s", jnp.exp(mu_log_sigma + tau_log_sigma*z_sigma))

        #--X likelihood
        numpyro.sample( "llx", dist.Normal(trend[:,:-1], sigma[:-1]), obs = X.reshape(T,S-1) )

        #--y likelihood
        with numpyro.handlers.mask(mask=jnp.isfinite(y.reshape(T,))):
            numpyro.sample( "lly", dist.Normal(trend[:,-1].reshape(T,), sigma[-1]), obs = y.reshape(T,) )

        if forecast:
            forecast = numpyro.sample("forecast", dist.Normal(trend[:,-1],sigma[-1]))
            numpyro.deterministic("y_pred", forecast*global_std + global_mu)

    def fit(self
            , M                          = 0
            , estimated_num_components_y = None):

        y, Y, X     = self.y, self.Y, self.X
        self.M      = M

        #--MCMC start
        dense_blocks = [
            ("repo_a_global", "repo_b_global", "gamma", "logit_I0"),
            ("mu_log_a", "tau_log_a", "mu_log_b", "tau_log_b", "mu_b", "tau_b"),
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
