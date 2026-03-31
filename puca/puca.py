

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

numpyro.enable_validation(True)


def logistic_registered_warp(a, b, logit_tau):
    """Shared warp: a, b shape (S,), logit_tau shape (T,) -> hs (S, T) in [0, 1]."""
    hs = jax.nn.sigmoid(a[:, None] + b[:, None] * logit_tau[None, :])
    return jnp.clip(hs, 0.0, 1.0)


def _clamped_uniform_knots(lb, ub, n_basis, degree, dtype=jnp.float64):
    n_int = n_basis - degree - 1
    if n_int < 0:
        raise ValueError("Need n_basis >= degree+1")
    if n_int == 0:
        interior = jnp.array([], dtype=dtype)
    else:
        interior = jnp.linspace(lb, ub, n_int + 2, dtype=dtype)[1:-1]
    return jnp.concatenate(
        [
            jnp.repeat(jnp.array(lb, dtype=dtype), degree + 1),
            interior,
            jnp.repeat(jnp.array(ub, dtype=dtype), degree + 1),
        ]
    )


def _bspline_basis_cols(x, knots, degree):
    """x: (T,), knots — same construction as ``puca.model``."""
    x = jnp.asarray(x, dtype=knots.dtype)
    n_basis = knots.shape[0] - degree - 1
    tlen = x.shape[0]
    right = knots[-1]
    x = jnp.minimum(x, jnp.nextafter(right, -jnp.inf))
    B = jnp.where(
        (x[:, None] >= knots[:n_basis]) & (x[:, None] < knots[1 : n_basis + 1]),
        1.0,
        0.0,
    )
    zeros_col = jnp.zeros((tlen, 1), dtype=knots.dtype)

    def body(d, Bcur):
        kd = jax.lax.dynamic_slice(knots, (d,), (n_basis,))
        kd1 = jax.lax.dynamic_slice(knots, (d + 1,), (n_basis,))
        k0 = knots[:n_basis]
        k1 = knots[1 : n_basis + 1]
        denom1 = kd - k0
        denom2 = kd1 - k1
        Bshift = jnp.concatenate([Bcur[:, 1:], zeros_col], axis=1)
        denom1_safe = jnp.where(denom1 > 0, denom1, 1.0)
        denom2_safe = jnp.where(denom2 > 0, denom2, 1.0)
        w1 = (x[:, None] - k0) / denom1_safe
        w2 = (kd1 - x[:, None]) / denom2_safe
        term1 = jnp.where(denom1 > 0, w1 * Bcur, 0.0)
        term2 = jnp.where(denom2 > 0, w2 * Bshift, 0.0)
        return term1 + term2

    return jax.lax.fori_loop(1, degree + 1, body, B)


def uniform01_bspline_design(x, n_basis, degree=3):
    """B-spline design on [0, 1], matching ``model`` knot layout; x shape (T,)."""
    knots = _clamped_uniform_knots(0.0, 1.0, n_basis, degree)
    return _bspline_basis_cols(jnp.asarray(x), knots, degree)


def embed_warped_U_in_spline_coeffs(U_warped, tau_warp_grid, n_basis, L, degree=3):
    """
    U_warped: (n_tau, k) left singular vectors on the warped τ grid.
    Return (n_basis, L) coefficients so Phi @ coeff ≈ first L columns of U.
    """
    U_warped = np.asarray(U_warped, dtype=float)
    Phi = np.asarray(uniform01_bspline_design(jnp.asarray(tau_warp_grid), n_basis, degree))
    k_avail = min(U_warped.shape[1], L)
    coeff = np.zeros((n_basis, L), dtype=float)
    if k_avail > 0 and L > 0:
        U_L = U_warped[:, :k_avail]
        c_part, *_ = np.linalg.lstsq(Phi, U_L, rcond=None)
        coeff[:, :k_avail] = c_part
    return jnp.array(coeff, dtype=jnp.float64)


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

    #@staticmethod
   
       
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


        def register(Y):
            """SVI registration using the same logistic warp as ``puca.model``."""

            tau_warp_grid = np.linspace(0.0, 1.0, 101)

            def model_register(Y_arr):
                Y_arr = jnp.asarray(Y_arr, dtype=jnp.float64)
                T, S = Y_arr.shape
                tau_grid = jnp.linspace(0.0, 1.0, T)
                tau_grid = jnp.clip(tau_grid, 1e-6, 1.0 - 1e-6)
                logit_tau = jnp.log(tau_grid / (1.0 - tau_grid))

                a_shared = numpyro.sample("warp_a_shared", dist.Normal(0.0, 1))
                b_shared = numpyro.sample("warp_b_shared", dist.Normal(0.0, 0.5))
                a_scale = numpyro.sample("warp_a_scale", dist.HalfNormal(0.25))
                b_scale = numpyro.sample("warp_b_scale", dist.HalfNormal(0.004))
                a_raw = numpyro.sample(
                    "warp_a_raw", dist.Normal(0.0, 1.0).expand([S])
                )
                b_raw = numpyro.sample(
                    "warp_b_raw", dist.Normal(0.0, 1.0).expand([S])
                )

                a_raw = a_raw - jnp.mean(a_raw)
                b_raw = b_raw - jnp.mean(b_raw)

                a = a_shared / 2 + a_scale * a_raw
                eta = b_shared / 2 + b_scale * b_raw
                b = jnp.exp(eta)

                hs = numpyro.deterministic(
                    "hs", logistic_registered_warp(a, b, logit_tau)
                )

                sd_hier = numpyro.sample("sd", dist.HalfNormal(1.0 / 2))
                z_hier = numpyro.sample("z", dist.Normal(0, 1).expand([T]))
                mean_curve = numpyro.deterministic(
                    "mean_curve", jnp.cumsum(z_hier * sd_hier)
                )

                ypred = jax.vmap(
                    lambda x: jnp.interp(x, tau_grid, mean_curve), in_axes=0
                )(hs)

                alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0).expand([S]))
                beta = numpyro.sample("beta", dist.LogNormal(0.0, 0.3).expand([S]))
                ypred = alpha[:, None] + beta[:, None] * ypred
                numpyro.deterministic("ypred", ypred)

                registered_curves = numpyro.deterministic(
                    "registered_curves",
                    jax.vmap(
                        lambda x, ycol: jnp.interp(tau_grid, x, ycol), in_axes=(0, 0)
                    )(hs, Y_arr.T),
                )

                sigma = numpyro.sample("sigma", dist.HalfNormal(5))
                with numpyro.handlers.mask(mask=~jnp.isnan(Y_arr.T)):
                    numpyro.sample("ll", dist.Normal(ypred, sigma), obs=Y_arr.T)

            guide = AutoDelta(model_register)
            optimizer = numpyro.optim.Adam(step_size=0.005)
            svi = SVI(model_register, guide, optimizer, loss=Trace_ELBO())
            svi_result = svi.run(jax.random.PRNGKey(20200320), 8000, Y)

            warp_sites = [
                "warp_a_shared",
                "warp_b_shared",
                "warp_a_scale",
                "warp_b_scale",
                "warp_a_raw",
                "warp_b_raw",
                "registered_curves",
                "hs",
            ]
            predictive = Predictive(
                model_register,
                guide=guide,
                params=svi_result.params,
                num_samples=1000,
                return_sites=warp_sites,
            )
            predictions = predictive(jax.random.PRNGKey(1), Y)

            registered_curves = predictions["registered_curves"].mean(0).T
            hs_mean = np.asarray(predictions["hs"].mean(axis=0))
            H = hs_mean.T
            self.H = H

            S = Y.shape[1]
            warped = np.full((len(tau_warp_grid), S), np.nan, dtype=float)
            y_np = np.asarray(Y, dtype=float)
            for s in range(S):
                hcol = hs_mean[s, :]
                ycol = y_np[:, s]
                m = np.isfinite(hcol) & np.isfinite(ycol)
                hcol = hcol[m]
                ycol = ycol[m]
                if hcol.size == 0:
                    continue
                order = np.argsort(hcol, kind="mergesort")
                h_ord = hcol[order]
                y_ord = ycol[order]
                warped[:, s] = np.interp(
                    tau_warp_grid, h_ord, y_ord, left=y_ord[0], right=y_ord[-1]
                )
            self.warped_Y = warped
            self.tau_warp_grid = tau_warp_grid

            current_season_map = H[:, -1]
            self.current_season_map = current_season_map

            self.registered_model_fit = {
                k: np.mean(v, axis=0) for k, v in predictions.items()
            }

            return registered_curves, current_season_map

        
        #--ROUTINE STARTED
        y           =  self.y__input[0] #<-- for now we assume one target only
        Y           =  self.Y__input[0]
        X           =  None  #self.X__input

        #--This block standardizes Y data to z-scores, collects the mean and sd, and smooths past Ys.
        smoothed_ys                                             = smooth_y_data( y,Y )

        #--remove the current season from this computation
        centered_smoothed_ys, (self.global_mu, self.global_std) = center_scale( smoothed_ys )

        register( centered_smoothed_ys )

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


    #--now build a basis and estimate nummber of latent factors
    def estimate_factors(self, D):
        D = np.asarray(D, dtype=float)
        if not np.isfinite(D).all():
            D = D.copy()
            col_med = np.nanmedian(D, axis=0)
            col_med = np.where(np.isfinite(col_med), col_med, 0.0)
            j = np.where(~np.isfinite(D))
            D[j] = col_med[j[1]]
        u, s, vt = np.linalg.svd(D, full_matrices=False)
        splain              = np.cumsum(s**2) / np.sum(s**2)
        estimated_factors_D = np.min(np.argwhere(splain > .95))

        if estimated_factors_D==0:
            estimated_factors_D=1

        print(estimated_factors_D)
        
        return estimated_factors_D, (u,s,vt)

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
        tau_log_a = numpyro.sample("tau_log_a", dist.HalfNormal(0.5 ))

        mu_b      = numpyro.sample("mu_b"     , dist.Normal(0.0, 1.0))
        tau_b     = numpyro.sample("tau_b"    , dist.HalfNormal(0.5 ))

        #main_var               = numpyro.sample("main_var", dist.Dirichlet(5*jnp.array([0.90,0.08,0.02])))
        spacing                = jnp.sqrt(tau_grid[1]) #<--this is only bc i start at zero and take same size increments.

        logit_I0 = numpyro.sample("logit_I0", dist.Normal(-4.0, 1.0))
        I0       = jax.nn.sigmoid(logit_I0)
        S0       = 1.0 - I0

        repo             = numpyro.sample("repo", dist.Gamma(2,1))
        repo_scale_local = numpyro.sample("repo_scale_local",dist.HalfNormal(1./2))
        with numpyro.plate("season", S):
            z_repos   = numpyro.sample( "z_repos", dist.Normal(0,1)  )
        repo = jnp.exp(jnp.log(repo) + repo_scale_local*z_repos)
        
        gamma    = numpyro.sample("gamma",dist.Gamma(2,1))

        beta              = repo*gamma
        #beta_scale_global = numpyro.sample("beta_scale_global",dist.HalfNormal(1./5))
        
        #z_betas          = numpyro.sample( "z_betas", dist.Normal(0,1).expand([T-1]) )
        #with numpyro.plate("season", S):
        #    beta_scale_local = numpyro.sample("beta_scale_local",dist.HalfNormal(1./10))

        #betas    = z_betas*(beta_scale_global)#*beta_scale_local)[:,None]
        #betas    = jnp.hstack([betas , jnp.zeros(S)[:,None] ])  #SXT

        #betas    = jnp.hstack([  jnp.zeros(1)[:,None] , betas])  #SXT
        #betas = jnp.append(0,betas)
        
        #beta_dev = jnp.flip(  jnp.cumsum(jnp.flip(betas, axis=1), axis=1),axis=1)
        #beta_dev = jnp.cumsum( betas)#, axis=1)
        #beta_dev = beta_dev - jnp.mean(beta_dev)#,axis=1)[:,None]

        #betas = jnp.repeat(beta,T) #jnp.exp( jnp.log(beta) + beta_dev)
        
        inc = jax.vmap( lambda beta: sir_euler_incidence(S0=S0, I0=I0, R0 = 0., beta=beta, gamma = gamma, T = T, dt=1) , in_axes=0 )(beta)
        #inc =  sir_euler_incidence(S0=S0, I0=I0, R0 = 0., beta=betas, gamma = gamma, T = T, dt=1) 

        print(inc.shape)
        numpyro.deterministic("inc",inc)

        peaks       = jnp.nanargmax( X, axis=0 )
        peak_deltas = (T+1) - 2*peaks
        
        delta  = numpyro.sample("delta", dist.Uniform( -T, T ) )
        deltas = numpyro.deterministic( "deltas", jnp.append( peak_deltas, delta))
        numpyro.sample("delta_fit", dist.Normal( jnp.mean(peak_deltas), jnp.std(peak_deltas) ), obs = delta )

        calendar_time = jnp.arange(0,T)
        original_taus = (2*calendar_time + T + 0)/(2*T)
        
        h             = (2*calendar_time[:,None]+T+deltas[None,:] )/(2*T)  #--maps from t to tau and is TXS
        h             = numpyro.deterministic( "h", h )
        

        # season_disc_scale_global = main_var[1]
        # season_disc_gp_scale = numpyro.sample("season_disc_gp_scale", dist.HalfNormal(1.0))
        # season_disc_gp_ls    = numpyro.sample("season_disc_gp_ls", dist.LogNormal(0.0, 0.5))
        # season_disc_time     = h[:, -1]
        # season_disc_dist     = jnp.abs(season_disc_time[:, None] - season_disc_time[None, :])
        # season_disc_sqrt5    = jnp.sqrt(5.0)
        # season_disc_scaled   = season_disc_dist / season_disc_gp_ls
        # season_disc_kernel   = (
        #     season_disc_gp_scale**2
        #     * (1.0 + season_disc_sqrt5 * season_disc_scaled + 5.0 * season_disc_scaled**2 / 3.0)
        #     * jnp.exp(-season_disc_sqrt5 * season_disc_scaled)
        # )
        # season_disc_kernel = season_disc_kernel + 1e-6 * jnp.eye(T)
        # season_disc_global_raw = numpyro.sample(
        #     "season_disc_global_raw",
        #     dist.MultivariateNormal(loc=jnp.zeros(T), covariance_matrix=season_disc_kernel),
        # )
        # season_disc_global = numpyro.deterministic(
        #     "season_disc_global",
        #     season_disc_global_raw - season_disc_global_raw[-1],
        # )

        # season_disc_dev_z = numpyro.sample("season_disc_dev_z", dist.Normal(0,1).expand([T-1, S]))
        # with numpyro.plate("season", S):
        #     season_disc_scale_local  = numpyro.sample("season_disc_scale_local", dist.HalfNormal(1.0))
        # season_disc_dev_steps = season_disc_scale_global * season_disc_scale_local[None, :] * season_disc_dev_z * spacing
        # season_disc_dev_rw = jnp.flip(
        #     jnp.cumsum(jnp.flip(season_disc_dev_steps, axis=0), axis=0),
        #     axis=0,
        # )
        # season_disc_dev_rw = season_disc_dev_rw - season_disc_dev_rw[-1:, :]
        # season_disc_dev = jnp.concatenate([season_disc_dev_rw, jnp.zeros((1, S))], axis=0)

        # season_disc = numpyro.deterministic("season_disc", season_disc_global[:, None] + season_disc_dev)

        #main_trend = jax.vmap(lambda h: jnp.interp( h, original_taus , inc  ), in_axes = (1))( h )
        main_trend = jax.vmap(lambda h,inc: jnp.interp( h, original_taus , inc  ), in_axes = (1,0))( h, inc )
        
        main_trend = main_trend.T
        numpyro.deterministic("main_trend", main_trend)

        with numpyro.plate("season", S):
            z_log_a = numpyro.sample("z_log_a", dist.Normal(0.0, 1.0))
            log_a   = mu_log_a + tau_log_a * z_log_a
            a       = numpyro.deterministic("a", jnp.exp(log_a))

            z_b     = numpyro.sample("z_b"     , dist.Normal(0.0, 1.0))
            b       = numpyro.deterministic("b", mu_b + tau_b * z_b)

        mu      = numpyro.deterministic( "mu", (a[None,:]*main_trend + b[None,:]) )
        #mu      = numpyro.deterministic( "mu", (main_trend + b[None,:]) )

        #--main trend
        trend = mu #+ season_disc  # TXS

        #--resiual is shrunk if possible.
        
        # global_sigma = numpyro.sample("gsigma", dist.HalfNormal(1./5))
        # local_sigma  = numpyro.sample("lsigma", dist.HalfNormal(1./10).expand([S]))
        # sigma        = numpyro.deterministic("sigma",global_sigma*local_sigma )


        mu_log_sigma = numpyro.sample("mu_log_sigma", dist.Normal(-2.0, 0.5))
        tau_log_sigma = numpyro.sample("tau_log_sigma", dist.HalfNormal(0.3))
        with numpyro.plate("season", S):
            z_sigma = numpyro.sample("z_sigma", dist.Normal(0,1))
            sigma = numpyro.deterministic("sigma_s", jnp.exp(mu_log_sigma + tau_log_sigma*z_sigma))
        #sigma = 0.10

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
            ("col_scale",),
            ("mu_log_ps","tau_log_ps"),
        ]


        from patsy import dmatrix
        #dense_grid = np.linspace(-1,1.5,200)
        #B          = dmatrix( "bs(x, df=10, degree=3, include_intercept=False) - 1", {"x":dense_grid} )
        #B          = jnp.asarray(B)
        #D          = jnp.diff(jnp.diff(jnp.eye( B.shape[-1] ),axis=0), axis=0)
 
        nuts_kernel = NUTS(self.model
                           , init_strategy = init_to_median(num_samples=100)
                           , dense_mass = [("repo","gamma","logit_I0")]
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

        #self.D = D
        #self.B = B
        #self.dense_grid = dense_grid
        
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
