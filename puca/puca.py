#mcandrew

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsp

import numpyro
import numpyro.distributions as dist

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
numpyro.enable_validation(True)

from numpyro.infer import MCMC, NUTS, Predictive, init_to_median, DiscreteHMCGibbs
from numpyro.infer.initialization import init_to_value
from functools import partial
import jax.scipy.linalg as jsp_linalg
from numpyro.distributions import transforms as Trans

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoIAFNormal, AutoBNAFNormal
from numpyro.optim import Adam, ClippedAdam
from numpyro.infer.reparam import NeuTraReparam


class puca( object ):

    def __init__(self
                 , y                 = None
                 , target_indicators = None
                 , X                 = None):

        self.X__input          = X
        self.y__input          = y
        self.target_indicators = target_indicators
        
        self.organize_data()

    @staticmethod
    def smooth_gaussian_anchored(x, sigma=2.0):
        """
        Heavy 1D/2D smoothing with a Gaussian kernel.
        - Uses reflect padding to avoid edge artifacts.
        - Forces first/last value of the smoothed series to equal the original.
        - Optimized to handle 2D arrays (smooths along axis 0)
        """
        x = np.asarray(x, float)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(-1, 1)
        
        radius = int(3 * sigma)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()

        # Optimization 3: Vectorize smoothing for 2D arrays
        # Process all columns at once
        x_pad = np.pad(x, pad_width=((radius, radius), (0, 0)), mode="reflect")
        
        # Apply convolution to each column
        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            y_full = np.convolve(x_pad[:, i], kernel, mode="same")
            y[:, i] = y_full[radius:-radius]

        # anchor endpoints
        y[0, :] = x[0, :]
        y[-1, :] = x[-1, :]
        
        return y.ravel() if is_1d else y
       
    def organize_data(self):

        y_input           = self.y__input
        target_indicators = self.target_indicators
        X                 = self.X__input

        #--split y data into examples from the past and the targets
        Y,y                 = zip(*[ (np.delete(_,t,axis=1), _[:,t])  for t,_ in zip(target_indicators, y_input)])
        
        #--This block standardizes Y data to z-scores, collects the mean and sd, and smooths past Ys.
        all_y             = np.array([])
        smooth_ys         = [] 
        y_means, y_scales = [] , []
        for n,(past,current) in enumerate(zip(Y,y)):
            means  = np.mean(past,axis=0)
            scales =  np.std(past,axis=0)

            y_means.append(means)
            y_scales.append(scales)

            smooth_y       =  self.smooth_gaussian_anchored(past,2)
            smooth_ys.append(smooth_y)
            
            if n==0:
                all_y = np.hstack([smooth_y])
            else:
                _     = np.hstack([smooth_y])
                all_y = np.hstack([all_y,_])
       

        self.global_mu  = np.nanmean( np.nanmean(all_y,0))
        self.global_std = np.nanmean( np.nanstd(all_y,0))

        all_y = (all_y - self.global_mu) / self.global_std

        tobs = []
        for target in y:
            tobs.append( int( min(np.argwhere(np.isnan(target))) ) )
        self.tobs = tobs

        copies      = [y.shape[-1] for y in Y]
        self.copies = copies
            
        #--STORE items
        self.y_means  = y_means
        self.y_scales = y_scales
        self.T        = Y[0].shape[0]  #<--assumption here is Y must be same row for all items in list

        self.all_y  = all_y 
        self.y      = y         #<--Target
        self.Y      = [all_y] # smooth_ys #<--past Y values
        self.X      = X         #<--covariate information 
        
        return y, Y, X, all_y

    
    @staticmethod
    def model( y_past            = None
              ,y_target          = None
              ,Xhat              = None
              ,B                 = None
              ,Sb                = None
              ,column_shifts     = None
              ,index_weights     = None 
              ,Ls                = None
              ,IK                = None
              ,IL                = None
              ,Kp_U              = None
              ,Kp_l              = None
              ,copies            = None
              ,tobs              = None
              ,target_indicators = None
              ,scales           = None
              ,centers          = None   
              ,forecast          = None):

        def bspline_basis(x, knots, degree):
            x = jnp.asarray(x, dtype=knots.dtype)
            n_basis = knots.shape[0] - degree - 1
            T = x.shape[0]

            # degree-0 indicators
            B = jnp.where(
                (x[:, None] >= knots[:n_basis]) & (x[:, None] < knots[1:n_basis+1]),
                1.0,
                0.0,
            )

            # If x hits the right endpoint exactly, put mass in last basis
            B = B.at[:, -1].set(jnp.where(x == knots[-1], 1.0, B[:, -1]))

            zeros_col = jnp.zeros((T, 1), dtype=knots.dtype)

            def body(d, Bcur):
                kd  = jax.lax.dynamic_slice(knots, (d,),   (n_basis,))
                kd1 = jax.lax.dynamic_slice(knots, (d+1,), (n_basis,))

                k0 = knots[:n_basis]
                k1 = knots[1:n_basis+1]

                denom1 = kd  - k0
                denom2 = kd1 - k1

                zeros_col = jnp.zeros((T, 1), dtype=knots.dtype)
                Bshift = jnp.concatenate([Bcur[:, 1:], zeros_col], axis=1)

                # --- SAFE DIVISIONS (avoid /0 producing inf even if later masked) ---
                denom1_safe = jnp.where(denom1 > 0, denom1, 1.0)
                denom2_safe = jnp.where(denom2 > 0, denom2, 1.0)

                w1 = (x[:, None] - k0) / denom1_safe
                w2 = (kd1 - x[:, None]) / denom2_safe

                term1 = jnp.where(denom1 > 0, w1 * Bcur, 0.0)
                term2 = jnp.where(denom2 > 0, w2 * Bshift, 0.0)

                return term1 + term2

            B = jax.lax.fori_loop(1, degree + 1, body, B)
            return B

        def clamped_uniform_knots(lb, ub, n_basis, degree, dtype=jnp.float64):
            # number of interior knots required for n_basis basis functions
            n_int = n_basis - degree - 1
            if n_int < 0:
                raise ValueError("Need n_basis >= degree+1")

            if n_int == 0:
                interior = jnp.array([], dtype=dtype)
            else:
                # equally spaced interior knots, excluding endpoints
                interior = jnp.linspace(lb, ub, n_int + 2, dtype=dtype)[1:-1]

            knots = jnp.concatenate([
                jnp.repeat(jnp.array(lb, dtype=dtype), degree + 1),
                interior,
                jnp.repeat(jnp.array(ub, dtype=dtype), degree + 1),
            ])
            return knots


        #--We need to build F as a set of P-splines that are rperesented as B @ beta
        Text,K          = B.shape
        T               = y_past.shape[0] 
        S_past_total    = y_past.shape[-1]
        S               = S_past_total + 1
        L               = IL.shape[0]
        M               = 15
        
        #--This is a smoothness penalty 
        base        = jnp.arange(0, T) 

        phi_raw         = numpyro.sample("phi_raw", dist.Normal(0, 1).expand([S]))
        phi_shifts_free = 9.0 * jnp.tanh(phi_raw)
        phi_shifts      = phi_shifts_free
        
        z          = numpyro.sample("z_log_a" , dist.Normal(0,1).expand([S]))
        sd         = numpyro.sample("sd_log_a", dist.HalfNormal(0.2))  # small!
        log_a_free = sd * z
        a          = numpyro.deterministic("a",0.5 + (2-0.5)*jax.nn.sigmoid( log_a_free ))
        
        #--the S-1 shift is a shift of zero
        phi_times = a*base[:,None] - phi_shifts
        phi_times = jnp.clip(phi_times, -M,T+M)

        xmin = -M
        xmax = (T - 1) + M

        n_basis = 15
        degree  = 3

        #knots = open_uniform_knots(xmin, xmax, n_basis=n_basis, degree=degree)
        lb, ub    = -M , T+M

        # integer interior knots but drop 2 (one near each boundary)
        #knots_int  = jnp.arange(lb+1, ub, 1)   # (lb+2, ..., ub-2)
        
        # knots_full = jnp.concatenate([
        #     jnp.repeat(jnp.array(lb, dtype=jnp.float64), degree+1),
        #     knots_int.astype(jnp.float64),
        #     jnp.repeat(jnp.array(ub, dtype=jnp.float64), degree+1),
        # ])


        knots_full = clamped_uniform_knots(lb, ub, n_basis=n_basis, degree=degree)

        B_shifted = jax.vmap(lambda x: bspline_basis(x=x, knots=knots_full, degree=degree), in_axes=1)(phi_times)

        ##B_shifted is SXTXL
        B_x = B_shifted[:-1,...]#<--SXTXL
        B_y = B_shifted[-1,...] #<--TXL

        numpyro.deterministic("B_shifted", B_shifted)
        numpyro.deterministic("B_y", B_y)

        #--This is the beta vector
        #sd_tau_diff = numpyro.sample("sd_tau_diff", dist.HalfNormal(0.2))
        #z__tau_diff = numpyro.sample("z__tau_diff", dist.Normal(0,1).expand([L]))
        tau_diff    =  numpyro.sample("sd_tau_diff", dist.HalfNormal( 1. ).expand([L])   ) #jnp.exp(jnp.log( 0.1 ) + sd_tau_diff*z__tau_diff )

        null        = 2
        U0          = Kp_U[:, :null]        
        Up          = Kp_U[:, null:]
        lp          = Kp_l[null:]

        beta0       = jnp.zeros( (U0.shape[1], L) ) #numpyro.sample("beta0", dist.Normal(0,1).expand([U0.shape[1], L])) # 
        z_beta      = numpyro.sample("z_beta", dist.Normal(0,1).expand([lp.shape[0] ,L]) )
        
        def produce_beta( tau_diff, z_beta, Up, lp ):
            on_diag    =   (lp/tau_diff**2) #+ 10**-6
            beta       =  ( Up * (1/jnp.sqrt(on_diag)  ))   @ z_beta
            return beta
        beta = jax.vmap( lambda x,y:produce_beta(x,y,Up,lp), in_axes = (0,1) )( tau_diff, z_beta  )
        beta = beta.T

        beta = numpyro.deterministic("beta", U0@beta0 + beta)

        Fx          = jnp.moveaxis( (B_x @ beta), [0,1], [1,0] )              #<--T X S X L   right now it comes in as S X T X L
        
        numpyro.deterministic("Fx",Fx)
       
        # #--P matrix so that X = Fnorm @ P
        #P_global_prec = numpyro.sample("P_global_prec" , dist.Gamma(2,2./4))
        #P_global_prec = 1./ jnp.sqrt(P_global_prec)
        #P_local_prec   = numpyro.sample("P_local_prec" , dist.Gamma(2, 2.).expand([L,1]) )
        
        gScale  = jnp.ones( (L,1) ) #(P_global_prec * P_local_prec)
        S_tot   = S  # include target season too
                
        # row scales (you already have gScale; shape (L,1) or (L,))
        row_scale = gScale.squeeze(-1)  # (L,)

        # exchangeable correlation across seasons
        rho_season_raw = numpyro.sample("rho_season_raw", dist.Normal(0, 1))
        rho_season     = 0.98 * jax.nn.sigmoid(1.+rho_season_raw)   # (0, 0.95)
        numpyro.deterministic("rho_season", rho_season)

        R      = (1. - rho_season) * jnp.eye(S_tot) + rho_season * jnp.ones((S_tot, S_tot))
        chol_R = jnp.linalg.cholesky(R + 1e-6 * jnp.eye(S_tot))  # jitter

        # optional per-season amplitude
        col_scale = numpyro.sample("col_scale", dist.Normal(0,1).expand([S_tot]))
        col_scale = col_scale*jnp.log(2)/2
        Lcol      = jnp.diag(col_scale) @ chol_R  # this is D * L_R

        # iid standard normal
        Z         = numpyro.sample("Z_A", dist.Normal(0,1).expand([L, S_tot]))

        # A = S_row * Z * (D L_R)^T
        A         = row_scale[:, None] * (Z @ Lcol.T)  # (L,S)

        P         = A[:,1:]
        Q         = A[:,0]

        numpyro.deterministic("A", A)
        numpyro.deterministic("P", P)
        numpyro.deterministic("Q", Q)

        #--need a long term local linear model
        # local_level_sigma   = numpyro.sample("local_level_sigma", dist.HalfNormal(1)) 
        # z_local_level       = numpyro.sample("local_level_z"    , dist.Normal(0,1).expand([T-1]))
        # z_local_level       = z_local_level*local_level_sigma 
        
        # local_trend_sigma   = numpyro.sample("local_trend_sigma", dist.HalfNormal(1)) 
        # z_local_trend       = numpyro.sample("local_trend_z"    , dist.Normal(0,1).expand([T-1]))
        # z_local_trend      = z_local_trend*local_trend_sigma

        # def LocalLin(carry,array):
        #     l,s      = carry
        #     z_l, z_s = array

        #     l_next = l + s + z_l
        #     s_next = s + z_s
        #     return (l_next,s_next), l_next
        # _, season_level_deviation = jax.lax.scan(LocalLin, init = (0,0), xs = ( z_local_level, z_local_trend )  )
        # season_level_deviation    = jnp.append(0, season_level_deviation).reshape(T,1)
        #numpyro.deterministic("season_level_deviation", season_level_deviation)
        #----------------------------------------------------------

        #--AR
        eps          = 1./10
        
        prec_log_ps = numpyro.sample("sd_log_path_scale", dist.Gamma(10, 0.10))
        sd_log_ps   = 1./jnp.sqrt(prec_log_ps)
        mu_log_ps   = numpyro.sample("mu_log_path_scale", dist.Normal(0, 1.0))  
        z           = numpyro.sample("z_log_path_scale" , dist.Normal(0,1).expand([S]))
        
        log_ps     = (mu_log_ps-3) + sd_log_ps * z
        
        path_scale = jnp.exp(log_ps)
       
        start_sigma  = numpyro.sample("start_sigma", dist.HalfNormal(1./5))  
        start_sigma  = jnp.ones((S,))*start_sigma
        
        qvar         = path_scale**2

        rho          = numpyro.sample("rho", dist.Uniform(0.01,1-0.01) )

        #start_intercept_center_sd = numpyro.sample("start_intercept_center_sd", dist.HalfNormal(1./5))         #jnp.ones( (S,) )
        #start_intercept_center    = numpyro.sample("start_intercept_center"   , dist.Normal(0,1))              #jnp.ones( (S,) )
        #z_start_intercept         = numpyro.sample("start_intercept"          , dist.Normal(0,1).expand([S,])) #jnp.ones( (S,) )
        #start_intercept           = start_intercept_center + z_start_intercept*start_intercept_center_sd

        start_intercept            = jnp.zeros((S,))
        start_mean                 = jnp.zeros((S,)) 

        Xtrend  = jnp.einsum("tsl,ls->ts",Fx,P) + start_intercept[:-1].reshape(1,S-1) #<-TXS
        numpyro.deterministic("Xtrend",Xtrend)

        Xtarget      = jnp.hstack(y_past).reshape(T,S_past_total)
        resid        = Xtarget - Xtrend

        Xmask        = jnp.isfinite(resid)
        resid_filled = jnp.where(Xmask, resid, 0.0)

        # #--kalm likelihood here
        def kf(carry, array, q, r, rho):
            mt,Pt    = carry
            yobs,mk  = array

            m_pred = rho*mt            #--This is mean from y (and also from x)
            p_pred = (rho**2)*Pt+q

            y_p_pred = (rho**2)*Pt+q+r #--This is variance from y
            
            S = p_pred + r
            K = p_pred / S

            innov      = (yobs-m_pred)*mk
            
            m_filt = m_pred + K*innov
            P_filt = p_pred * (1-K*mk)

            LOG2PI = jnp.log(2.0 * jnp.pi)
            ll     = mk*(-0.5 * (LOG2PI + jnp.log(S) + (innov**2) / S))

            mpost = m_filt
            ppost = P_filt 
            
            ll_ttl = ll
            
            return (mpost,ppost), (ll_ttl, m_pred, p_pred, mpost, ppost, y_p_pred  )
        
        _,(LL_xs,_,_,_,_,_) = jax.vmap( lambda path_scale_indiv, data, mask, start_m,start_s : jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=path_scale_indiv**2,rho=rho)
                                                                                                           , init = ( start_m, start_s     )
                                                                                                           , xs   = ( data, mask.squeeze() )   )
                            , in_axes=(0,1,1,0,0) )(path_scale[:-1], resid_filled[::-1,:] , Xmask[::-1,:], start_mean[:-1], start_sigma[:-1] )
        
        numpyro.factor( "LL_x", jnp.sum(LL_xs) )

        #numpyro.deterministic("Xtarget", Xtarget)

        # ll = dist.Normal(Xtrend, eps).log_prob(Xtarget)
        # numpyro.deterministic("llx",ll)

        # with numpyro.handlers.mask(mask=Xmask):
        #     numpyro.sample("LLX", dist.Normal( Xtrend, eps ), obs = Xtarget )

        eps_y     = (2*eps) * scales

        ymask_list    = [ jnp.isfinite(y) for y in y_target]
        y_target_list = [jnp.where(m, y, 0.0) for y, m in zip(y_target, ymask_list)]

        y_target      = jnp.hstack(y_target_list).reshape(-1,)
        ymask         = jnp.hstack(ymask_list).reshape(-1,)
        
        y_path_scale = path_scale[-1]
        
        F_y      = (B_y@beta@Q).reshape(T,1)
        
        y_trend  =  F_y + start_intercept[-1]
        numpyro.deterministic("y_trend",y_trend)

        resid        =  ((y_target.reshape(T,1) - centers)/scales) - y_trend
        ymask = ymask.reshape(T,1)
        
        resid_filled =  jnp.where(ymask, resid, 0.0)

        resid_filled_rev = resid_filled[::-1]
        ymask_rev        = ymask[::-1] 
        
        #--The end is time 0
        (m_last, P_last),(LL_y,mp1, Pp1, m_t, P_t, y_p_pred) = jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=y_path_scale**2,rho=rho)
                                                                                              , init  = ( start_mean[-1], start_sigma[-1] )
                                                                                              , xs    = ( resid_filled_rev.squeeze(), ymask_rev.squeeze() ) )
 
        numpyro.factor("LLY", jnp.sum(LL_y))

        numpyro.deterministic("mp1"         , mp1)
        numpyro.deterministic("Pp1"         , Pp1)
        numpyro.deterministic("m_t"         , m_t)
        numpyro.deterministic("P_t"         , P_t)
        numpyro.deterministic("m_last"      , m_last)
        numpyro.deterministic("P_last"      , P_last)
        numpyro.deterministic("y_p_pred_rev", y_p_pred)

        # z_y    = numpyro.sample("z_y", dist.Normal(0,1).expand([T-1]))
        # y_path = jnp.cumsum( jnp.append( start_mean[-1], z_y*y_path_scale ) )[::-1]

        # y_trend = (y_trend.reshape(-1,) + y_path.reshape(-1,))*scales + centers

        # with numpyro.handlers.mask(mask=ymask):
        #      numpyro.sample("LLY", dist.Normal( y_trend ,  eps_y ), obs = y_target )
       
        if forecast:
            #numpyro.sample("y_pred", dist.Normal(y_trend, eps_y))

            innov_z      = numpyro.sample("new_innov_z"    , dist.Normal(0,1).expand([ T-1 ])) #<--one for P and one for Q
            eps          = innov_z

            xT           = numpyro.sample("xT", dist.Normal(m_t[-1], jnp.sqrt(P_t[-1]+10**-6) )) #<--this is time zero
            def step_ffbs(x_next, inputs):
                eps_t, m_t, P_t, mp1, Pp1 = inputs  # all scalars
                J    = rho * P_t / (Pp1+10**-6 )
                mean = m_t + J * (x_next - mp1)

                var  = P_t - (J * J) * Pp1
                var  = jnp.maximum(var, 10**-6)
                
                x_t  = mean + jnp.sqrt(var) * eps_t
                return x_t, x_t

            inputs = (eps,m_t[1:],P_t[1:],mp1[:-1],Pp1[:-1])
            _, xs_rev  = jax.lax.scan(step_ffbs, init=xT, xs=inputs, reverse=True)
            x_path     = jnp.concatenate([  xs_rev[::-1], xT[None] ], axis=0)

            numpyro.deterministic("sxT"    , xT)
            numpyro.deterministic("xs_rev", xs_rev)
            numpyro.deterministic("x_path", x_path)
            
            numpyro.sample( "y_pred", dist.Normal( (y_trend+x_path.reshape(T,1) )*scales + centers, eps_y))
            
            # #----------------------------------------------------------------------------------------

    def estimate_factors(self,D):
        u, s, vt            = np.linalg.svd(D, full_matrices=False)
        splain              = np.cumsum(s**2) / np.sum(s**2)
        estimated_factors_D = np.min(np.argwhere(splain > .95))

        if estimated_factors_D==0:
            estimated_factors_D=1

        print(estimated_factors_D)
        
        return estimated_factors_D, (u,s,vt)


    def find_best_alignment(self):
        y, Y, X     = self.y, self.Y, self.X
        y_past      = Y[0]
        y_target    = ( y - self.global_mu )/self.global_std

        #--pick reference trajectory from past
        y_candidate =  y_past[:,-1]  #y_target
        y_curves    = np.hstack([y_past, y_target.reshape(-1,1) ])

        def apply_shift_1d(y, d, fill=np.nan):
            """Return shifted series y_shifted[t] = y[t + d] (out of bounds -> fill)."""
            T        = y.shape[0]
            out      = np.full(T, fill, dtype=float)
            t        = np.arange(T)
            idx      = t + d
            inb      = (idx >= 0) & (idx < T)
            out[inb] = y[idx[inb]]
            return out

        def best_shift_corr(col, ref, max_shift=10, min_overlap=15):
            best = (-np.inf, 0)
            for d in range(-max_shift, max_shift+1):
                sh = apply_shift_1d(col, d)
                mask = np.isfinite(sh) & np.isfinite(ref)
                n = mask.sum()
                if n < min_overlap:
                    continue
                x = sh[mask]; y = ref[mask]
                x = (x - x.mean()) / (x.std() + 1e-8)
                y = (y - y.mean()) / (y.std() + 1e-8)
                corr = np.mean(x * y)
                if corr > best[0]:
                    best = (corr, d)
            return best[1]

        column_shifts = []
        for n,column in enumerate(y_curves.T):
            best = best_shift_corr(column, y_candidate)
            column_shifts.append(best)

        return column_shifts


    def build_basis_for_F(self, phi_times, both=False):
        # time grid
        T = self.T
        M = self.M

        from patsy import dmatrix

        def bs_basis_zero_padded(tvals):
            def bs_basis(tvals):
                return np.asarray(
                    dmatrix(
                        "bs(t, knots=knots, degree=3, include_intercept=True, lower_bound=lb, upper_bound=ub)-1",
                        {"t": tvals, "knots": knots, "lb": lb, "ub": ub},
                    )
                )

            tvals = np.asarray(tvals, float)
            ok = (tvals >= lb) & (tvals <= ub)

            # build basis for clipped values (any in-range values ok)
            B = np.zeros((len(tvals), bs_basis(np.array([lb])).shape[1]), float)
            if ok.any():
                B_ok = bs_basis(tvals[ok])  # uses fixed knots/bounds
                B[ok] = B_ok
            return B

        def bs_basis_numpy(tvals, knots, lb, ub):
            return np.asarray(
                dmatrix(
                    "bs(t, knots=knots, degree=3, include_intercept=True, lower_bound=lb, upper_bound=ub)-1",
                    {"t": np.asarray(tvals, float), "knots": knots, "lb": lb, "ub": ub},
                )
            )

        #--Choose fixed spline settings and pad them from -M to M for a total of 2M+1 potential reference points (dont forget zero)
        lb, ub = -M, (T) + M
        t      = np.arange(lb,ub+1)
        knots  = np.arange(lb, ub, 1)[1:] #was originally 2

        def clamped_uniform_knots(lb, ub, n_basis, degree, dtype=jnp.float64):
            # number of interior knots required for n_basis basis functions
            n_int = n_basis - degree - 1
            if n_int < 0:
                raise ValueError("Need n_basis >= degree+1")

            if n_int == 0:
                interior = jnp.array([], dtype=dtype)
            else:
                # equally spaced interior knots, excluding endpoints
                interior = jnp.linspace(lb, ub, n_int + 2, dtype=dtype)[1:-1]

            knots = jnp.concatenate([
                jnp.repeat(jnp.array(lb, dtype=dtype), degree + 1),
                interior,
                jnp.repeat(jnp.array(ub, dtype=dtype), degree + 1),
            ])
            return knots, interior

        def bspline_basis(x, knots, degree):
            x = jnp.asarray(x, dtype=knots.dtype)
            n_basis = knots.shape[0] - degree - 1
            T = x.shape[0]

            # degree-0 indicators
            B = jnp.where(
                (x[:, None] >= knots[:n_basis]) & (x[:, None] < knots[1:n_basis+1]),
                1.0,
                0.0,
            )

            # If x hits the right endpoint exactly, put mass in last basis
            B = B.at[:, -1].set(jnp.where(x == knots[-1], 1.0, B[:, -1]))

            zeros_col = jnp.zeros((T, 1), dtype=knots.dtype)

            def body(d, Bcur):
                kd  = jax.lax.dynamic_slice(knots, (d,),   (n_basis,))
                kd1 = jax.lax.dynamic_slice(knots, (d+1,), (n_basis,))

                k0 = knots[:n_basis]
                k1 = knots[1:n_basis+1]

                denom1 = kd  - k0
                denom2 = kd1 - k1

                zeros_col = jnp.zeros((T, 1), dtype=knots.dtype)
                Bshift = jnp.concatenate([Bcur[:, 1:], zeros_col], axis=1)

                # --- SAFE DIVISIONS (avoid /0 producing inf even if later masked) ---
                denom1_safe = jnp.where(denom1 > 0, denom1, 1.0)
                denom2_safe = jnp.where(denom2 > 0, denom2, 1.0)

                w1 = (x[:, None] - k0) / denom1_safe
                w2 = (kd1 - x[:, None]) / denom2_safe

                term1 = jnp.where(denom1 > 0, w1 * Bcur, 0.0)
                term2 = jnp.where(denom2 > 0, w2 * Bshift, 0.0)

                return term1 + term2

            B = jax.lax.fori_loop(1, degree + 1, body, B)
            return B

        knots_full, knots_interior = clamped_uniform_knots(lb, ub, n_basis=15, degree=3)
        
        #B                = bs_basis_numpy(t,knots,lb,ub)
        B                 = bspline_basis(t,knots_full,3)

        #--Normalize the basis 
        #D            = jnp.std(B, 0) 
        #Sb           = jnp.diag( D )
        #Sbinv        = jnp.diag(1./D)

        #B_norm       = B @ Sbinv

        self.B      = jnp.array(B)    #<--Normalized B
        self.Sb     = 1#Sb        #<--Scaling matrix for B

        #--While we're here we should compute the matrix of second differences (D) and the penalty matrix Kp
        D          = jnp.diff(jnp.diff(jnp.eye(B.shape[-1]),axis=0),axis=0)
        
        Kp         = D.T@D
        
        self.Kp                 = Kp
        self.Kp_l, self.Kp_U    = jnp.linalg.eigh(Kp)

        if both:
            return B,Kp
            #return B_norm, Kp
        else:
            return B
            #B_norm

    def fit(self
            , M                          = 15
            , estimated_num_components_y = None):

        y, Y, X     = self.y, self.Y, self.X
        all_y       = self.all_y

        self.M      = M

        #--SVD for X
        if estimated_num_components_y is None:
            y_svd_components = { "U":[], "VT":[], "LAMBDA":[] }
            num_components   = []
            for _ in Y:
                nfactors, (u,s,vt) = self.estimate_factors(_)

                num_components.append(nfactors)
                
                y_svd_components["U"].append(u)
                y_svd_components["LAMBDA"].append(s)
                y_svd_components["VT"].append(vt)
                
            self.estimated_num_components_y = num_components
            
        else:
            self.estimated_num_components_y = estimated_num_components_y

        column_shifts      = self.find_best_alignment()
        self.column_shifts = jnp.array(column_shifts)

        y_counts    = np.zeros( (2*M+1,))
        for shift in column_shifts:
            y_counts[ M - shift ]+=1
        self.index_weights = y_counts

        B,_ = self.build_basis_for_F( phi_times = jnp.arange(self.T) ,  both=True)

        model      = self.model
        copies     = self.copies
        
        #--collect helpful parameters
        S_past_total      = int(sum(copies))
        num_targets       = len(copies)
        S                 = S_past_total + num_targets

        #--we will flateen target_indicators
        copies_j                    = jnp.array(self.target_indicators)        
        starts                      = jnp.concatenate([jnp.array([0]), jnp.cumsum(copies_j + 1)[:-1]])
        target_indicators           = starts + copies_j
        self.target_indicators_mcmc = target_indicators

        IL                = jnp.eye( int(sum(self.estimated_num_components_y)))
        self.IL           = IL

        IK                 = jnp.eye(B.shape[1])
        self.IK           = IK
        
        rng_post          = jax.random.PRNGKey(100915)

        if X is not None:
           y_past = np.hstack([ X, Y[0] ])
           target_indicators = [target_indicators[0]+X.shape[-1]]
           self.target_indicators_mcmc = target_indicators
        else:
            y_past = Y[0]
        self.y_past = y_past

        print(self.tobs)

        #--SVI start
        # guide = AutoIAFNormal(
        #     self.model
        #     ,num_flows=4
        #     ,init_loc_fn=init_to_median(num_samples=100)
        # )
 
        # optimizer = ClippedAdam(step_size= 1*10**-3, clip_norm=1)  # common starting point: 1e-3 to 1e-2
        # loss      = Trace_ELBO(num_particles=3)

        # svi = SVI(self.model, guide, optimizer, loss)

        # svi_result = svi.run(
        #     jax.random.PRNGKey(20200320),
        #     10*10**3,
        #     y_past            = y_past,
        #     y_target          = y,
        #     B                 = self.B,
        #     Sb                = self.Sb,
        #     Ls                = self.estimated_num_components_y,
        #     column_shifts     = self.column_shifts,
        #     index_weights     = self.index_weights,
        #     IL                = self.IL,
        #     IK                = self.IK,
        #     Kp_l              = self.Kp_l,
        #     Kp_U              = self.Kp_U,
        #     copies            = self.copies,
        #     tobs              = self.tobs[0],
        #     target_indicators = target_indicators,
        #     scales            = self.global_std,
        #     centers           = self.global_mu,
        #     forecast          = None,
        #     stable_update=True
        # )

        # params = svi_result.params
        # losses = svi_result.losses
        # print("final loss:", float(losses[-1]))

        # self.params = params
        # self.guide = guide

        #--SVI Stop

        
        #--MCMC start
        dense_blocks = [
            #("z_beta",)
            #("local_level_sigma","local_trend_sigma")
            #("P_global_prec", "P_local_prec")
            #,("rho_season_raw","col_scale")
            #,("phi_raw","z_log_a","sd_log_a")
            
        ]
        
        nuts_kernel = NUTS(self.model#neutra_model
                           , init_strategy = init_to_median(num_samples=100)
                           #, dense_mass = dense_blocks
                           ,  find_heuristic_step_size=True)     
        kernel      = nuts_kernel 
        mcmc        = MCMC(kernel
                    , num_warmup     = 1000
                    , num_samples    = 1000
                    , num_chains     = 1
                    , jit_model_args = False)

        mcmc.run(jax.random.PRNGKey(20200320)
                              ,y_past            = y_past
                              ,y_target          = y
                              ,B                 = self.B
                              ,Sb                = self.Sb
                              ,Ls                = self.estimated_num_components_y
                              ,column_shifts     = self.column_shifts
                              ,index_weights     = self.index_weights
                              ,IL                = self.IL
                              ,IK                = self.IK
                              ,Kp_l              = self.Kp_l
                              ,Kp_U              = self.Kp_U
                              ,copies            = self.copies
                              ,tobs              = self.tobs[0]
                              ,target_indicators = target_indicators
                              ,scales            = self.global_std
                              ,centers           = self.global_mu
                              ,forecast          = None 
                              ,extra_fields      = ("diverging", "num_steps", "accept_prob", "energy","adapt_state.step_size"))

        self.mcmc = mcmc
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.posterior_samples = samples
        #--MCMC end

        
        return self

    def forecast(self):

        #--SVI START
        # predictive = Predictive(self.model
        #                        , guide=self.guide
        #                        , params=self.params
        #                        , num_samples=5000
        #                        , return_sites  = ["y_pred"] )
        #--SVI END

        #--MCMC START
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred","xT","xs_rev","x_path"] )
        #--MCMC END

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,y_past            = self.y_past
                              ,y_target          = self.y
                              ,B                 = self.B
                              ,Sb                = self.Sb
                              ,Ls                = self.estimated_num_components_y
                              ,column_shifts     = self.column_shifts
                              ,index_weights     = self.index_weights
                              ,IL                = self.IL
                              ,IK                = self.IK
                              ,Kp_l              = self.Kp_l
                              ,Kp_U              = self.Kp_U
                              ,copies            = self.copies
                              ,tobs              = self.tobs[0]
                              ,target_indicators = self.target_indicators_mcmc
                              ,scales            = self.global_std
                              ,centers           = self.global_mu
                              ,forecast          = True
                                  )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        self.pred_samples = pred_samples
        self.forecast     = yhat_draws
        return yhat_draws
    



if __name__ == "__main__":

    pass
