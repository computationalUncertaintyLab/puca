#mcandrew

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
numpyro.enable_validation(True)

from numpyro.infer import MCMC, NUTS, Predictive, init_to_median


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
              ,B                 = None
              ,IL                = None
              ,Kp_U              = None
              ,Kp_l              = None
              ,scales           = None
              ,centers          = None   
              ,forecast          = None):

        def bspline_basis(x, knots, degree):
            x = jnp.asarray(x, dtype=knots.dtype)
            n_basis = knots.shape[0] - degree - 1
            T = x.shape[0]

            right = knots[-1]
            eps   = 1e-8 * (right - knots[0] + 1.0)

            x = jnp.minimum(x, jnp.nextafter(right, -jnp.inf))  # best: one ulp below right
            # or: x = jnp.minimum(x, right - eps)

            # degree-0 indicators
            B = jnp.where(
                (x[:, None] >= knots[:n_basis]) & (x[:, None] < knots[1:n_basis+1]),
                1.0,
                0.0,
            )

            # If x hits the right endpoint exactly, put mass in last basis
            #B = B.at[:, -1].set(jnp.where(x == knots[-1], 1.0, B[:, -1]))

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
        K               = B.shape[1]
        T               = y_past.shape[0] 
        S_past_total    = y_past.shape[-1]
        S               = S_past_total + 1
        L               = IL.shape[0]
        M               = 15
        
        #--This is a smoothness penalty 
        base            = jnp.arange(0, T) 

        phi_raw         = numpyro.sample("phi_raw", dist.Normal(0, 1).expand([S]))
        phi_shifts_free = 9.0 * jnp.tanh(phi_raw)
        phi_shifts      = phi_shifts_free
        
        #--the S-1 shift is a shift of zero
        phi_times = 1*base[:,None] - phi_shifts
        phi_times = jnp.clip(phi_times, -M, (T+M) - 1e-6)

        n_basis = 10
        degree  =  3

        lb, ub    = -M , T+M

        knots_full      = clamped_uniform_knots(lb, ub, n_basis=n_basis, degree=degree)
        
        B_shifted       = jax.vmap(lambda x: bspline_basis(x=x, knots =  knots_full , degree=degree), in_axes=1)(phi_times)

        ##B_shifted is SXTXL
        B_x = B_shifted[:-1,...]#<--SXTXL
        B_y = B_shifted[-1,...] #<--TXL

        numpyro.deterministic("B_shifted", B_shifted)
        numpyro.deterministic("B_y", B_y)

        #--This is the beta vector
        tau_diff    =  numpyro.sample("sd_tau_diff", dist.HalfNormal( 1. ).expand([L])   ) #jnp.exp(jnp.log( 0.1 ) + sd_tau_diff*z__tau_diff )

        null = 2
        U0 = Kp_U[:, :null]          # (K,2)
        Up = Kp_U[:, null:]          # (K,K-2)
        lp = jnp.clip(Kp_l[null:], 1e-10)   # (K-2,)  <-- avoid tiny/negative numerical junk

        tau    = numpyro.sample("tau_diff", dist.HalfNormal(1./10).expand([L]))  # smaller -> smoother

        z      = numpyro.sample("z_beta"  , dist.Normal(0,1).expand([lp.shape[0], L]))  # (K-2,L)

        scale = (tau[None, :] / jnp.sqrt(lp)[:, None])      # (K-2,L)
        beta_pen = Up @ (scale * z)                          # (K,L)

        centered_t = jnp.arange(n_basis)
        centered_t = (centered_t-jnp.mean(centered_t))/jnp.std(centered_t)
        U0         = jnp.vstack([ jnp.ones(n_basis,), centered_t ] ).T
        
        beta = numpyro.deterministic("beta",   beta_pen)  # (K,L)


        Fx          = jnp.moveaxis( (B_x @ beta), [0,1], [1,0] )              #<--T X S X L   right now it comes in as S X T X L
        
        numpyro.deterministic("Fx",Fx)
       
        gScale  = jnp.ones( (L,1) )
        S_tot   = S  # include target season too
                
        # row scales (you already have gScale; shape (L,1) or (L,))
        row_scale = gScale.squeeze(-1)  # (L,)

        rho_season     =  0.90 # jax.nn.sigmoid(1.+rho_season_raw)   # (0, 0.95)
        numpyro.deterministic("rho_season", rho_season)

        R      = (1. - rho_season) * jnp.eye(S_tot) + rho_season * jnp.ones((S_tot, S_tot))
        chol_R = jnp.linalg.cholesky(R + 1e-6 * jnp.eye(S_tot))  # jitter

        col_scale = numpyro.sample("col_scale", dist.Normal(0,1).expand([S_tot]))

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

        #--AR
        eps          = 1./10
        
        mu_log_ps = numpyro.sample("mu_log_ps", dist.Normal(0.0, 1.0))

        # precision on log scale: tau = 1/sd^2
        tau_log_ps = numpyro.sample("tau_log_ps"      , dist.Gamma(concentration=10.0, rate= 0.01))
        sd_log_ps  = numpyro.deterministic("sd_log_ps", 1.0 / jnp.sqrt(tau_log_ps))

        z          = numpyro.sample("z_log_ps", dist.Normal(0.0, 1.0).expand([S]))
        log_ps     = mu_log_ps -1  + sd_log_ps * z
        path_scale = jnp.exp(log_ps)
      
        start_sigma  = 10**-2#numpyro.sample("start_sigma", dist.HalfNormal(1./5))
        start_sigma  = jnp.ones((S,))*start_sigma
        
        #qvar         = path_scale**2
        rho = 0.95 * jax.nn.sigmoid(numpyro.sample("rho_raw", dist.Normal(0,1))) + 0.02

        start_intercept_center_sd = numpyro.sample("start_intercept_center_sd", dist.HalfNormal(1./5))         #jnp.ones( (S,) )
        start_intercept_center    = numpyro.sample("start_intercept_center"   , dist.Normal(0,1))              #jnp.ones( (S,) )
        z_start_intercept         = numpyro.sample("start_intercept"          , dist.Normal(0,1).expand([S,])) #jnp.ones( (S,) )
        start_intercept           = start_intercept_center + z_start_intercept*start_intercept_center_sd

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
                            , in_axes=(0,1,1,0,0) )(path_scale[:-1], resid_filled[::-1,:] , Xmask[::-1,:], start_mean[:-1], start_sigma[:-1]**2 )
        
        numpyro.factor( "LL_x", jnp.sum(LL_xs) )

        eps_y     = (eps) * scales

        ymask_list    = [ jnp.isfinite(y) for y in y_target]
        y_target_list = [jnp.where(m, y, 0.0) for y, m in zip(y_target, ymask_list)]

        y_target      = jnp.hstack(y_target_list).reshape(-1,)
        ymask         = jnp.hstack(ymask_list).reshape(-1,)
        
        y_path_scale = path_scale[-1]
        
        F_y      = (B_y@beta@Q).reshape(T,1)
        
        y_trend  =  F_y + start_intercept[-1]
        numpyro.deterministic("y_trend",y_trend)

        resid        =  ((y_target.reshape(T,1) - centers)/scales) - y_trend
        ymask        = ymask.reshape(T,1)
        
        resid_filled =  jnp.where(ymask, resid, 0.0)

        resid_filled_rev = resid_filled[::-1]
        ymask_rev        = ymask[::-1] 
        
        #--The end is time 0
        (m_last, P_last),(LL_y,mp1, Pp1, m_t, P_t, y_p_pred) = jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=y_path_scale**2,rho=rho)
                                                                                              , init  = ( start_mean[-1], start_sigma[-1]**2 )
                                                                                              , xs    = ( resid_filled_rev.squeeze(), ymask_rev.squeeze() ) )
 
        numpyro.factor("LLY", jnp.sum(LL_y))

        if forecast:
            innov_z      = numpyro.sample("new_innov_z"    , dist.Normal(0,1).expand([ T-1 ])) #<--one for P and one for Q
            eps          = innov_z

            xT           = numpyro.sample("xT", dist.Normal(m_t[-1], jnp.sqrt(P_t[-1]+10**-6) )) #<--this is time zero
            def step_ffbs(x_next, inputs,q):
                eps_t, m, P = inputs  # these are at time t

                mp1 = rho * m
                Pp1 = (rho**2) * P + q          # <-- use your q here

                J    = rho * P / (Pp1 + 1e-6)
                mean = m + J * (x_next - mp1)

                var  = P - (J * J) * Pp1
                var  = jnp.clip(var, 1e-6, 1e6)

                x_t  = mean + jnp.sqrt(var) * eps_t
                return x_t, x_t

            # use t = 0..T-2 (so m_t[:-1], P_t[:-1])
            inputs = (eps, m_t[:-1], P_t[:-1])
            _, xs_rev = jax.lax.scan( lambda x,y: step_ffbs(x,y,q=y_path_scale**2), init=xT, xs=inputs, reverse=True)
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


    def build_basis_for_F(self, both=False):
        # time grid
        T = self.T
        M = self.M

        #--Choose fixed spline settings and pad them from -M to M for a total of 2M+1 potential reference points (dont forget zero)
        lb, ub = -M, (T) + M
        t      = np.arange(lb,ub+1)

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

        knots_full = clamped_uniform_knots(lb, ub, n_basis=10, degree=3)
        
        B                 = bspline_basis(t,knots_full,3)

        self.B      = jnp.array(B)
        self.Sb     = 1

        #--While we're here we should compute the matrix of second differences (D) and the penalty matrix Kp
        D          = jnp.diff(jnp.diff(jnp.eye(B.shape[-1]),axis=0),axis=0)
        
        Kp         = D.T@D
        
        self.Kp                 = Kp
        self.Kp_l, self.Kp_U    = jnp.linalg.eigh(Kp)

        if both:
            return B,Kp
        else:
            return B

    def fit(self
            , M                          = 15
            , estimated_num_components_y = None):

        y, Y, X     = self.y, self.Y, self.X
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

        B,_ = self.build_basis_for_F(both=True)

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
        
        if X is not None:
           y_past = np.hstack([ X, Y[0] ])
           target_indicators = [target_indicators[0]+X.shape[-1]]
           self.target_indicators_mcmc = target_indicators
        else:
            y_past = Y[0]
        self.y_past = y_past

        print(self.tobs)

        #--MCMC start
        dense_blocks = [
            ("col_scale",),
            ("mu_log_ps","tau_log_ps"),
        ]
        
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
                              ,y_past            = y_past
                              ,y_target          = y
                              ,B                 = self.B
                              ,IL                = self.IL
                              ,Kp_l              = self.Kp_l
                              ,Kp_U              = self.Kp_U
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

        #--MCMC START
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred","xT","xs_rev","x_path"] )
        #--MCMC END

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,y_past            = self.y_past
                              ,y_target          = self.y
                              ,B                 = self.B
                              ,IL                = self.IL
                              ,Kp_l              = self.Kp_l
                              ,Kp_U              = self.Kp_U
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
