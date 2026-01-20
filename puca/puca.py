#mcandrew

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsp

import numpyro
import numpyro.distributions as dist
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

        #--We need to build F as a set of P-splines that are rperesented as B @ beta
        Text,K          = B.shape
        T               = y_past.shape[0] 
        S_past_total    = y_past.shape[-1]
        S               = S_past_total + 1
        L               = IL.shape[0]

        M               = 10
        
        #--This is a smoothness penalty 
        #tau_diff = jnp.ones((L,))*0.1

        sd_tau_diff = numpyro.sample("sd_tau_diff", dist.HalfNormal(1))
        z__tau_diff = numpyro.sample("z__tau_diff", dist.Normal(0,1).expand([L]))
        tau_diff    = jnp.exp(jnp.log( 0.1 ) + sd_tau_diff*z__tau_diff )

        null        = 2
        U0          = Kp_U[:, :null]        
        Up          = Kp_U[:, null:]
        lp          = Kp_l[null:]

        beta0       =  jnp.zeros( (U0.shape[1], L) ) #
        
        z_beta      = numpyro.sample("beta", dist.Normal(0,1).expand([lp.shape[0] ,L]) )
        
        def produce_beta( tau_diff, z_beta, Up, lp ):
            on_diag    =   (lp/tau_diff**2) #+ 10**-6
            beta       =  ( Up * (1/jnp.sqrt(on_diag)  ))   @ z_beta
            return beta
        beta = jax.vmap( lambda x,y:produce_beta(x,y,Up,lp), in_axes = (0,1) )( tau_diff, z_beta  )
        beta = beta.T

        beta = U0@beta0 + beta

        X_shifts, y_shift = column_shifts[:S_past_total], column_shifts[-1]
        
        base        = jnp.arange(T)
        
        B_shifted   = B[ (M+base)[:,None] - X_shifts[None,:]  ]
        B_shifted   = B_shifted - B_shifted[0,...][None,...] # <- enforce a zero start
        
        Fx          = B_shifted @ beta              #<--T X S X L
        
        mFx         = jnp.mean( Fx, axis=0 ).reshape(1, S-1, L)
        sFx         = jnp.std(  Fx, axis=0 ).reshape(1, S-1, L)
        Fx          = (Fx - mFx ) / sFx
        
        numpyro.deterministic("Fx",Fx)
       
        # #--P matrix so that X = Fnorm @ P
        P_global_prec = numpyro.sample("P_global_prec" , dist.Gamma(2,2./4))
        P_global_prec = 1./ jnp.sqrt(P_global_prec)
        
        P_local_prec   = numpyro.sample("P_local_prec" , dist.Gamma(2, 2.).expand([L,1]) )
        P_local_prec   = Trans.OrderedTransform()(P_local_prec)
        
        gScale        = (P_global_prec * P_local_prec)

        # n_tril = L * (L - 1) // 2

        # if n_tril > 0:
        #     r,c          = jnp.tril_indices(L,k=-1)
        #     P_zeros      = jnp.zeros((L,L))
        #     z_P_blk_raw  = numpyro.sample("z_P_blk_raw", dist.Normal(0,1).expand([ int(L*(L-1)/2) ] ))
        #     P_blk        = gScale*( P_zeros.at[r,c].set(z_P_blk_raw) ) + IL
        # else:
        #     P_blk = IL

        # P_rest       = numpyro.sample("P_rest", dist.Normal(0,1).expand([L, S - L]))
        # P_rest       = gScale*P_rest
        
        # A            = jnp.concatenate([P_rest, P_blk], axis=1) #<--LXS
        
        # Q_scale = numpyro.sample("Q_scale", dist.Normal(0,1) )
        # Q_scale = jnp.exp( Q_scale * jnp.log(2)/2 )

        S_tot = S  # include target season too
                
        # row scales (you already have gScale; shape (L,1) or (L,))
        row_scale = gScale.squeeze(-1)  # (L,)

        # exchangeable correlation across seasons
        rho = numpyro.sample("rho_season", dist.Beta(5., 2.))  # encourages rho > 0
        R = (1. - rho) * jnp.eye(S_tot) + rho * jnp.ones((S_tot, S_tot))
        chol_R = jnp.linalg.cholesky(R + 1e-6 * jnp.eye(S_tot))  # jitter

        # optional per-season amplitude
        col_scale = numpyro.sample("col_scale", dist.HalfNormal(1.).expand([S_tot]))  # (S,)
        Lcol = jnp.diag(col_scale) @ chol_R  # this is D * L_R

        # iid standard normal
        Z = numpyro.sample("Z_A", dist.Normal(0,1).expand([L, S_tot]))

        # A = S_row * Z * (D L_R)^T
        A = row_scale[:, None] * (Z @ Lcol.T)  # (L,S)

        P = A[:,1:]
        Q = A[:,0]

        numpyro.deterministic("A", A)
        numpyro.deterministic("P", P)
        numpyro.deterministic("Q", Q)

        #--AR
        eps         = 1.
        
        q            = numpyro.sample("qh", dist.Uniform(0.001,2).expand([S]))
        path_scale   = 0.5+q
       
        start_sigma = jnp.ones((S,))*0.0001
        start_mean  = jnp.zeros( (S,) )  

        Xtarget = jnp.hstack(y_past).reshape(T,S_past_total)

        Xtrend  = jnp.einsum("tsl,ls->ts",Fx,P) #<-TXS
        Xtrend  = Xtrend #+ season_specific_intercept[None, :-1]
        numpyro.deterministic("Xtrend",Xtrend)

        resid          = Xtarget - Xtrend

        Xmask        = jnp.isfinite(resid)
        resid_filled = jnp.where(Xmask, resid, 0.0)

        rho          = 1

        sd_r_pins = numpyro.sample("sd_r_pins" , dist.HalfNormal(1/2))
        mn_r_pins = numpyro.sample("mn_r_pins" , dist.Normal(0,1))
        z__r_pins = numpyro.sample("z__r_pins" , dist.Normal(0,1).expand([S]))
        
        r_pins    = 0.01 + jnp.exp( (mn_r_pins) + z__r_pins*sd_r_pins)

        #--kalm likelihood here
        def kf(carry, array, q, r, rho, r_pin):
            mt,Pt    = carry
            yobs,mk  = array

            m_pred = rho*mt
            p_pred = (rho**2)*Pt+q

            y_p_pred = (rho**2)*Pt+q+r
            
            S = p_pred + r
            K = p_pred / S

            innov      = (yobs-m_pred)*mk
            
            m_filt = m_pred + K*innov
            P_filt = p_pred * (1-K*mk)

            LOG2PI = jnp.log(2.0 * jnp.pi)
            ll     = mk*(-0.5 * (LOG2PI + jnp.log(S) + (innov**2) / S))

            #pseudo-observation: z = 0 ~ N(m, P + r_pin)

            S_pin = P_filt + r_pin
            Kp    = P_filt / (S_pin)
            
            ll_pin = -0.5 * (jnp.log(2*jnp.pi) + jnp.log(S_pin) + (0-m_filt)**2 / (S_pin))

            mpost = m_filt + Kp*(0-m_filt)
            ppost = P_filt - Kp*P_filt
            
            ll_ttl = ll + ll_pin
            
            return (mpost,ppost), (ll_ttl, m_pred, p_pred, mpost, ppost  )
            #return (xp_y_E,xp_y_P), (ll, xp_E, xp_P, xp_y_E, xp_y_P  )
        
        _,(LL_xs,_,_,_,_) = jax.vmap( lambda path_scale_indiv, data, mask, start_m,start_s, r_pin : jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=path_scale_indiv**2,rho=rho, r_pin=r_pin**2) , init = ( start_m,start_s  ), xs = ( data, mask.squeeze() ), reverse=True   )
                            , in_axes=(0,1,1,0,0,0) )(path_scale[:-1], resid_filled, Xmask, start_mean[:-1], start_sigma[:-1], r_pins[:-1])
        numpyro.factor( "LL_x", jnp.sum(LL_xs) )

        # with numpyro.handlers.mask(mask=Xmask):
        #     numpyro.sample("LLX", dist.Normal( Xtrend, eps ), obs = Xtarget )

        eps_scale = 1
        eps_y     = (eps*eps_scale) * scales

        y_target = jnp.hstack(y_target).reshape(T,1)
        
        ymask     = jnp.isfinite(y_target)

        y_pin = r_pins[-1]
        
        #--shifting probabilities
        #index_weights = index_weights+10**-2
        #index_probs   = numpyro.sample( "index_weights", dist.Dirichlet( index_weights) )
        #index_probs = jnp.clip(index_probs, 1e-12, 1.0)

        #print(self.column_shifts)
        #print(y_shift)
        #index_probs = jnp.zeros( (len(index_weights)) )
        #index_probs = index_probs.at[y_shift].set(1.)
        
        def from_shift_to_prob(shift, y_path_scale, y_pin):
            B_y      = B[M+base-shift,:] #<--   y_target is the last shifted index
            B_y      = B_y - B_y[0,:]

            F_y      = (B_y@beta@Q).reshape(T,1)
            y_trend      = F_y 
            
            resid        =  ((y_target - centers)/scales) - y_trend
            ymask        =  jnp.isfinite(resid)
            resid_filled =  jnp.where(ymask, resid, 0.0)

            (m_last, P_last),(LL_y,xstep_E,xstep_P,xE,xP) = jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=y_path_scale**2,rho=rho,r_pin=y_pin**2) , init = ( start_mean[-1], start_sigma[-1] ), xs = ( resid_filled.squeeze(), ymask.squeeze() ), reverse=True)

            return jnp.sum( LL_y ), xstep_E,xstep_P,xE,xP,m_last, P_last

        start        = jnp.zeros((1,))  
        shift        = y_shift
        y_path_scale = path_scale[-1]

        
        B_y      = B[M+base-shift,:] #<--   y_target is the last shifted index
        B_y      = B_y - B_y[0,:]

        F_y      = (B_y@beta@Q).reshape(T,1)
        y_trend  =  F_y 
        numpyro.deterministic("y_trend", y_trend)

        shift                                   = y_shift  # deterministic shift value
        LLY, mp1, Pp1, m_t, P_t, m_last, P_last = from_shift_to_prob(shift, y_path_scale, y_pin)
        numpyro.factor( "LLY", LLY )

        # shift = y_shift
        
        # B_y      = B[M+base-shift,:] #<--   y_target is the last shifted index
        # B_y      = B_y - B_y[0,:]
        
        # Fy      = (B_y@beta).reshape(T,L)

        # mFy         = jnp.mean( Fy, axis=0 ).reshape(1,L)
        # sFy         = jnp.std( Fy, axis=0 ).reshape(1,L)
        # Fy          = (Fy - mFy ) / sFy

        # Fy = (Fy @ Q).reshape(T,1)
        # y_trend      = numpyro.deterministic("y_trend", Fy )

        # with numpyro.handlers.mask(mask=ymask):
        #     numpyro.sample("LLY", dist.Normal( y_trend, eps_y ), obs = y_target )
        
       
        if forecast:
            innov_z      = numpyro.sample("new_innov_z"    , dist.Normal(0,1).expand([ T ])) #<--one for P and one for Q
            eps          = y_path_scale*innov_z

            xT           = numpyro.sample("xT", dist.Normal(m_t[-1], jnp.sqrt(P_t[-1]) ))
            
            def step_ffbs(x_next, inputs):
                eps_t, m_t, P_t, mp1, Pp1 = inputs  # all scalars
                J    = rho * P_t / (Pp1 )
                mean = m_t + J * (x_next - mp1)
                var  = P_t - (J * J) * Pp1
                #var  = jnp.maximum(var, tiny)
                x_t  = mean + jnp.sqrt(var) * eps_t
                return x_t, x_t

            inputs_rev = (eps[:-1][::-1], m_t[:-1][::-1], P_t[:-1][::-1], mp1[:-1][::-1], Pp1[:-1][::-1])
            _, xs_rev  = jax.lax.scan(step_ffbs, init=xT, xs=inputs_rev)
            x_path     = jnp.concatenate([xs_rev[::-1], xT[None]], axis=0)
            
            numpyro.sample( "y_pred", dist.Normal( (y_trend+x_path.reshape(T,1) )*scales + centers, eps_y))
                           
            #--trned component
            #shift_post = jax.nn.softmax(logw)              # normalized responsibilities
            #shift_idx   = numpyro.sample("shift", dist.Categorical(shift_post) )
            #shift       = shifts[shift_idx]
            # shift = y_shift

            #blarp
            
            # B_y      = B[M + base - shift,:]
            # B_y      = B_y - B_y[0,:] 
            
            # F_y      = numpyro.deterministic("F_y", (B_y@beta@Q).reshape(T,1))

            # #--noise path component
            

            # start        = numpyro.sample("pred_start", dist.Normal( start_mean[-1], start_sigma[-1])).reshape(1,1)
            # 
            # numpyro.deterministic("innovations",innovations)

            # y_path = numpyro.sample( "y_path", dist.Normal(ye,yp)  ) 
            
            # y  = F_y + season_specific_intercept[-1] +  y_path[:,None]
            # yhat_    = y *scales + centers

            # #numpyro.deterministic("y_trend_pred", y_trend*scales + centers)
            # numpyro.sample("y_pred", dist.Normal(yhat_, eps_y) )

            # numpyro.deterministic("path_scale", path_scale)
            # numpyro.deterministic("y_path_scale", y_path_scale)


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

        column_shifts = []
        for n,column in enumerate(y_curves.T):
            sses = []
            for shift in np.arange(-10,10+1,1):
                shifted_column = apply_shift_1d(column, shift)
                sses.append( (np.nansum( (shifted_column - y_candidate)**2 ) , shift) )
            best = sorted(sses)[0][-1]

            column_shifts.append(best)
        return column_shifts

    def build_basis_for_F(self):
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
        lb, ub = -M, (T-1) + M
        t      = np.arange(lb,ub+1)
        knots  = np.arange(lb, ub, 1)[1:] #was originally 2

        B            = bs_basis_numpy(t,knots,lb,ub)

        #--Normalize the basis 
        D            = jnp.std(B, 0) 
        Sb           = jnp.diag( D )
        Sbinv        = jnp.diag(1./D)

        B_norm       = B @ Sbinv

        self.B      = jnp.array(B_norm)    #<--Normalized B
        self.Sb     = Sb        #<--Scaling matrix for B

        #--While we're here we should compute the matrix of second differences (D) and the penalty matrix Kp
        D          = jnp.diff(jnp.diff(jnp.eye(B.shape[-1]),axis=0),axis=0)
        #D         = jnp.diff(D,axis=0)
        
        Kp         = D.T@D
        
        self.Kp                 = Kp
        self.Kp_l, self.Kp_U    = jnp.linalg.eigh(Kp)

        return B_norm, Kp

    def fit(self
            , M                          = 10
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

        B,_ = self.build_basis_for_F()

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
 
        # optimizer = ClippedAdam(step_size= 4*10**-5, clip_norm=1)  # common starting point: 1e-3 to 1e-2
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
            ("z_y",),
            ("beta","alpha"),                            # spline shape + smoothness + level
            ("z_P_blk_raw", "P_global_prec", "P_local_prec"),          # P hierarchy
            ("Qraw", "Q_global_prec"),
            ("intercept_perc", "season_specific_intercept"),
            ("global_prec", "local_prec")#,"innov_z"),  # RW scales (optionally add "start_z")
        ]
        
        nuts_kernel = NUTS(self.model#neutra_model
                           , init_strategy = init_to_median(num_samples=100)
                           , dense_mass = [("beta",)]
                           ,  find_heuristic_step_size=True)     
        kernel      = nuts_kernel 
        mcmc        = MCMC(kernel
                    , num_warmup     = 2000
                    , num_samples    = 2500
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
        #                        , return_sites  = ["y_pred","y_trend_pred","Fx","F_y","path_scale","y_path_scale","innovations","eps_kf","kappa","log_kappa_sigma","log_kappa_mu"] )
        #--SVI END

        #--MCMC START
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred","y_trend_pred"] )
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
