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
            #past_smooth_means =  np.mean(past_smooth,axis=0)
            #past_smooth_stds  =  np.std(past_smooth,axis=0)

            #smooth_y          = (past_smooth - past_smooth_means)/past_smooth_stds
            smooth_ys.append(smooth_y)
            
            if n==0:
                all_y = np.hstack([smooth_y])
            else:
                _     = np.hstack([smooth_y])
                all_y = np.hstack([all_y,_])
       

        self.global_mu  = np.nanmean( np.nanmean(all_y,0))
        self.global_std = np.nanmean( np.nanstd(all_y,0))

        #all_y = np.hstack([ all_y,  current.reshape(-1,1)])
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
              ,target_centers    = None
              ,target_scales     = None
              ,Ls                = None
              ,IK                = None
              ,IL                = None
              ,Kp                = None 
              ,copies            = None
              ,tobs              = None
              ,target_indicators = None
              ,Kmix              = 1
               ,scales           = None
               ,centers          = None   
              ,forecast          = None):

        #--We need to build F as a set of P-splines that are rperesented as B @ beta
        Text,K          = B.shape
        T               = y_past.shape[0] 
        S_past_total    = y_past.shape[-1]
        S               = S_past_total + 1
        L               = IL.shape[0]

        M=20
        
        #--This is a smoothness penalty 
        meps       = 10**-3
        tau_diff   = numpyro.sample("tau_diff"  , dist.HalfNormal(1))    #smoothness_switch))  # small => smooth (scale depends on your standardization!)
        L_prec     = jnp.linalg.cholesky( (Kp / (tau_diff**2)) + meps * IK )

        z_beta     = numpyro.sample("beta", dist.Normal(0,1).expand([K,L]) )
        beta       = jsp_linalg.solve_triangular(L_prec, z_beta, lower=True)

        b           = numpyro.sample("b", dist.Normal(0, 1.).expand([L]))   # (L,)
        
        F_extended = B@beta
        Fbase      = F_extended[M:M+T,:]
        sF         = jnp.std(Fbase,0)                       #-- L We should normalize the columns of F should that X and y can share it
        Fx         = Fbase/sF[None,:] + b
        numpyro.deterministic("Fx",Fx)

        shift_select = numpyro.sample("shift_select", dist.Categorical( jnp.ones(2*M+1)/(2*M+1) ).expand([L]) )
        idx = jnp.arange(T, dtype=jnp.int32)[None, :] + shift_select[:, None]  # (L, T)
        
        #idx          = jnp.arange(T, dtype=jnp.int32) + shift_select
        #F_select     = jnp.take(F_extended, idx, axis=0)
        F_select = jnp.take_along_axis(F_extended.T, idx, axis=1).T  # (T, L)
        
        Fy        = F_select/sF[None,:] + b
        numpyro.deterministic("Fy",Fy)
        
        #--P matrix so that X = Fnorm @ P
        tau_P     = numpyro.sample("tau_P"   , dist.HalfNormal(1.0))
        lambda_P  = numpyro.sample("lambda_P", dist.HalfNormal(1.0).expand([L,1]))

        P_blk_raw = numpyro.sample("P_blk_raw", dist.Normal(0,1).expand([L,L]))
        P_blk     = jnp.tril(P_blk_raw, k=-1) + IL
        P_blk     = P_blk * tau_P * lambda_P  

        P_rest    = numpyro.sample("P_rest", dist.Normal(0,1).expand([L, S -1 - L]))
        
        P         = jnp.concatenate([P_rest, P_blk], axis=1)
        P         = numpyro.deterministic("P", P)# * sF[:,None])
        
        Qraw      = numpyro.sample("Qraw", dist.Normal(0,1).expand([L,1])  )
        Q_scales  = numpyro.sample("Q_scales", dist.HalfNormal(1).expand([L,1]))
        Q         = Q_scales*Qraw
        numpyro.deterministic("Q", Q)
        
        #--pbsevration space
        global_scale      = numpyro.sample("global_scale", dist.HalfNormal(1 * jnp.sqrt(jnp.pi/2)  ))
        local_scale       = numpyro.sample("local_scale" , dist.HalfNormal(1./2).expand([S]))

        #log_global_scale = numpyro.sample("log_global_scale", dist.Normal(0, 1))
        #global_scale = jnp.exp( log_global_scale - 2)
        #numpyro.deterministic("global_scale", global_scale)

        #log_local_scale = numpyro.sample("log_local_scale", dist.Normal(0, 1).expand([S]))
        #local_scale = jnp.exp(log_local_scale - 0.5)
        #numpyro.deterministic("local_scale", local_scale)


        #start_scale       = numpyro.sample("start_scale", dist.HalfNormal(1./5)) 
        start_z           = jnp.zeros((S,1))#numpyro.sample("start_z", dist.Normal(0,1).expand([S,1]) )
        innov_z           = numpyro.sample("innov_scale", dist.Normal(0,1).expand([S,T-1])) #<--one for P and one for Q

        innovations       =  jnp.hstack([ (global_scale*local_scale)[:,None]*innov_z, start_z]) #<-- S,T
       
        paths             =  jnp.cumsum(innovations[:,::-1],1)[:,::-1] #<--this is  S X T
        numpyro.deterministic("paths", paths)

        X            = (Fx@P).T 
        X            = (X + paths[:-1,:]).T

        X_target   = jnp.hstack(y_past).reshape(T,S_past_total)
        present    = jnp.isfinite(X_target)

        maskX  = jnp.isfinite(X_target)
        X_fill = jnp.where(maskX, X_target, 0.0)

        #log_eps          = numpyro.sample( "eps", dist.Normal(0,1) ) # 1./5
        #log_eps =  numpyro.sample("log_eps", dist.Normal(0,jnp.log(2)/2) )#  (1./10)   #jnp.exp( -jnp.log(10) + log_eps*jnp.log(2/2) )
        #eps     = jnp.exp(log_eps-1)

        eps = 0.05
        with numpyro.handlers.mask(mask=maskX):
            numpyro.sample("X_ll", dist.Normal(X, eps), obs=X_fill)
       
        y = (Fy@Q).reshape(T,1)
        numpyro.deterministic("y",y)

        y = y+paths[-1,:][:,None]
        
        yhat_    = (y*scales) + centers
        numpyro.deterministic("yhat_", yhat_)

        eps_y  = (eps) * scales

        y_target = jnp.hstack(y_target).reshape(T,1)
        
        mask     = jnp.isfinite(y_target)
        y_obs    = jnp.where(mask, y_target, 0.0)

        numpyro.deterministic("y_target", y_target)
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("y_ll", dist.Normal(yhat_.reshape(T,1), eps_y), obs = y_obs.reshape(T,1))

        if forecast:
            y_old_innovations = innov_z[-1,:]  #1,T-1
            y_old_innovations = y_old_innovations[:tobs]

            y_new_innovations = numpyro.sample("new_y_innov_scale"  , dist.Normal(0,1).expand([T-tobs-1 ,1]))

            this_local_scale  = local_scale[-1]
            innovations       = jnp.vstack([ (global_scale*this_local_scale)* y_old_innovations[:,None], (global_scale*this_local_scale)*y_new_innovations, start_z[-1][None,:] ]) #<-- T,1
            reverse_path      = jnp.cumsum(innovations[::-1,:], axis=0)[::-1,:]

            new_paths = reverse_path
            y         = (Fy@Q).reshape(T,1) + new_paths

           
            yhat_    = jnp.clip( y*scales + centers, 0, None)
            numpyro.sample("y_pred", dist.Normal(yhat_, eps_y) )



        # def shift_zero_int(x, s):
        #     """
        #     x: (T,) or (T,L)
        #     s: integer (can be traced)
        #     positive s shifts RIGHT (later)
        #     """
        #     T = x.shape[0]
        #     y = jnp.roll(x, s, axis=0)
        #     idx = jnp.arange(T)

        #     def pos(_):
        #         m = (idx >= s)
        #         m = m.reshape((T,) + (1,)*(x.ndim-1))
        #         return y * m

        #     def neg(_):
        #         m = (idx < T + s)   # since s is negative
        #         m = m.reshape((T,) + (1,)*(x.ndim-1))
        #         return y * m

        #     return jax.lax.cond(s >= 0, pos, neg, operand=None)

        # def shift_zero_continuous(x, delta):
        #     k = jnp.floor(delta).astype(jnp.int32)
        #     a = delta - jnp.floor(delta)
        #     x0 = shift_zero_int(x, k)
        #     x1 = shift_zero_int(x, k + 1)
        #     return (1.0 - a) * x0 + a * x1

        # def sample_shifted(F_ext, delta, M):
        #     """
        #     F_ext: (Text, L)
        #     delta: scalar float (can be traced)
        #     Returns F_shift: (T, L) using linear interpolation
        #     """
        #     T   = F_ext.shape[0] - 2*M
        #     pos = jnp.arange(T) + M - delta          # positions in F_ext
        #     i0  = jnp.floor(pos).astype(jnp.int32)
        #     a   = pos - jnp.floor(pos)

        #     f0 = F_ext[i0, :]
        #     f1 = F_ext[i0 + 1, :]
        #     return (1.0 - a)[:, None] * f0 + a[:, None] * f1

        # def sample_shifted_1d(f_ext, delta, M, T):
        #     """
        #     f_ext: (Text,) where Text = T + 2*M
        #     delta: scalar shift (float). Positive shifts RIGHT (later).
        #     returns: (T,)
        #     """
        #     Text = f_ext.shape[0]

        #     # positions in f_ext we want to sample
        #     pos = jnp.arange(T) + M - delta
        #     i0  = jnp.floor(pos).astype(jnp.int32)
        #     a   = pos - jnp.floor(pos)

        #     # keep i0 in-bounds so i0+1 is valid
        #     i0 = jnp.clip(i0, 0, Text - 2)

        #     f0 = jnp.take(f_ext, i0)
        #     f1 = jnp.take(f_ext, i0 + 1)
        #     return (1.0 - a) * f0 + a * f1



        #delta     = numpyro.sample("delta_fixed", dist.Uniform(-20+1,20-1)  )
        #Fshifted  = sample_shifted(F_extended, delta, M)   #jax.vmap(lambda fcol, d: sample_shifted_1d(fcol, d, 20,T),in_axes=(1, 0), out_axes=1)( F_extended  , deltas)
 

        #F = shift_zero_continuous( B @ beta, delta_fixed)
        
        #F = B @ beta #<< This is Tpadded X L

        # delta = numpyro.sample("delta", dist.Uniform(-M+1., M-1.))  # keep within support
        # F = sample_shifted(F, delta, M)         # (T, L)

        
        #beta         = Sb @ beta #<-- (B @ Sbinv) @ (Sb @ beta). In other words, normalize the columns of our B matrix
        
        #--Lets assume that the F columns (it the latent time series) are correlated
        #F          = B@beta                             #--(T,K) @ (K,L) -> T X L
        #Fnorm      = F/sF[None,:] 
 

        #--center and scale for y
        #Kmix = 2
        # centers, scales = [],[]
        # for signal, (cents, scals) in enumerate( zip(target_centers, target_scales) ):
        #     N           = len(cents)
        #     log_scals   = jnp.log(1+scals)

        #     mu_cents    = jnp.mean(cents)
        #     sigma_cents = jnp.std(cents) 
        #     zc          = ((cents - mu_cents) / sigma_cents) .reshape(-1,1)

        #     mu_scals    = jnp.mean(log_scals)
        #     sigma_scals = jnp.std(log_scals)
        #     zs = ((log_scals - mu_scals)/ sigma_scals).reshape(-1,1)

        #     cs_vec = jnp.hstack([zc,zs])

        #     # --- Bayesian Gaussian mixture in z-space -----------------------------
        #     # weights
        #     pi = numpyro.sample(f"pi_{signal}", dist.Dirichlet((1./Kmix)*jnp.ones(Kmix)))

        #     # component means (in z-space)
        #     mu_k = numpyro.sample(
        #         f"mu_k_{signal}",
        #         dist.Normal(0.0, 1.5).expand([Kmix, 2])
        #     )

        #     # component covariance via (scales * LKJ correlation)
        #     # per-component marginal scales
        #     sigma_k = numpyro.sample(
        #         f"sigma_k_{signal}",
        #         dist.HalfNormal(1.0).expand([Kmix, 2])
        #     )
        #     # per-component correlation Cholesky
        #     Lcorr_k = numpyro.sample(
        #         f"Lcorr_k_{signal}",
        #         dist.LKJCholesky(dimension=2, concentration=2.0).expand([Kmix])
        #     )  # (Kmix, 2, 2)

        #     D_k = jax.vmap(jnp.diag)(sigma_k)                      # (Kmix, 2, 2)
        #     L_k = jax.vmap(lambda D, L: D @ L)(D_k, Lcorr_k)       # (Kmix, 2, 2) scale_tril

        #     comp = dist.MultivariateNormal(loc=mu_k, scale_tril=L_k)  # batch (Kmix,), event (2,)
        #     mix  = dist.MixtureSameFamily(dist.Categorical(probs=pi), comp)

        #     numpyro.sample(f"cs_obs_{signal}", mix.expand([N]), obs=cs_vec)

        #     choice = numpyro.sample(f"choice_{signal}", mix)  # shape (2,)
        #     center_choice_z, scal_choice_z = choice[0], choice[1]

        #     #map back to natural scale
        #     center_choice = center_choice_z * sigma_cents + mu_cents
        #     scal_choice   = jnp.expm1(scal_choice_z * sigma_scals + mu_scals)

        #     centers.append(center_choice)
        #     scales.append(scal_choice)
        
        # centers = jnp.hstack(centers)
        # #scales  = jnp.hstack(scales) 




        # y_paths = paths[-1,:][:,None]
        # shifts  = jnp.arange(0,2*M+1)

        # def compute_LL_for_shift(shift, y_paths, present):
        #     idx     = jnp.arange(T, dtype=jnp.int32) + shift
        #     #F_      = jnp.take(F_extended , idx , axis=0 )
        #     F_      = jax.lax.dynamic_slice(F_extended, (shift, 0), (T, F_extended.shape[1]))

        #     mu        = (F_/sF[None,:] + b)@Q + y_paths
        #     mu_scaled = mu*scales + centers 
            
        #     loglike = dist.Normal( mu_scaled , eps_y ).log_prob( y_obs )
        #     return jnp.sum(jnp.where(present, loglike, 0.0))  
        # all_LLS = jax.vmap( lambda x: compute_LL_for_shift(x,y_paths,mask) )( shifts  )

        # LLy = 0.5*jax.scipy.special.logsumexp( all_LLS/ (0.5) ) - jnp.log( shifts.size )
        # numpyro.factor("LLy",LLy)

        #numpyro.deterministic("Fy",Fy)



        # # #--sample this for y so that its not annoying in predcition
        # start_z = numpyro.sample("start_scale", dist.Normal(0,1).expand([1,L]) )
        # innov_z = numpyro.sample("innov_scale", dist.Normal(0,1).expand([T-1,L])) #<--one for P and one for Q

    # innovations =  (global_scale*local_scale).reshape(1,L) * jnp.vstack([ innov_z[::-1,:], start_z]) 

        # y_path       = jnp.cumsum(innovations,0)[::-1,:] #<--this is  T X L
        # numpyro.deterministic("y_path", y_path)
        
 

        # def Kalmanfilter_Xt( carry, array, A, H, R, Q ):
        #     m_, P_      = carry
        #     yt,mask     = array   # S X 1

        #     yt   = jnp.squeeze(yt)                        # ensure (S,)
        #     mask = jnp.squeeze(mask).astype(y.dtype)  # ensure (S,)

        #     #Reff = R + jnp.diag((1.0 - mask) * 10**-10)

        #     #--assume we are at time t now
        #     #--we assume that we carried foreward xt-1|yt-1
            
        #     #--p(yt | y1:t-1) = sum p(yt | xt) p(xt|y1:t-1) dxt
            
        #     #--xt|y1:t-1
        #     mu_xt_t1 = A@m_                       # LX1
        #     P_xt_t1  = A @ P_ @ A.T + Q           # LXL

        #     #--p(yt|y1:t-1)
        #     mu_yt_yt1 = H @ mu_xt_t1              # SX1
        #     P_yt_yt1  = H @ P_xt_t1 @ H.T + R     # SXS

        #     #--p(xt|y1:t) = p(xt|yt,y1:t-1) = p(yt|xt) p(xt|y1:t-1) or form p( xt_t1, yt ) and condition
        #     eps2 = R[0, 0]                      # assumes R is eps2 * I
        #     rinv = 1.0 / eps2

        #     HtH = H.T @ H                       # (L,L)

        #     # P^{-1} (small L×L)
        #     Lp    = jnp.linalg.cholesky(P_xt_t1)
        #     P_inv = jsp.cho_solve((Lp, True), jnp.eye(P_xt_t1.shape[0]))

        #     # A = P^{-1} + H^T R^{-1} H  (L×L)
        #     A_small = P_inv + rinv * HtH
        #     LA      = jnp.linalg.cholesky(A_small + 1e-9 * jnp.eye(A_small.shape[0]))

        #     # K = A^{-1} H^T R^{-1}  (L×S)
        #     K   = jsp.cho_solve((LA, True), rinv * H.T)

        #     #inv_P_yt_yt1 = jnp.linalg.inv(P_yt_yt1)
        #     #K            = P_xt_t1 @ H.T @ inv_P_yt_yt1

        #     mu_xt_t = mu_xt_t1 + K @ (yt - H@mu_xt_t1)
        #     I       = jnp.eye(len(K))
        #     P_xt_t  = P_xt_t1  - K @ H @ P_xt_t1

        #     #log_prob = dist.MultivariateNormal( mu_yt_yt1, P_yt_yt1 ).mask(mask).log_prob( yt )

        #     L = jnp.linalg.cholesky(P_yt_yt1)
        #     innov      = yt - H@mu_xt_t1
        #     Sinv_innov = jsp.cho_solve((L, True), innov)
        #     logdetS    = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        #     d          = y.shape[0]
        #     log_prob   = -0.5 * (d * jnp.log(2*jnp.pi) + logdetS + innov.T @ Sinv_innov)
            
        #     return (mu_xt_t, P_xt_t), log_prob

        # def Kalmanfilter_Xt( carry, array, H, eps, Is, sigma,Il, sigma_HHT, sigmaH, Htsigma, S, HTH ):
        #     m_, P_      = carry
        #     yt,mask     = array   # S X 1

        #     yt   = jnp.squeeze(yt)                    # ensure (S,)
        #     mask = jnp.squeeze(mask).astype(y.dtype)  # ensure (S,)

        #     #--assume we are at time t now
        #     #--we assume that we carried foreward xt-1|yt-1
            
        #     #--Goal -> p(yt | y1:t-1) = sum p(yt | xt) p(xt|y1:t-1) dxt
            
        #     #--xt|y1:t-1
        #     mu_xt_t1 = m_                       # LX1
        #     P_xt_t1  = P_ + sigma*Il          # LXL

        #     #--p(yt|y1:t-1)
        #     mu_yt_yt1 = H @ m_
        #     P_yt_yt1  = H@ P_ @ H.T + sigma_HHT + eps*Is     # SXS

        #     #--p(xt|y1:t) = p(xt|yt,y1:t-1) = p(yt|xt) p(xt|y1:t-1) or form p( xt_t1, yt ) and condition
        #     temp = jnp.linalg.inv(P_xt_t1) + (1./eps)*HTH
        #     Sinv = (1./eps)*Is - (1./eps**2)*H@jnp.linalg.solve(temp, H.T )
            
        #     K    = (H@P_ + sigmaH).T @ Sinv  

        #     innov   = (yt - H@m_)
        #     mu_xt_t = m_ + K @ innov
            
        #     P_xt_t  = P_xt_t1  - K @ ( P_@H.T + Htsigma ).T

        #     jitter = 1e-7

        #     # --- stabilize (4x4 only)
        #     P_pred = 0.5 * (P_xt_t1 + P_xt_t1.T) + jitter * Il
        #     temp_  = 0.5 * (temp + temp.T)       + jitter * Il

        #     Lp = jnp.linalg.cholesky(P_pred)   # 4x4
        #     Lt = jnp.linalg.cholesky(temp_)    # 4x4

        #     # logdet(S) = S*log(eps) + logdet(P_pred) + logdet(temp)
        #     logdetP    = 2.0 * jnp.sum(jnp.log(jnp.diag(Lp)))
        #     logdetTemp = 2.0 * jnp.sum(jnp.log(jnp.diag(Lt)))
        #     logdetS    = S_past_total * jnp.log(eps) + logdetP + logdetTemp

        #     # quad = innov^T S^{-1} innov
        #     a = H.T @ innov                                # (4,)
        #     temp_inv_a = jsp.cho_solve((Lt, True), a)
        #     quad = (innov @ innov) / eps - (a @ temp_inv_a) / (eps**2)

        #     log_prob = -0.5 * (S_past_total * jnp.log(2*jnp.pi) + logdetS + quad)
           
        #     return (mu_xt_t, P_xt_t), log_prob

        # def Kalmanfilter_Xt(carry, array, H, eps, sigma, Il, HTH, jitter=1e-7, Sdim=None):
        #     m_, P_, log_total = carry
        #     yt, mask = array

        #     yt = yt.reshape((Sdim,))
        #     mask = mask.reshape((Sdim,)).astype(yt.dtype)

        #     # Predict
        #     P_pred = P_ + sigma * Il
        #     P_pred = 0.5*(P_pred + P_pred.T) + jitter*Il

        #     # Cholesky of P_pred (reuse for both update + low-rank likelihood)
        #     Lp = jnp.linalg.cholesky(P_pred)

        #     # --- Kalman update (4x4)
        #     invP_pred = jsp.cho_solve((Lp, True), Il)
        #     temp = invP_pred + (1.0/eps) * HTH
        #     temp = 0.5*(temp + temp.T) + jitter*Il
        #     Lt = jnp.linalg.cholesky(temp)

        #     innov = yt - (H @ m_)
        #     a = H.T @ innov

        #     rhs = jnp.concatenate([Il, a[:, None]], axis=1)
        #     sol = jsp.cho_solve((Lt, True), rhs)
        #     P_post = sol[:, :Il.shape[0]]
        #     # temp_inv_a = sol[:, -1]  # you can keep if needed

        #     m_post = m_ + (1.0/eps) * (P_post @ a)

        #     # --- Likelihood via LowRankMVN (no 33x33)
        #     mu_y = H @ m_
        #     U = H @ Lp
        #     cov_diag = eps * jnp.ones((Sdim,))   # or eps*mask + big*(1-mask) if missing
        #     log_prob = dist.LowRankMultivariateNormal(mu_y, U, cov_diag).log_prob(yt)

        #     log_total = log_total + log_prob
        #     return (m_post, P_post, log_total), None

        # H   = P.T
        # HTH = H.T @ H
        # Il  = jnp.eye(L)
        # res = X_fill - Fx@P
        # sigma     = (global_scale * local_scale)**2
        # eps       = eps**2

        # (_, _, log_total),_ = jax.lax.scan(
        #     lambda c, a: Kalmanfilter_Xt(c, a, H=H, eps=eps, sigma=sigma, Il=Il, HTH=HTH,Sdim=S_past_total),
        #     init=(jnp.zeros((L,)), sigma * Il,0.),
        #     xs=(res, maskX),
        #     reverse=True
        # )
        # numpyro.factor("ll_x", log_total)


        # H         = P.T
        # 
        # Is        = jnp.eye(S_past_total)
        # 
        # Il        = jnp.eye(L)
        # sigma_HHT = H@ (sigma[:,None]*H.T)
        # sigmaH    = sigma*H
        # Htsigma   = sigmaH.T
        # HTH       = H.T@H 
        # _, logys = jax.lax.scan( lambda x,y: Kalmanfilter_Xt(x,y, H, eps, Is,sigma, Il, sigma_HHT, sigmaH, Htsigma, S_past_total, HTH )
        #                          , init = ( jnp.zeros( (L,) ) , sigma*Il )
        #                          , xs = (res[::-1,:] ,maskX[::-1,:]) )
        
        # numpyro.factor("ll_x", jnp.sum(logys) )

        # H      = Q.T
        # HTH    = H.T @ H
        # Il     = jnp.eye(L)
        # res    = (y_obs-centers.reshape(-1,1)/scales.reshape(-1,1)) -  (Fy@Q).reshape(-1,1)
        # sigma  = (global_scale * local_scale)**2
        # eps    = eps**2

        # (_, _), logys = jax.lax.scan(
        #     lambda c, a: Kalmanfilter_Xt(c, a, H=H, eps=eps, sigma=sigma, Il=Il, HTH=HTH, Sdim=1 ),
        #     init=(jnp.zeros((L,)), sigma * Il),
        #     xs=(res[::-1, :], mask[::-1, :]),
        # )
        # numpyro.factor("ll_y", jnp.sum(logys))

        
    def estimate_factors(self,D):
        u, s, vt            = np.linalg.svd(D, full_matrices=False)
        splain              = np.cumsum(s**2) / np.sum(s**2)
        estimated_factors_D = (np.min(np.argwhere(splain > .95)) + 1)
        return estimated_factors_D, (u,s,vt)

    def fit(self
            , estimated_lmax_x = True
            , estimated_lmax_y = None
            , Kmix             = 1
            , run_SVI          = True
            , use_anchor       = False):

        y, Y, X     = self.y, self.Y, self.X
        all_y       = self.all_y
        self.Kmix = Kmix

        #--SVD for X
        if estimated_lmax_y is None:
            Ls, us, vts, lambdas = [],[],[], []
            for _ in Y:
                nfactors, (u,s,vt) = self.estimate_factors(_)
                Ls.append(nfactors)
                us.append(u)
                lambdas.append(s)
                vts.append(vt)
                
            self.estimated_factors_y = Ls
            self.us                  = us
            self.lambdas             = lambdas 
            self.vts                 = vts
        else:
            self.estimated_factors_y = estimated_lmax_y

        from patsy import dmatrix

        def bs_basis_zero_padded(tvals):
            def bs_basis(tvals):
                return np.asarray(
                    dmatrix(
                        "bs(t, knots=knots, degree=3, include_intercept=True, "
                        "lower_bound=lb, upper_bound=ub) - 1",
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
                    "bs(t, knots=knots, degree=3, include_intercept=True, "
                    "lower_bound=lb, upper_bound=ub) - 1",
                    {"t": np.asarray(tvals, float), "knots": knots, "lb": lb, "ub": ub},
                )
            )


        # time grid
        T = self.T

        # choose fixed spline settings
        lb, ub = -20, (T-1) + 20
        t      = np.arange(lb,ub+1)
        knots  = np.arange(lb, ub, 2)[1:]
        
        B            = bs_basis_numpy(t,knots,lb,ub)
        self.B       = B

        D            = jnp.std(B, 0) 
        Sb           = jnp.diag( D )
        B_norm       = B@jnp.diag(1./D)

        self.B_norm = B_norm
        self.Sb     = Sb

        D          = jnp.diff(jnp.diff(jnp.eye(B.shape[-1]),axis=0),axis=0)
        Kp         = D.T@D
        self.Kp    = Kp
        
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

        IL                = jnp.eye( int(sum(self.estimated_factors_y)))
        self.IL           = IL

        IK                 = jnp.eye(B_norm.shape[1])
        self.IK           = IK
        
        rng_post          = jax.random.PRNGKey(100915)
        num_draws         = 5*10**3

        print("4")

        dense_blocks = [
            ("beta", "tau_diff","b"),                          # spline shape + smoothness + level
            ("P_blk_raw", "P_rest", "tau_P", "lambda_P"),       # P hierarchy
            ("Qraw", "Q_scales"),                               # target loading
            ("global_scale", "local_scale"),     # RW scales (optionally add "start_z")
            # optionally:
            # ("innov_scale", "start_z")                        # only if S*(T-1) is not too big
        ]
        # for s in range(num_targets):
        #     dense_blocks.append( (f"pi_{s}",f"mu_k_{s}",f"sigma_k_{s}",f"Lcorr_k_{s}",f"choice_{s}") )
        
        nuts_kernel = NUTS(self.model, init_strategy = init_to_median(num_samples=100) , dense_mass = dense_blocks)        #, dense_mass =   [("innov_scale",), ("Lcorr_k_0",), ("beta",), ("P_rest",)] )
        kernel      = DiscreteHMCGibbs(nuts_kernel)
        mcmc        = MCMC(kernel
                    , num_warmup     = 3000
                    , num_samples    = 3500
                    , num_chains     = 1
                    , jit_model_args = False)



        if X is not None:
           y_past = np.hstack([ X, Y[0] ])
           target_indicators = [target_indicators[0]+X.shape[-1]]
           self.target_indicators_mcmc = target_indicators
        else:
            y_past = Y[0]
        self.y_past = y_past

        print(self.tobs)
        
        mcmc.run(jax.random.PRNGKey(20200320)
                              ,y_past            = y_past
                              ,y_target          = y
                              ,B                 = B_norm
                              ,Sb                = Sb
                              ,target_centers    = self.y_means
                              ,target_scales     = self.y_scales
                              ,Ls                = self.estimated_factors_y
                              ,IL                = IL
                              ,IK                = IK
                              ,Kp                = Kp
                              ,copies            = self.copies
                              ,tobs              = self.tobs[0]
                              ,target_indicators = target_indicators
                              , scales = self.global_std
                              ,centers = self.global_mu
                              , Kmix             = self.Kmix 
                              ,forecast          = None )
                              #,extra_fields      = ("diverging", "num_steps", "accept_prob", "energy"))

        self.mcmc = mcmc
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.posterior_samples = samples

        #--diagnostics
        # extra = mcmc.get_extra_fields()

        # print("divergences:", int(extra["diverging"].sum()))

        # steps = np.asarray(extra["num_steps"])
        # print("num_steps median / 90% / max:", np.median(steps), np.quantile(steps, 0.9), steps.max())

        # acc = np.asarray(extra["accept_prob"])
        # print("accept_prob mean:", acc.mean(), "min:", acc.min(), "max:", acc.max())

        # # Step size (after adaptation)
        # print("step_size:", float(mcmc.last_state.adapt_state.step_size))

        # if "energy" in extra:
        #     E = np.asarray(extra["energy"])
        #     ebfmi = np.mean(np.diff(E)**2) / np.var(E)
        #     print("E-BFMI:", ebfmi)  # rule of thumb: < 0.3 is concerning

        # div_idx = np.where(np.asarray(extra["diverging"]))[0]
        # print("divergent draws:", div_idx[:20], "count:", len(div_idx))

        # for name in ["A_logit_0", "Q_diags", "global_sigma_rw", "global_f_sigma"]:
        #     if name in samples:
        #         vals = np.asarray(samples[name])
        #         print(name, "divergent mean:", vals[div_idx].mean(), "overall mean:", vals.mean())

        return self

    def forecast(self):
        from numpyro.infer import Predictive
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred"])

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,y_past            = self.y_past
                              ,y_target          = self.y
                              ,B                 = self.B_norm
                              ,Sb                = self.Sb
                              ,target_centers    = self.y_means
                              ,target_scales     = self.y_scales
                              ,Ls                = self.estimated_factors_y
                              ,IL                = self.IL
                              ,IK                = self.IK
                              ,Kp                = self.Kp
                              ,copies            = self.copies
                              ,tobs              = self.tobs[0]
                              ,target_indicators = self.target_indicators_mcmc
                              , scales = self.global_std
                              , centers = self.global_mu
                              , Kmix             = self.Kmix 
                              ,forecast          = True
                                  )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        self.pred_samples = pred_samples
        self.forecast     = yhat_draws
        return yhat_draws
    

if __name__ == "__main__":

    pass
