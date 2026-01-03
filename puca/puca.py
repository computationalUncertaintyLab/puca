#mcandrew

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, init_to_median
from numpyro.infer.initialization import init_to_value
from functools import partial


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

            past_smooth       =  self.smooth_gaussian_anchored(past,2)
            past_smooth_means =  np.mean(past_smooth,axis=0)
            past_smooth_stds  =  np.std(past_smooth,axis=0)

            smooth_y          = (past_smooth - past_smooth_means)/past_smooth_stds
            smooth_ys.append(smooth_y)
            
            if n==0:
                all_y = np.hstack([smooth_y,current.reshape(-1,1)])
            else:
                _     = np.hstack([smooth_y,current.reshape(-1,1)])
                all_y = np.hstack([all_y,_])

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
        self.Y      = smooth_ys #<--past Y values
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
              ,IL                = None
              ,Kp                = None 
              ,copies            = None
              ,tobs              = None
              ,target_indicators = None
              ,forecast          = None):

        #--We need to build F as a set of P-splines that are rperesented as B @ beta
        T,K          = B.shape
        S_past_total = y_past.shape[-1]
        S            = S_past_total + 1
        L            = IL.shape[0] 
        
        #--This is a smoothness penalty 
        meps       = 10**-5
        smoothness_switch = numpyro.sample("smoothness_switch", dist.Exponential(1) )
        tau_diff          = numpyro.sample("tau_diff", dist.HalfNormal(smoothness_switch))  # small => smooth (scale depends on your standardization!)
        prec              = (Kp / (tau_diff**2)) + meps * jnp.eye(K)
        
        beta       = numpyro.sample("beta", dist.MultivariateNormal(0, precision_matrix = prec).expand([L]) )
        beta       = beta.T    #<- K X L
        beta       = Sb @ beta #<-- (B @ Sbinv) @ (Sb @ beta). In other words, normalize the columns of our B matrix
        
        #--Lets assume that the F columns (it the latent time series) are correlated
        F          = B@beta                             #--(T,K) @ (K,L) -> T X L
        sF         = jnp.std(F,0)                       #-- L We should normalize the columns of F should that X and y can share it
        Fnorm      = F/sF[None,:] 
        
        #--discrepanacyes
        global_scale = numpyro.sample("global_scale", dist.HalfNormal(0.5 * jnp.sqrt(jnp.pi/2)  ))
        local_scale  = numpyro.sample("local_scale" , dist.HalfNormal(1.).expand([L]))

        start_z = numpyro.sample("start_scale", dist.Normal(0,1).expand([S,1,L]) )
        innov_z = numpyro.sample("innov_scale", dist.Normal(0,1).expand([S,T-1,L])) #<--one for P and one for Q

        innovations =  (global_scale*local_scale)[None,None,:] * jnp.hstack([ innov_z[:, ::-1,:], start_z]) #<-- 2,T,L

        d_paths       = jnp.cumsum(innovations,1)[:,::-1,:] #<--this is  S X T X L
        numpyro.deterministic("d_paths", d_paths)

        b     = numpyro.sample("rw_center", dist.Normal(0, 1.).expand([L]))   # (L,)
        Fnorm = Fnorm + b + d_paths # T X L
        Fx    = Fnorm[:-1,...] 
        Fy    = Fnorm[-1,...]

        # Fx = Fnorm
        # Fy = Fnorm + y_path 
        # Fy = Fnorm + b[None,:] + y_path_centered #--yes path here just bc its easier to forecast

        numpyro.deterministic("Fnorm",Fnorm)
        
        #--P matrix so that X = Fnorm @ P
        tau_P     = numpyro.sample("tau_P"   , dist.HalfNormal(1.0))
        lambda_P  = numpyro.sample("lambda_P", dist.HalfCauchy(1.0).expand([L,1]))

        P_blk_raw = numpyro.sample("P_blk_raw", dist.Normal(0,1).expand([L,L]))
        P_blk     = jnp.tril(P_blk_raw, k=-1) + IL

        P_blk     = P_blk * tau_P * lambda_P  

        P_rest    = numpyro.sample("P_rest", dist.Normal(0,1).expand([L, S-1 - L]))

        P         = jnp.concatenate([P_rest, P_blk], axis=1)
        P         = P * sF[:,None]

        Q         = numpyro.sample("Qraw", dist.Normal(0,1).expand([L,1])  )
        Q         = Q * sF[:,None] 
        
        numpyro.deterministic("P", P)
        numpyro.deterministic("Q", Q)

        X            = jnp.einsum("stl,sl->st",Fx,P.T).T  #- S X T X L  and L X S -> T X S  P is LXS
        eps          = 1./4#numpyro.sample("eps_x", dist.HalfNormal( (1./4)*jnp.sqrt(jnp.pi/2) )  )  

        X_target   = jnp.hstack(y_past).reshape(T,S_past_total)
        present    = jnp.isfinite(X_target)

        maskX  = jnp.isfinite(X_target)
        X_fill = jnp.where(maskX, X_target, 0.0)
        
        with numpyro.handlers.mask(mask=maskX):
            numpyro.sample("X_ll", dist.Normal(X, eps), obs=X_fill)

        #--center and scale for y
        Kmix = 2
        centers, scales = [],[]
        for signal, (cents, scals) in enumerate( zip(target_centers, target_scales) ):
            N           = len(cents)
            log_scals   = jnp.log(1+scals)

            mu_cents    = jnp.mean(cents)
            sigma_cents = jnp.std(cents) 
            zc          = ((cents - mu_cents) / sigma_cents) .reshape(-1,1)

            mu_scals    = jnp.mean(log_scals)
            sigma_scals = jnp.std(log_scals)
            zs = ((log_scals - mu_scals)/ sigma_scals).reshape(-1,1)

            cs_vec = jnp.hstack([zc,zs])

            # --- Bayesian Gaussian mixture in z-space -----------------------------
            # weights
            pi = numpyro.sample(f"pi_{signal}", dist.Dirichlet((1./Kmix)*jnp.ones(Kmix)))

            # component means (in z-space)
            mu_k = numpyro.sample(
                f"mu_k_{signal}",
                dist.Normal(0.0, 1.5).expand([Kmix, 2])
            )

            # component covariance via (scales * LKJ correlation)
            # per-component marginal scales
            sigma_k = numpyro.sample(
                f"sigma_k_{signal}",
                dist.HalfNormal(1.0).expand([Kmix, 2])
            )
            # per-component correlation Cholesky
            Lcorr_k = numpyro.sample(
                f"Lcorr_k_{signal}",
                dist.LKJCholesky(dimension=2, concentration=2.0).expand([Kmix])
            )  # (Kmix, 2, 2)

            D_k = jax.vmap(jnp.diag)(sigma_k)                      # (Kmix, 2, 2)
            L_k = jax.vmap(lambda D, L: D @ L)(D_k, Lcorr_k)       # (Kmix, 2, 2) scale_tril

            comp = dist.MultivariateNormal(loc=mu_k, scale_tril=L_k)  # batch (Kmix,), event (2,)
            mix  = dist.MixtureSameFamily(dist.Categorical(probs=pi), comp)

            numpyro.sample(f"cs_obs_{signal}", mix.expand([N]), obs=cs_vec)

            choice = numpyro.sample(f"choice_{signal}", mix)  # shape (2,)
            center_choice_z, scal_choice_z = choice[0], choice[1]

            #map back to natural scale
            center_choice = center_choice_z * sigma_cents + mu_cents
            scal_choice   = jnp.expm1(scal_choice_z * sigma_scals + mu_scals)

            # corr   = (cs_vec.T@cs_vec) / (N-1)
            # Lcov   = jnp.linalg.cholesky(corr)

            # center_choice, scal_choice = numpyro.sample(f"choice_{signal}", dist.MultivariateStudentT(N-2, 0,Lcov) )

            # center_choice = center_choice*sigma_cents + mu_cents
            # scal_choice   = jnp.exp(scal_choice*sigma_scals + mu_scals)-1

            centers.append(center_choice)
            scales.append(scal_choice)
        
        centers = jnp.hstack(centers)
        scales  = jnp.hstack(scales) 

        y = (Fy@Q).reshape(T,1)
        numpyro.deterministic("y",y)

        yhat_    = (y*scales.reshape(-1,1)) + centers.reshape(-1,1)
        numpyro.deterministic("yhat_", yhat_)
        
        eps_y0 = eps                      
        eps_y  = eps_y0 * scales

        y_target = jnp.hstack(y_target).reshape(T,1)
        
        mask     = jnp.isfinite(y_target)
        y_obs    = jnp.where(mask, y_target, 0.0)

        numpyro.deterministic("y_target", y_target)

        print(yhat_.shape)
        print(y_obs.shape)
       
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("y_ll", dist.Normal(yhat_.reshape(T,1), eps_y), obs = y_obs.reshape(T,1))

        if forecast:
            numpyro.sample("y_pred", dist.Normal(yhat_, eps_y) )


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
            , estimate_lmax_x = True
            , estimate_lmax_y = True
            , run_SVI         = True
            , use_anchor      = False):

        y, Y, X     = self.y, self.Y, self.X
        all_y       = self.all_y

        #--SVD for X
        if estimate_lmax_y:
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

        # time grid
        T = self.T
        t = jnp.arange(T) #<--move one time unit back

        # choose fixed spline settings
        lb, ub = 0, T-1
        knots  = np.arange(lb,ub,2)[1:]#-1]
        
        B            = bs_basis_zero_padded(t)
        self.B       = B

        D            = jnp.std(B, 0) 
        Sb           = jnp.diag( D )
        B_norm       = B@jnp.diag(1./D)

        self.B_norm = B_norm
        self.Sb     = Sb

        D          = jnp.diff(jnp.diff(jnp.eye(B.shape[-1]),axis=0),axis=0)
        Kp         = D.T@D
        self.Kp    = Kp
        
        model        = self.model
        copies = self.copies
        
        #--collect helpful parameters
        S_past_total      = int(sum(copies))
        num_targets       = len(copies)
        S                 = S_past_total + num_targets

        #--we will flateen target_indicators
        copies_j                    = jnp.array(self.target_indicators)        
        starts                      = jnp.concatenate([jnp.array([0]), jnp.cumsum(copies_j + 1)[:-1]])
        target_indicators           = starts + copies_j
        self.target_indicators_mcmc = target_indicators

        IL                = np.eye( int(sum(self.estimated_factors_y)))
        self.IL           = IL
        
        rng_post          = jax.random.PRNGKey(100915)
        num_draws         = 5*10**3
        
        nuts_kernel = NUTS(self.model, init_strategy = init_to_median(num_samples=100)) 
        mcmc = MCMC(nuts_kernel
                    , num_warmup     = 2000
                    , num_samples    = 4000
                    , num_chains     = 1
                    , jit_model_args = True)

        mcmc.run(jax.random.PRNGKey(20200320)
                              ,y_past            = Y[0]
                              ,y_target          = y
                              ,B                 = B_norm
                              ,Sb                = Sb
                              ,target_centers    = self.y_means
                              ,target_scales     = self.y_scales
                              ,Ls                = self.estimated_factors_y
                              ,IL                = IL
                              ,Kp                = Kp
                              ,copies            = self.copies
                              ,tobs              = self.tobs
                              ,target_indicators = target_indicators 
                              ,forecast          = None 
                              ,extra_fields      = ("diverging", "num_steps", "accept_prob", "energy"))

        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.posterior_samples = samples

        #--diagnostics
        extra = mcmc.get_extra_fields()

        print("divergences:", int(extra["diverging"].sum()))

        steps = np.asarray(extra["num_steps"])
        print("num_steps median / 90% / max:", np.median(steps), np.quantile(steps, 0.9), steps.max())

        acc = np.asarray(extra["accept_prob"])
        print("accept_prob mean:", acc.mean(), "min:", acc.min(), "max:", acc.max())

        # Step size (after adaptation)
        print("step_size:", float(mcmc.last_state.adapt_state.step_size))

        if "energy" in extra:
            E = np.asarray(extra["energy"])
            ebfmi = np.mean(np.diff(E)**2) / np.var(E)
            print("E-BFMI:", ebfmi)  # rule of thumb: < 0.3 is concerning

        div_idx = np.where(np.asarray(extra["diverging"]))[0]
        print("divergent draws:", div_idx[:20], "count:", len(div_idx))

        for name in ["A_logit_0", "Q_diags", "global_sigma_rw", "global_f_sigma"]:
            if name in samples:
                vals = np.asarray(samples[name])
                print(name, "divergent mean:", vals[div_idx].mean(), "overall mean:", vals.mean())

        return self

    def forecast(self):
        from numpyro.infer import Predictive
        predictive = Predictive(self.model,posterior_samples = self.posterior_samples
                                , return_sites               = list(self.posterior_samples.keys()) + ["y_pred"])

        rng_key    = jax.random.PRNGKey(100915)
        pred_samples = predictive( rng_key
                              ,y_past            = self.Y[0]
                              ,y_target          = self.y
                              ,B                 = self.B_norm
                              ,Sb                = self.Sb
                              ,target_centers    = self.y_means
                              ,target_scales     = self.y_scales
                              ,Ls                = self.estimated_factors_y
                              ,IL                = self.IL
                              ,Kp                = self.Kp
                              ,copies            = self.copies
                              ,tobs              = self.tobs
                              ,target_indicators = self.target_indicators_mcmc
                              ,forecast          = True
                                  )
        yhat_draws = pred_samples["y_pred"]      # (draws, T, S)

        self.pred_samples = pred_samples
        self.forecast     = yhat_draws
        return yhat_draws
    

if __name__ == "__main__":

    pass
