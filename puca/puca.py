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
        #alpha     = numpyro.sample("alpha", dist.Gamma(2, 2./100 ).expand([K,L]))
        alpha     = numpyro.sample("alpha", dist.Gamma(1, 0.005).expand([L]))
        
        tau_diff  = 1./jnp.sqrt(alpha)
        
        null        = 2
        U0          = Kp_U[:, :null]        
        Up          = Kp_U[:, null:]
        lp          = Kp_l[null:]

        #a  = 20
        #s0 = 0.10
        #beta0_raw   = numpyro.sample("beta0_raw", dist.Normal(0, 1).expand([U0.shape[1], L]))
        #beta0_prec  = numpyro.sample("beta0_prec", dist.Gamma(a, a*s0**2))  # mean = 1/s0^2
        #beta0       = beta0_raw / jnp.sqrt(beta0_prec)
        
        z_beta      = numpyro.sample("beta", dist.Normal(0,1).expand([lp.shape[0] ,L]) )
        
        def produce_beta( tau_diff, z_beta, Up, lp ):
            on_diag    =   (lp/tau_diff**2) + 10**-6
            beta       =  ( Up * (1/jnp.sqrt(on_diag)  ))   @ z_beta
            return beta
        beta = jax.vmap( lambda x,y:produce_beta(x,y,Up,lp), in_axes = (0,1) )( tau_diff, z_beta  )
        beta = beta.T

        #beta = U0@beta0 + beta

        X_shifts, y_shift = column_shifts[:S_past_total], column_shifts[-1]

        base        = jnp.arange(T)
        
        B_shifted   = B[ (M+base)[:,None] - X_shifts[None,:]  ]
        Fx          = B_shifted @ beta              #<--T X S X L
        
        sFx = jnp.std(Fx, axis=(0,1))
        mFx         = jnp.mean(Fx, axis=0)

        Fx          = Fx 

        numpyro.deterministic("Fx",Fx)
       
        #--P matrix so that X = Fnorm @ P
        P_global_prec = numpyro.sample("P_global_prec" , dist.Gamma(2,2./4))
        P_global_prec = 1./ jnp.sqrt(P_global_prec)
        
        P_local_prec   = numpyro.sample("P_local_prec" , dist.Gamma(2, 2.).expand([L,1]) )
        P_local_prec   = Trans.OrderedTransform()(P_local_prec)
        
        gScale        = (P_global_prec * P_local_prec)
        
        r,c          = jnp.tril_indices(L,k=-1)
        P_zeros      = jnp.zeros((L,L))
        z_P_blk_raw  = numpyro.sample("z_P_blk_raw", dist.Normal(0,1).expand([ int(L*(L-1)/2) ] ))
        P_blk        = gScale*( P_zeros.at[r,c].set(z_P_blk_raw) ) + IL  

        P_rest       = numpyro.sample("P_rest", dist.Normal(0,1).expand([L, S - 1 - L]))
        P_rest       = gScale*P_rest
        P            = jnp.concatenate([P_rest, P_blk], axis=1) #<--LXS
        P            = P.at[0,:].set( jax.nn.softplus(P[0,:]) )
        
        numpyro.deterministic("P",P)
        #P            = P #* sFx[:,None]

        Q_global_prec = numpyro.sample("Q_global_prec" , dist.Gamma(2,2.).expand([L,1]))
        Q_scales      = 1./ jnp.sqrt(Q_global_prec)

        Qraw          = numpyro.sample("Qraw"    , dist.Normal(0,1).expand([L,1])  )
        Q             = Q_scales*Qraw
        numpyro.deterministic("Q", Q)

        #--compute trends
        Xtrend = jnp.einsum("tsl,ls->ts",Fx,P) #<-TXS
        numpyro.deterministic("Xtrend",Xtrend)
        
        #ytrend = (Fy@Q).reshape(T,1)
        #numpyro.deterministic("ytrend",ytrend)
        
        #--AR
        eps  = 0.025
        flex = 2
        global_prec  = numpyro.sample("global_prec", dist.Gamma( 5, 5.*(flex*eps)**2))#  /500  ) )
        global_scale = 1./jnp.sqrt(global_prec)
        numpyro.deterministic("global_scale", global_scale)

        local_prec = numpyro.sample("local_prec", dist.Gamma(2,2).expand([S]) )
        local_scale = 1./jnp.sqrt(local_prec)
        numpyro.deterministic("local_scale", local_scale)

        path_scale = global_scale * local_scale

        # innov_z           = numpyro.sample("innov_z"    , dist.Normal(0,1).expand([1,T-1])) #<--one for P and one for Q

        start              = jnp.zeros((1,1)) 
        #innovations        = path_scale[-1,None]*innov_z
        phi=1.

        intercept_perc            = numpyro.sample("intercept_perc"           , dist.Gamma(2,2)) 
        season_specific_intercept = numpyro.sample("season_specific_intercept", dist.Normal(0, 1.).expand([S]))   # (L,)
        season_specific_intercept = season_specific_intercept / jnp.sqrt(intercept_perc)

        # def step(x_next, z_t, phi):
        #     x_t = phi * (x_next) + z_t
        #     return x_t, x_t
        # _, xs_rev = jax.lax.scan( lambda x,y: step(x,y,phi), init=start, xs=innovations.T )  # xs_rev is [x_{T-2}, x_{T-3}, ..., x_0]

        # y_path = jnp.concatenate([jnp.flip(xs_rev.squeeze(), axis=0), start.squeeze()[None,]], axis=0)  # [x_0,...,x_{T-1}]  T X S

        X_target = jnp.hstack(y_past).reshape(T,S_past_total)
        Xtrend   = Xtrend + season_specific_intercept[None, :-1]

        resid = X_target - Xtrend
        Xmask = jnp.isfinite(resid)
        resid_filled = jnp.where(Xmask, resid, 0.0)
        
        #--kalm likelihood here
        def kf(carry, array, q, r):
            m,P     = carry
            yobs,mk = array

            xtp1_E = m
            xtp1_P = P+q

            ytp1_E = m
            ytp1_P = P+q+r

            K = 1./( 1 + (r/(P+q)) )

            innov      = (yobs-m)
            xtp1_yp1_E = m + K*innov
            xtp1_yp1_P = (P+q)*(1-K)

            LOG2PI = jnp.log(2.0 * jnp.pi)
            ll     = -0.5 * (LOG2PI + jnp.log(ytp1_P) + (innov * innov) / ytp1_P)
            ll     = mk * ll

            return (xtp1_yp1_E,xtp1_yp1_P), ll
            
        _,LL_xs = jax.vmap( lambda path_scale_indiv, data, mask : jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=path_scale_indiv**2) , init = ( jnp.array(0.), 10**-6  ), xs = ( data, mask.squeeze() ), reverse=True   )
                            , in_axes=(0,1,1) )(path_scale[:-1], resid_filled, Xmask)
        numpyro.factor( "LL_x", jnp.sum(LL_xs) )

        eps_y  = (eps) * scales

        y_target = jnp.hstack(y_target).reshape(T,1)
        
        ymask     = jnp.isfinite(y_target)

        #--shifting probabilities
        index_weights = index_weights+10**-2
        index_probs   = numpyro.sample( "index_weights", dist.Dirichlet( index_weights) )

        def from_shift_to_prob(shift, y_path_scale):
            B_y      = B[M+base-shift,:] #<--   y_target is the last shifted index
            y_trend  = (B_y@beta@Q).reshape(T,1) + season_specific_intercept[-1]  #+  y_path)

            resid        =  ((y_target - centers)/scales) - y_trend
            ymask        =  jnp.isfinite(resid)
            resid_filled = jnp.where(ymask, resid, 0.0)

            _,LL_y = jax.lax.scan( lambda x,y: kf(x,y,r=eps**2,q=y_path_scale**2) , init = ( jnp.array(0.), 10**-6  ), xs = ( resid_filled.squeeze(), ymask.squeeze() ), reverse=True)
            return jnp.sum( LL_y )

        y_path_scale = path_scale[-1]
        shifts       = jnp.arange(-M,M+1) 
        probs        = jax.vmap( lambda x: from_shift_to_prob(x, y_path_scale))( shifts )
        logw         = probs + jnp.log(index_probs)            # (2M+1,)

        prior = jnp.log( index_probs )
        LLY   = jax.scipy.special.logsumexp( (probs + prior)  )
        numpyro.factor( "LLY", LLY )
        
        if forecast:
            #--trned component
            shift_post = jax.nn.softmax(logw)              # normalized responsibilities
            shift_idx   = numpyro.sample("shift", dist.Categorical(shift_post) )
            shift       = shifts[shift_idx]
            
            B_y      = B[M + base - shift,:] 
            F_y      = B_y@beta 

            #--noise path component
            innov_z      = numpyro.sample("new_innov_z"    , dist.Normal(0,1).expand([1,T-1])) #<--one for P and one for Q

            start        = start[-1].reshape(-1,1)
            innovations  = y_path_scale*innov_z
           
            def step(x_next, z_t, phi):
                x_t = phi * (x_next) + z_t
                return x_t, x_t
            _, xs_rev = jax.lax.scan( lambda x,y: step(x,y,phi), init=start, xs=innovations.T)  # xs_rev is [x_{T-2}, x_{T-3}, ..., x_0]

            y_path = jnp.concatenate([jnp.flip(xs_rev.squeeze(), axis=0), start.squeeze()[None,]], axis=0)  # [x_0,...,x_{T-1}]  T X S

            y  = (F_y@Q).reshape(T,1)  + season_specific_intercept[-1] +  y_path[:,None]
            yhat_    = y *scales + centers

            #numpyro.deterministic("y_trend_pred", y_trend*scales + centers)
            numpyro.sample("y_pred", dist.Normal(yhat_, eps_y) )


    def estimate_factors(self,D):
        u, s, vt            = np.linalg.svd(D, full_matrices=False)
        splain              = np.cumsum(s**2) / np.sum(s**2)
        estimated_factors_D = (np.min(np.argwhere(splain > .95)) + 1)
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
        knots  = np.arange(lb, ub, 2)[1:]

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

        # guide = AutoBNAFNormal(
        #     self.model
        #     ,num_flows=2
        #     ,init_loc_fn=init_to_median(num_samples=100)
        # )

        guide = AutoIAFNormal(
            self.model
            ,num_flows=2
            ,init_loc_fn=init_to_median(num_samples=100)
        )
 
        optimizer = ClippedAdam(step_size= 10**-4, clip_norm=1)  # common starting point: 1e-3 to 1e-2
        loss      = Trace_ELBO(num_particles=3)

        svi = SVI(self.model, guide, optimizer, loss)

        svi_result = svi.run(
            jax.random.PRNGKey(20200320),
            10*10**3,
            y_past            = y_past,
            y_target          = y,
            B                 = self.B,
            Sb                = self.Sb,
            Ls                = self.estimated_num_components_y,
            column_shifts     = self.column_shifts,
            index_weights     = self.index_weights,
            IL                = self.IL,
            IK                = self.IK,
            Kp_l              = self.Kp_l,
            Kp_U              = self.Kp_U,
            copies            = self.copies,
            tobs              = self.tobs[0],
            target_indicators = target_indicators,
            scales            = self.global_std,
            centers           = self.global_mu,
            forecast          = None,
        )

        params = svi_result.params
        losses = svi_result.losses
        print("final loss:", float(losses[-1]))

        self.params = params
        self.guide = guide

        # neutra       = NeuTraReparam(guide, svi_result.params)
        # neutra_model = neutra.reparam(self.model)
        
        # dense_blocks = [
        #     ("beta","alpha"),                            # spline shape + smoothness + level
        #     ("z_P_blk_raw", "P_global_prec", "P_local_prec"),          # P hierarchy
        #     ("Qraw", "Q_global_prec"),
        #     ("intercept_perc", "season_specific_intercept"),
        #     ("global_prec", "local_prec")#,"innov_z"),  # RW scales (optionally add "start_z")
        # ]
        
        # nuts_kernel = NUTS(self.model#neutra_model
        #                    , init_strategy = init_to_median(num_samples=100)
        #                    , dense_mass = dense_blocks
        #                    ,  find_heuristic_step_size=True)     
        # kernel      = nuts_kernel 
        # mcmc        = MCMC(kernel
        #             , num_warmup     = 2000
        #             , num_samples    = 2500
        #             , num_chains     = 1
        #             , jit_model_args = False)

        # mcmc.run(jax.random.PRNGKey(20200320)
        #                       ,y_past            = y_past
        #                       ,y_target          = y
        #                       ,B                 = self.B
        #                       ,Sb                = self.Sb
        #                       ,Ls                = self.estimated_num_components_y
        #                       ,column_shifts     = self.column_shifts
        #                       ,index_weights     = self.index_weights
        #                       ,IL                = self.IL
        #                       ,IK                = self.IK
        #                       ,Kp_l              = self.Kp_l
        #                       ,Kp_U              = self.Kp_U
        #                       ,copies            = self.copies
        #                       ,tobs              = self.tobs[0]
        #                       ,target_indicators = target_indicators
        #                       ,scales            = self.global_std
        #                       ,centers           = self.global_mu
        #                       ,forecast          = None 
        #                       ,extra_fields      = ("diverging", "num_steps", "accept_prob", "energy","adapt_state.step_size"))

        # self.mcmc = mcmc
        # mcmc.print_summary()
        # samples = mcmc.get_samples()
        # self.posterior_samples = samples
        
        return self

    def forecast(self):

        predictive = Predictive(self.model
                               , guide=self.guide
                               , params=self.params
                               , num_samples=5000
                               , return_sites  = ["y_pred","y_trend_pred"] )
        
        # predictive = Predictive(self.model,posterior_samples = self.posterior_samples
        #                         , return_sites               = list(self.posterior_samples.keys()) + ["y_pred","y_trend_pred"] )

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


if __name__ == "__main__":

    pass
