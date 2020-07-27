__all__ = ['hists_from_nn','nn_hepdata_like', 'nn_histosys']

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .smooth import kde_hist as hist

def hists_from_nn(
    data_generator,  
    predict,
    method='softmax',
    LUMI=10,
    sig_scale=2,
    bkg_scale=10):
        """
        Initialize a function `hist_maker` that returns a 'soft' histogram based on a neural network
        with a softmax output. Choose which example problem to try by setting the `example` argument.

        Args:
                data_generator: Callable that returns generated data (in jax array format).

                predict: Decision function for a parameterized observable, e.g. neural network.

                method: A string to specify the method to use for constructing soft histograms. Either 'softmax'
                or 'kde'.

                LUMI: 'Luminosity' scaling factor for the yields.

                sig_scale: Individual scaling factor for the signal yields.

                bkg_scale: Individual scaling factor for the signal yields.

        Returns:
                hist_maker: A callable function that takes the parameters of the observable (and optional hyperpars),
                then constructs signal, background, and background uncertainty yields.
        """
        data = data_generator()
        if len(data) == 3:
            if method == 'softmax':
                def hist_maker(hm_params):
                    '''
                    Uses the nn decision function `predict` to form histograms from signal and background
                    data, all drawn from multivariate normal distributions with different means. Two
                    background distributions are sampled from, which is meant to mimic the situation in
                    particle physics where one has a 'nominal' prediction for a nuisance parameter 
                    (taken here as the mean of two modes) and then alternate values (e.g. from varying 
                    up/down by one standard deviation), which then modifies the background pdf. Here, we
                    take that effect to be a shift of the mean of the distribution. The value for the 
                    background histogram is then the mean of the resulting counts of the two modes, and 
                    the uncertainty can be quantified through the count standard deviation.

                    Arguments: 
                        hm_params: a list containing:
                            nn: jax array of observable parameters.
                    '''
                    nn, _ = hm_params
                    s, b_up, b_down = data
                    NMC = len(s)
                    s_hist = predict(nn, s).sum(axis=0) * sig_scale / NMC * LUMI

                    b_hists = [
                        predict(nn, b_up).sum(axis=0) * bkg_scale / NMC * LUMI,
                        predict(nn, b_down).sum(axis=0) * bkg_scale / NMC * LUMI
                    ]

                    b_mean = jnp.mean(jnp.asarray(b_hists), axis=0)
                    b_unc = jnp.std(jnp.asarray(b_hists), axis=0)

                    return s_hist, b_mean, b_unc
                
            elif method == 'kde':
                def hist_maker(hm_params):
                    '''
                    Uses the nn decision function `predict` to form histograms from signal and background
                    data using a kde, all drawn from multivariate normal distributions with different means. Two
                    background distributions are sampled from, which is meant to mimic the situation in
                    particle physics where one has a 'nominal' prediction for a nuisance parameter 
                    (taken here as the mean of two modes) and then alternate values (e.g. from varying 
                    up/down by one standard deviation), which then modifies the background pdf. Here, we
                    take that effect to be a shift of the mean of the distribution. The value for the 
                    background histogram is then the mean of the resulting counts of the two modes, and 
                    the uncertainty can be quantified through the count standard deviation.

                    Arguments:
                        hm_params: Array-like, consisting of:
                            nn: jax array of observable parameters.
                            
                            bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
                                edges at 0, 0.5, 1.

                            bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
                            this fairly similar to the bin width to avoid oversmoothing the distribution. Going too low
                            will cause things to break, as the gradients of the kde become unstable.

                    '''
                    nn, hpar_dict = hm_params
                    bins, bandwidth = hpar_dict['bins'], hpar_dict['bandwidth']
                    s, b_up, b_down = data
                    NMC = len(s)
                    
                    nn_s, nn_b_up, nn_b_down = (
                        predict(nn, s).ravel(),
                        predict(nn, b_up).ravel(),
                        predict(nn, b_down).ravel(),
                    )
                        
                    s_hist = hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI

                    b_hists = jnp.asarray([
                        hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
                        hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
                    ])

                    kde_counts = [
                        s_hist,
                        jnp.mean(b_hists, axis=0),
                        jnp.std(b_hists, axis=0)
                    ]
                    
                    return kde_counts

            else:
                assert False, f'Unknown soft histogram method \'{method}\'. Currently only \'softmax\' or \'kde\' are available.'

        elif len(data) == 4:
            if method == 'softmax':
                def hist_maker(hm_params):
                    '''
                    Uses the nn decision function `predict` to form histograms from signal and background
                    data, all drawn from multivariate normal distributions with different means. Three
                    background distributions are sampled from, which mimics the situation in
                    particle physics where one has a 'nominal' prediction for a nuisance parameter 
                    (taken here as the mean of two modes) and then alternate values (e.g. from varying 
                    up/down by one standard deviation), which then modifies the background pdf. Here, we
                    take that effect to be a shift of the mean of the distribution. The HistFactory 
                    'histosys' nusiance parameter will then be constructed from the yields downstream by
                    interpolating between them using pyhf.

                    Arguments:
                        hm_params: a list containing:
                            nn: jax array of observable parameters.

                    Returns:
                        Set of 4 counts for signal, background, and up/down modes.
                    '''
                    nn, _ = hm_params 
                    s, b_nom, b_up, b_down = data
                    NMC = len(s)
                    counts = [
                            predict(nn, s).sum(axis=0)* sig_scale / NMC * LUMI,
                            predict(nn, b_nom).sum(axis=0)* bkg_scale / NMC * LUMI,
                            predict(nn, b_up).sum(axis=0)* bkg_scale / NMC * LUMI,
                            predict(nn, b_down).sum(axis=0)* bkg_scale / NMC * LUMI
                    ]

                    return counts

            elif method == 'kde':
                def hist_maker(hm_params):
                    '''
                    Uses the nn decision function `predict` to form histograms from signal and background
                    data, all drawn from multivariate normal distributions with different means. Three
                    background distributions are sampled from, which mimics the situation in
                    particle physics where one has a 'nominal' prediction for a nuisance parameter 
                    (taken here as the mean of two modes) and then alternate values (e.g. from varying 
                    up/down by one standard deviation), which then modifies the background pdf. Here, we
                    take that effect to be a shift of the mean of the distribution. The HistFactory 
                    'histosys' nusiance parameter will then be constructed from the yields downstream by
                    interpolating between them using pyhf.

                    Arguments:
                        hm_params: Array-like, consisting of:
                            nn: jax array of observable parameters.
                            
                            bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
                                edges at 0, 0.5, 1.

                            bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
                            this fairly similar to the bin width to avoid oversmoothing the distribution. Going too low
                            will cause things to break, as the gradients of the kde become unstable.

                    Returns:
                        Set of 4 counts for signal, background, and up/down modes.
                    '''
                    nn, hpar_dict = hm_params
                    bins, bandwidth = hpar_dict['bins'], hpar_dict['bandwidth']
                    s, b_nom, b_up, b_down = data
                    NMC = len(s)
                    
                    nn_s, nn_b_nom, nn_b_up, nn_b_down = (
                        predict(nn, s).ravel(),
                        predict(nn, b_nom).ravel(),
                        predict(nn, b_up).ravel(),
                        predict(nn, b_down).ravel(),
                    )
                        
                    kde_counts = [
                        hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
                        hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
                        hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
                        hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
                    ]
                    
                    return kde_counts
            else:
                assert False, f'Unknown soft histogram method \'{method}\'. Currently only \'softmax\' or \'kde\' are available.'

        else:
            assert False, f'Unknown blob size (only using 3 or 4 blobs for this example).'
                
        return hist_maker

# def kde_hists_from_nn(
#     data_generator,  
#     predict,
#     example,
#     LUMI=10,
#     sig_scale=2,
#     bkg_scale=10):
#         """
#         Initialize a function `hist_maker` that returns a 'soft' histogram based on kdes and a neural network
#         with a single output. The yields are then calculated as the integral of the kde's cumulative density 
#         function between the bin edges, which should be specified using the argument `bins`. Choose which example 
#         problem to try by setting the `example` argument.

#         Args:
#                 data_generator: Callable that returns generated data (in jax array format).

#                 predict: Decision function for a parameterized observable. Assumed softmax here.

#                 example: A string to specify which example to test. Either 'three_blobs' or 'histosys'.

#         Returns:
#                 hist_maker: A callable function that takes the parameters of the observable,
#                 then constructs signal, background, and background uncertainty yields.
#         """
        
#         if example == 'three_blobs':
#             def hist_maker(nn):
#                 '''
#                 Uses the nn decision function `predict` to form histograms from signal and background
#                 data using a kde, all drawn from multivariate normal distributions with different means. Two
#                 background distributions are sampled from, which is meant to mimic the situation in
#                 particle physics where one has a 'nominal' prediction for a nuisance parameter 
#                 (taken here as the mean of two modes) and then alternate values (e.g. from varying 
#                 up/down by one standard deviation), which then modifies the background pdf. Here, we
#                 take that effect to be a shift of the mean of the distribution. The value for the 
#                 background histogram is then the mean of the resulting counts of the two modes, and 
#                 the uncertainty can be quantified through the count standard deviation.

#                 Arguments:
#                     nn: jax array of observable parameters.
                      
#                     bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
#                         edges at 0, 0.5, 1.

#                     bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
#                     this fairly similar to the bin width to avoid oversmoothing the distribution. Going too low
#                     will cause things to break, as the gradients of the kde become unstable.

#                 '''
                              
#                 s, b_up, b_down = data_generator()
#                 NMC = len(s)
                
#                 nn_s, nn_b_up, nn_b_down = (
#                     predict(nn, s).ravel(),
#                     predict(nn, b_up).ravel(),
#                     predict(nn, b_down).ravel(),
#                 )
                    
#                 s_hist = hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,

#                 b_hists = jnp.asarray([
#                     hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                     hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 ])

#                 kde_counts = [
#                     s_hist,
#                     jnp.mean(b_hists, axis=0),
#                     jnp.std(b_hists, axis=0)
#                 ]
                
#                 return kde_counts

#         elif example == 'histosys':
#             def hist_maker(nn):
#                 '''
#                 Uses the nn decision function `predict` to form histograms from signal and background
#                 data, all drawn from multivariate normal distributions with different means. Three
#                 background distributions are sampled from, which mimics the situation in
#                 particle physics where one has a 'nominal' prediction for a nuisance parameter 
#                 (taken here as the mean of two modes) and then alternate values (e.g. from varying 
#                 up/down by one standard deviation), which then modifies the background pdf. Here, we
#                 take that effect to be a shift of the mean of the distribution. The HistFactory 
#                 'histosys' nusiance parameter will then be constructed from the yields downstream by
#                 interpolating between them using pyhf.

#                 Arguments:
#                     nn: jax array of observable parameters.
                    
#                     bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
#                         edges at 0, 0.5, 1.

#                     bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
#                     this fairly similar to the bin width to avoid oversmoothing the distribution. Going too low
#                     will cause things to break, as the gradients of the kde become unstable.

#                 Returns:
#                     Set of 4 counts for signal, background, and up/down modes.
#                 '''
#                 def hist_maker(nn):
#                     s, b_nom, b_up, b_down = data_generator()
#                     NMC = len(s)
                    
#                     nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#                         predict(nn, s).ravel(),
#                         predict(nn, b_nom).ravel(),
#                         predict(nn, b_up).ravel(),
#                         predict(nn, b_down).ravel(),
#                     )
                        
#                     kde_counts = jax.numpy.asarray([
#                         smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#                         smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                         smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                         smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                     ])
                    
#                     return kde_counts

#         else:
#             print(f'Unknown example \'{example}\'. Currently only \'three_blobs\' or \'histosys\' are available.')
#             raise
                
#         return hist_maker

#     def hists_from_nn_three_blobs(data_generator, predict, bins, method='kde', bandwidth=None, LUMI=10, sig_scale = 2, bkg_scale = 10):
    
#         def hist_maker(nn):
#             s, b_nom, b_up, b_down = data_generator()
#             NMC = len(s)
            
#             nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#                 predict(nn, s).ravel(),
#                 predict(nn, b_nom).ravel(),
#                 predict(nn, b_up).ravel(),
#                 predict(nn, b_down).ravel(),
#             )
                
#             kde_counts = jax.numpy.asarray([
#                 smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#                 smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             ])
        
#         return kde_counts

#     def hists_from_nn_three_blobs(data_generator, predict, bins, method='kde', bandwidth=None, LUMI=10, sig_scale = 2, bkg_scale = 10):
    
#         def hist_maker(nn):
#             s, b_nom, b_up, b_down = data_generator()
#             NMC = len(s)
            
#             nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#                 predict(nn, s).ravel(),
#                 predict(nn, b_nom).ravel(),
#                 predict(nn, b_up).ravel(),
#                 predict(nn, b_down).ravel(),
#             )
                
#             kde_counts = jax.numpy.asarray([
#                 smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#                 smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             ])
        
#         return kde_counts

#     def hists_from_nn_three_blobs(data_generator, predict, bins, method='kde', bandwidth=None, LUMI=10, sig_scale = 2, bkg_scale = 10):
    
#         def hist_maker(nn):
#             s, b_nom, b_up, b_down = data_generator()
#             NMC = len(s)
            
#             nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#                 predict(nn, s).ravel(),
#                 predict(nn, b_nom).ravel(),
#                 predict(nn, b_up).ravel(),
#                 predict(nn, b_down).ravel(),
#             )
                
#             kde_counts = jax.numpy.asarray([
#                 smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#                 smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             ])
        
#         return kde_counts

#     def softmax_hists_from_nn(data_generator, predict, bins, bandwidth, LUMI=10, sig_scale = 2, bkg_scale = 10):
    
#         def hist_maker(nn):
#             s, b_nom, b_up, b_down = data_generator()
#             NMC = len(s)
            
#             nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#                 predict(nn, s).ravel(),
#                 predict(nn, b_nom).ravel(),
#                 predict(nn, b_up).ravel(),
#                 predict(nn, b_down).ravel(),
#             )
                
#             kde_counts = jax.numpy.asarray([
#                 smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#                 smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#                 smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             ])
        
#         return kde_counts
    
#     return hist_maker
    
#     def get_hists(network, s, bs):
#         NMC = len(s)
#         s_hist = predict(network, s).sum(axis=0) * sig_scale / NMC * LUMI

#         b_hists = tuple(
#             (predict(network, b).sum(axis=0) * bkg_scale / NMC * LUMI) for b in bs
#         )

#         b_mean = jax.numpy.mean(jax.numpy.asarray(b_hists), axis=0)
#         b_unc = jax.numpy.std(jax.numpy.asarray(b_hists), axis=0)
#         results = s_hist, b_mean, b_unc
#         return results

#     def hist_maker():
#         bkg1 = np.random.multivariate_normal(b1_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg2 = np.random.multivariate_normal(b2_mean, [[1, 0], [0, 1]], size=(NMC,))
#         sig = np.random.multivariate_normal(sig_mean, [[1, 0], [0, 1]], size=(NMC,))

#         def make(network):
#             return get_hists(network, sig, (bkg1, bkg2))

#         make.bkg1 = bkg1
#         make.bkg2 = bkg2
#         make.sig = sig
#         return make

#     return hist_maker


# def kde_three_blobs(
#     predict,
#     bins,
#     bandwidth,
#     NMC=500,
#     sig_mean=[-1, 1],
#     b1_mean=[2, 2],
#     b2_mean=[-1, -1],
#     LUMI=10,
#     sig_scale=2,
#     bkg_scale=10,
# ):
#     """
#     Exactly the same as `hists_from_nn_three_blobs`, but takes in a regression network, and
#     forms a kernel density estimate (kde) for the output. The yields are then calculated as
#     the integral of the kde's cumulative density function between the bin edges, which should
#     be specified using the argument `bins`.

#     Args:
#             predict: Decision function for a parameterized observable. When evaluated, the
#             output should be one number per event, i.e. a regression network or similar.

#             bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
#             edges at 0, 0.5, 1.

#             bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
#             this lower than the bin width to avoid oversmoothing the distribution. Going too low
#             will cause things to break, as the gradients of the kde become unstable. 0.1*bin_width
#             is a good rule of thumb, but we have yet to properly validate this practically.

#     Returns:
#             hist_maker: A callable function that takes the parameters of the observable,
#             then constructs signal, background, and background uncertainty yields.
#     """
#     # grab bin edges
#     edge_lo = bins[:-1]
#     edge_hi = bins[1:]

#     # get counts from gaussian cdfs centered on each event, evaluated binwise
#     def to_hist(events):
#         cdf_up = jsc.stats.norm.cdf(edge_hi.reshape(-1, 1), loc=events, scale=bandwidth)
#         cdf_dn = jsc.stats.norm.cdf(edge_lo.reshape(-1, 1), loc=events, scale=bandwidth)
#         summed = (cdf_up - cdf_dn).sum(axis=1)
#         return summed

#     def get_hists(network, s, b1, b2):
#         NMC = len(s)
#         nn_s, nn_b1, nn_b2 = (
#             predict(network, s).ravel(),
#             predict(network, b1).ravel(),
#             predict(network, b2).ravel(),
#         )

#         kde_counts = jax.numpy.asarray(
#             [
#                 to_hist(nn_s) * sig_scale / NMC * LUMI,
#                 to_hist(nn_b1) * bkg_scale / NMC * LUMI,
#                 to_hist(nn_b2) * bkg_scale / NMC * LUMI,
#             ]
#         )

#         b_mean = jax.numpy.mean(kde_counts[1:], axis=0)
#         b_unc = jax.numpy.std(kde_counts[1:], axis=0)
#         results = kde_counts[0], b_mean, b_unc
#         return results

#     def hist_maker():
#         bkg1 = np.random.multivariate_normal(b1_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg2 = np.random.multivariate_normal(b2_mean, [[1, 0], [0, 1]], size=(NMC,))
#         sig = np.random.multivariate_normal(sig_mean, [[1, 0], [0, 1]], size=(NMC,))

#         def make(network):
#             return get_hists(network, sig, bkg1, bkg2)

#         make.bkg1 = bkg1
#         make.bkg2 = bkg2
#         make.sig = sig
#         return make

#     return hist_maker


# def kde_histosys(
#     predict,
#     bins,
#     bandwidth,
#     NMC=500,
#     sig_mean=[-1, 1],
#     b1_mean=[2.5, 2],
#     b_mean=[1, -1],
#     b2_mean=[-2.5, -1.5],
#     LUMI=10,
#     sig_scale=2,
#     bkg_scale=10,
# ):
#     """
#     Exactly the same as `hists_from_nn_three_blobs`, but takes in a regression network, and
#     forms a kernel density estimate (kde) for the output. The yields are then calculated as
#     the integral of the kde's cumulative density function between the bin edges, which should
#     be specified using the argument `bins`.

#     Args:
#             predict: Decision function for a parameterized observable. When evaluated, the
#             output should be one number per event, i.e. a regression network or similar.

#             bins: Array of bin edges, e.g. np.linspace(0,1,3) defines a two-bin histogram with
#             edges at 0, 0.5, 1.

#             bandwidth: Float that controls the 'smoothness' of the kde. It's recommended to keep
#             this lower than the bin width to avoid oversmoothing the distribution. Going too low
#             will cause things to break, as the gradients of the kde become unstable. 0.1*bin_width
#             is a good rule of thumb, but we have yet to properly validate this practically.

#     Returns:
#             hist_maker: A callable function that takes the parameters of the observable,
#             then constructs signal, background, and background uncertainty yields.
#     """

#     def get_hists(network, s, b_nom, b_up, b_down):
#         NMC = len(s)
#         nn_s, nn_b_nom, nn_b_up, nn_b_down = (
#             predict(network, s).ravel(),
#             predict(network, b_nom).ravel(),
#             predict(network, b_up).ravel(),
#             predict(network, b_down).ravel(),
#         )

#          kde_counts = jax.numpy.asarray([
#             smooth.hist(nn_s, bins, bandwidth) * sig_scale / NMC * LUMI,
#             smooth.hist(nn_b_nom, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             smooth.hist(nn_b_up, bins, bandwidth) * bkg_scale / NMC * LUMI,
#             smooth.hist(nn_b_down, bins, bandwidth) * bkg_scale / NMC * LUMI,
#         ])

#         return kde_counts

#     def hist_maker():
#         bkg_up = np.random.multivariate_normal(b1_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg_down = np.random.multivariate_normal(b2_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg_nom = np.random.multivariate_normal(b_mean, [[1, 0], [0, 1]], size=(NMC,))
#         sig = np.random.multivariate_normal(sig_mean, [[1, 0], [0, 1]], size=(NMC,))

#         def make(network):
#             return get_hists(network, sig, bkg_nom, bkg_up, bkg_down)

#         make.bkg_nom = bkg_nom
#         make.bkg_up = bkg_up
#         make.bkg_down = bkg_down
#         make.sig = sig
#         return make

#     return hist_maker

  
# def softmax_histosys(
#     predict,
#     NMC=500,
#     sig_mean=[-1, 1],
#     b1_mean=[2.5, 2],
#     b_mean=[1, -1],
#     b2_mean=[-2.5, -1.5],
#     LUMI=10,
#     sig_scale=2,
#     bkg_scale=10,
# ):
#     """
#     Exactly the same as `hists_from_nn_three_blobs`, but takes in a regression network, and
#     forms a kernel density estimate (kde) for the output. The yields are then calculated as
#     the integral of the kde's cumulative density function between the bin edges, which should
#     be specified using the argument `bins`.

#     Args:
#             predict: Decision function for a parameterized observable. When evaluated, the
#             output should be one number per bin.

#     Returns:
#             hist_maker: A callable function that takes the parameters of the observable,
#             then constructs signal, background, and background uncertainty yields.
#     """

#     def get_hists(network, s, b_nom, b_up, b_down):
#         NMC = len(s)
#         counts = jax.numpy.asarray(
#             [
#                 predict(network, s).sum(axis=0)* sig_scale / NMC * LUMI,
#                 predict(network, b_nom).sum(axis=0)* bkg_scale / NMC * LUMI,
#                 predict(network, b_up).sum(axis=0)* bkg_scale / NMC * LUMI,
#                 predict(network, b_down).sum(axis=0)* bkg_scale / NMC * LUMI
#             ]
#         )

#         return counts

#     def hist_maker():
#         bkg_up = np.random.multivariate_normal(b1_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg_down = np.random.multivariate_normal(b2_mean, [[1, 0], [0, 1]], size=(NMC,))
#         bkg_nom = np.random.multivariate_normal(b_mean, [[1, 0], [0, 1]], size=(NMC,))
#         sig = np.random.multivariate_normal(sig_mean, [[1, 0], [0, 1]], size=(NMC,))

#         def make(network):
#             return get_hists(network, sig, bkg_nom, bkg_up, bkg_down)

#         make.bkg_nom = bkg_nom
#         make.bkg_up = bkg_up
#         make.bkg_down = bkg_down
#         make.sig = sig
#         return make

#     return hist_maker

  
import pyhf
from .models import hepdata_like

pyhf.set_backend('jax')


def nn_hepdata_like(histogram_maker):
    """
    Returns a function that constructs a typical 'hepdata-like' statistical model
    with signal, background, and background uncertainty yields when evaluated at
    the parameters of the observable.

    Args:
            histogram_maker: A function that, when called, returns a secondary function
            that takes the observable's parameters as argument, and returns yields.

    Returns:
            nn_model_maker: A function that returns a Model object (either from
            `neos.models` or from `pyhf`) when evaluated at the observable's parameters,
            along with the background-only parameters for use in downstream inference.
    """
    def nn_model_maker(hm_params):
        s, b, db = histogram_maker(hm_params)
        m = hepdata_like(s, b, db)  # neos 'pyhf' model
        nompars = m.config.suggested_init()
        bonlypars = jnp.asarray([x for x in nompars])
        bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
        return m, bonlypars

    return nn_model_maker


def nn_histosys(histogram_maker):
    """
    Returns a function that constructs a HEP statistical model using a 'histosys'
    uncertainty for the background (nominal background, up and down systematic variations)
    when evaluated at the parameters of the observable.

    Args:
            histogram_maker: A function that, when called, returns a secondary function
            that takes the observable's parameters as argument, and returns yields.

    Returns:
            nn_model_maker: A function that returns a Model object (either from
            `neos.models` or from `pyhf`) when evaluated at the observable's parameters,
            along with the background-only parameters for use in downstream inference.
    """
    def from_spec(yields):

        s, b, bup, bdown = yields

        spec = {
            "channels": [
                {
                    "name": "nn",
                    "samples": [
                        {
                            "name": "signal",
                            "data": s,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "bkg",
                            "data": b,
                            "modifiers": [
                                {
                                    "name": "nn_histosys",
                                    "type": "histosys",
                                    "data": {"lo_data": bdown, "hi_data": bup,},
                                }
                            ],
                        },
                    ],
                },
            ],
        }

        return pyhf.Model(spec)

    def nn_model_maker(hm_params):
        yields = histogram_maker(hm_params)
        print(yields,len(yields))
        m = from_spec(yields)
        nompars = m.config.suggested_init()
        bonlypars = jnp.asarray([x for x in nompars])
        bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
        return m, bonlypars

    return nn_model_maker