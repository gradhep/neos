__all__ = ['hists_from_nn','hepdata_like_from_hists', 'histosys_model_from_hists']

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .smooth import kde_hist as hist

## Soft histogram makers

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


## Model makers
  
# gotta patch stuff
import sys
from unittest.mock import patch
# let's get started
import pyhf
jax_backend = pyhf.tensor.jax_backend(precision='64b')
pyhf.set_backend(jax_backend)

from .models import hepdata_like


def hepdata_like_from_hists(histogram_maker):
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


def histosys_model_from_hists(histogram_maker):
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

    # bunch of patches to make sure we use jax in pyhf
    @patch('pyhf.default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.interpolators.code0'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.interpolators.code1'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.interpolators.code2'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.interpolators.code4'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.interpolators.code4p'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.modifiers.shapefactor'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.modifiers.shapesys'], 'default_backend', new=jax_backend)
    @patch.object(sys.modules['pyhf.modifiers.staterror'], 'default_backend', new=jax_backend)
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
        m = from_spec(yields)
        nompars = m.config.suggested_init()
        bonlypars = jnp.asarray([x for x in nompars])
        bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
        return m, bonlypars

    return nn_model_maker