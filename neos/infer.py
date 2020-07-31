__all__ = ["expected_CLs_upper_limit"]

import jax
import jax.numpy as jnp
import pyhf
from functools import partial

pyhf.set_backend('jax')
# avoid those precision errors!
jax.config.update("jax_enable_x64", True)

from .fit import global_fit, constrained_fit
from .transforms import to_bounded_vec, to_inf_vec


def expected_CLs_upper_limit(model_maker, solver_kwargs):
    """
    Args:
        model_maker: Function that returns a Model object using the `params` arg.

    Returns:
        get_expected_CLs: A callable function that takes the parameters of the observable as argument,
        and returns an expected p-value from testing the background-only model against the
        nominal signal hypothesis (or whatever corresponds to the value of the arg 'test_mu')
    """

    @jax.jit
    def get_expected_CLs(params, test_mu, hyperparams=None, pvalues = ['CLs']):
        '''
        A callable function that takes the parameters of the observable as argument,
        and returns an expected CLs (or p-value if you set `return_pvalue`=True) from testing the background-only 
        model against the nominal signal hypothesis (or whatever corresponds to the value of the arg 'test_mu')

        Args:
            test_mu: a float for the value of mu to test against.

            params: jax array of the parameters of the observable.

            hyperparams: dict of hyperparameters for the model construction, e.g. bandwidth & binning for kde histograms.

            pvalues: list of strings corresponding to the pvalues that should be returned. Defaults to CL_s.

        '''
        #g_fitter = global_fit(model_maker, **solver_kwargs)
        c_fitter = constrained_fit(model_maker, **solver_kwargs)

        m, bonlypars = model_maker([params,hyperparams])
        exp_data = m.expected_data(bonlypars)
        bounds = m.config.suggested_bounds()

        # map these
        initval = jnp.asarray([test_mu, 1.0])
        transforms = solver_kwargs.get("pdf_transform", False)
        if transforms:
            initval = to_inf_vec(initval, bounds)

        # the constrained fit
        numerator = (
            to_bounded_vec(c_fitter(initval, [[params,hyperparams], test_mu]), bounds)
            if transforms
            else c_fitter(initval, [[params,hyperparams], test_mu])
        )

        # don't have to fit these -- we know them for expected limits!
        denominator = bonlypars  
        # denominator = to_bounded_vec(g_fitter(initval, params), bounds) if transforms else g_fitter(initval, params)

        # compute test statistic (lambda(µ))
        profile_likelihood = -2 * (
            m.logpdf(numerator, exp_data)[0] - m.logpdf(denominator, exp_data)[0]
        )

        # in exclusion fit zero out test stat if best fit µ^ is larger than test µ
        muhat = denominator[0]
        sqrtqmu = jnp.sqrt(jnp.where(muhat < test_mu, profile_likelihood, 0.0))
        CLsb =  1 - pyhf.tensorlib.normal_cdf(sqrtqmu)
        altval = 0
        CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
        CLs = CLsb/CLb
        
        pdict = dict(CLs=CLs,p_sb=CLsb,p_b=CLb)
        return [pdict[key] for key in pvalues]

    return get_expected_CLs