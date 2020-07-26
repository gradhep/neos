__all__ = ["expected_CLs_upper_limit"]

import jax
import jax.numpy as jnp
import pyhf
from functools import partial

pyhf.set_backend('jax')

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
    def get_expected_CLs(test_mu, params, hyperparams=None, return_pvalue=False):
        '''
        A callable function that takes the parameters of the observable as argument,
        and returns an expected CLs (or p-value if you set `return_pvalue`=True) from testing the background-only 
        model against the nominal signal hypothesis (or whatever corresponds to the value of the arg 'test_mu')
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
        
        if not return_pvalue:
            altval = 0
            CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
            return CLsb / CLb
          
        return CLsb

    return get_expected_CLs