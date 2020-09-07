__all__ = ["Model", "hepdata_like"]

import jax
import jax.numpy as jnp
import pyhf

pyhf.set_backend("jax")

# class-based
class _Config(object):
    def __init__(self):
        self.poi_index = 0
        self.npars = 2

    def suggested_init(self):
        return jnp.asarray([1.0, 1.0])

    def suggested_bounds(self):
        return jnp.asarray([jnp.asarray([0.0, 10.0]), jnp.asarray([0.0, 10.0])])


class Model(object):
    def __init__(self, spec):
        self.sig, self.nominal, self.uncert = spec
        self.factor = (self.nominal / self.uncert) ** 2
        self.aux = 1.0 * self.factor
        self.config = _Config()

    def expected_data(self, pars, include_auxdata=True):
        mu, gamma = pars
        expected_main = jnp.asarray([gamma * self.nominal + mu * self.sig])
        aux_data = jnp.asarray([self.aux])
        return jnp.concatenate([expected_main, aux_data])

    def logpdf(self, pars, data):
        maindata, auxdata = data
        main, _ = self.expected_data(pars)
        _, gamma = pars
        main = pyhf.probability.Poisson(main).log_prob(maindata)
        constraint = pyhf.probability.Poisson(gamma * self.factor).log_prob(auxdata)
        # sum log probs over bins
        return jnp.asarray([jnp.sum(main + constraint, axis=0)])


def hepdata_like(signal_data, bkg_data, bkg_uncerts, batch_size=None):
    return Model([signal_data, bkg_data, bkg_uncerts])


# # functional version for fun :)
# from collections import namedtuple

# _Config = namedtuple("_Config", ["poi_index","npars","suggested_init","suggested_bounds"])

# def init_config():
#     return _Config(0,2,jnp.asarray([1.0, 1.0]),jnp.asarray(
#             [jnp.asarray([0.0, 10.0]), jnp.asarray([0.0, 10.0])]
#         ))

# Model = namedtuple("Model", ["sig", "nominal", "uncert", "factor", "aux", "config"])

# def init_model(spec):
#     sig, nominal, uncert = spec
#     factor = (nominal / uncert) ** 2
#     aux = 1.0 * factor
#     config = init_config()
#     return Model(sig, nominal, uncert, factor, aux, config)

# def expected_data(model, pars, include_auxdata=True):
#     mu, gamma = pars
#     expected_main = jnp.asarray([gamma * model.nominal + mu * model.sig])
#     aux_data = jnp.asarray([model.aux])
#     return jnp.concatenate([expected_main, aux_data])

# @jax.jit
# def logpdf(model, pars, data):
#     maindata, auxdata = data
#     main, _ = expected_data(model,pars)
#     mu, gamma = pars
#     main = pyhf.probability.Poisson(main).log_prob(maindata)
#     constraint = pyhf.probability.Poisson(gamma * model.factor).log_prob(auxdata)
#     # sum log probs over bins
#     return jnp.asarray([jnp.sum(main + constraint,axis=0)])


# def hepdata_like(signal_data, bkg_data, bkg_uncerts, batch_size=None):
#     return init_model([signal_data, bkg_data, bkg_uncerts])
