import time

import jax
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
import jax.random
jax.config.update("jax_enable_x64", True)
from jax.random import PRNGKey
import numpy as np
from functools import partial

from neos import data, cls, makers

rng = PRNGKey(22)

init_random_params, predict = stax.serial(
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1),
    stax.Sigmoid
)

dgen = data.generate_blobs(rng,blobs=4) 
hmaker = makers.hists_from_nn(dgen, predict, method='kde')
nnm = makers.nn_histosys(hmaker)
get_cls = cls.expected_CLs_upper_limit(nnm, solver_kwargs=dict(pdf_transform=True))
loss = partial(get_cls,hyperparams=dict(bins=np.linspace(0,1,4),bandwidth=0.26666666666666666))

_, network = init_random_params(jax.random.PRNGKey(13), (-1, 2))

print(loss(test_mu = 1.0, params = network))