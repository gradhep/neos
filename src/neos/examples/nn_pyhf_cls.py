# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # KDE demo, with histosys!
#
# > It works :)
# %% [markdown]
# ![](assets/kde_pyhf_animation.gif)

# %%
import time

import jax
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
import jax.random
import numpy as np
import pyhf
from jax.random import PRNGKey
from relaxed import infer

from .. import data, makers

pyhf.set_backend("jax")
pyhf.default_backend = pyhf.tensor.jax_backend(precision="64b")


def neos_pyhf_example(rng=PRNGKey(1), maxN=10):

    # regression net
    init_random_params, predict = stax.serial(
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(1),
        stax.Sigmoid,
    )

    # %% [markdown]
    # ## Compose differentiable workflow

    # %%
    dgen = data.generate_blobs(rng, blobs=4)

    # Specify our hyperparameters ahead of time for the kde histograms
    bins = np.linspace(0, 1, 4)
    bandwidth = 0.27
    reflect_infinite_bins = True

    hmaker = makers.hists_from_nn(
        dgen,
        predict,
        hpar_dict=dict(bins=bins, bandwidth=bandwidth),
        method="kde",
        reflect_infinities=reflect_infinite_bins,
    )

    # %%
    nnm = makers.histosys_model_from_hists(hmaker)
    get_cls = infer.make_hypotest(nnm, solver_kwargs=dict(pdf_transform=True))

    # loss returns a list of metrics -- let's just index into one (CLs)
    def loss(params, test_mu):
        return get_cls(params, test_mu)["CLs"]

    # %% [markdown]
    # ### Randomly initialise nn weights and check that we can get the gradient of the loss wrt nn params

    # %%
    _, network = init_random_params(jax.random.PRNGKey(2), (-1, 2))

    # gradient wrt nn weights
    jax.value_and_grad(loss)(network, test_mu=1.0)

    # %% [markdown]
    # ### Define training loop!

    # %%
    opt_init, opt_update, opt_params = optimizers.adam(1e-3)

    def train_network(N):
        _, network = init_random_params(jax.random.PRNGKey(1), (-1, 2))
        state = opt_init(network)
        losses = []

        # parameter update function
        # @jax.jit
        def update_and_value(i, opt_state, mu):
            net = opt_params(opt_state)
            value, grad = jax.value_and_grad(loss)(net, mu)
            return opt_update(i, grad, state), value, net

        for i in range(N):
            start_time = time.time()
            state, value, network = update_and_value(i, state, 1.0)
            epoch_time = time.time() - start_time
            losses.append(value)
            metrics = {"loss": losses}

            yield network, metrics, epoch_time

    # Training
    for i, (network, metrics, epoch_time) in enumerate(train_network(maxN)):
        pass  # print(f"epoch {i}:", f'CLs = {metrics["loss"][-1]}, took {epoch_time}s')
    # %%

    return metrics["loss"][-1]
