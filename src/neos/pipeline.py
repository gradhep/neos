from __future__ import annotations

__all__ = ("Pipeline",)

import time
from functools import partial
from pprint import pprint
from typing import Any, Callable, NamedTuple

import jax.numpy as jnp
import jaxopt
import numpy.random as npr
import optax
import pyhf
import relaxed
from chex import Array
from jax import jit
from sklearn.model_selection import train_test_split


@partial(
    jit, static_argnames=["model", "return_mle_pars", "return_constrained_pars"]
)  # forward pass
def hypotest(
    test_poi: float,
    data: Array,
    model: pyhf.Model,
    lr: float,
    bonly_pars: Array,
    return_constrained_pars: bool = False,
) -> tuple[Array, Array] | Array:
    # hard-code 1 as inits for now
    # TODO: need to parse different inits for constrained and global fits
    init_pars = jnp.asarray(model.config.suggested_init())[
        model.config.par_slice("correlated_bkg_uncertainty")
    ]
    conditional_pars = relaxed.mle.fixed_poi_fit(
        data, model, poi_condition=test_poi, init_pars=init_pars, lr=lr
    )
    mle_pars = bonly_pars
    profile_likelihood = -2 * (
        model.logpdf(conditional_pars, data)[0] - model.logpdf(mle_pars, data)[0]
    )

    poi_hat = mle_pars[model.config.poi_index]
    qmu = jnp.where(poi_hat < test_poi, profile_likelihood, 0.0)

    CLsb = 1 - pyhf.tensorlib.normal_cdf(jnp.sqrt(qmu))
    altval = 0.0
    CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    if return_constrained_pars:
        return CLs, conditional_pars
    else:
        return CLs


class Pipeline(NamedTuple):
    """Class to compose the pipeline for training a learnable summary statistic."""

    yields_from_pars: Callable[..., tuple[Array, ...]]
    model_from_yields: Callable[..., pyhf.Model]
    init_pars: Array
    data: Array | None = None
    yield_kwargs: dict[str, Any] | None = None
    nuisance_parname: str = "correlated_bkg_uncertainty"
    random_state: int = 0
    num_epochs: int = 20
    batch_size: int = 500
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: Callable[[dict], float] = lambda x: x["CLs"]
    test_size: float = 0.2
    per_epoch_callback: Callable = lambda x: None
    first_epoch_callback: Callable = lambda x: None
    last_epoch_callback: Callable = lambda x: None
    post_training_callback: Callable = lambda x: None
    plot_setup: Callable = lambda x: None
    possible_metrics: tuple[str, ...] = (
        "CLs",
        "mu_uncert",
        "1-pull_width**2",
        "gaussianity",
    )
    animate: bool = True
    plotname: str = "neos_demo.png"
    animationname: str = "neos_demo.gif"

    def run(self):
        pyhf.set_backend("jax", default=True)

        def pipeline(pars, data):
            yields = self.yields_from_pars(pars, data, **self.yield_kwargs)
            model = self.model_from_yields(*yields)
            state: dict[str, Any] = {}
            state["yields"] = yields
            bonly_pars = (
                jnp.asarray(model.config.suggested_init())
                .at[model.config.poi_index]
                .set(0.0)
            )
            data = jnp.asarray(model.expected_data(bonly_pars))
            state["CLs"], constrained = hypotest(
                1.0,
                data,
                model,
                return_constrained_pars=True,
                bonly_pars=bonly_pars,
                lr=1e-2,
            )
            uncerts = relaxed.cramer_rao_uncert(model, bonly_pars, data)
            state["mu_uncert"] = uncerts[model.config.poi_index]
            pull_width = uncerts[model.config.par_slice(self.nuisance_parname)][0]
            state["pull_width"] = pull_width
            state["1-pull_width**2"] = (1 - pull_width) ** 2
            # state["gaussianity"] = relaxed.gaussianity(model, bonly_pars, data, rng_key=PRNGKey(self.random_state))
            state["pull"] = jnp.array(
                [
                    (constrained - jnp.array(model.config.suggested_init()))[
                        model.config.par_order.index(k)
                    ]
                    / model.config.param_set(k).width()[0]
                    for k in model.config.par_order
                    if model.config.param_set(k).constrained
                ]
            )
            loss = self.loss(state)
            state["loss"] = loss
            return loss, state

        if self.data is not None:
            split = train_test_split(
                *self.data, test_size=self.test_size, random_state=self.random_state
            )
            train, _ = split[::2], split[1::2]

            num_train = train[0].shape[0]
            num_complete_batches, leftover = divmod(num_train, self.batch_size)
            num_batches = num_complete_batches + bool(leftover)

            # batching mechanism
            def data_stream():
                rng = npr.RandomState(self.random_state)
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[
                            i * self.batch_size : (i + 1) * self.batch_size
                        ]
                        yield [points[batch_idx] for points in train]

            batches = data_stream()
        else:

            def blank_data():
                while True:
                    yield None

            batches = blank_data()

        solver = jaxopt.OptaxSolver(
            fun=pipeline, opt=optax.adam(self.learning_rate), has_aux=True
        )
        params, state = solver.init(self.init_pars)

        plot_kwargs = self.plot_setup(self)

        metrics = {"CLs": [], "mu_uncert": [], "1-pull_width**2": [], "loss": []}
        metric_keys = list(metrics.keys())
        for epoch_num in range(self.num_epochs):
            batch_data = next(batches)
            print(f"epoch {epoch_num}: ", end="")
            start = time.perf_counter()
            params, state = solver.update(params=params, state=state, data=batch_data)
            end = time.perf_counter()
            t = end - start
            print(f"took {t:.4f}s. state:")
            pprint(state.aux)
            for key in state.aux:
                if key in metric_keys:
                    metrics[key].append(state.aux[key])
                else:
                    metrics[key] = state.aux[key]

            if epoch_num == 0:
                plot_kwargs["camera"] = self.first_epoch_callback(
                    params,
                    this_batch=batch_data,
                    metrics=metrics,
                    maxN=self.num_epochs,
                    **self.yield_kwargs,
                    **plot_kwargs,
                )
            elif epoch_num == self.num_epochs - 1:
                plot_kwargs["camera"] = self.last_epoch_callback(
                    params,
                    this_batch=batch_data,
                    metrics=metrics,
                    maxN=self.num_epochs,
                    pipeline=self,
                    **self.yield_kwargs,
                    **plot_kwargs,
                )
            else:
                plot_kwargs["camera"] = self.per_epoch_callback(
                    params,
                    this_batch=batch_data,
                    metrics=metrics,
                    maxN=self.num_epochs,
                    **self.yield_kwargs,
                    **plot_kwargs,
                )
        if self.animate:
            plot_kwargs["camera"].animate().save(
                f"{self.animationname}", writer="imagemagick", fps=9
            )
