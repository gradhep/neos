from __future__ import annotations

__all__ = (
    "loss_from_model",
    "hists_from_nn",
)

from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import pyhf
import relaxed

from neos.losses import (
    cls_value,
    discovery_significance,
    generalised_variance,
    poi_uncert,
)

Array = jnp.ndarray


def hists_from_nn(
    pars: Array,
    data: dict[str, Array],
    nn: Callable,
    bandwidth: float,
    bins: Array,
    scale_factors: dict[str, float] | None = None,
    overall_scale: float = 1.0,
) -> dict[str, Array]:
    """Function that takes in data + analysis config parameters, and constructs yields."""
    # apply the neural network to each data sample, and keep track of the sample names in a dict
    nn_output = {k: nn(pars, data[k]).ravel() for k in data}

    # The next two lines allow you to also optimise your binning:
    bins_new = jnp.concatenate(
        (
            jnp.array([bins[0]]),
            jnp.where(bins[1:] > bins[:-1], bins[1:], bins[:-1] + 1e-4),
        ),
        axis=0,
    )
    # define our histogram-maker with some hyperparameters (bandwidth, binning)
    make_hist = partial(relaxed.hist, bandwidth=bandwidth, bins=bins_new)

    # every histogram is scaled to the number of points from that data source in the batch
    # so we have more control over the scaling of sig/bkg for realism
    scale_factors = scale_factors or {k: 1.0 for k in nn_output}
    hists = {
        k: make_hist(nn_output[k]) * scale_factors[k] * overall_scale / len(v)
        + 1e-3  # add a floor so no zeros in any bin!
        for k, v in nn_output.items()
    }
    return hists


def loss_from_model(
    model: pyhf.Model,
    loss: str | Callable[[dict[str, Any]], float] = "neos",
    fit_lr: float = 1e-3,
) -> float:
    if isinstance(loss, Callable):
        # everything
        return 0
    # loss specific
    if loss.lower() == "discovery":
        return discovery_significance(model, fit_lr)
    elif loss.lower() in ["neos", "cls"]:
        return cls_value(model, fit_lr)
    elif loss.lower() in ["inferno", "poi_uncert", "mu_uncert"]:
        return poi_uncert(model)
    elif loss.lower() in [
        "general_variance",
        "generalised_variance",
        "generalized_variance",
    ]:
        return generalised_variance(model)
    else:
        raise ValueError(f"loss function {loss} not recognised")
