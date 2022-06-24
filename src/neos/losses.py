from __future__ import annotations

__all__ = (
    "poi_uncert",
    "discovery_significance",
    "cls_value",
    "generalised_variance",
    "bce",
)

import jax.numpy as jnp
import pyhf
import relaxed

Array = jnp.ndarray


def poi_uncert(model: pyhf.Model) -> float:
    hypothesis_pars = (
        jnp.asarray(model.config.suggested_init()).at[model.config.poi_index].set(1.0)
    )
    observed_hist = jnp.asarray(model.expected_data(hypothesis_pars))
    return relaxed.cramer_rao_uncert(model, hypothesis_pars, observed_hist)[
        model.config.poi_index
    ]


def discovery_significance(model: pyhf.Model, fit_lr: float) -> float:
    test_stat = "q0"
    test_poi = 0.0  # background-only as the alternative
    # nominal s+b as the null
    hypothesis_pars = (
        jnp.asarray(model.config.suggested_init()).at[model.config.poi_index].set(1.0)
    )
    observed_hist = jnp.asarray(model.expected_data(hypothesis_pars))
    return relaxed.infer.hypotest(
        test_poi=test_poi,
        data=observed_hist,
        model=model,
        test_stat=test_stat,
        expected_pars=hypothesis_pars,
        lr=fit_lr,
    )


def cls_value(model: pyhf.Model, fit_lr: float) -> float:
    test_stat = "q"
    test_poi = 1.0  # nominal s+b as the null
    # background-only as the alternative
    hypothesis_pars = (
        jnp.asarray(model.config.suggested_init()).at[model.config.poi_index].set(0.0)
    )
    observed_hist = jnp.asarray(model.expected_data(hypothesis_pars))
    return relaxed.infer.hypotest(
        test_poi=test_poi,
        data=observed_hist,
        model=model,
        test_stat=test_stat,
        expected_pars=hypothesis_pars,
        lr=fit_lr,
    )


def generalised_variance(model: pyhf.Model) -> float:
    hypothesis_pars = (
        jnp.asarray(model.config.suggested_init()).at[model.config.poi_index].set(0.0)
    )
    observed_hist = jnp.asarray(model.expected_data(hypothesis_pars))
    return 1 / jnp.linalg.det(
        relaxed.fisher_info(model, hypothesis_pars, observed_hist)
    )


def sigmoid_cross_entropy_with_logits(preds, labels):
    return jnp.mean(
        jnp.maximum(preds, 0) - preds * labels + jnp.log1p(jnp.exp(-jnp.abs(preds)))
    )


def bce(data, nn, pars):
    preds = {k: nn(pars, data[k]).ravel() for k in data}
    bkg = jnp.concatenate([preds[k] for k in preds if "sig" not in k])
    sig = preds["sig"]
    labels = jnp.concatenate([jnp.ones_like(sig), jnp.zeros_like(bkg)])
    return sigmoid_cross_entropy_with_logits(
        jnp.concatenate(list(preds.values())).ravel(), labels
    ).mean()
