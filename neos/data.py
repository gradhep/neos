__all__ = ["generate_blobs"]

import jax
import jax.numpy as jnp


def generate_blobs(
    rng,
    blobs,
    NMC=500,
    sig_mean=jnp.asarray([-1, 1]),
    bup_mean=jnp.asarray([2.5, 2]),
    bdown_mean=jnp.asarray([-2.5, -1.5]),
    b_mean=jnp.asarray([1, -1]),
):
    """
    Function that returns a callable to generate a set of 2D normally distributed blobs, corresponding to signal, background, and background uncertainty modes.

    Args:
        rng: jax PRNG key (random seed).
        blobs: Number of blobs to generate (3 or 4).
        NMC: Number of 'monte carlo' samples to generate.
        sig_mean: jax array of the mean of the signal distribution.
        bup_mean: jax array of the mean of the 'up' background distribution.
        bdown_mean: jax array of the mean of the 'up' background distribution.
        b_mean: jax array of the mean of the nominal background distribution.
    """
    if blobs == 3:

        def gen_blobs():
            sig = jax.random.multivariate_normal(
                rng, sig_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )
            bkg_up = jax.random.multivariate_normal(
                rng, bup_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )
            bkg_down = jax.random.multivariate_normal(
                rng, bdown_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )

            return sig, bkg_up, bkg_down

    elif blobs == 4:

        def gen_blobs():
            sig = jax.random.multivariate_normal(
                rng, sig_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )
            bkg_up = jax.random.multivariate_normal(
                rng, bup_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )
            bkg_down = jax.random.multivariate_normal(
                rng, bdown_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )
            bkg_nom = jax.random.multivariate_normal(
                rng, b_mean, jnp.asarray([[1, 0], [0, 1]]), shape=(NMC,)
            )

            return sig, bkg_nom, bkg_up, bkg_down

    else:
        assert False, (
            f"Unsupported number of blobs: {blobs}"
            " (only using 3 or 4 blobs for these examples)."
        )

    return gen_blobs

