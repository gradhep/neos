__all__ = ["kde_hist", "cut"]

import jax.numpy as jnp
import jax.scipy as jsc

def kde_hist(events, bins, bandwidth=None, density=False):
    """
    Args:
            events: (jax array-like) data to filter.

            bins: (jax array-like) intervals to calculate counts.

            bandwidth: (float) value that specifies the width of the individual
            distributions (kernels) whose cdfs are averaged over each bin. Defaults
            to Scott's rule -- the same as the scipy implementation of kde.

            density: (bool) whether or not to normalize the histogram to unit area.
    Returns:
            binned counts, calculated by kde!
    """
    bandwidth = bandwidth or events.shape[-1]**-.25 # Scott's rule

    edge_hi = bins[1:] # ending bin edges ||<-
    edge_lo = bins[:-1] # starting bin edges ->||

    # get cumulative counts (area under kde) for each set of bin edges
    cdf_up = jsc.stats.norm.cdf(edge_hi.reshape(-1,1),loc = events, scale = bandwidth)
    cdf_dn = jsc.stats.norm.cdf(edge_lo.reshape(-1,1),loc = events, scale = bandwidth)
    # sum kde contributions in each bin
    counts = (cdf_up - cdf_dn).sum(axis=1) 

    if density: # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float) # bin spacing
        return counts/db/counts.sum(axis=0)

    return counts


def cut(events, sign, cut_val, slope=1.0):
    """
    Event weights from cutting `events` at `cut_val` with logical operator `sign` = '>' or '<'.

    Chain cuts by multiplying their output: `evt_weights = cut(data1, sign1, c1) * cut(data2, sign2, c2) etc.    

    Args:
            events: (jax array-like) data to filter.
    Returns:
            event weights!
    """
    if sign == ">":
        passed = 1 / (1 + jnp.exp(-slope * (events - cut_val)))
    elif sign == "<":
        passed = 1 - (1 / (1 + jnp.exp(-slope * (events - cut_val))))
    else:
        print("Invalid cut sign -- use > or <.")

    return passed
