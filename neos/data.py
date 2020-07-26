__all__ = ['blobs']

import jax

def blobs(rng, gen_nominal=True, NMC=500, sig_mean = [-1, 1], bup_mean=[2.5, 2], b_mean=[1, -1], bdown_mean=[-2.5, -1.5]):
    '''
    Two background distributions are sampled from, which is meant to mimic the situation in
    particle physics where one has a 'nominal' prediction for a nuisance parameter and then
    an alternate value (e.g. from varying up/down by one standard deviation), which then
    modifies the background pdf. Here, we take that effect to be a shift of the mean of the
    distribution. The value for the background histogram is then the mean of the resulting
    counts of the two modes, and the uncertainty can be quantified through the count
    standard deviation.
    '''
    
    def generate_blobs():
        sig = jax.random.multivariate_normal(rng, sig_mean, [[1, 0], [0, 1]], shape=(NMC,))
        bkg_up = jax.random.multivariate_normal(rng, bup_mean, [[1, 0], [0, 1]], shape=(NMC,))
        bkg_down = jax.random.multivariate_normal(rng, bdown_mean, [[1, 0], [0, 1]], shape=(NMC,))

        if gen_nominal:
            bkg_nom = jax.random.multivariate_normal(rng, b_mean, [[1, 0], [0, 1]], shape=(NMC,))
            return sig, bkg_nom, bkg_up, bkg_down
        
        return sig, bkg_up, bkg_down
    
    return generate_blobs