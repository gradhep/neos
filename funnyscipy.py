import jax.experimental.optimizers as optimizers
import jax.numpy as jnp
import jax
from jax import lax
from fax.implicit import twophase

def objective_funcs(func,x0):
    init, update, get_params  = optimizers.adam(1e-2)
    def myupdate(i,state):
        params = get_params(state)
        return update(i,jax.grad(func)(params),state)
    return init(x0), myupdate, get_params

def minimize(func,x0):
    state, myupdate, get_params = objective_funcs(func,x0)

    state = lax.fori_loop(0,int(1e7),myupdate,state)
    a,b,c = state

    result = get_params(state)
    return result

