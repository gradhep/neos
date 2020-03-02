import neos.transforms as transforms
import jax.numpy as jnp
import neos.models as models
import jax
import scipy.optimize
import neos.fit as fit
from neos.cls import cls_maker
import pyhf
import funnyscipy


bounds = jnp.array([
    [0,10],
    [0,20]
])


# check that we map to inf space (i.e. -pi/2 to pi/2)
w = jnp.linspace(0,10)
x = transforms.toinf(w,bounds[0])
print(x.min(),x.max())


# check that we can map very large values to bounded space
w = jnp.linspace(-1000,1000,1001)
x = transforms.to_bounded(w,bounds[0])
print(x.min(),x.max())

# define NLL functions in both parameter spaces

def make_nll_boundspace(hyperpars):
    s,b,db = hyperpars
    def nll_boundspace(pars):
        truth_pars = [0,1]
        m = models.hepdata_like([s],[b],[db])
        val = m.logpdf(pars,m.expected_data(truth_pars))
        return -val[0]
    return nll_boundspace

def make_nll_infspace(hyperpars):
    s,b,db = hyperpars
    def nll_infspace(pars):
        truth_pars = [0,1]

        pars = transforms.to_bounded_vec(pars,bounds)
        
        m = models.hepdata_like([s],[b],[db])
        val = m.logpdf(pars,m.expected_data(truth_pars))
        return -val[0]
    return nll_infspace


nll_boundspace = make_nll_boundspace([1,50,7])
nll_infspace   = make_nll_infspace([1,50,7])

# define a point and compute it in both spaces
apoint_bnd = jnp.array([0.5,0.5])
apoint_inf = transforms.toinf_vec(apoint_bnd,bounds)

# check consistency in both spaces
print(nll_boundspace(apoint_bnd))
print(nll_infspace(apoint_inf))

# check gradiends in bounded
print(jax.grad(nll_boundspace)(apoint_bnd))

# check gradients in inf
print(jax.grad(nll_infspace)(apoint_inf))

# check consistency of gradients
print(jax.grad(nll_infspace)(apoint_inf)*jnp.array([jax.grad(lambda x,b: transforms.toinf_vec(x,b)[i])(apoint_bnd,bounds)[i] for i in range(2)]))


def fit_nll_bounded(init, hyperpars):
    mu, model_pars = hyperpars[0],hyperpars[1:]
    objective = make_nll_boundspace(model_pars)
    return scipy.optimize.minimize(objective, x0 = init, bounds = bounds).x

def fit_nll_infspace(init, hyperpars):
    mu, model_pars = hyperpars[0],hyperpars[1:]
    objective = make_nll_infspace(model_pars)
    # result = scipy.optimize.minimize(objective, x0 = init).x
    result = funnyscipy.minimize(objective, x0 = init)
    return transforms.to_bounded_vec(result,bounds)

# fit in bounded space
if False:
    print('scipy minim in bounded space')
    print(fit_nll_bounded(apoint_bnd,[1.0,5,50,7]))
    print(fit_nll_bounded(apoint_bnd,[1.0,5,50,2]))
    print(fit_nll_bounded(apoint_bnd,[1.0,5,50,1]))
    print(fit_nll_bounded(apoint_bnd,[1.0,5,50,.1]))
    print(fit_nll_bounded(apoint_bnd,[1.0,5,50,.01]))

# fit in inf space
if False:
    print('scipy minim in inf space')
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,7]))
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,2]))
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,1]))
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,.1]))
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,.01]))
    print(fit_nll_infspace(apoint_inf,[1.0,5,50,.001]))


def nn_model_maker(nn_params):
    s,b,db = nn_params
    m = models.hepdata_like([s], [b], [db])
    nompars = m.config.suggested_init()
    bonlypars = jax.numpy.asarray([x for x in nompars])
    bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
    return m, bonlypars

g_fitter, c_fitter = fit.get_solvers(nn_model_maker,pdf_transform=True, learning_rate=1e-4)

bounds = jnp.array([[0.,10],[0.,10.]])

if False:
    print('diffable minim in inf space')
    apoint_bnd = jnp.array([0.5,0.5])
    apoint_inf = transforms.toinf_vec(apoint_bnd,bounds)
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,7.0]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,2.0]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,1.0]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,0.5]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,0.1]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,50,0.01]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[5,55,1.5]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[10,5,1.5]]),bounds))
    print(transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,[2,90,1.5]]),bounds))


print('global fit grad')
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,x]),bounds)[0])([5.,50.,15.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,x]),bounds)[0])([5.,50.,10.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,x]),bounds)[0])([5.,50.,7.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(g_fitter(apoint_inf,[1.0,x]),bounds)[0])([5.,50.,1.0]))

print('constrained!')

apoint_bnd = jnp.array([1.0,1.0])
apoint_inf = transforms.toinf_vec(apoint_bnd,bounds)
print(transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,[5,50,15.0]]),bounds))
print(transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,[5,50,10.0]]),bounds))
print(transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,[5,50,7.0]]),bounds))
print(transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,[5,50,1.0]]),bounds))
print(transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,[5,50,0.1]]),bounds))


print('constrained fit grad')
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,x]),bounds)[1])([5.,50.,15.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,x]),bounds)[1])([5.,50.,10.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,x]),bounds)[1])([5.,50.,7.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,x]),bounds)[1])([5.,50.,1.0]))
print(jax.value_and_grad(lambda x: transforms.to_bounded_vec(c_fitter(apoint_inf,[1.0,x]),bounds)[1])([5.,50.,0.1]))

def fit_nll_bounded_constrained(init, hyperpars,fixed_val):
    mu, model_pars = hyperpars[0],hyperpars[1:]
    objective = make_nll_boundspace(model_pars)
    return scipy.optimize.minimize(objective, x0 = init, bounds = bounds, constraints=[{'type': 'eq', 'fun': lambda v: v[0] - fixed_val}]).x

print('reference')
print(fit_nll_bounded_constrained(apoint_bnd,[1.0,5,50,15.0],1.0))
print(fit_nll_bounded_constrained(apoint_bnd,[1.0,5,50,10.0],1.0))
print(fit_nll_bounded_constrained(apoint_bnd,[1.0,5,50,7.0],1.0))
print(fit_nll_bounded_constrained(apoint_bnd,[1.0,5,50,1.0],1.0))
print(fit_nll_bounded_constrained(apoint_bnd,[1.0,5,50,0.1],1.0))



print('diffable cls')

print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([5.,50.,15.0],1.0))
print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([5.,50.,10.0],1.0))
print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([5.,50.,7.0],1.0))
print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([5.,50.,1.0],1.0))
print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([5.,50.,0.1],1.0))

print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([10.,5.,0.1],1.0))
print(jax.value_and_grad(cls_maker(nn_model_maker,solver_kwargs=dict(pdf_transform=True)))([15.,5.,0.1],1.0))


print('cross check cls')
def pyhf_cls(nn_params,mu):
    s,b,db =  nn_params
    m = pyhf.simplemodels.hepdata_like([s],[b],[db])
    return pyhf.infer.hypotest(1.0,[b]+m.config.auxdata,m)[0]
    
print(pyhf_cls([5.,50.,15.0],1.0))
print(pyhf_cls([5.,50.,10.0],1.0))
print(pyhf_cls([5.,50.,7.0],1.0))
print(pyhf_cls([5.,50.,1.0],1.0))
print(pyhf_cls([5.,50.,0.1],1.0))

print(pyhf_cls([10.,5.,0.1],1.0))
print(pyhf_cls([15.,5.,0.1],1.0))
