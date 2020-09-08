i# hide
import scipy

import pyhf
from neos import infer, fit

pyhf.set_backend(pyhf.tensor.jax_backend())


def fit_nll_bounded(init, hyperpars):
    mu, model_pars = hyperpars[0], hyperpars[1:]
    objective = make_nll_boundspace(model_pars)
    return scipy.optimize.minimize(objective, x0=init, bounds=bounds).x


def fit_nll_infspace(init, hyperpars):
    mu, model_pars = hyperpars[0], hyperpars[1:]
    objective = make_nll_infspace(model_pars)
    # result = scipy.optimize.minimize(objective, x0 = init).x
    result = funnyscipy.minimize(objective, x0=init)
    return to_bounded_vec(result, bounds)


# fit in bounded space
if False:
    print("scipy minim in bounded space")
    print(fit_nll_bounded(apoint_bnd, [1.0, 5, 50, 7]))
    print(fit_nll_bounded(apoint_bnd, [1.0, 5, 50, 2]))
    print(fit_nll_bounded(apoint_bnd, [1.0, 5, 50, 1]))
    print(fit_nll_bounded(apoint_bnd, [1.0, 5, 50, 0.1]))
    print(fit_nll_bounded(apoint_bnd, [1.0, 5, 50, 0.01]))

# fit in inf space
if False:
    print("scipy minim in inf space")
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 7]))
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 2]))
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 1]))
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 0.1]))
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 0.01]))
    print(fit_nll_infspace(apoint_inf, [1.0, 5, 50, 0.001]))


def nn_model_maker(nn_params):
    s, b, db = nn_params
    m = models.hepdata_like(jnp.asarray([s]), jnp.asarray([b]), jnp.asarray([db]))
    nompars = m.config.suggested_init()
    bonlypars = jax.numpy.asarray([x for x in nompars])
    bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
    return m, bonlypars


g_fitter, c_fitter = fit.get_solvers(
    nn_model_maker, pdf_transform=True, learning_rate=1e-4
)

bounds = jnp.array([[0.0, 10], [0.0, 10.0]])

if False:
    print("diffable minim in inf space")
    apoint_bnd = jnp.array([0.5, 0.5])
    apoint_inf = to_inf_vec(apoint_bnd, bounds)
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 7.0]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 2.0]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 1.0]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 0.5]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 0.1]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 50, 0.01]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [5, 55, 1.5]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [10, 5, 1.5]]), bounds))
    print(to_bounded_vec(g_fitter(apoint_inf, [1.0, [2, 90, 1.5]]), bounds))


print("global fit grad")
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(g_fitter(apoint_inf, [1.0, x]), bounds)[0]
    )([5.0, 50.0, 15.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(g_fitter(apoint_inf, [1.0, x]), bounds)[0]
    )([5.0, 50.0, 10.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(g_fitter(apoint_inf, [1.0, x]), bounds)[0]
    )([5.0, 50.0, 7.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(g_fitter(apoint_inf, [1.0, x]), bounds)[0]
    )([5.0, 50.0, 1.0])
)

print("constrained!")

apoint_bnd = jnp.array([1.0, 1.0])
apoint_inf = to_inf_vec(apoint_bnd, bounds)
print(to_bounded_vec(c_fitter(apoint_inf, [1.0, [5, 50, 15.0]]), bounds))
print(to_bounded_vec(c_fitter(apoint_inf, [1.0, [5, 50, 10.0]]), bounds))
print(to_bounded_vec(c_fitter(apoint_inf, [1.0, [5, 50, 7.0]]), bounds))
print(to_bounded_vec(c_fitter(apoint_inf, [1.0, [5, 50, 1.0]]), bounds))
print(to_bounded_vec(c_fitter(apoint_inf, [1.0, [5, 50, 0.1]]), bounds))


print("constrained fit grad")
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(c_fitter(apoint_inf, [1.0, x]), bounds)[1]
    )([5.0, 50.0, 15.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(c_fitter(apoint_inf, [1.0, x]), bounds)[1]
    )([5.0, 50.0, 10.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(c_fitter(apoint_inf, [1.0, x]), bounds)[1]
    )([5.0, 50.0, 7.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(c_fitter(apoint_inf, [1.0, x]), bounds)[1]
    )([5.0, 50.0, 1.0])
)
print(
    jax.value_and_grad(
        lambda x: to_bounded_vec(c_fitter(apoint_inf, [1.0, x]), bounds)[1]
    )([5.0, 50.0, 0.1])
)


def fit_nll_bounded_constrained(init, hyperpars, fixed_val):
    mu, model_pars = hyperpars[0], hyperpars[1:]
    objective = make_nll_boundspace(model_pars)
    return scipy.optimize.minimize(
        objective,
        x0=init,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda v: v[0] - fixed_val}],
    ).x


print("reference")
print(fit_nll_bounded_constrained(apoint_bnd, [1.0, 5, 50, 15.0], 1.0))
print(fit_nll_bounded_constrained(apoint_bnd, [1.0, 5, 50, 10.0], 1.0))
print(fit_nll_bounded_constrained(apoint_bnd, [1.0, 5, 50, 7.0], 1.0))
print(fit_nll_bounded_constrained(apoint_bnd, [1.0, 5, 50, 1.0], 1.0))
print(fit_nll_bounded_constrained(apoint_bnd, [1.0, 5, 50, 0.1], 1.0))


print("diffable cls")


get_cls = infer.expected_CLs(nn_model_maker, solver_kwargs=dict(pdf_transform=True))

j_cls = []

j_cls.append(
    jax.value_and_grad(
        infer.expected_CLs(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([5.0, 50.0, 15.0], 1.0)[0]
)
j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([5.0, 50.0, 10.0], 1.0)[0]
)
j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([5.0, 50.0, 7.0], 1.0)[0]
)
j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([5.0, 50.0, 1.0], 1.0)[0]
)
j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([5.0, 50.0, 0.1], 1.0)[0]
)

j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([10.0, 5.0, 0.1], 1.0)[0]
)
j_cls.append(
    jax.value_and_grad(
        cls.cls_maker(nn_model_maker, solver_kwargs=dict(pdf_transform=True))
    )([15.0, 5.0, 0.1], 1.0)[0]
)


print("cross check cls")


def pyhf_cls(nn_params, mu):
    s, b, db = nn_params
    m = pyhf.simplemodels.hepdata_like([s], [b], [db])
    return pyhf.infer.hypotest(1.0, [b] + m.config.auxdata, m)[0]


p_cls = []

p_cls.append(pyhf_cls([5.0, 50.0, 15.0], 1.0))
p_cls.append(pyhf_cls([5.0, 50.0, 10.0], 1.0))
p_cls.append(pyhf_cls([5.0, 50.0, 7.0], 1.0))
p_cls.append(pyhf_cls([5.0, 50.0, 1.0], 1.0))
p_cls.append(pyhf_cls([5.0, 50.0, 0.1], 1.0))

p_cls.append(pyhf_cls([10.0, 5.0, 0.1], 1.0))
p_cls.append(pyhf_cls([15.0, 5.0, 0.1], 1.0))

assert np.allclose(np.asarray(j_cls), np.asarray(p_cls)), "cls values don't match pyhf"