__all__ = ["global_fit", "constrained_fit"]

import jax
import jax.numpy as jnp

from fax.implicit import twophase
import jax.experimental.optimizers as optimizers

from .transforms import to_bounded_vec, to_inf_vec, to_bounded, to_inf


@jax.jit
def global_fit(
    model_constructor,
    pdf_transform=False,
    default_rtol=1e-10,
    default_atol=1e-10,
    default_max_iter=int(1e7),
    learning_rate=1e-6,
):
    """Wraps a series of functions that perform maximum likelihood fitting in
    the `two_phase_solver` method found in the `fax` python module. This allows
    for the calculation of gradients of the best-fit parameters with respect to
    upstream parameters that control the underlying model, i.e. the event
    yields (which are then parameterized by weights or similar).

    Args:
            model_constructor: Function that takes in the parameters of the observable,
            and returns a model object (and background-only parameters)
    Returns:
            global_fitter: Callable function that performs global fits.
            Differentiable :)
    """

    adam_init, adam_update, adam_get_params = optimizers.adam(learning_rate)

    def make_model(model_pars):
        m, bonlypars = model_constructor(model_pars)

        bounds = m.config.suggested_bounds()

        exp_bonly_data = m.expected_data(bonlypars, include_auxdata=True)

        def expected_logpdf(pars):  # maps pars to bounded space if pdf_transform = True

            return (
                m.logpdf(to_bounded_vec(pars, bounds), exp_bonly_data)
                if pdf_transform
                else m.logpdf(pars, exp_bonly_data)
            )

        def global_fit_objective(pars):  # NLL
            return -expected_logpdf(pars)[0]

        return global_fit_objective

    def global_bestfit_minimized(hyper_param):
        nll = make_model(hyper_param)

        def bestfit_via_grad_descent(i, param):  # gradient descent
            g = jax.grad(nll)(param)
            # param = param - g * learning_rate
            param = adam_get_params(adam_update(i, g, adam_init(param)))
            return param

        return bestfit_via_grad_descent

    global_solve = twophase.two_phase_solver(
        param_func=global_bestfit_minimized,
        default_rtol=default_rtol,
        default_atol=default_atol,
        default_max_iter=default_max_iter,
    )

    def global_fitter(init, hyper_pars):
        solve = global_solve(init, hyper_pars)
        return solve.value

    return global_fitter


@jax.jit
def constrained_fit(
    model_constructor,
    pdf_transform=False,
    default_rtol=1e-10,
    default_atol=1e-10,
    default_max_iter=int(1e7),
    learning_rate=1e-6,
):
    """Wraps a series of functions that perform maximum likelihood fitting in
    the `two_phase_solver` method found in the `fax` python module. This allows
    for the calculation of gradients of the best-fit parameters with respect to
    upstream parameters that control the underlying model, i.e. the event
    yields (which are then parameterized by weights or similar).

    Args:
            model_constructor: Function that takes in the parameters of the observable,
            and returns a model object (and background-only parameters)
    Returns:
            constrained_fitter: Callable function that performs constrained fits.
            Differentiable :)
    """

    adam_init, adam_update, adam_get_params = optimizers.adam(learning_rate)

    def make_model(hyper_pars):

        model_pars, constrained_mu = hyper_pars
        m, bonlypars = model_constructor(model_pars)

        bounds = m.config.suggested_bounds()
        constrained_mu = (
            to_inf(constrained_mu, bounds[0]) if pdf_transform else constrained_mu
        )

        exp_bonly_data = m.expected_data(bonlypars, include_auxdata=True)

        def expected_logpdf(pars):  # maps pars to bounded space if pdf_transform = True

            return (
                m.logpdf(to_bounded_vec(pars, bounds), exp_bonly_data)
                if pdf_transform
                else m.logpdf(pars, exp_bonly_data)
            )

        def constrained_fit_objective(nuis_par):  # NLL
            pars = jnp.concatenate([jnp.asarray([constrained_mu]), nuis_par])
            return -expected_logpdf(pars)[0]

        return constrained_mu, constrained_fit_objective

    def constrained_bestfit_minimized(hyper_pars):
        mu, cnll = make_model(hyper_pars)

        def bestfit_via_grad_descent(i, param):  # gradient descent
            _, np = param[0], param[1:]
            g = jax.grad(cnll)(np)
            np = adam_get_params(adam_update(i, g, adam_init(np)))
            param = jnp.concatenate([jnp.asarray([mu]), np])
            return param

        return bestfit_via_grad_descent

    constrained_solver = twophase.two_phase_solver(
        param_func=constrained_bestfit_minimized,
        default_rtol=default_rtol,
        default_atol=default_atol,
        default_max_iter=default_max_iter,
    )

    def constrained_fitter(init, hyper_pars):
        solve = constrained_solver(init, hyper_pars)
        return solve.value

    return constrained_fitter
