# neos
> ~neural~ nice end-to-end optimized statistics

[![Actions Status][actions-badge]][actions-link]
[![Coverage Status](https://codecov.io/gh/gradhep/neos/branch/main/graph/badge.svg?token=NHT2SRRJLV)](https://codecov.io/gh/gradhep/neos)
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]





[actions-badge]:            https://github.com/gradhep/neos/workflows/CI/badge.svg
[actions-link]:             https://github.com/gradhep/neos/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/neos
[conda-link]:               https://github.com/conda-forge/neos-feedstock
[codecov-badge]:            https://app.codecov.io/gh/gradhep/neos/branch/main/graph/badge.svg
[codecov-link]:             https://app.codecov.io/gh/gradhep/neos
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/gradhep/neos/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/gradhep/neos/community.svg
[gitter-link]:              https://gitter.im/https://github.com/gradhep/neos/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/neos/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/neos
[pypi-version]:             https://badge.fury.io/py/neos.svg
[rtd-badge]:                https://readthedocs.org/projects/neos/badge/?version=latest
[rtd-link]:                 https://neos.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg


[![DOI](https://zenodo.org/badge/235776682.svg)](https://zenodo.org/badge/latestdoi/235776682)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gradhep/neos/main?filepath=demo_kde_pyhf.ipynb)

<img src="nbs/assets/neos_logo.png" alt="neos logo" width="250">

![](nbs/assets/pyhf_3.gif)

## About

Leverages the shoulders of giants ([`jax`](https://github.com/google/jax/), [`fax`](https://github.com/gehring/fax), and [`pyhf`](https://github.com/scikit-hep/pyhf)) to differentiate through a high-energy physics analysis workflow, including the construction of the frequentist profile likelihood.

Documentation can be found at [http://gradhep.github.io/neos](http://gradhep.github.io/neos)!

To see examples of `neos` in action, look for the notebooks in the nbs folder with the `demo_` prefix.

If you're more of a video person, see [this talk](https://www.youtube.com/watch?v=3P4ZDkbleKs) given by [Nathan](https://github.com/phinate) on the broader topic of differentiable programming in high-energy physics, which also covers `neos`.

## Install

Just run

```
python -m pip install neos
```

## Contributing

**Please read** [`CONTRIBUTING.md`](https://github.com/pyhf/neos/blob/master/CONTRIBUTING.md) **before making a PR**, as this project is maintained using [`nbdev`](https://github.com/fastai/nbdev), which operates completely using Jupyter notebooks. One should make their changes in the corresponding notebooks in the [`nbs`](nbs) folder (including `README` changes -- see `nbs/index.ipynb`), and not in the library code, which is automatically generated.

## Example usage -- train a neural network to optimize an expected p-value

```python
# bunch of imports:
import time

import jax
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
import jax.random
from jax.random import PRNGKey
import numpy as np
from functools import partial

import pyhf
pyhf.set_backend('jax')
pyhf.default_backend = pyhf.tensor.jax_backend(precision='64b')

from neos import data, infer, makers

rng = PRNGKey(22)
```

Let's start by making a basic neural network for regression with the `stax` module found in `jax`:

```python
init_random_params, predict = stax.serial(
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1),
    stax.Sigmoid,
)
```

Now, let's compose a workflow that can make use of this network in a typical high-energy physics statistical analysis.

Our workflow is as follows:
- From a set of normal distributions with different means, we'll generate four blobs of `(x,y)` points, corresponding to a signal process, a nominal background process, and two variations of the background from varying the background distribution's mean up and down.
- We'll then feed these points into the previously defined neural network for each blob, and construct a histogram of the output using kernel density estimation. The difference between the two background variations is used as a systematic uncertainty on the nominal background.
- We can then leverage the magic of `pyhf` to construct an [event-counting statistical model](https://scikit-hep.org/pyhf/intro.html#histfactory) from the histogram yields.
- Finally, we calculate the p-value of a test between the nominal signal and background-only hypotheses. This uses a [profile likelihood-based test statistic](https://arxiv.org/abs/1007.1727).

In code, `neos` can specify this workflow through function composition:

```python
# data generator
data_gen = data.generate_blobs(rng,blobs=4)
# histogram maker
hist_maker = makers.hists_from_nn(data_gen, predict, method='kde')
# statistical model maker
model_maker = makers.histosys_model_from_hists(hist_maker)
# CLs value getter
get_cls = infer.expected_CLs(model_maker, solver_kwargs=dict(pdf_transform=True))
```

A peculiarity to note is that each of the functions used in this step actually return functions themselves. The reason we do this is that we need a skeleton of the workflow with all of the fixed parameters to be in place before calculating the loss function, as the only 'moving parts' here are the weights of the neural network.

`neos` also lets you specify hyperparameters for the histograms (e.g. binning, bandwidth) to allow these to be tuned throughout the learning process if neccesary (we don't do that here).

```python
bins = np.linspace(0,1,4) # three bins in the range [0,1]
bandwidth = 0.27 # smoothing parameter
get_loss = partial(get_cls, hyperparams=dict(bins=bins,bandwidth=bandwidth))
```

Our loss currently returns a list of metrics -- let's just index into the first one (the CLs value).

```python
def loss(params, test_mu):
    return get_loss(params, test_mu)[0]
```

Now we just need to initialize the network's weights, and construct a training loop & optimizer:

```python
# init weights
_, network = init_random_params(jax.random.PRNGKey(2), (-1, 2))

# init optimizer
opt_init, opt_update, opt_params = optimizers.adam(1e-3)

# define train loop
def train_network(N):
    cls_vals = []
    _, network = init_random_params(jax.random.PRNGKey(1), (-1, 2))
    state = opt_init(network)
    losses = []

    # parameter update function
    def update_and_value(i, opt_state, mu):
        net = opt_params(opt_state)
        value, grad = jax.value_and_grad(loss)(net, mu)
        return opt_update(i, grad, state), value, net

    for i in range(N):
        start_time = time.time()
        state, value, network = update_and_value(i, state, 1.0)
        epoch_time = time.time() - start_time
        losses.append(value)
        metrics = {"loss": losses}
        yield network, metrics, epoch_time
```

It's time to train!

```python
maxN = 10  # make me bigger for better results (*nearly* true ;])

for i, (network, metrics, epoch_time) in enumerate(train_network(maxN)):
    print(f"epoch {i}:", f'CLs = {metrics["loss"][-1]:.5f}, took {epoch_time:.4f}s')
```

    epoch 0: CLs = 0.06885, took 13.4896s
    epoch 1: CLs = 0.03580, took 1.9772s
    epoch 2: CLs = 0.01728, took 1.9912s
    epoch 3: CLs = 0.00934, took 1.9947s
    epoch 4: CLs = 0.00561, took 1.9548s
    epoch 5: CLs = 0.00378, took 1.9761s
    epoch 6: CLs = 0.00280, took 1.9500s
    epoch 7: CLs = 0.00224, took 1.9844s
    epoch 8: CLs = 0.00190, took 1.9913s
    epoch 9: CLs = 0.00168, took 1.9928s


And there we go!

You'll notice the first epoch seems to have a much larger training time. This is because `jax` is being used to just-in-time compile some of the code, which is an overhead that only needs to happen once.

If you want to reproduce the full animation from the top of this README, a version of this code with plotting helpers can be found in [`demo_kde_pyhf.ipynb`](https://github.com/pyhf/neos/blob/master/demo_kde_pyhf.ipynb)! :D

## Thanks

A big thanks to the teams behind [`jax`](https://github.com/google/jax/), [`fax`](https://github.com/gehring/fax), and [`pyhf`](https://github.com/scikit-hep/pyhf) for their software and support.
