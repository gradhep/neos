# neos
> nice end-to-end optimized statistics ;)


[![DOI](https://zenodo.org/badge/235776682.svg)](https://zenodo.org/badge/latestdoi/235776682) ![CI](https://github.com/pyhf/neos/workflows/CI/badge.svg) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyhf/neos/master?filepath=demo_training.ipynb)

![](nbs/assets/neos_logo.png)

![](nbs/assets/training.gif)

## Install

Just run

```
python -m pip install neos
```

## Contributing

**Please read** [`CONTRIBUTING.md`](https://github.com/pyhf/neos/blob/master/CONTRIBUTING.md) **before making a PR**, as this project is maintained using [`nbdev`](https://github.com/fastai/nbdev), which operates completely using Jupyter notebooks. One should make their changes in the corresponding notebooks in the [`nbs`](nbs) folder (including `README` changes -- see `nbs/index.ipynb`), and not in the library code, which is automatically generated.

## How to use (and reproduce the results from the cool animation)

```python
import jax
import neos.makers as makers
import neos.cls as cls
import numpy as np
import jax.experimental.stax as stax
import jax.experimental.optimizers as optimizers
import jax.random
import time
```

### Initialise network using `jax.experimental.stax`

```python
init_random_params, predict = stax.serial(
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(2),
    stax.Softmax,
)
```

### Initialse tools from `neos`:

The way we initialise in `neos` is to define functions that make a statistical model from histograms, which in turn are themselves made from a predictive model, such as a neural network. Here's some detail on the unctions used below:

- `hists_from_nn_three_blobs(predict)` uses the nn decision function `predict` defined in the cell above to form histograms from signal and background data, all drawn from multivariate normal distributions with different means. Two background distributions are sampled from, which is meant to mimic the situation in particle physics where one has a 'nominal' prediction for a nuisance parameter and then an alternate value (e.g. from varying up/down by one standard deviation), which then modifies the background pdf. Here, we take that effect to be a shift of the mean of the distribution. The value for the background histogram is then the mean of the resulting counts of the two modes, and the uncertainty can be quantified through the count standard deviation.
- `nn_hepdata_like(hmaker)` uses `hmaker` to construct histograms, then feeds them into the `neos.models.hepdata_like` function that constructs a pyhf-like model. This can then be used to call things like `logpdf` and `expected_data` downstream.
- `cls_maker` takes a model-making function as it's primary argument, which is fed into functions from `neos.fit` that minimise the `logpdf` of the model in both a constrained (fixed parameter of interest) and a global way. Moreover, these fits are wrapped in a function that allows us to calculate gradients through the fits using *fixed-point differentiation*. This allows for the calculation of both the profile likelihood and its gradient, and then the same for cls :)

All three of these methods return functions. in particular, `cls_maker` returns a function that differentiably calculates cls values, which is our desired objective to minimise.

```python
hmaker = makers.hists_from_nn_three_blobs(predict)
nnm = makers.nn_hepdata_like(hmaker)
loss = cls.cls_maker(nnm, solver_kwargs=dict(pdf_transform=True))
```

```python
_, network = init_random_params(jax.random.PRNGKey(2), (-1, 2))
```

    /home/phinate/envs/neos/lib/python3.7/site-packages/jax-0.1.59-py3.7.egg/jax/lib/xla_bridge.py:122: UserWarning: No GPU/TPU found, falling back to CPU.


### Define training loop!

```python
opt_init, opt_update, opt_params = optimizers.adam(1e-3)

def update_and_value(i, opt_state, mu):
    net = opt_params(opt_state)
    value, grad = jax.value_and_grad(loss)(net, mu)
    return opt_update(i, grad, opt_state), value, net

def train_network(N):
    cls_vals = []
    _, network = init_random_params(jax.random.PRNGKey(1), (-1, 2))
    state = opt_init(network)
    losses = []
    
    for i in range(N):
        start_time = time.time()
        state, value, network = update_and_value(i,state,1.0)
        epoch_time = time.time() - start_time
        losses.append(value)
        metrics = {"loss": losses}
        yield network, metrics, epoch_time
```

### Let's run it!!

```python
maxN = 20 # make me bigger for better results!

# Training
for i, (network, metrics, epoch_time) in enumerate(train_network(maxN)):
    print(f"epoch {i}:", f'CLs = {metrics["loss"][-1]}, took {epoch_time}s') 
```

    epoch 0: CLs = 0.06680655092981347, took 5.355436325073242s
    epoch 1: CLs = 0.4853891149072429, took 1.5733795166015625s
    epoch 2: CLs = 0.3379355596004474, took 1.5171947479248047s
    epoch 3: CLs = 0.1821927415636535, took 1.5081253051757812s
    epoch 4: CLs = 0.09119136931683047, took 1.5193650722503662s
    epoch 5: CLs = 0.04530559823843272, took 1.5008423328399658s
    epoch 6: CLs = 0.022572851867672883, took 1.499192476272583s
    epoch 7: CLs = 0.013835564056077887, took 1.5843737125396729s
    epoch 8: CLs = 0.01322058601444187, took 1.520324468612671s
    epoch 9: CLs = 0.013407422454837725, took 1.5050244331359863s
    epoch 10: CLs = 0.011836452218993765, took 1.509469985961914s
    epoch 11: CLs = 0.00948507486266359, took 1.5089364051818848s
    epoch 12: CLs = 0.007350505632595539, took 1.5106918811798096s
    epoch 13: CLs = 0.005755974539907838, took 1.5267891883850098s
    epoch 14: CLs = 0.0046464301411786035, took 1.5851080417633057s
    epoch 15: CLs = 0.0038756402968267434, took 1.8452086448669434s
    epoch 16: CLs = 0.003323640670405803, took 1.9116990566253662s
    epoch 17: CLs = 0.0029133909840759475, took 1.7648999691009521s
    epoch 18: CLs = 0.002596946123608612, took 1.6314191818237305s
    epoch 19: CLs = 0.0023454051342963744, took 1.5911424160003662s


And there we go!!

If you want to reproduce the full animation, a version of this code with plotting helpers can be found in [`demo_training.ipynb`](https://github.com/pyhf/neos/blob/master/demo_training.ipynb)! :D

## Thanks

A big thanks to the teams behind [`jax`](https://github.com/google/jax/), [`fax`](https://github.com/gehring/fax), and [`pyhf`](https://github.com/scikit-hep/pyhf) for their software and support.
