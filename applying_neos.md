# The current state of `neos` (written by Nathan)

At the moment, **there is no working version of the full `neos` pipeline with `pyhf`**. The main reason is that due to a number of errors and bugs when trying to make `pyhf` model building properly use `jax`, it was rabbit hole after rabbit hole of trying to fix things, and it's become clear that there's too much to change to hack it together in a way that makes sense.


## i want to use `neos` with `pyhf` now

If you want to do something *right now*, there's a pretty simple but time-consuming solution to this: write a new likelihood function in `jax` that can be used in the `neos` pipeline. This is a non-trivial task, and will require a significant amount of time to complete.

If you're doing unbinned fits, it's likely that this was always the case (there's no HEP-driven unbinned likelihoods in JAX yet, unless you can somehow make [`zfit`](https://github.com/zfit/zfit) work, but this would be another `pyhf`-scale operation). I'd recommend you make your model class a JAX PyTree by inheriting from `equinox.Module` -- this will make it work out-the-box with `relaxed` (see below for more on this).

If you need some inspiration for a HistFactory-based solution, there are a couple places I can point you to:

- the [`dummy_pyhf` file in `relaxed`](https://github.com/gradhep/relaxed/blob/main/tests/dummy_pyhf.py) has a working example of a HistFactory-based likelihood function that can be used roughly interchangeably with the `pyhf` one for simple likelihoods with one (bin-uncorrelated) background systematic. It's not perfect, but it could serve as a starting point to try testing your pipelines.
- [`dilax`](https://github.com/pfackeldey/dilax) is a slighly more mature version of this, but does not use the same naming conventions as `pyhf`. It's a nice first attempt at what could be the right way to go about this in future.

## long-term plans
I'm not working very actively in the field right now, but I've tried my best to indicate the direction I think things should go in [this discussion on the `pyhf` repo](https://github.com/scikit-hep/pyhf/discussions/2196). The key ingredient is PyTrees (read the issue for more details). If you're interested in working on this, I'd be happy to help out, but I don't have the time (or the HistFactory expertise) to do it fully myself. I think it's a really important thing to do, though -- probably essential if this is going to be truly used in HEP!

I've just released [`relaxed` v0.3.0](https://github.com/gradhep/relaxed), which has been tested with dummy PyTree models to work. It's designed for a `pyhf` that doesn't exist yet, and may never exist at all. But it will work with any PyTree model, so if you can write a PyTree model, you can use `relaxed` to do your fits, then backpropagate through them.

## reaching out

If you're interested in working on this, please reach out to me through [Mattermost](https://mattermost.web.cern.ch/signup_user_complete/?id=zf7w5rb1miy85xsfjqm68q9hwr&md=link&sbr=su), or by email.

