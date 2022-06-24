from neos._version import version as __version__

__all__ = (
    "__version__",
    "hists_from_nn",
    "loss_from_model",
    "losses",
)

from neos import losses
from neos.top_level import hists_from_nn, loss_from_model
