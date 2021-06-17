import neos
from neos.examples import neos_pyhf_example


def test_version():
    assert neos.__version__


def test_workflow():
    assert neos_pyhf_example()
