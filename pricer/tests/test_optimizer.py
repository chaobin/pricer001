import numpy as np
import pytest

from pricer.optimizer import root


@pytest.mark.optimizer
def test_newton():
    four = root.newton(
        2, 16,
        lambda x: x**2,
        lambda x: 2*x)
    assert np.isclose(four , 4), "newton failed"
