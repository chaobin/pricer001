import numpy as np
import pytest

from pricer.option import models


@pytest.mark.option
def test_price_eu_option():
    p = "C"; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.64, c_option.price(), atol=1e2), "obtained incorrect c_opt px from bs"
    p = 'P' 
    p_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.39, p_option.price(), atol=1e2), "obtained incorrect p_opt px from bs"
