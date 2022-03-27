import numpy as np
import pytest

from pricer.option import models


@pytest.mark.option
def test_price_eu_option():
    p = "C"; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.64, c_option.price(), atol=1e-2), "obtained incorrect c_opt px from bs"
    p = 'P' 
    p_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.39, p_option.price(), atol=1e-2), "obtained incorrect p_opt px from bs"

@pytest.mark.option
def test_price_asian_option():
    p = "C"; n = 5; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.AsianGeometric(p, n, S, K, T, sigma, r)
    assert np.isclose(2.34, c_option.price(), atol=1e-2), "obtained incorrect c_opt px from bs"

@pytest.mark.option
def test_imply_vol():
    p = "C"; S = 50; K = 50; T = 0.5; sigma = None; r = 0.01
    c_option = models.European(p, S, K, T, sigma, r)
    iv = c_option.imply_sigma(px=3.64)
    assert np.isclose(iv, 0.25, atol=1e-2), f"implied vol incorrect, {iv} vs. {0.25}"
    