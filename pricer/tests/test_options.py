import numpy as np
import pytest

from pricer.option import models


@pytest.mark.option
def test_price_eu_option():
    # Black-Scholes
    p = "C"; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.64, c_option.price(), atol=1e-2), "obtained incorrect c_opt px from bs"
    p = 'P' 
    p_option = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.39, p_option.price(), atol=1e-2), "obtained incorrect p_opt px from bs"

@pytest.mark.option
def test_price_eu_option_with_binomial_tree():
    # Binomial
    p = "C"; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option_bin = models.BinomialTree(p, S, K, T, sigma, r)
    assert np.isclose(3.64, c_option_bin.price(100), atol=1e-2), "obtained incorrect c_opt px from bs"
    p = 'P' 
    p_option_bin = models.European(p, S, K, T, sigma, r)
    assert np.isclose(3.39, p_option_bin.price(100), atol=1e-2), "obtained incorrect p_opt px from bs"

@pytest.mark.option
def test_imply_vol():
    p = "C"; S = 50; K = 50; T = 0.5; sigma = None; r = 0.01
    c_option = models.European(p, S, K, T, sigma, r)
    iv = c_option.imply_sigma(px=3.64)
    assert np.isclose(iv, 0.25, atol=1e-2), f"implied vol incorrect, {iv} vs. {0.25}"

@pytest.mark.option
def test_price_asian_geometric():
    p = "C"; n = 5; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.AsianGeometric(p, n, S, K, T, sigma, r)
    assert np.isclose(2.34, c_option.price(), atol=1e-2), "obtained incorrect c_opt px from bs"

@pytest.mark.option
def test_price_asian_arithmetic():
    p = "C"; n = 5; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    c_option = models.Asian(p, n, S, K, T, sigma, r)
    px, lower, upper = c_option.price(simulations=10000, control_variate=True)
    print(f"95% CV: {lower} < {px} < {upper}")
    assert np.isclose(2.34, px, atol=1e-2), "obtained incorrect c_opt px from bs"

@pytest.mark.option
def test_price_basket_gmean_option():
    S1, S2 = 50, 50
    sigma1, sigma2 = 0.25, 0.25
    corr = 0.1
    p = "C"; n = 5; K = 50; T = 0.5; r = 0.01
    c_option = models.GeometricBasketWithTwoAssets(p, S1, S2, K, T, sigma1, sigma2, corr, r)
    assert np.isclose(2.34, c_option.price(), atol=1e-2), "obtained incorrect c_opt px from bs"

@pytest.mark.option
def test_price_basket_mean_option():
    S1, S2 = 50, 50
    sigma1, sigma2 = 0.25, 0.25
    corr = 0.1
    p = "C"; n = 5; K = 50; T = 0.5; r = 0.01
    c_option = models.ArithmeticBasketWithTwoAssets(p, S1, S2, K, T, sigma1, sigma2, corr, r)
    px, lower, upper = c_option.price(simulations=10000, control_variate=True)
    assert np.isclose(2.34, px, atol=1e-2), "obtained incorrect c_opt px from bs"
