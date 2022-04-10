import os
import numpy as np
import pandas as pd
import pytest
import copy

from pricer.option import models


HOME = os.environ['HOME']


@pytest.mark.report
def test_price_asian_geometric():
    p = "C"; n = 5; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    S = 100; T = 3; r = 0.05; M = int(1e4)
    tests = [
        ['P', 50,  S, 100, T, 0.3, r],
        ['P', 100, S, 100, T, 0.3, r],
        ['P', 50,  S, 100, T, 0.4, r],

        ['C', 50,  S, 100, T, 0.3, r],
        ['C', 100, S, 100, T, 0.3, r],
        ['C', 50,  S, 100, T, 0.4, r]
    ]
    tests_control_variate = copy.deepcopy(tests)
    for test in tests:
        option = models.Asian(*test)
        px, l, u = option.price(simulations=M)
        test.extend([round(x, 3) for x in [0, l, u, px]])
    for test in tests_control_variate:
        option = models.Asian(*test)
        px, l, u = option.price(simulations=M, control_variate=True)
        test.extend([round(x, 3) for x in [1, l, u, px]])

    tests.extend(tests_control_variate)

    COLUMNS = ['type', 'n', 'S', 'K', 'T', 'sigma', 'r', 'cv', 'lower', 'upper', 'px']
    report = pd.DataFrame(tests, columns=COLUMNS)
    filename = os.path.join(HOME, 'report_asian.csv')
    with open(filename, 'w') as out:
        report.to_csv(out)

@pytest.mark.report
def test_price_basket_mean_option():

    p = "C"; n = 5; S = 50; K = 50; T = 0.5; sigma = 0.25; r = 0.01
    S = 100; T = 3; r = 0.05; M = int(1e4)
    tests = [
        ['P', 100, 100, 100, T, 0.3, 0.3, 0.5, r],
        ['P', 100, 100, 100, T, 0.3, 0.3, 0.9, r],
        ['P', 100, 100, 100, T, 0.1, 0.3, 0.5, r],
        ['P', 100, 100, 80,  T, 0.3, 0.3, 0.5, r],
        ['P', 100, 100, 120, T, 0.3, 0.3, 0.5, r],
        ['P', 100, 100, 100, T, 0.5, 0.3, 0.5, r],

        ['C', 100, 100, 100, T, 0.3, 0.3, 0.5, r],
        ['C', 100, 100, 100, T, 0.3, 0.3, 0.9, r],
        ['C', 100, 100, 100, T, 0.1, 0.3, 0.5, r],
        ['C', 100, 100, 80,  T, 0.3, 0.3, 0.5, r],
        ['C', 100, 100, 120, T, 0.3, 0.3, 0.5, r],
        ['C', 100, 100, 100, T, 0.5, 0.3, 0.5, r]
    ]
    tests_control_variate = copy.deepcopy(tests)
    for test in tests:
        option = models.ArithmeticBasketWithTwoAssets(*test)
        px, l, u = option.price(simulations=M)
        test.extend([round(x, 3) for x in [0, l, u, px]])
    for test in tests_control_variate:
        option = models.ArithmeticBasketWithTwoAssets(*test)
        px, l, u = option.price(simulations=M, control_variate=True)
        test.extend([round(x, 3) for x in [1, l, u, px]])

    tests.extend(tests_control_variate)

    COLUMNS = [
        'type', 'S1', 'S2', 'K', 'T', 'sigma1', 'sigma2', 'p',
        'r', 'cv', 'lower', 'upper', 'px']
    report = pd.DataFrame(tests, columns=COLUMNS)
    filename = os.path.join(HOME, 'report_basket.csv')
    with open(filename, 'w') as out:
        report.to_csv(out)
