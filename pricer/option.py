import numpy as np
import pandas as pd
import scipy as si
from scipy.stats import norm


def bs(p:str, S, K, T, sigma, r, q=0):
    '''
    p
        str, "call" or "put"
    q
        float, incorporates dividend and funding cost
    '''
    def d1():
        return (np.log(S/K) + (r - q + sigma**2 / 2) * T) / sigma * np.sqrt(T)
    d_1 = d1()
    d_2 = d_1 - sigma * np.sqrt(T)
    if p == 'C':
        return S * np.exp(-q*T) * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * np.exp(-(q)*T) * norm.cdf(-d_1)

data = pd.DataFrame(data={
    'p': 5*['C', 'P'],
    'S': 10*[50],
    'K': 2*[50, 60, 50, 50, 50],
    'T': 2*[.5,.5,1.0,.5,.5],
    'sigma': 2*[.2,.2,.2,.3,.2],
    'r': 2*[.01,.01,.01,-.01,.02],
    'q': 10*[0]
})

data['v'] = data.apply(lambda row: bs(*row), axis=1)

def Z(x, y, p) -> float:
    return p * x + np.sqrt(1 - p**2) * y
def verify(p=0.5) -> pd.DataFrame:
    sample = pd.DataFrame(
        np.random.normal(size=(200, 2)),
        columns=["X", "Y"])
    sample["Z"] = sample.apply(lambda row: Z(row['X'], row['Y'], p), axis=1)
    p_XZ = sample["Z"].corr(sample["X"])
    assert np.isclose(p_XZ, p, rtol=5e-1), "mathematically wrong"
    print(f"verified {p_XZ}~{p}")

def vega(S, K, T, sigma, r, q=0):
    '''
    Se−q(T−t)√T − tN′(d1)
    
    >>> vega(50, 50, 0.5, 0.25, 0.01)
    '''
    def d1():
        return (np.log(S/K) + (r - q + sigma**2 / 2) * T) / sigma * np.sqrt(T)
    return S * np.exp(-q*T) * norm.pdf(d1()) * np.sqrt(T)

def newton(x, y, f, f_prime, tolerance=0.00001, n_iter=1000):
    '''
    x
        can be initial guess

    >>> newton(
    >>> 0.4, 4.3569,
    >>> lambda sigma: bs('P', 50, 50, 0.5, sigma, 0.01),
    >>> lambda sigma: vega(50, 50, 0.5, sigma, 0.01)
    >>> )
    '''
    diff = f(x) - y
    m = 0
    while abs(diff) > tolerance and m < n_iter:
        x = x - diff / f_prime(x)
        m += 1
        diff = f(x) - y
    return x

# Q3

import os
PATH_ROOT = os.path.normpath("/Users/chaobintang/Documents/Course/MScFTDA/FITE7405")

from dateutil.relativedelta import relativedelta

PATH_INSTRUMENT = os.path.join(PATH_ROOT,'instruments.csv')
PATH_MARKET = os.path.join(PATH_ROOT, 'marketdata.csv')
MATURITY = (24-16) / 365
RATE_RISKFREE = .04
DIVDEND = 0.2
UNDERLYING = 510050


def get_data() -> pd.DataFrame:
    instrument = pd.read_csv(PATH_INSTRUMENT)
    chain = pd.read_csv(PATH_MARKET)

    chain = chain.set_index("Symbol").join(instrument.set_index("Symbol"))

    chain.LocalTime = chain.LocalTime.astype('datetime64[m]')
    chain.LocalTime = chain.LocalTime + pd.Timedelta(minutes=1)
    chain = chain.groupby(["Symbol", "LocalTime"]).last()
    chain['Maturity'] = MATURITY

    chain["Mid"] = (chain["Bid1"] + chain["Ask1"]) / 2
    chain.reset_index(inplace=True)
    underlying = chain[chain.Symbol==UNDERLYING]
    chain = chain[chain.Symbol!=UNDERLYING]
    chain.drop(columns=["Mid"], inplace=True)
    underlying = underlying[["LocalTime", "Mid", "Symbol"]]
    underlying = underlying.rename(columns={"Mid": "S", "Symbol": "Underlying"})
    chain = chain.set_index("LocalTime").join(underlying.set_index("LocalTime"))
    
    chain["r"] = RATE_RISKFREE
    chain["Divd"] = DIVDEND
    chain.reset_index(inplace=True)
    return chain

def imply(row, bid_or_ask:str, guess=0.2):
    return newton(
        guess, row['Bid1'] if bid_or_ask == 'B' else row['Ask1'],
        lambda sigma: bs(row['OptionType'], row['S'], row['Strike'], row['Maturity'], sigma, row['r'], q=row['Divd']),
        lambda sigma: vega(row['S'], row['Strike'], row['Maturity'], sigma, row['r'], q=row['Divd']))

data = get_data()

data['BidVol'] = data.apply(lambda row: imply(row, 'B'), axis=1)

data['AskVol'] = data.apply(lambda row: imply(row, 'A'), axis=1)

KEY = ["LocalTime", "Strike"]
puts  = data[data.OptionType=="P"][KEY + ["BidVol", "AskVol"]] \
    .rename(columns={"BidVol": "BidVolP", "AskVol": "AskVolP"})
calls = data[data.OptionType=="C"][KEY + ["BidVol", "AskVol"]] \
    .rename(columns={"BidVol": "BidVolC", "AskVol": "AskVolC"})
results = puts.set_index(KEY).join(calls.set_index(KEY)) 
results = results.sort_index(ascending=True)
results.reset_index(inplace=True)

ticks = results.LocalTime.unique()
segments = {}
for tick in ticks:
    segment = results[results.LocalTime == tick]
    segment = segment.drop(columns=["LocalTime"]).set_index("Strike")
    filename = pd.to_datetime(tick).minute
    # with open(os.path.join(PATH_ROOT, f'{filename}.csv'), 'w') as f:
    #     segment.to_csv(f)
    segments[filename] = segment.reset_index()

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

for (t, segment) in segments.items():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    plt.title(f't={t}')
    ax.set_xlabel('Strike')
    ax.set_ylabel('IV')
    plot_p = plt.plot(segment['Strike'], segment['BidVolP'], 'r', label="P")
    plot_c = plt.plot(segment['Strike'], segment['BidVolC'], 'b', label="C")
    ax.legend(loc="best")
    # show the plot
    plt.show()

    analysis = data[["LocalTime", "Symbol", "Bid1", "Ask1", "OptionType", "Strike", "Maturity", "S", "r"]]
calls = analysis[analysis.OptionType=="C"]
calls = calls.drop(columns=["OptionType"])
calls.rename(inplace=True, columns={
    "Symbol": "Call",
    "Bid1": "BidC",
    "Ask1": "AskC"
    })
puts = analysis[analysis.OptionType=="P"].drop(columns=["OptionType", "Maturity", "S", "r"])
puts.rename(inplace=True, columns={
    "Symbol": "Put",
    "Bid1": "BidP",
    "Ask1": "AskP"
    })
KEY = ["LocalTime", "Strike"]
A = calls.set_index(KEY).join(puts.set_index(KEY)).reset_index()
A = A.sort_values("Strike")


A['BidCallParity'] = A["BidC"] + A["Strike"]*np.exp(-A["r"]*A["Maturity"])
A['BidPutParity']  = A["BidP"] + A["S"]
A['AskCallParity'] = A["AskC"] + A["Strike"]*np.exp(-A["r"]*A["Maturity"])
A['AskPutParity']  = A["AskP"] + A["S"]
A['SprdLongCallShortPut'] = A['AskCallParity'] - A['BidPutParity']
A['SprdShortCallLongPut'] = A['BidCallParity'] - A['AskPutParity']

for tick in A.LocalTime.unique():
    quotes = A[A.LocalTime == tick]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.axhline(y=0.0003, linewidth=1, color='y', linestyle="--")
    plt.plot(quotes["Strike"], quotes["SprdLongCallShortPut"], 'r.', label="Spread LongC-ShortP")
    plt.plot(quotes["Strike"], quotes["SprdShortCallLongPut"], 'b.', label="Spread ShortC-LongP")
    ax.legend(loc="best")
    # show the plot
    plt.show()
