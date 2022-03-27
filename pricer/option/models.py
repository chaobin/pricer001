from typing import Sequence

import numpy as np
import pandas as pd
import scipy as si
from scipy.stats import norm

from ..optimizer import root


__all__ = ['European']


class Model(object): pass

class BlackScholes(Model):

    @classmethod
    def v_price(cls, p:str, S, K, T, sigma, r, q=0):
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
            return S * np.exp(-q*T) * norm.cdf(d_1) - \
                K * np.exp(-r * T) * norm.cdf(d_2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d_2) - \
                S * np.exp(-(q)*T) * norm.cdf(-d_1)

    @classmethod
    def v_delta_sigma(cls, S, K, T, sigma, r, q=0):
        
        '''
        Se−q(T−t)√T − tN′(d1)
        
        >>> vega(50, 50, 0.5, 0.25, 0.01)
        '''
        def d1():
            return (np.log(S/K) + (r - q + sigma**2 / 2) * T) / sigma * np.sqrt(T)
        return S * np.exp(-q*T) * norm.pdf(d1()) * np.sqrt(T)

class European(BlackScholes):

    def __init__(self, p:str, S, K, T, sigma, r, q=0):
        '''
        p
            str, "call" or "put"
        q
            float, incorporates dividend and funding cost
        '''
        self.p = p # buy or sell
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q

    def price(self, p=None, S=None, K=None, T=None, sigma=None, r=None, q=None):
        return self.v_price(
            p or self.p, S or self.S, K or self.K,
            T or self.T, sigma or self.sigma, r or self.r, q or     self.q)

    def delta_sigma(self, S=None, K=None, T=None, sigma=None, r=None, q=None):
        return self.v_delta_sigma(
            S or self.S, K or self.K, T or self.T, sigma or self.sigma,
            r or self.r, q or self.q)
    
    def imply_sigma(self, px, guess=0.2):
        return root.newton(
            guess, px,
            lambda sigma: self.price(sigma=sigma),
            lambda sigma: self.delta_sigma(sigma=sigma))

class AsianGeometric(BlackScholes):

    def __init__(self, p:str, n, S, K, T, sigma, r):
        '''
        n
            number of observations in the mean basket
        p
            str, "call" or "put"
        '''
        self.p = p
        self.n = n
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
    
    def price(self, n=None, S=None, K=None, T=None, sigma=None, r=None):
        n = n or self.n
        sigma = sigma or self.sigma
        r = r or self.r
        T = T or self.T
        
        sigma_m = sigma * np.sqrt(( (n + 1) * (2*n + 1) )/ (6*n**2))
        r_m = (r - sigma**2 / 2) * ((n + 1) / (2 * n)) + sigma_m**2 / 2
        
        return self.v_price(
            self.p, S or self.S, K or self.K,
            T or self.T, sigma_m, r_m)

class AsianArithemetic(BlackScholes):

    def __init__(self, p:str, n, S, K, T, sigma, r):
        '''
        n
            number of observations in the mean basket
        p
            str, "call" or "put"
        '''
        self.p = p
        self.n = n
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r

class BasketGeometric(BlackScholes):

    def __init__(self, p:str, S:Sequence, K, T, sigma:Sequence, r):
        '''
        S
            vector of prices of a basket of assets
        sigma
            vector of std of a basket of assets
        p
            str, "call" or "put"
        '''
        self.p = p
        self.n = n
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r

    def price(self, S:Sequence=None, K=None, T=None, sigma=None, r=None):
        n = n or self.n
        sigma = sigma or self.sigma
        r = r or self.r
        T = T or self.T
        
        sigma_m =  np.sqrt() / n
        r_m = (r - sigma**2 / 2) * ((n + 1) / (2 * n)) + sigma_m**2 / 2
        
        return self.v_price(
            self.p, S or self.S, K or self.K,
            T or self.T, sigma_m, r_m)
