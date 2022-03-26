import numpy as np
import pandas as pd
import scipy as si
from scipy.stats import norm

from ..optimizer import root


__all__ = ['European']


class Model(object): pass

class European(Model):

    @staticmethod
    def v_price(p:str, S, K, T, sigma, r, q=0):
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

    @staticmethod
    def v_delta_sigma(S, K, T, sigma, r, q=0):
        '''
        Se−q(T−t)√T − tN′(d1)
        
        >>> vega(50, 50, 0.5, 0.25, 0.01)
        '''
        def d1():
            return (np.log(S/K) + (r - q + sigma**2 / 2) * T) / sigma * np.sqrt(T)
        return S * np.exp(-q*T) * norm.pdf(d1()) * np.sqrt(T)

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
        return self.v_price(p or self.p, S or self.S, K or self.K,
                            T or self.T, sigma or self.sigma, r or self.r, q or self.q)

    def delta_sigma(self):
        return self.v_delta_sigma(self.S, self.K, self.T, self.sigma, self.r, self.q)
    
    def imply_sigma(self, px, guess=0.2):
        return root.newton(
            guess, px,
            lambda sigma: self.price(sigma=sigma),
            lambda sigma: self.delta_sigma())

class Asian(Model):

    def price():
        pass
