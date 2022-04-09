import math
from typing import Sequence
from itertools import permutations

import numpy as np
import pandas as pd
import scipy as si
from scipy.stats import norm, gmean

from ..optimizer import root


__all__ = [
    'European',
    'Asian',
    'BinomialTree',
    'AsianGeometric',
    'GeometricBasketWithTwoAssets',
    'ArithmeticBasketWithTwoAssets']


class Model(object):

    def df(self):
        return np.exp(-self.r * self.T)
    
    def brownian(self, r, T, sigma, z):
        return np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*z)

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
        
        sigma_m = sigma * np.sqrt(( (n + 1) * (2*n + 1) )/ (6*n**2))
        r_m = (r - sigma**2 / 2) * ((n + 1) / (2 * n)) + sigma_m**2 / 2
        T = T or self.T
        
        df = np.exp(-r * T)
        return self.v_price(
            self.p, S or self.S, K or self.K,
            T or self.T, sigma_m, r_m)

class MonteCarlo(Model):

    def reduce_variance(self, X, Y):
        XY = X * Y
        covXY = np.mean(XY) - (np.mean(X) * np.mean(Y))
        theta = covXY / np.var(Y)

        target = X + theta*(self.control_variate() - Y)
        return target
        
    def confidence(self, result):
        M = len(result)
        mean = np.mean(result)
        std = np.std(result)
        
        upper = mean + 1.96 * std/np.sqrt(M)
        lower = mean - 1.96 * std/np.sqrt(M)
        return mean, lower, upper

class Asian(MonteCarlo):

    def __init__(self, p:str, n, S, K, T, sigma, r, M:int=1000, type_="arithmetic"):
        self.p = p
        self.n = n
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.M = M
        self.type_ = type_

    # override
    def control_variate(self):
        geo = AsianGeometric(self.p, self.n, self.S, self.K, self.K, self.sigma, self.r)
        return geo.price()
    
    def interval(self):
        return self.T / self.n
    
    def generate_path(self, simulations:int):
        '''
        In a Asian option with geometric average, the observed price
        is defined as:
            S_t1 = S_t * e**brownian
        '''
        z = np.random.randn(simulations, self.n)
        S_T = self.S * np.cumprod(
            self.brownian(self.r, self.T, self.sigma, z),
            1)
        return S_T, z

    def payoff(self, S_T):
        if self.p == 'C':
            option_payoff = self.df() * np.maximum(S_T - self.K, 0)
        else:
            option_payoff = self.df() * np.maximum(self.K - S_T, 0)
        return option_payoff
    
    def price(self, simulations:int=None, control_variate=None):
        simulations = simulations or self.M
        if control_variate:
            return self.price_with_control_variate(simulations)
            
        S_T, z = self.generate_path(simulations)
        if self.type_ == 'arithmetic':
            avg = np.mean(S_T, 1)
        elif self.type_ == 'geometric':
            avg = gmean(S_T, axis=1)
        payoff = self.payoff(avg)
        return self.confidence(payoff)
    
    def price_with_control_variate(self, simulations:int=None):
        simulations = simulations or self.M
        S_T, z = self.generate_path(simulations)

        avg_geo = gmean(S_T, axis=1)
        avg_ari = np.mean(S_T, axis=1)
        
        payoff_geo = self.payoff(avg_geo)
        payoff_ari = self.payoff(avg_ari)

        payoff = self.reduce_variance(payoff_ari, payoff_geo)
        return self.confidence(payoff)
        
class GeometricBasketWithTwoAssets(BlackScholes):

    def __init__(self, p:str, S1, S2, K, T, sigma1, sigma2, corr, r):
        '''
        S
            vector of prices of a basket of assets
        sigma
            vector of std of a basket of assets
        p
            str, "call" or "put"
        '''
        self.p = p
        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.T = T
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.corr = corr
        self.r = r

    def price(self, S1=None, S2=None, K=None, T=None, sigma1=None, sigma2=None, corr=None, r=None):
        sigma1 = sigma1 or self.sigma1
        sigma2 = sigma2 or self.sigma2
        
        S1 = S1 or self.S1
        S2 = S2 or self.S2
        sigma1 = sigma1 or self.sigma1
        sigma2 = sigma2 or self.sigma2
        r = r or self.r
        T = T or self.T
        corr = corr or self.corr

        # basket 
        B = np.sqrt(S1*S2)
        sigma = np.sqrt(sigma1 ** 2 + sigma1 * sigma2 * corr * 2 + sigma2 ** 2) / 2
        mu = r - (sigma1**2 + sigma2**2)/4 + 0.5*sigma**2
        df = np.exp(-r * T)
        
        return df * self.v_price(
            self.p, B, K or self.K,
            T or self.T, sigma, mu)

class ArithmeticBasketWithTwoAssets(MonteCarlo):

    def __init__(self, p:str, S1, S2, K, T, sigma1, sigma2, corr, r, M:int=1000, type_="arithmetic"):
        self.p = p
        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.T = T
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.corr = corr
        self.r = r
        self.M = M
        self.type_ = type_

    def control_variate(self):
        geo = GeometricBasketWithTwoAssets(
            self.p, self.S1, self.S2, self.K, self.T,
            self.sigma1, self.sigma2, self.corr, self.r)
        return geo.price()
    
    def generate_path(self, simulations, seed=100):
        np.random.seed(seed)
        z1 = np.random.randn(simulations)
        z  = np.random.randn(simulations)
        z2 = self.corr*z1 + np.sqrt(1-self.corr**2)*z

        S1_T = self.S1 * self.brownian(self.r, self.T, self.sigma1, z1)
        S2_T = self.S2 * self.brownian(self.r, self.T, self.sigma2, z2)
        
        return S1_T, S2_T
    
    def price_with_control_variate(self):

        XY = [0.0]*path
        for i in range(0, path):
            XY[i] = arith_payoff[i]*geo_payoff[i]
        covXY = np.mean(XY) - np.mean(arith_payoff)*np.mean(geo_payoff)
        theta = covXY/np.var(geo_payoff)

        geo = geo_basket(S1, S2, sigma1, sigma2, r, T, K ,corr, type)
        Z = arith_payoff + theta * (geo - geo_payoff)
        z_mean = np.mean(Z)
        z_std = np.std(Z)
        return z_mean

    def payoff(self, S_T):
        if self.p == 'C':
            option_payoff = self.df() * np.maximum(S_T - self.K, 0)
        else:
            option_payoff = self.df() * np.maximum(self.K - S_T, 0)
        return option_payoff
    
    def price(self, simulations:int=1000, control_variate=None):
        simulations = simulations or self.M
        if control_variate:
            return self.price_with_control_variate(simulations)
            
        S1_T, S2_T = self.generate_path(simulations)
        
        if self.type_ == 'arithmetic':
            B_T = (S1_T + S2_T) / 2
        elif self.type_ == 'geometric':
            B_T = np.exp((np.log(S1_T) + np.log(S1_T)) / 2)

        payoff = self.payoff(B_T)
        return self.confidence(payoff)
    
    def price_with_control_variate(self, simulations:int=1000):
        S1_T, S2_T = self.generate_path(simulations)
        
        B_T_mean = (S1_T + S2_T) / 2
        B_T_geo  = np.exp((np.log(S1_T) + np.log(S1_T)) / 2)
        
        payoff_geo = self.payoff(B_T_geo)
        payoff_mean = self.payoff(B_T_mean)

        payoff = self.reduce_variance(payoff_mean, payoff_geo)
        return self.confidence(payoff)

class BinomialTree(Model):

    def __init__(self, p:str, S, K, T, sigma, r, N=10):
        '''
        p
            str, "call" or "put"
        '''
        self.p = p # buy or sell
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.N = N

    def payoff(self, S_T):
        if self.p == 'C':
            return max(S_T - self.K,0)
        else:
            return max(self.K - S_T, 0)

    def combos(self, n, i):
        return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))
    
    def price(self, N=None):
        N = N or self.N
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt))
        P = (  np.exp(self.r*dt) - d )  /  (  u - d )
        v = 0
        for i in range(N+1):
            p = self.combos(N, i)*P**i*(1-P)**(N-i)
            S_T = self.S*(u)**i*(d)**(N-i)
            v += self.payoff(S_T) * p
        
        return self.df() * v
