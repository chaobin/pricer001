__all__ = ['newton']


def newton(x, y, f, f_prime, tolerance=0.00001, n_iter=1000):
    '''
    x
        can be initial guess

    >>> newton(
    >>> 1, 4,
    >>> lambda x: x**2,
    >>> lambda x: 2*x)
    '''
    diff = f(x) - y
    m = 0
    while abs(diff) > tolerance and m < n_iter:
        x = x - diff / f_prime(x)
        m += 1
        diff = f(x) - y
    return x
