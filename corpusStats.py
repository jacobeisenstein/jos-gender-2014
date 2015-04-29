from scipy.special import gammaln
from scipy.misc import factorial
import numpy as np


def log_dcm(counts,alpha):
    """
    Compute the log probability of counts under a Dirichlet-Compound Multinomial distribution with parameter alpha
    http://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution

    :type counts: numpy.ndarray
    :param counts: event counts
    
    :type alpha: numpy.ndarray
    :param alpha: pseudo-count parameters of the DCM distribution

    :rtype: float
    :return: log probability of counts under DCM distribution
    
    """
    N = sum(counts)
    A = sum(alpha)
    return gammaln(N+1) - sum(gammaln(counts+1)) + gammaln(A) - gammaln(N + A) + sum(gammaln(counts+alpha) - gammaln(alpha))

def log_betabin(k,N,a,b):
    """
    Compute the log probability of k successes in N trials under a beta-binomial distribution with parameters a,b
    http://en.wikipedia.org/wiki/Beta-binomial_distribution

    :type k: int
    :param k: number of "successes" (i.e. counts of a variable)
    
    :type N: int
    :param N: number of "trials" (i.e. total word counts)

    :type a: float
    :param a: prior parameter on the beta distribution
    
    :type b: float
    :param b: prior parameter on the beta distribution

    :rtype: float
    :return: log probability under the beta-binomial distribution

    """
    return log_dcm(np.array([k,N-k]),np.array([a,b]))

def betabin_cdf(k,N,a,b):
    """
    Compute the probability of X \leq k successes in N trials under a beta-binomial distribution with parameters a,b
        
    :type k: int
    :param k: number of "successes" (i.e. counts of a variable)
    
    :type N: int
    :param N: number of "trials" (i.e. total word counts)

    :type a: float
    :param a: prior parameter on the beta distribution
    
    :type b: float
    :param b: prior parameter on the beta distribution

    :rtype: float
    :return: log probability under the beta-binomial distribution

    """
    if k > 0.5 * N:
        p = 1. - sum([np.exp(log_betabin(x,N,a,b)) for x in range(k+1,N)])
    else:
        p = sum([np.exp(log_betabin(x,N,a,b)) for x in range(k+1)])
    return p

def printSF(x,n):
    """
    Print x up to n significant figures.
    I think I got this from stackoverflow somewhere, but I can't find the page now. Sorry.

    :type x: float
    :param x: number to print
    
    :type n: int
    :param N: number of significant figures
    """
    nsf = -int(np.floor(np.log10(x))) + n - 1
    fmtstr = "%."+str(nsf)+"f"
    return fmtstr%(x)
