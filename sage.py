from numpy import ones, zeros, exp, log, tile, array, dot
from scipy.optimize import minimize
import deltaIterator as di

"""
This is a python implementation of the SAGE model from Eisenstein et al 2011.

Eisenstein, Jacob, Amr Ahmed, and Eric P. Xing. "Sparse additive generative models of text." In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pp. 1041-1048. 2011.

@inproceedings{eisenstein2011sparse,
  title={Sparse additive generative models of text},
  author={Eisenstein, Jacob and Ahmed, Amr and Xing, Eric P},
  booktitle={Proceedings of the 28th International Conference on Machine Learning (ICML-11)},
  pages={1041--1048},
  year={2011}
}

This implementation does not include the Newton optimization from the paper; that's available in the Matlab implementation, https://github.com/jacobeisenstein/SAGE

"""

# for this to work, ecounts must be
# eq_m must be a 1D vector of shape (W,)
def estimate(ecounts,eq_m,max_its=25):
    """
    Compute the parameters of a SAGE distribution.

    :type ecounts: numpy.ndarray
    :param ecounts: must be a 2D vector of shape (W,K). If K=1, it still has to be 2d. Use np.reshape to get this. W is the number of words in the vocabulary, K is the number of vectors.

    :type eq_m: numpy.ndarray
    :param eq_m: must be a vector of shape (W,1), the expected log-probabilities
    
    :rtype: numpy.ndarray
    :returns: parameters of a SAGE distribution
    
    """
    [W,K] = ecounts.shape
    eta = zeros(W)
    eq_inv_tau = ones(W)
    exp_eq_m = exp(eq_m)
    max_inv_tau = 1e5
    it = di.DeltaIterator(debug=False,max_its=max_its,thresh=1e-4)
    while not(it.done):
        fLogNormal = lambda x : fLogNormalAux(x,ecounts,exp_eq_m,eq_inv_tau)
        gLogNormal = lambda x : gLogNormalAux(x,ecounts,exp_eq_m,eq_inv_tau)
        min_out = minimize(fLogNormal,eta,method='L-BFGS-B',jac=gLogNormal,options={'disp':False})
        #TODO: implement newton step optimization from paper
        eta = min_out.x
        eq_inv_tau = 1/(eta**2)
        eq_inv_tau[eq_inv_tau > max_inv_tau] = max_inv_tau
        it.update(eta)
    return(eta)

def fLogNormalAux(eta,ecounts,exp_eq_m,eq_inv_tau):
    C = ecounts.sum(axis=0)
    [W,K] = ecounts.shape
    denom = tile(exp(eta),(K,1)).dot(exp_eq_m.T)
    out = -(eta.T.dot(ecounts).sum(axis=0) - C * log(denom.sum(axis=0)) - 0.5 * eq_inv_tau.T.dot(eta ** 2))
    return(out[0])
           
def gLogNormalAux(eta,ecounts,exp_eq_m,eq_inv_tau):
    C = ecounts.sum(axis=0)
    [W,K] = ecounts.shape
    denom = tile(exp(eta),(K,1)) * exp_eq_m
    denom_norm = (denom.T / denom.sum(axis=1))
    beta = C * denom_norm / (C + 1e-10)
    g = -(ecounts.sum(axis=1) - beta.dot(C) - eq_inv_tau * eta)
    return(g)

    
