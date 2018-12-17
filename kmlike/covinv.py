import numpy as np
import sys
from numpy.linalg   import inv, pinv, slogdet, svd

def detinv_pseudo(cov):
    u, s, vh = svd(cov)

    uh = np.matrix(u).H
    v  = np.matrix(vh).H
    si = pinv(np.diagflat(s))
    tol = sys.float_info.epsilon

    covi = np.matmul(v,np.matmul(si, uh))
    plogdet = np.sum(np.log(s[s > tol]))

    return plogdet, covi

def detinv(cov):
    s, logdet = slogdet(cov)
    covi = inv(cov)

    return logdet, covi

def covreg_1(cov, lamb=1e-10): ## regularize with a constant matrix
    l = len(cov)
    CC = np.full((l,l), lamb)
    covr = cov + lamb*CC
    
    return covr

def covreg_I(cov, lamb=1e-10): ## regularize with an identity matrix
    l = len(cov)
    II = np.eye(l)
    covr = cov + lamb*II
    
    return covr

def covreg_R(cov, lamb=1e-10): ## regularize with a random matrix
    l = len(cov)
    II = np.eye(l)
    covr = cov + lamb*II
    
    return covr

def covreg_D(cov, lamb=1e-10): ## regularize with a random diagonal matrix
    l = len(cov)
    CC = np.full((l,l), lamb )
    covr = cov + lamb*CC
    
    return covr

def covreg_none(cov, lamb=1e-10):
    return cov
