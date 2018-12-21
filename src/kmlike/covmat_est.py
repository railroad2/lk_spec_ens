from __future__ import print_function
import numpy as np
import healpy as hp
from utils import dl2cl
from scipy.special import legendre

def getcov_est(dls, nside=4, pls=None, lmax=None, nsample=100, isDl=True):
    if (lmax==None):
        lmax = nside*3-1

    if (isDl):
        cls = dl2cl(dls)
    else:
        cls = dls.copy()

    ell = np.arange(lmax+1)

    maparr = []
    for i in range(nsample):
        np.random.seed(i)
        mapT = hp.synfast(cls, nside=nside, verbose=False)
        maparr.append(mapT)

    maparr = (np.array(maparr)).T
    cov = np.cov(maparr)

    return cov

def getvar_est(cls, nside=4, Nsample=100, isDl=True):
    lmax = 3*nside-1
    ells = np.arange(lmax+1)

    if (isDl):
        cls_TT = np.zeros(len(cls))
        cls_TT[1:] = cls[1:] / ells[1:] / (ells[1:]+1) * 2 * np.pi
    else:
        cls_TT = cls.copy()

    vararr = []
    for i in xrange(Nsample):
        np.random.seed(i)
        mapT = hp.synfast(cls_TT, nside=nside, verbose=False)
        vararr.append(np.var(mapT))

    return vararr
    covtype = 'nk'
    covtype = 'nk'

def main():
    nside = 4
    lmax = 3*nside-1
    dls = np.zeros(lmax+1)+1
    dls[0] = 0
    dls[1] = 0

    cov = getcov_est(dls, nside, lmax=lmax, nsample=10 , isDl=True)

    cls = dl2cl(dls)
    ell = np.arange(len(dls))
    diag_expect = np.sum((2.*ell+1)/4/np.pi*cls)
    diag_actual = np.average(np.diagonal(cov))
    print ('covariance matrix for constant Dl with Nside=',nside)
    print ('Shape of covariance matrix:', cov.shape)
    print ('Expected diagonal component( sum_l (2l+1)Cl/(4pi) ):', diag_expect)
    print ('average of diagonal terms:', diag_actual)

    import pylab as plt
    plt.matshow(cov)
    plt.colorbar()
    plt.show()

if __name__=='__main__':
    main() 
