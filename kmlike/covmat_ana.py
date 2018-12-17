from __future__ import print_function
import numpy as np
import healpy as hp
from utils import dl2cl
from scipy.special import legendre

def dl2cl(dls):
    ell = np.arange(len(dls))
    cls = dls.copy()
    cls[1:] = cls[1:] * 2. * np.pi / ell[1:] / (ell[1:] + 1)
    return cls

def get_cosbeta(v1, v2):
    return np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)

def gen_pl_ana(nside=4, lmax=None):
    if (lmax==None):
        lmax = nside*3-1

    npix = 12 * nside * nside
    pixarr = np.arange(npix)

    ## npix array -> vectors
    vecarr = np.array(hp.pix2vec(nside, pixarr)).T

    ## two vectors' pairs -> Pls
    cosbeta = np.zeros((npix, npix))
    for i in xrange(npix):
        for j in xrange(npix):
            cosbeta[i][j] = get_cosbeta(vecarr[i], vecarr[j])

    ell = np.arange(lmax+1)
    legs = [legendre(l) for l in ell]
    pl = np.array([fnc(cosbeta) for fnc in legs])

    return pl

def getcov_ana(dls, nside=4, pls=None, lmax=None, isDl=True):
    if (lmax==None):
        lmax = nside*3-1

    if (isDl):
        cls = dl2cl(dls)
    else:
        cls = dls.copy()

    if (type(pls)==type(None)):
        pls = gen_pl_ana(nside, lmax)

    ## l sum with (2l+1)Cl/4pi -> covariance
    ell = np.arange(len(dls))
    bls = (2.*ell+1)/4/np.pi * cls
    cov = np.einsum('l,lij->ij', bls, pls)

    return cov

def main():
    nside = 4
    lmax = 3*nside-1
    dls = np.zeros(lmax+1)+1
    dls[0] = 0
    dls[1] = 0

    cov = getcov_ana(dls, nside=nside, lmax=lmax, isDl=True)

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
