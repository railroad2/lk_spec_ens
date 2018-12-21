from __future__ import print_function
import numpy as np
import healpy as hp
from utils import dl2cl

def gen_ylm(nside=4):
    npix = 12 * nside * nside
    oop  = float(npix)/(4*np.pi) # 1/(solid angle of a pixel) 
    marr = np.eye(npix) * oop # delta function map array

    ylm = [hp.map2alm(m, use_weights=True, iter=100) for m in marr]
    return np.array(ylm)

def gen_pl_nk(nside, ylm=None):
    if (ylm==None):
        ylm = gen_ylm(nside)

    ylm = np.array(ylm)

    pl = []
    lmax = 3 * nside -1
    for l in range(lmax+1):
        y0  = ylm[:,l]
        y0c = np.conj(y0)
        pltmp = np.outer(y0, y0c)
        if (l > 0):
            m   = np.arange(l)+1
            idx = [hp.Alm.getidx(lmax, l, mm) for mm in m]
            ym  = ylm[:, idx]
            ymc = np.conj(ym)
            pltmp += np.einsum('lm,km->lk', ym, ymc) \
                   + np.einsum('lm,km->lk', ymc, ym)
        pl.append(pltmp.real)
        
    return np.array(pl)

def getcov_nk(dls, nside=4, pls=None, lmax=None, isDl=True):
    if (lmax==None):
        lmax = nside * 3 - 1

    if (isDl):
        cls = dl2cl(dls)
    else:
        cls = dls.copy()

    if (type(pls)==type(None)):
        pls = gen_pl_nk(nside)

    cov = np.einsum('l,lij->ij', cls, pls)

    return cov

def main():
    nside = 4
    lmax = 3*nside-1
    dls = np.zeros(lmax+1)+1
    dls[0] = 0
    dls[1] = 0

    cov = getcov_nk(dls, nside=nside, lmax=lmax, isDl=True)

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
