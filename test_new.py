import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from iminuit import Minuit

from src import lk_hamimeche as lh
from src.kmlike.spectrum import get_spectrum_camb

def test_new_single_As():
    lmax = 100
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=1e-9) * scale

    map = hp.synfast(cl_fid, nside=64, new=True)
    cl_est = hp.anafast(map, lmax=lmax)

    L1 = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)

    print ('L1=', L1)

    def fnc(As):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=As) * scale
        return lh.n2logL_new_single(cl_ana, cl_est, cl_fid)
         
    m = Minuit(fnc, As=2e-9, errordef=1, limit_As=(1e-9, 4e-9), error_As=5e-10)
    #m.migrad()
    m.draw_profile('As', bins=100)
    plt.savefig('profile.png')

def test_new_single_logAs():
    lmax = 100
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=1e-9) * scale

    map = hp.synfast(cl_fid, nside=64, new=True)
    cl_est = hp.anafast(map, lmax=lmax)

    L1 = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)

    print ('L1=', L1)

    def fnc(logAs):
        print ('logAs=', logAs, )
        As = np.exp(logAs)/1e10
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=As) * scale
        return lh.n2logL_new_single(cl_ana, cl_est, cl_fid)
         
    limit = (2.9, 3.1)
    m = Minuit(fnc, logAs=3.0, errordef=1, limit_logAs=limit, error_logAs=1)
    #m.migrad()
    m.draw_profile('logAs', bins=100, bound=limit)
    plt.savefig('profile.png')

if __name__=='__main__':
    test_new_single_logAs()
