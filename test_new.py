import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from iminuit import Minuit

from src import lk_hamimeche as lh
from src.kmlike.spectrum import get_spectrum_camb

def test_new_single():
    lmax = 100
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=2e-9) 
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=1e-9) 

    map = hp.synfast(cl_fid, nside=64, new=True)
    cl_est = hp.anafast(map, lmax=lmax)

    L1 = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)

    print ('L1=', L1)

    def fnc(As):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=As)
        return lh.n2logL_new_single(cl_ana, cl_est, cl_fid)
         
    m = Minuit(fnc, As=2e-9, errordef=1, limit_As=(0, 1e-8), error_As=5e-10)
    #m.migrad()
    m.draw_profile('As', bins=100)
    plt.savefig('profile.png')


if __name__=='__main__':
    test_new_single()
