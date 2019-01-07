import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time

from iminuit import Minuit

from src import lk_hamimeche as lh
from src.kmlike.spectrum import get_spectrum_camb
from src.kmlike.utils import print_msg, print_debug

def test_new_single_As(mapT=None, lmax=100, scale=1e12):
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=1e-9) * scale

    if mapT is None:
        mapT = hp.synfast(cl_fid, nside=64, new=True)

    cl_est = hp.anafast(mapT, lmax=lmax)

    L1 = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)

    print_debug ('Test L1=', L1)

    def fnc(As):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=As) * scale
        res = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)
        print_msg ('As = {}, n2logL = {} '.format(As, res))
        return res
         
    limit = (np.exp(2.9)/1e10, np.exp(3.1)/1e10)
    m = Minuit(fnc, As=2e-9, errordef=1, limit_As=limit, error_As=5e-10)
    #m = Minuit(fnc, As=2e-9, errordef=1, limit_As=(1e-9, 4e-9), error_As=5e-10)
    #m.migrad()
    plt.figure()
    m.draw_profile('As', bins=100, bound=limit)
    plt.savefig('profile_single_As_lmax{}.png'.format(lmax))

def test_new_single_logAs(mapT=None, lmax=100, scale=1e12):
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=1e-9) * scale

    if mapT is None:
        mapT = hp.synfast(cl_fid, nside=64, new=True)
        
    cl_est = hp.anafast(mapT, lmax=lmax)

    L1 = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)

    print_debug ('Test L1=', L1)

    def fnc(logAs):
        As = np.exp(logAs)/1e10
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=True, As=As) * scale
        res = lh.n2logL_new_single(cl_ana, cl_est, cl_fid)
        print_msg ('log As = {}, n2logL = {} '.format(logAs, res))
        return res
         
    limit = (2.9, 3.1)
    m = Minuit(fnc, logAs=3.0, errordef=1, limit_logAs=limit, error_logAs=1)
    #m.migrad()
    plt.figure()
    m.draw_profile('logAs', bins=100, bound=limit)
    plt.savefig('profile_single_logAs_lmax{}.png'.format(lmax))

def test_new_multi_cov():
    lmax = 100
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=2e-9) * scale

    M = lh.covMat_full(cl_fid)

    plt.matshow(np.log10(M))
    plt.colorbar()
    plt.show()

def test_new_multi_fit_As(mapTQU=None, lmax=100, scale=1e12):
    scale = 1e12
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=1e-9) * scale

    if mapTQU is None:
        mapTQU = hp.synfast(cl_fid, nside=64, new=True)

    cl_est = hp.anafast(mapTQU, lmax=lmax)[:4]

    L1 = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)

    print_debug ('Test L1=', L1)

    def fnc(As):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=As) * scale
        res = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)
        print_msg ('As = {}, n2logL = {} '.format(As, res))
        return res
         
    limit = (np.exp(2.9)/1e10, np.exp(3.1)/1e10)
    m = Minuit(fnc, As=2e-9, errordef=1, limit_As=limit, error_As=5e-10)
    #m = Minuit(fnc, As=2e-9, errordef=1, limit_As=(1e-9, 4e-9), error_As=5e-10)
    #m.migrad()
    plt.figure()
    m.draw_profile('As', bins=100, bound=limit)
    plt.savefig('profile_multi_As_lmax{}.png'.format(lmax))

def test_new_multi_fit_logAs(mapTQU=None, lmax=100, scale=1e12):
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=1e-9) * scale

    if mapTQU is None:
        mapTQU = hp.synfast(cl_fid, nside=64, new=True)

    cl_est = hp.anafast(mapTQU, lmax=lmax)[:4]

    L1 = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)

    print_debug ('Test L1=', L1)
    
    def fnc(logAs):
        st_tot = time.time()
        As = np.exp(logAs)/1e10
        
        st = time.time()
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=As) * scale
        print_debug ('Elapsed time for CAMB: ', time.time() - st)

        st = time.time()
        res = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)
        print_debug ('Elapsed time for Likelihood calculation: ', time.time() - st)

        print_msg ('log As = {}, n2logL = {} '.format(logAs, res))
        print_debug ('total elapsed time = ', time.time() - st_tot)

        return res
         
    limit = (2.9, 3.1)
    m = Minuit(fnc, logAs=3.0, errordef=1, limit_logAs=limit, error_logAs=1)
    #m.migrad()
    plt.figure()
    m.draw_profile('logAs', bins=100, bound=limit)
    plt.savefig('profile_multi_logAs_lmax{}_faster.png'.format(lmax))

if __name__=='__main__':
    lmax = 100
    As_in = 2e-9
    logAs_in = np.log(As_in * 1e10)
    
    print ('As_in = {}, logAs_in = {}'.format(As_in, logAs_in))
    scale = 1e12

    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=As_in) * scale
    mapTQU = hp.synfast(cl_fid, nside=1024, new=True)
    mapT = mapTQU[0]

    #test_new_single_As(mapT=mapT, lmax=lmax, scale=scale)
    #test_new_single_logAs(mapT=mapT, lmax=lmax, scale=scale)
    #test_new_multi_fit_As(mapTQU=mapTQU, lmax=lmax, scale=scale)
    test_new_multi_fit_logAs(mapTQU=mapTQU, lmax=lmax, scale=scale)




