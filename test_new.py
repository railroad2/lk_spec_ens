import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time

from iminuit import Minuit

from src import lk_hamimeche as lh
from src.kmlike.spectrum import get_spectrum_camb, get_spectrum_noise
from src.kmlike.utils import print_msg, print_debug, cl2dl

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

def test_new_multi_fit_As_wnoise(mapTQU=None, lmax=100, scale=1e12):
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=2e-9) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=1e-9) * scale

    if mapTQU is None:
        mapTQU = hp.synfast(cl_fid, nside=64, new=True)

    cl_est = hp.anafast(mapTQU, lmax=lmax)[:4]

    L1 = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)

    print_debug ('Test L1=', L1)

    def fnc(As):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=As) * scale
        nl     = get_spectrum_noise(lmax, wp=10, isDl=False, TTonly=False) * scale
        cl_ana = cl_ana + nl
        res = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid)
        print_msg ('As = {}, n2logL = {} '.format(As, res))
        return res
         
    #limit = (np.exp(2.9)/1e10, np.exp(3.1)/1e10)
    limit = (1.5e-9, 2.5e-9)
    m = Minuit(fnc, As=2e-9, errordef=1, limit_As=limit, error_As=5e-10)
    #m = Minuit(fnc, As=2e-9, errordef=1, limit_As=(1e-9, 4e-9), error_As=5e-10)
    m.migrad()
    #plt.figure()
    #m.draw_profile('As', bins=100, bound=limit)
    #plt.savefig('profile_multi_As_lmax{}.png'.format(lmax))
        

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

## par = r, with white noise
def test_new_multi_fit_r_wnoise(mapTQU=None, lmax=100, scale=1e12, spectra=['TT','EE','BB','TE'], wp=0, r_fid=0.1):
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, r=r_fid) * scale
    cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, r=r_fid) * scale

    if mapTQU is None:
        mapTQU = hp.synfast(cl_fid, nside=64, new=True)

    cl_est = hp.anafast(mapTQU, lmax=lmax)[:4]

    L1 = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid, spectra=spectra)
    #L1 = lh.n2logL_new_multi(cl_ana[1:3], cl_est[1:3], cl_fid[1:3])

    print_debug ('Test L1=', L1)

    def fnc(r):
        cl_ana = get_spectrum_camb(lmax, isDl=False, TTonly=False, r=r) * scale
        nl     = get_spectrum_noise(lmax, wp=wp, isDl=False, TTonly=False) * scale
        cl_ana = cl_ana + nl
        res = lh.n2logL_new_multi(cl_ana, cl_est, cl_fid, spectra=spectra)
        #res = lh.n2logL_new_multi(cl_ana[1:3], cl_est[1:3], cl_fid[1:3])
        print_msg ('r = {}, n2logL = {} '.format(r, res))
        return res
         
    limit = (0.0, 0.07)
    m = Minuit(fnc, r=0.009, errordef=1, limit_r=limit, error_r=0.01)
    m.migrad() 
    return m


def As_fit():
    lmax = 100
    nside = 1024
    As_in = 2e-9
    logAs_in = np.log(As_in * 1e10)
    
    print ('As_in = {}, logAs_in = {}'.format(As_in, logAs_in))
    scale = 1e12

    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, As=As_in) * scale
    nl_3   = get_spectrum_noise(lmax, wp=10, isDl=False, TTonly=False) * scale

    np.random.seed(42)
    mapTQU = hp.synfast(cl_fid, nside=nside, new=True)
    mapWN  = hp.synfast(nl_3,   nside=nside, new=True)
    mapT = mapTQU[0]

    mapTot = mapTQU + mapWN
    hp.mollview (mapTot[0])
    
    cl_ana = hp.anafast(mapTot, lmax=lmax)

    plt.figure()
    plt.loglog(cl2dl(cl_ana)[:3].T)
    plt.loglog(cl2dl(cl_fid)[:3].T)
    plt.loglog(cl2dl(nl_3)[:3].T)
    plt.xlabel('Multipole moment, $l$')
    plt.ylabel('$D_l (\mu K^2)$')

    #test_new_single_As(mapT=mapT, lmax=lmax, scale=scale)
    #test_new_single_logAs(mapT=mapT, lmax=lmax, scale=scale)
    #test_new_multi_fit_As(mapTQU=mapTQU, lmax=lmax, scale=scale)
    test_new_multi_fit_As_wnoise(mapTQU=mapTot, lmax=lmax, scale=scale)
    #test_new_multi_fit_logAs(mapTQU=mapTQU, lmax=lmax, scale=scale)

    #cl_est = hp.anafast(mapTQU, lmax=lmax)
    #cl_fit = get_spectrum_camb(lmax, isDl

    plt.show()

def r_fit():
    nside = 64
    lmax = 100
    r_in = 0.1
    
    scale = 1e12

    wp = 0
    cl_fid = get_spectrum_camb(lmax, isDl=False, TTonly=False, r=r_in) * scale
    nl_3   = get_spectrum_noise(lmax, wp=wp, isDl=False, TTonly=False) * scale

    np.random.seed(42)
    mapTQU = hp.synfast(cl_fid, nside=nside, new=True)
    mapWN  = hp.synfast(nl_3,   nside=nside, new=True)
    mapT = mapTQU[0]

    mapTot = mapTQU + mapWN
    hp.mollview (mapTot[0])
    
    cl_ana = hp.anafast(mapTot, lmax=lmax)

    plt.figure()
    plt.loglog(cl2dl(cl_ana)[:3].T, '+')
    plt.loglog(cl2dl(cl_fid)[:3].T, '--', linewidth=0.5)
    plt.loglog(cl2dl(nl_3)[:3].T, '--', linewidth=0.5)
    plt.loglog(cl2dl(cl_fid[:3] + nl_3[:3]).T, '--')
    plt.xlabel('Multipole moment, $l$')
    plt.ylabel('$D_l (\mu K^2)$')

    spectra = ['EE','BB']

    res = test_new_multi_fit_r_wnoise(mapTQU=mapTot, lmax=lmax, scale=scale, wp=wp, spectra=spectra)

    print (res.values)
    print (res.values['r'])
    r_fit = res.values['r']

    #cl_est = hp.anafast(mapTQU, lmax=lmax)
    cl_fit = get_spectrum_camb(lmax, r=r_fit, isDl=False) * scale
    plt.loglog(cl2dl(cl_fit[:3] + nl_3[:3]).T)

    plt.show()

if __name__=='__main__':
    r_fit()
