import healpy as hp
import numpy as np
import pylab as plt

from src import lk_hamimeche as lh
from src.kmlike.spectrum import get_spectrum_camb, get_spectrum_noise
from src.kmlike.utils import cl2dl

def noised_map():
    nside = 1024
    lmax = 3*nside-1
    CMB_unit = 'muK'
    np.random.seed(42)

    cls = get_spectrum_camb(lmax=lmax, r=0.01, isDl=False, CMB_unit=CMB_unit)
    Nls = get_spectrum_noise(lmax=lmax, wp=10, isDl=False, TTonly=False)
    print (Nls.shape)

    mapTQU = hp.synfast(cls, nside=nside, new=True) 
    noiTQU = np.random.normal(0, 1, mapTQU.shape) 
    noiTQU2 = hp.synfast(Nls, nside=nside, new=True)

    map_tot = mapTQU + noiTQU 
    #hp.mollview(map_tot[0])
    hp.mollview(noiTQU2[0])

    cl_sig = hp.anafast(mapTQU, lmax=lmax)
    cl_noi = hp.anafast(noiTQU, lmax=lmax)
    cl_noi2 = hp.anafast(noiTQU2, lmax=lmax)

    plt.figure()
    plt.loglog(cl2dl(cl_sig[:3]).T)
    #plt.loglog(cl2dl(cl_noi[:3]).T)
    plt.loglog(cl2dl(cl_noi2[:3]).T)
    plt.loglog(cl2dl(Nls[:3]).T)
    plt.figure()
    plt.loglog((cl_sig[:3]).T)
    #plt.loglog((cl_noi[:3]).T)
    plt.loglog((cl_noi2[:3]).T)
    plt.loglog((Nls[:3]).T)
    plt.show()

def check_relation():
    nside = 64
    lmax = nside * 3 -1
    CMB_unit = 'muK'
    np.random.seed(42)

    cls = get_spectrum_camb(lmax=lmax, r=0.01, isDl=False, CMB_unit=CMB_unit)
    mapTQU = hp.synfast(cls, nside=nside, new=True) 
    noiTQU = np.random.normal(0, 1, mapTQU.shape) 

    cl_sig = hp.anafast(mapTQU, lmax=lmax)
    cl_noi = hp.anafast(noiTQU, lmax=lmax)

    mapT = mapTQU[0]
    noiT = noiTQU[0]

    """
    #print ('calculating Cl')
    #from scipy.special import legendre
    #leg = legendre(1)
    #Cl = 0
    #Npix = len(noiT)
    #idx = np.arange(Npix)
    #ni = np.array(hp.pix2vec(nside, idx))
    #Cl = np.sum(np.outer(noiT[idx] , noiT[idx]) * leg(np.dot(ni.T, ni)))
    #Cl *= 4. * np.pi / Npix 
    #print (Cl)
    """

    noiscale = []
    scales = np.arange(1000) / 100.

    for i in scales:

        noiTQU = np.random.normal(0, i, mapTQU.shape) 
        cl_noi = hp.anafast(noiTQU, lmax=lmax)
        noiscale.append(np.mean(cl_noi[0]))

    plt.plot(scales, noiscale)

    plt.show()
     

def main():
    noised_map()
    #check_relation()

if __name__=='__main__':
    main()

