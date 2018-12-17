import numpy as np
import pylab as plt
import healpy as hp

import kmlike.spectrum as spectrum
import kmlike.utils as utils

# compare cmb spectrum and white noise Dl's
# 2018-10-30

def compare_cmb_noise():
    nside = 1024
    lmax = 3*nside - 1#1500#nside*3 - 1 
    cl_r01 = spectrum.get_spectrum_camb(lmax, r=0.01)
    cl_r001 = spectrum.get_spectrum_camb(lmax, r=0.001)
    nl_wp10 = spectrum.get_spectrum_whitenoise(lmax, wp=10, bw=30,)# 1.5*60)
    nl_wp2 = spectrum.get_spectrum_whitenoise(lmax, wp=2, bw=30, )#1.5*60)

    print (cl_r01.shape)
    print (cl_r001.shape)
    print (nl_wp10.shape)
    print (nl_wp2.shape)

    ell = np.arange(lmax+1)
    plt.loglog(ell, cl_r01[:3].T*1e12)
    plt.loglog(ell, cl_r001[:3].T*1e12)
    plt.loglog(ell, nl_wp10[:3].T*1e12)
    plt.loglog(ell, nl_wp2[:3].T*1e12)

    cls = utils.dl2cl(nl_wp10)

    m = hp.synfast(cls, nside=nside, new=True, fwhm=0.0*np.pi/180)
    hp.mollview(m[0])

    std_wnT = np.std(m[0])
    std_wnQ = np.std(m[1])
    std_wnU = np.std(m[2])
    std_wn  = np.std(m)
    
    #mr = np.random.random(len(m[0]))
    mrT = np.random.normal(size=len(m[0]), scale=std_wnT)
    mrQ = np.random.normal(size=len(m[0]), scale=std_wnQ)
    mrU = np.random.normal(size=len(m[0]), scale=std_wnU)
    mr = [mrT, mrQ, mrU]
    
    std_wnTr = np.std(mr[0])
    std_wnQr = np.std(mr[1])
    std_wnUr = np.std(mr[2])
    std_wnr  = np.std(mr)

    print ('Standard deviations of synthesized noise maps')
    print (std_wnT, std_wnQ, std_wnU)
    print (std_wn) 
    print ('Standard deviations of Gaussian random maps')
    print (std_wnTr, std_wnQr, std_wnUr)
    print (std_wnr) 

    cls_rec = hp.anafast(m, lmax=lmax)
    cls_ran = hp.anafast(mr, lmax=lmax)

    names = ['TT', 'EE', 'BB']
    for i, name in enumerate(names):
        plt.figure()
        plt.loglog(ell, utils.cl2dl(cls)[i].T)
        plt.loglog(ell, utils.cl2dl(cls_rec)[i].T, '*', label='Synthesized map')
        plt.loglog(ell, utils.cl2dl(cls_ran)[i].T, '+', label='Gaussian random number map')
        plt.title(name)

    plt.show()

if __name__=='__main__':
    compare_cmb_noise()

