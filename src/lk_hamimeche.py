from __future__ import print_function
import numpy as np
import healpy as hp
import camb
import pylab as plt
import time
import os

from pprint import pprint
from scipy.linalg import sqrtm, eigvals
from scipy.optimize import minimize
from iminuit import Minuit

from .kmlike.spectrum import get_spectrum_camb
from .kmlike.utils import cl2dl, dl2cl, print_error, print_debug
from .kmlike.covinv import detinv, detinv_pseudo


# Global parameters
Tcmb = 2.7255
nside = 512
lmin = 2
lmax = 3*nside-1
if (lmax > 2000):
    lmax = 2000
lmax = 11
parname = 'ns'
pars = np.arange(0.8, 1.1, 0.01) 

DEBUG=0

#==================================================
#
# calculate covariance matrix
#
#==================================================


def get_cov(cls_ana, cls_est):
    dim = len(cls_est)

    cov = np.zeros((dim,dim))

    cls_diff = np.array(cls_est) - np.array(cls_ana)
    cov = np.outer(cls_diff, cls_diff)

    return cov


#==================================================
#
# Likelihood functions by Hamimeche 2008, 
#
#==================================================


def TTonly(cls):
    if (len(cls.shape) == 2):
        if (len(cls) == 4 or len(cls) == 6):
            res = cls[0]
        else:
            res = cls.T[0]
    else:
        res = cls

    return res


## Exact likelihood by Hamimeche 2008, eq. (15)

def n2logL_exact(cls_ana, cls_est): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    ell = np.arange(len(cls_ana))
    n=1

    res = np.sum((2*ell[lmin:] + 1) 
                 * ( cls_est[lmin:] / cls_ana[lmin:] 
                     - np.log(np.abs(cls_est[lmin:] / cls_ana[lmin:])) 
                     - n ))
    return res


## Likelihood by Hamimeche 2008, eq. (22)

def n2logL_approx_eq22(cls_ana, cls_est): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    ell = np.arange(len(cls_ana))
    res = np.sum((2*ell[lmin:] + 1)/2 
                 *((cls_est[lmin:] - cls_ana[lmin:]) / cls_est[lmin:])**2) 

    return res


## Likelihood by Hamimeche 2008, eq. (23)

def n2logL_approx_eq23(cls_ana, cls_est): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    ell = np.arange(len(cls_ana))
    lmin = 2
    res = np.sum((2*ell[lmin:] + 1)/2 
                 *((cls_est[lmin:] - cls_ana[lmin:]) / cls_ana[lmin:])**2)

    return res


## Likelihood by Hamimeche 2008, eq. (24) 

def n2logL_approx_eq24(cls_ana, cls_est, cls_fid=None): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    # idk what is the fiducial model 
    # 2018-10-22 Now I know.
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    if cls_fid is None:
        cls_fid = get_spectrum_camb(lmax=lmax)

    cls_fid = TTonly(cls_fid)

    ell = np.arange(len(cls_ana))
    res = np.sum( (2*ell[lmin:] + 1)/2 
                   * ((cls_est[lmin:] - cls_ana[lmin:])/cls_fid[lmin:])**2 )

    return res


## Likelihood by Hamimeche 2008, eq. (25)

def n2logL_approx_eq25(cls_ana, cls_est): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    ell = np.arange(len(cls_ana))
    res = np.sum( (2*ell[lmin:] + 1)/2 
                   * ((cls_est[lmin:] - cls_ana[lmin:])/cls_ana[lmin:])**2 
                       + np.log(np.abs(cls_ana[lmin:])) )

    return res


## Likelihood by Hamimeche 2008, eq. (26)

def n2logL_approx_eq26(cls_ana, cls_est): 
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    if (len(cls_ana) != len(cls_est)):
        print ('length of two power spectra are different')

    ell = np.arange(len(cls_ana))
    res = np.sum( (2*ell[lmin:] + 1)/2 
                   * (np.log(cls_est[lmin:]/cls_ana[lmin:]))**2 ) 

    return res


## Likelihood by Hamimeche 2008, eq. (27)

def n2logL_approx_eq27(cls_ana, cls_est):
    cls_ana = TTonly(cls_ana)
    cls_est = TTonly(cls_est)
    n2logL_Q  = n2logL_approx_eq23(cls_ana, cls_est) 
    n2logL_LN = n2logL_approx_eq26(cls_ana, cls_est)
    n2logL_WMAP = 1./3*n2logL_Q + 2./3*n2logL_LN
    return n2logL_WMAP


## Gaussian approximation for correlated fields

def n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid, n_field=3):
    print ('test approximation TEB')

    ## input Cls's are array of matrices 
    ## [[TT, TE,  0], 
    ##  [TE, EE,  0], 
    ##  [ 0,  0, BB]] for l's

    # mock up
    # dif = (cls_ana - cls_est) 
    # inp = dif * inv_cls_fid * dif * inv_cls_fid
    # n2logLf = (2.0*l+1)/2.0 * trace(inp) + (n+1) * np.log(det_cls_fid)

    Cls_dif = (Cls_ana - Cls_est)
    inp1 = np.matmul(Cls_dif, inv_Cls_fid)
    inp2 = np.matmul(inp1, inp1)
    trace = np.trace(inp2, axis1=1, axis2=2)
    n = n_field   # number of correlated fields
    
    n2logLf = 0
    for l, (Tr, logdet) in enumerate(zip(trace, det_Cls_fid)):
        #print_debug ('l=%d, Tr=%e, logdet=%f ' % (l, Tr, logdet))
        
        if (l >= lmin):
            n2logLf += (2.0*l + 1.0)/2.0 * Tr + (n + 1) * logdet 

    return n2logLf


def g(x):
    return np.sign(x-1)*np.sqrt(2*(x-np.log(x)-1))


def n2logL_new_single(cl_ana, cl_est, cl_fid):
    cl_ana = np.array(cl_ana)[2:]
    cl_est = np.array(cl_est)[2:]
    cl_fid = np.array(cl_fid)[2:]

    l = np.arange(len(cl_ana))+2
    v = (g(cl_est/cl_ana)*cl_fid)
      
    ## 1. vectors and summation
    #st = time.time()
    Mv = 2./(2.*l - 1.) * cl_fid #* cl_fid
    Mvi = 1/Mv
    n2lnL1 = np.sum(v * Mvi * v)
    #print ('elapsed time for method 1 (summation):', time.time()-st, 's')
    
    ## 2. Matrix operation
    #st = time.time()
    #M = np.diag(2/(2*l - 1) * cl_fid)
    #Mi = np.linalg.inv(M) 
    #M = 2/(2*l - 1) * cl_fid
    #Mi = np.diag(1/M) 
    #n2lnL2 = np.dot(v, np.dot(Mi, v))
    #print ('elapsed time for method 2 (matrix):', time.time()-st, 's')

    #plt.loglog(l, cl_ana)
    #plt.loglog(l, cl_fid)
    #plt.loglog(l, cl_est)
    #plt.savefig('cls.png')
    #print ('-2 log likelihood=', n2lnL1) 

    return n2lnL1
     

def covMat_full(cl_fid, diagonal=True, simple=True):
    if len(cl_fid) == 4 or len(cl_fid) == 6:
        cl = cl_fid.T 
    else:
        cl = cl_fid

    if len(cl[0]) == 4:
        Ncl = 4
    elif len(cl[0]) == 6:
        Ncl = 6
    else:
        Ncl = 1

    ndim = (len(cl)-2)
    ell = np.arange(ndim) + 2
    M_l = np.full((ndim, ndim, Ncl, Ncl), 0.)

    for l in ell:
        M_l[l-2,l-2] = 2./(2.*l-1.) * np.outer(cl[l], cl[l])

    if (diagonal):
        res = np.array([M_l[i,i] for i in range(ndim)])
    else:
        if (simple):
            res = np.concatenate(np.concatenate(M_l, axis=1), axis=1)
        else:
            res = M_l

    return res


def g_(X):
    V, U = np.linalg.eig(X)
    gV = np.sign(V-1) * np.sqrt(2*(V-np.log(V)-1))
    return np.array([np.diag(gl) for gl in gV])


def vecp(Cls):
    return np.array([[Cl[0][0], Cl[1][1], Cl[2][2], Cl[0][1]] for Cl in Cls])

"""
def getVector_Xg(cl_ana, cl_est, cl_fid):
    Cl_ana = cls2Cls(cl_ana)[2:]
    Cl_est = cls2Cls(cl_est)[2:]
    Cl_fid = cls2Cls(cl_fid)[2:]

    Cl_ana_inv = np.linalg.inv(Cl_ana)
    Cl_ana_nsqrt = [sqrtm(ml) for ml in Cl_ana_inv]
    Cl_fid_sqrt = [sqrtm(ml) for ml in Cl_fid]

    CCC = np.matmul(Cl_ana_nsqrt, np.matmul(Cl_est, Cl_ana_nsqrt))

    gCCC = g_(CCC)
    
    CgCCCC = np.matmul(Cl_fid_sqrt, np.matmul(gCCC, Cl_fid_sqrt))

    Xg = vecp(CgCCCC)

    return Xg
"""

def sqrtm_km(X):
    D, V = np.linalg.eig(X)
    D_sqrt_arr = np.sqrt(D)
    if (len(X.shape) == 3):
        D_sqrt = np.array([np.diag(mi) for mi in D_sqrt_arr])
    elif (len(X.shape) == 2):
        D_sqrt = np.diag(np.sqrt(D))
    else:
        print_error('Matrix with invaild dimension.')
        
    Vi = np.linalg.inv(V)

    res = np.matmul(V, np.matmul(D_sqrt, Vi))
    return res
     

# for speed test and improvement
def sqrtm_(x):
    return sqrtm(x)

def getVector_Xg(cl_ana, cl_est, cl_fid):
    st = time.time()
    Cl_ana = cls2Cls(cl_ana)[2:]
    Cl_est = cls2Cls(cl_est)[2:]
    Cl_fid = cls2Cls(cl_fid)[2:]
    print_debug ('---- Elapsed time for cls2Cls: ', time.time() - st)

    st = time.time()
    Cl_ana_inv = np.linalg.inv(Cl_ana)
    print_debug ('---- Elapsed time for Cl_ana inversion: ', time.time() - st)

    ## default (0.4 s)
    #st = time.time()
    #Cl_ana_nsqrt = [sqrtm(ml) for ml in Cl_ana_inv]
    #Cl_fid_sqrt = [sqrtm(ml) for ml in Cl_fid] 
    #print_debug ('---- Elapsed time for sqrt of matrices: ', time.time() - st)

    ## using sqrtm_km() (0.04 s : 10x faster than default)
    st = time.time()
    Cl_ana_nsqrt = sqrtm_km(Cl_ana_inv)
    Cl_fid_sqrt = sqrtm_km(Cl_fid)
    print_debug ('---- Elapsed time for sqrt of matrices: ', time.time() - st)

    ## single large matrix and disintegrate (40 s: takes long for sqrtm(large matrix))
    #st = time.time()
    #Cl_ana_inv_large = np.array([[[[0]*3]*3]*len(Cl_ana_inv)]*len(Cl_ana_inv))
    #Cl_fid_large = np.array([[[[0]*3]*3]*len(Cl_fid)]*len(Cl_fid))
    #for i in range(len(Cl_ana_inv)):
    #    Cl_ana_inv_large[i,i] = Cl_ana_inv[i]
    #    Cl_fid_large[i,i] = Cl_fid[i]
    #Cl_ana_inv_large = np.concatenate(np.concatenate(Cl_ana_inv_large, 1), 1)
    #Cl_fid_large = np.concatenate(np.concatenate(Cl_fid_large, 1), 1)

    #st_p = time.time()
    #Cl_ana_inv_nsqrt_large = sqrtm(Cl_ana_inv_large)
    #Cl_fid_sqrt_large = sqrtm(Cl_fid_large)
    #print_debug ('---- Elapsed time for sqrt of large matrices: ', time.time() - st_p)

    #Cl_ana_nsqrt = [sqrtm(ml) for ml in Cl_ana_inv]
    #Cl_fid_sqrt = [sqrtm(ml) for ml in Cl_fid]
    #print_debug ('---- Elapsed time for sqrt of matrices: ', time.time() - st)

    ## with multithreading (0.3 s: ~ 20 % improvement)
    #from multiprocessing import Pool
    #st = time.time()
    #with Pool(processes=32) as pool:
    #    Cl_ana_nsqrt = pool.map(sqrtm_, Cl_ana_inv)
    #    Cl_fid_sqrt = pool.map(sqrtm_, Cl_fid)
    #print_debug ('---- Elapsed time for sqrt of matrices: ', time.time() - st)

    st = time.time()
    CCC = np.matmul(Cl_ana_nsqrt, np.matmul(Cl_est, Cl_ana_nsqrt))
    print_debug ('---- Elapsed time for CCC: ', time.time() - st)

    st = time.time()
    gCCC = g_(CCC)
    print_debug ('---- Elapsed time for gCCC: ', time.time() - st)

    st = time.time()
    CgCCCC = np.matmul(Cl_fid_sqrt, np.matmul(gCCC, Cl_fid_sqrt))
    print_debug ('---- Elapsed time for CgCCCC: ', time.time() - st)

    st = time.time()
    Xg = vecp(CgCCCC)
    print_debug ('---- Elapsed time for Xg: ', time.time() - st)

    return Xg

def n2logL_new_multi(cl_ana, cl_est, cl_fid):
    st = time.time()
    M = covMat_full(cl_fid, diagonal=True)
    print_debug ('elapsed time for covariance matrix calculation: ', time.time()-st)

    st = time.time()
    #Mi = np.linalg.inv(M)
    Mi = np.linalg.pinv(M)
    print_debug ('elapsed time for covariance matrix inversion: ', time.time()-st)

    st = time.time()
    Xg = getVector_Xg(cl_ana, cl_est, cl_fid)
    print_debug ('elapsed time for Xg vector calculation: ', time.time()-st)

    st = time.time()
    L = sum(np.einsum('ij,ij->i', Xg, np.einsum('ijk,ij->ik', Mi, Xg)))
    print_debug ('elapsed time for Likelihood calculation: ', time.time()-st)

    return L


## convert cls to Cls

def cls2Cls(cls, T=True):
    cls = np.array(cls)
    if len(cls.shape) != 2:
        print_error ('The input should be 2-D array')
        return
    else:
        if (len(cls) == 4 or len(cls) == 6):
            cls_tmp = cls.T[:,:4]
        else:
            cls_tmp = cls[:,:4]


    if T:
        Cls = [ [[TT, TE, 0], [TE, EE, 0], [0, 0, BB]] for TT, EE, BB, TE in cls_tmp ]
    else:
        Cls = [ [[EE, 0], [0, BB]] for __, EE, BB, __ in cls_tmp ]

    Cls = np.array(Cls)

    return Cls


def invdet_fid(cls_fid, T=True):
    Cls_fid = cls2Cls(cls_fid, T) 
    inv_Cls_fid = np.zeros(Cls_fid.shape)
    inv_Cls_fid[2:] = np.linalg.inv(Cls_fid[2:])
    det_Cls_fid = np.full(len(Cls_fid), 1.0)
    s, det_Cls_fid[2:] = np.linalg.slogdet(Cls_fid[2:])

    return inv_Cls_fid, det_Cls_fid


#==================================================
#
# Likelihood wrapper 
#
#==================================================


def lk_for_par(ns, cls_est, fnc_type):
    cls_ana = get_spectrum_camb(lmax=lmax, ns=ns) 
    lk = fnc_type(cls_ana, cls_est)
    return lk


#==================================================
#
# Fit with minuit
#
#==================================================


## Fit with minuit for n_s

def fit_minuit(ns0, cls_est, fnc_type):
    def fit_minuit_1(ns):
        cls_ana = get_spectrum_camb(lmax=lmax, ns=ns) 
        lk = fnc_type(cls_ana, cls_est)
        return lk
    m = Minuit(fit_minuit_1, ns=ns0)
    res = m.migrad()
    return res


## Fit with minuit for two parameters, n_s and Omega_b h^2

def fit_minuit_2par(ns0, cls_est, fnc_type):
    def fit_minuit_1(ns, ombh2):
        cls_ana = get_spectrum_camb(lmax=lmax, ns=ns, ombh2=ombh2) 
        lk = fnc_type(cls_ana, cls_est)
        return lk

    m = Minuit(fit_minuit_1, ns=ns0, ombh2=0.022, limit_ns=(0.9, 1.1), 
               limit_ombh2=(0.01, 0.03))
    res = m.migrad()
    param = m.hesse()
    pprint(m.matrix())
    param2 = m.minos()
    m.print_param()

    #m.draw_mncontour('ns', 'ombh2', nsigma=4)

    return res


## Fit with minuit for A_s and tau

def fit_minuit_AsTau(As0, tau0, cls_est, fnc_type):
    def fit_minuit_1(As, tau):
        cls_ana = get_spectrum_camb(lmax=lmax, As=As, tau=tau) 
        lk = fnc_type(cls_ana, cls_est)
        return lk

    m = Minuit(fit_minuit_1, As=As0, tau=tau0, limit_As=(1e-9, 3e-9), 
               limit_tau=(0, 0.1))
    res = m.migrad()
    param = m.hesse()
    pprint(m.matrix())
    param2 = m.minos()
    m.print_param()

    #m.draw_mncontour('ns', 'ombh2', nsigma=4)

    return res


def fit_minuit_TEB(r, cls_est, fnc_type):
    def fit_minuit_1(ns):
        cls_ana = get_spectrum_camb(lmax=lmax, ns=ns) 
        lk = fnc_type(cls_ana, cls_est)
        return lk
    m = Minuit(fit_minuit_1, ns=ns0)
    res = m.migrad()
    return res



#==================================================
#
# Test modules
#
#==================================================


## test of covariance calculation

def test_1():
    nside = 2048
    lmax = 1000;#3*nside-1
    m = hp.read_map('cmb_map.fits')
    m = hp.ud_grade(m, nside_out=nside)

    ell = np.arange(lmax - 2) + 2

    cls_est = hp.anafast(m)
    cls_est = cls_est[2:lmax] * (ell*(ell + 1)) / (2*np.pi) * 1e12
    
    cls_ana = get_spectrum_camb(omk=0)
    cls_ana = cls_ana[0][2:lmax] * 1e12
    
    plt.loglog(cls_est)
    plt.loglog(cls_ana)

    cov = get_cov(cls_est, cls_ana)
    plt.matshow(cov)

    plt.show()


## Test of likelihood profile of likelihood functions
    ## analytic power spectrum
    cls_ana0 = get_spectrum_camb(lmax=lmax, TTonly=True)
    cls_ana1 = get_spectrum_camb(lmax=lmax, ns=0.9, TTonly=True)

    ## estimated power spectrum
    np.random.seed(42)
    
    bls_ana0 = dl2cl(cls_ana0)
    m = hp.synfast(bls_ana0, nside, new=True, verbose=False)    
    hp.mollview(m)
    cls_est = TTonly(cl2dl(hp.anafast(m, lmax=lmax)))

    ## log-likelihood
    n2logL_approx_eqNN = n2logL_approx_eq22
    n2logL0 = n2logL_approx_eqNN(cls_ana0, cls_est)
    print ('Fiducial value of likelihood = ', n2logL0)
             
    ## likelihood function
    lkfnc22 = []
    lkfnc23 = []
    lkfnc24 = []
    lkfnc25 = []
    lkfnc26 = []
    lkfnc27 = []
    lkfncEx = []
    for p in pars:
        kwargs = {parname:p}
        cls_ana = get_spectrum_camb(lmax=lmax, **kwargs)
        n2logL22 = n2logL_approx_eq22(cls_ana, cls_est)
        n2logL23 = n2logL_approx_eq23(cls_ana, cls_est)
        n2logL24 = n2logL_approx_eq24(cls_ana, cls_est, cls_ana1)
        n2logL25 = n2logL_approx_eq25(cls_ana, cls_est)
        n2logL26 = n2logL_approx_eq26(cls_ana, cls_est)
        n2logL27 = n2logL_approx_eq27(cls_ana, cls_est)
        n2logLex = n2logL_exact(cls_ana, cls_est)

        lkfnc22.append(n2logL22)
        lkfnc23.append(n2logL23)
        lkfnc24.append(n2logL24)
        lkfnc25.append(n2logL25)
        lkfnc26.append(n2logL26)
        lkfnc27.append(n2logL27)
        lkfncEx.append(n2logLex)

        print (parname, "=", p, 
               ' n2logL22=', n2logL22, 
               ' n2logL23=', n2logL23,
               ' n2logL24=', n2logL24,
               ' n2logL25=', n2logL25,
               ' n2logL26=', n2logL26,
               ' n2logL27=', n2logL27,
               ' n2logLex=', n2logLex)

    minidx22 = np.argmin(lkfnc22)
    minidx23 = np.argmin(lkfnc23)
    minidx24 = np.argmin(lkfnc24)
    minidx25 = np.argmin(lkfnc25)
    minidx26 = np.argmin(lkfnc26)
    minidx27 = np.argmin(lkfnc27)
    minidxEx = np.argmin(lkfncEx)

    parmin22 = pars[minidx22]
    parmin23 = pars[minidx23]
    parmin24 = pars[minidx24]
    parmin25 = pars[minidx25]
    parmin26 = pars[minidx26]
    parmin27 = pars[minidx27]
    parminEx = pars[minidxEx]
   
    cls_ana22 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin22})
    cls_ana23 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin23})
    cls_ana24 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin24})
    cls_ana25 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin25})
    cls_ana26 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin26})
    cls_ana27 = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parmin27})
    cls_anaEx = get_spectrum_camb(lmax=lmax, TTonly=True, **{parname:parminEx})
    
    print ('Min lk at ', parname, "=\n", 
           parmin22, 
           "+", pars[np.argmin((np.array(lkfnc22[minidx22:])
                                - min(lkfnc22) - 1)**2) + minidx22] - parmin22, 
           "-", pars[np.argmin((np.array(lkfnc22[:minidx22])
                                - min(lkfnc22) - 1)**2)] *-1 + parmin22, 
           " for eq.(22) least sq = ", np.sum((cls_ana22 - cls_est)**2), "\n", 
           parmin23,  
           "+", pars[np.argmin((np.array(lkfnc23[minidx23:])
                                - min(lkfnc23) - 1)**2) + minidx23] - parmin23, 
           "-", pars[np.argmin((np.array(lkfnc23[:minidx23])
                                - min(lkfnc23) - 1)**2)] *-1 + parmin23, 
           " for eq.(23) least sq = ", np.sum((cls_ana23 - cls_est)**2), "\n", 
           parmin24,  
           "+", pars[np.argmin((np.array(lkfnc24[minidx24:])
                                - min(lkfnc24) - 1)**2) + minidx24] - parmin24,
           "-", pars[np.argmin((np.array(lkfnc24[:minidx24])
                                - min(lkfnc24) - 1)**2)] *-1 + parmin24, 
           " for eq.(24) least sq = ", np.sum((cls_ana24 - cls_est)**2), "\n", 
           parmin25,  
           "+", pars[np.argmin((np.array(lkfnc25[minidx25:])
                                - min(lkfnc25) - 1)**2) + minidx25] - parmin25,
           "-", pars[np.argmin((np.array(lkfnc25[:minidx25])
                                - min(lkfnc25) - 1)**2)] *-1 + parmin25, 
           " for eq.(25) least sq = ", np.sum((cls_ana25 - cls_est)**2), "\n", 
           parmin26,  
           "+", pars[np.argmin((np.array(lkfnc26[minidx26:])
                                - min(lkfnc26) - 1)**2) + minidx26] - parmin26,
           "-", pars[np.argmin((np.array(lkfnc26[:minidx26])
                                - min(lkfnc26) - 1)**2)] *-1 + parmin26, 
           " for eq.(26) least sq = ", np.sum((cls_ana26 - cls_est)**2), "\n", 
           parmin27,  
           "+", pars[np.argmin((np.array(lkfnc27[minidx27:])
                                - min(lkfnc27) - 1)**2) + minidx27] - parmin27,
           "-", pars[np.argmin((np.array(lkfnc27[:minidx27])
                                - min(lkfnc27) - 1)**2)] *-1 + parmin27, 
           " for eq.(27) least sq = ", np.sum((cls_ana27 - cls_est)**2), "\n", 
           parminEx, 
           "+", pars[np.argmin((np.array(lkfncEx[minidxEx:])
                                - min(lkfncEx)-1)**2) + minidxEx] - parminEx,
           "-", pars[np.argmin((np.array(lkfncEx[:minidxEx])    
                                - min(lkfncEx)-1)**2)] *-1 + parminEx, 
           " for exact likelihood least sq = ", np.sum((cls_anaEx - cls_est)**2), ) 
             
    ## plot likelihood functions
    plt.figure()
    plt.plot(pars, lkfnc22, label='eq. (22)')
    plt.plot(pars, lkfnc23, label='eq. (23)')
    plt.plot(pars, lkfnc24, label='eq. (24)')
    plt.plot(pars, lkfnc25, label='eq. (25)')
    plt.plot(pars, lkfnc26, label='eq. (26)')
    plt.plot(pars, lkfnc27, label='eq. (27)')
    plt.plot(pars, lkfncEx, label='Exact')
    
    plt.xlabel(parname)
    plt.ylabel('-2 log likelihood')
    plt.legend()

    ## plot power spectra
    plt.figure()
    ell = np.arange(len(cls_est))
    plt.loglog(ell[2:], cls_est[2:], label='estimated')
    plt.loglog(ell[2:], cls_ana0[2:], label='fiducial')
    plt.loglog(ell[2:], cls_ana22[2:], label='eq.(22)')
    plt.loglog(ell[2:], cls_ana23[2:], label='eq.(23)')
    plt.loglog(ell[2:], cls_ana24[2:], label='eq.(24)')
    plt.loglog(ell[2:], cls_ana25[2:], label='eq.(25)')
    plt.loglog(ell[2:], cls_ana26[2:], label='eq.(26)')
    plt.loglog(ell[2:], cls_ana27[2:], label='eq.(27)')
    plt.loglog(ell[2:], cls_anaEx[2:], label='Exact')
    plt.xlabel('multipole moment, l')
    plt.ylabel('D_l')
    plt.legend()

    plt.show()


## Test of likelihood fit using scipy minimizer 

def test_lkmin():
    print ('Calculating analytic spectrum')
    cls_ana0 = get_spectrum_camb(lmax=lmax)
    #cls_ana1 = get_spectrum_camb(lmax=lmax, ns=0.9)

    ## estimated power spectrum
    bls_ana0 = dl2cl(cls_ana0)
    m = hp.synfast(bls_ana0, nside, new=True, verbose=False)    
    cls_est = cl2dl(hp.anafast(m, lmax=lmax))

    print ('Calculating estimated spectrum')
    np.random.seed(42)
    m = hp.synfast(cls_ana0, nside, new=True, verbose=False)    
    cls_est = hp.anafast(m, lmax=lmax)

    ## log-likelihood
    """
    n2logL_approx_eqNN = n2logL_approx_eq22
    n2logL0 = n2logL_approx_eqNN(cls_ana0, cls_est)
    print ('Fiducial value of likelihood = ', n2logL0)
    """

    ## negative log likelihood minimization
    method = 'Nelder-Mead'
    method = 'BFGS'
    bnds = ((0.9,1.1),)
    tole  = 1e-10

    lkfunc = n2logL_approx_eq27
    starttime = time.time()
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print ('elapsed time = ', time.time() - starttime)
    print (res)

    return

    lkfunc = n2logL_approx_eq22
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    lkfunc = n2logL_approx_eq23
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)
    
    lkfunc = n2logL_approx_eq24
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    lkfunc = n2logL_approx_eq25
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    lkfunc = n2logL_approx_eq26
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    lkfunc = n2logL_approx_eq27
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    lkfunc = n2logL_exact
    res = minimize(lk_for_par, (0.90,), method=method, 
                   args=(cls_est, lkfunc), bounds=bnds, tol=tole)
    print ('likelihood fitting for', lkfunc.__name__)
    print (res)

    return res

             
## Test of likelihood fit using minuit

def test_lkminuit():
    As0 = 1.92e-9
    tau0 = 0.067

    print ('Calculating analytic spectrum')
    cls_ana0 = get_spectrum_camb(lmax=lmax, As=As0, tau=tau0)
    #cls_ana1 = get_spectrum_camb(lmax=lmax, ns=0.9)

    ## estimated power spectrum
    bls_ana0 = dl2cl(cls_ana0)
    m = hp.synfast(bls_ana0, nside, new=True, verbose=False)    
    cls_est = cl2dl(hp.anafast(m, lmax=lmax))

    print ('Calculating estimated spectrum')
    np.random.seed(42)
    m = hp.synfast(cls_ana0, nside, new=True, verbose=False)    
    cls_est = hp.anafast(m, lmax=lmax)

    ## log-likelihood
    """
    n2logL_approx_eqNN = n2logL_approx_eq22
    n2logL0 = n2logL_approx_eqNN(cls_ana0, cls_est)
    print ('Fiducial value of likelihood = ', n2logL0)
    """

    ## negative log likelihood minimization
    lkfunc = n2logL_approx_eq27
    print ('likelihood fitting using', lkfunc.__name__+'()')
    starttime = time.time()
    fmin, par = fit_minuit_AsTau(As0, tau0, cls_est, lkfunc)

    print ('elapsed time = ', time.time() - starttime)
    pprint (fmin)
    pprint (par)

    print ('end')

    return


## Test of three correlated fields fit

def test_n2logLf_TEB():
    K2uK = 1e12
    clscale = K2uK * 1.0
    cls_fid = get_spectrum_camb(lmax, isDl=False) * clscale

    cls_syn = get_spectrum_camb(lmax, tau=0.0522, As=2.092e-9, r=0.01, isDl=False) * clscale 
    map_syn = hp.synfast(cls_syn, nside=nside, new=True)
    cls_est = hp.anafast(map_syn, lmax=lmax)

    cls_ana = get_spectrum_camb(lmax, isDl=False) * clscale

    Cls_ana = cls2Cls(cls_ana)
    Cls_est = cls2Cls(cls_est)
    inv_Cls_fid, det_Cls_fid = invdet_fid(cls_fid)

    n2logLf = n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)

    print ('Likelihood for scale %e = %e' % (clscale, n2logLf))

    def fit_minuit_1(tau, As, r):
        cls_ana = get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False) * clscale
        Cls_ana = cls2Cls(cls_ana)
        lk = n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)
        print ('tau = %e, As = %e, r = %e, lk = %e' % (tau, As, r, lk)) 
        return lk

    tau0 = 0.0522
    tau_limit = (0.02, 0.08)
    As0 = 2.1e-9
    As_limit = (1.5e-9, 2.5e-9)
    r0 = 0.01
    r_limit = (0.0, 0.4)
    m = Minuit(fit_minuit_1, tau=tau0, As=As0, r=r0, 
               limit_tau=tau_limit, limit_As=As_limit, limit_r=r_limit)

    st = time.time()
    res = m.migrad()
    print ('Elapsed time for migrad: %fs' % (time.time()-st))

    st = time.time()
    res = m.hesse()
    print ('Elapsed time for hesse: %fs' % (time.time()-st))

    st = time.time()
    #res = m.minos()
    print ('Elapsed time for minos: %fs' % (time.time()-st))


    plt.figure()
    m.draw_profile('tau')
    plt.figure()
    m.draw_profile('As')
    plt.figure()
    m.draw_profile('r')

    tau_min = m.values['tau']
    tau_err = m.errors['tau']
    cls_min = get_spectrum_camb(lmax=lmax, tau=tau_min, isDl=False) * clscale
    cls_upp = get_spectrum_camb(lmax=lmax, tau=tau_min + tau_err, isDl=False) * clscale
    cls_low = get_spectrum_camb(lmax=lmax, tau=tau_min - tau_err, isDl=False) * clscale

    ell = np.arange(len(cls_est[0]))
    plt.figure()
    plt.loglog(ell, cl2dl(cls_est[:3].T), '*')
    plt.loglog(ell, cl2dl(cls_syn[:3].T), '--', linewidth=1.0)
    plt.loglog(ell, cl2dl(cls_min[:3].T), '-', linewidth=2.0)
    plt.loglog(ell, cl2dl(cls_upp[:3].T), '--', linewidth=0.5)
    plt.loglog(ell, cl2dl(cls_low[:3].T), '--', linewidth=0.5)
    
    pprint(res)
    plt.show()


## Test of 2 correlated fields fit (polarizations)
 
def test_n2logLf_EB():
    K2uK = 1e12
    clscale = K2uK * 1.0
    cls_fid = get_spectrum_camb(lmax, tau=0.05, As=2e-9, r=0.01, isDl=False) * clscale

    cls_syn = get_spectrum_camb(lmax, tau=0.0522, As=2.092e-9, r=0.01, isDl=False) * clscale 
    #np.random.seed(42)
    map_syn = hp.synfast(cls_syn, nside=nside, new=True)
    cls_est = hp.anafast(map_syn, lmax=lmax)

    cls_ana = get_spectrum_camb(lmax, isDl=False) * clscale

    Cls_ana = cls2Cls(cls_ana, T=False)
    Cls_est = cls2Cls(cls_est, T=False)
    Cls_syn = cls2Cls(cls_syn, T=False)
    inv_Cls_fid, det_Cls_fid = invdet_fid(cls_fid, T=False)

    n2logLf = n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)

    print ('Likelihood for scale %e = %e' % (clscale, n2logLf))

    def fit_minuit_1(tau, As, r):
        cls_ana = get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False) * clscale
        Cls_ana = cls2Cls(cls_ana, T=False)
        lk = n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)
        #lk = n2logL_approx_TEB(Cls_ana, Cls_syn, inv_Cls_fid, det_Cls_fid)
        print ('tau = %e, As = %e, r = %e, lk = %e' % (tau, As, r, lk)) 
        return lk

    tau0 = 0.052
    tau_limit = (0.02, 0.08)
    As0 = 2.09e-9
    As_limit = (1e-9, 3e-9)
    r0 = 0.009
    r_limit = (0.0, 0.1)
    m = Minuit(fit_minuit_1, tau=tau0, As=As0, r=r0, 
               limit_tau=tau_limit, limit_As=As_limit, limit_r=r_limit)

    st = time.time()
    res = m.migrad()
    print ('Elapsed time for migrad: %fs' % (time.time()-st))

    st = time.time()
    #res = m.hesse()
    print ('Elapsed time for hesse: %fs' % (time.time()-st))

    st = time.time()
    #res = m.minos()
    print ('Elapsed time for minos: %fs' % (time.time()-st))


    ''' 
    plt.figure()
    m.draw_profile('tau')
    plt.figure()
    m.draw_profile('As')
    plt.figure()
    m.draw_profile('r')

    tau_min = m.values['tau']
    tau_err = m.errors['tau']
    cls_min = get_spectrum_camb(lmax=lmax, tau=tau_min, isDl=False) * clscale
    cls_upp = get_spectrum_camb(lmax=lmax, tau=tau_min + tau_err, isDl=False) * clscale
    cls_low = get_spectrum_camb(lmax=lmax, tau=tau_min - tau_err, isDl=False) * clscale

    ell = np.arange(len(cls_est[0]))
    plt.figure()
    plt.loglog(ell, cl2dl(cls_est[:3].T), '*')
    plt.loglog(ell, cl2dl(cls_syn[:3].T), '--', linewidth=1.0)
    plt.loglog(ell, cl2dl(cls_min[:3].T), '-', linewidth=2.0)
    plt.loglog(ell, cl2dl(cls_upp[:3].T), '--', linewidth=0.5)
    plt.loglog(ell, cl2dl(cls_low[:3].T), '--', linewidth=0.5)
    '''
    
    pprint(res)
    #plt.show()
    return res

#==================================================
#
# Main
#
#==================================================


def main():
    #test_lkminuit()
    #test_lkmin()
    #test_lks()
    #test_n2logLf_TEB()
    #test_n2logLf_EB()
    pass

if __name__=='__main__':
    main()
