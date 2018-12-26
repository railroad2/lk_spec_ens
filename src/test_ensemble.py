from __future__ import print_function
import numpy as np
import healpy as hp
import camb
import pylab as plt
import time
import os, sys

from pprint import pprint
from scipy.optimize import minimize
from iminuit import Minuit

from .kmlike.spectrum import get_spectrum_camb
from .kmlike.utils import cl2dl, dl2cl, print_error, print_debug
from .kmlike.covinv import detinv, detinv_pseudo
from . import lk_hamimeche as lh


# Global parameters
Tcmb = 2.7255
nside = 512
lmin = 2
lmax = 3*nside-1
if (lmax > 2000):
    lmax = 2000

DEBUG=0

#==================================================
#
# Test modules
#
#==================================================

## Test of three correlated fields fit
def test_n2logLf_TEB():
    K2uK = 1e12
    clscale = K2uK * 1.0
    cls_fid = get_spectrum_camb(lmax, isDl=False) * clscale

    cls_syn = get_spectrum_camb(lmax, tau=0.0522, As=2.092e-9, r=0.01, isDl=False) * clscale 
    map_syn = hp.synfast(cls_syn, nside=nside, new=True)
    cls_est = hp.anafast(map_syn, lmax=lmax)

    cls_ana = get_spectrum_camb(lmax, isDl=False) * clscale

    Cls_ana = lh.cls2Cls(cls_ana)
    Cls_est = lh.cls2Cls(cls_est)
    inv_Cls_fid, det_Cls_fid = lh.invdet_fid(cls_fid)

    n2logLf = n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)

    print ('Likelihood for scale %e = %e' % (clscale, n2logLf))

    def fit_minuit_1(tau, As, r):
        cls_ana = get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False) * clscale
        Cls_ana = lh.cls2Cls(cls_ana)
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
def test_n2logLf_EB(As_in=2.092e-9, tau_in=0.0522, r_in=0.01,
                    fix_As=False, fix_tau=False, fix_r=False, spec=False):
    K2uK = 1e12
    clscale = K2uK * 1.0
    # fiducial model
    cls_fid = get_spectrum_camb(lmax, tau=0.05, As=2e-9, r=0.05, isDl=False) * clscale

    # synthesize a random map
    print_debug(As_in, tau_in, r_in)
    cls_syn = get_spectrum_camb(lmax, tau=tau_in, As=As_in, r=r_in, isDl=False) * clscale 
    map_syn = hp.synfast(cls_syn, nside=nside, new=True)
    cls_est = hp.anafast(map_syn, lmax=lmax)

    # analytic 
    cls_ana = get_spectrum_camb(lmax, isDl=False) * clscale

    Cls_ana = lh.cls2Cls(cls_ana, T=False)
    Cls_est = lh.cls2Cls(cls_est, T=False)
    Cls_syn = lh.cls2Cls(cls_syn, T=False)
    inv_Cls_fid, det_Cls_fid = lh.invdet_fid(cls_fid, T=False)

    n2logLf = lh.n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)

    print ('Likelihood for scale %e = %e' % (clscale, n2logLf))

    def fit_minuit_1(tau, As, r):
        cls_ana = get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False) * clscale
        Cls_ana = lh.cls2Cls(cls_ana, T=False)
        lk = lh.n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)
        print ('tau = %e, As = %e, r = %e, lk = %e' % (tau, As, r, lk)) 
        return lk

    tau0 = tau_in * 0.99 
    tau_sig = 2e-2
    tau_limit = tau0 + 5*np.array((-tau_sig, tau_sig))
    if tau_limit[0] < 0:
        tau_limit[0] = 0

    As0 = As_in * 0.99
    As_sig = 1e-2
    As_limit = As0 + 5*np.array((-As_sig, As_sig))
    if As_limit[0] < 0:
        As_limit[0] = 0

    r0 = r_in * 0.99
    r_sig = 2e-2
    r_limit = r0 + 5*np.array((-r_sig, r_sig))
    if r_limit[0] < 0:
        r_limit[0] = 0

    if fix_tau:
        tau0 = tau_in

    if fix_As:
        As0 = As_in

    if fix_r:
        r0 = r_in
        

    # fit fnc check
    print (fit_minuit_1(tau0, As0, r0))

    m = Minuit(fit_minuit_1, tau=tau0, As=As0, r=r0, 
               limit_tau=tau_limit, limit_As=As_limit, limit_r=r_limit,
               fix_tau=fix_tau, fix_As=fix_As, fix_r=fix_r)

    st = time.time()
    res = m.migrad()
    print ('Elapsed time for migrad: %fs' % (time.time()-st))

    st = time.time()
    res_h = m.hesse()
    print ('Elapsed time for hesse: %fs' % (time.time()-st))

    st = time.time()
    res_m = m.minos()
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
    
    #pprint(res)
    #plt.show()
    if (spec == True):
        return res, res_h, res_m, cls_est/clscale
    else:
        return res, res_h, res_m


def test_n2logLf_EB_nonfid(As_in=2.092e-9, tau_in=0.0522, r_in=0.01,
                    fix_As=False, fix_tau=False, fix_r=False, spec=False):
    K2uK = 1e12
    clscale = K2uK * 1.0
    # fiducial model
    cls_fid = get_spectrum_camb(lmax, tau=0.05, As=2e-9, r=0.05, isDl=False) * clscale

    # synthesize a random map
    print_debug(As_in, tau_in, r_in)
    cls_syn = get_spectrum_camb(lmax, tau=tau_in, As=As_in, r=r_in, isDl=False) * clscale 
    map_syn = hp.synfast(cls_syn, nside=nside, new=True)
    cls_est = hp.anafast(map_syn, lmax=lmax)

    # analytic 
    cls_ana = get_spectrum_camb(lmax, isDl=False) * clscale

    Cls_ana = lh.cls2Cls(cls_ana, T=False)
    Cls_est = lh.cls2Cls(cls_est, T=False)
    Cls_syn = lh.cls2Cls(cls_syn, T=False)
    inv_Cls_fid, det_Cls_fid = lh.invdet_fid(cls_fid, T=False)

    n2logLf = lh.n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_fid, det_Cls_fid)

    print ('Likelihood for scale %e = %e' % (clscale, n2logLf))

    def fit_minuit_1(tau, As, r):
        cls_ana = get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False) * clscale
        inv_Cls_ana, det_Cls_ana = lh.invdet_fid(cls_ana, T=False)
        Cls_ana = lh.cls2Cls(cls_ana, T=False)
        lk = lh.n2logL_approx_TEB(Cls_ana, Cls_est, inv_Cls_ana, det_Cls_ana)
        print ('tau = %e, As = %e, r = %e, lk = %e' % (tau, As, r, lk)) 
        return lk

    tau0 = tau_in * 0.99 
    tau_sig = 2e-3
    tau_limit = tau0 + 5*np.array((-tau_sig, tau_sig))

    As0 = As_in * 0.99
    As_sig = 1e-10
    As_limit = As0 + 5*np.array((-As_sig, As_sig))

    r0 = r_in * 0.99
    r_sig = 1.5e-3
    r_limit = r0 + 5*np.array((-r_sig, r_sig))

    if fix_tau:
        tau0 = tau_in

    if fix_As:
        As0 = As_in

    if fix_r:
        r0 = r_in
        

    # fit fnc check
    print (fit_minuit_1(tau0, As0, r0))

    m = Minuit(fit_minuit_1, tau=tau0, As=As0, r=r0, 
               limit_tau=tau_limit, limit_As=As_limit, limit_r=r_limit,
               fix_tau=fix_tau, fix_As=fix_As, fix_r=fix_r)

    st = time.time()
    res = m.migrad()
    print ('Elapsed time for migrad: %fs' % (time.time()-st))

    st = time.time()
    res_h = m.hesse()
    print ('Elapsed time for hesse: %fs' % (time.time()-st))

    st = time.time()
    res_m = m.minos()
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
    
    #pprint(res)
    #plt.show()
    if (spec == True):
        return res, res_h, res_m, cls_est/clscale
    else:
        return res, res_h, res_m


## Ensemble test
def test_ensemble(pname='As', par_in=2.09e-9, ntest=1, dir='.', specplt=False, nonfid=False):
    f=open(dir + '/' + pname + ('_%2.3e.dat' % par_in), 'w')
    f.write("#\t" + pname + "_val\tmigrad_err\thesse_err\tminos_lower\tminos_upper\n")

    As = 2.092e-9
    tau = 0.0522
    r = 0.01
    fix_As = fix_tau = fix_r = True

    cls_arr = []

    for i in range(ntest):
        try:
            if pname=='As':
                As = par_in
                fix_As = False
                pn = 1
            elif pname=='tau':
                tau = par_in
                fix_tau = False
                pn = 0
            elif pname=='r':
                r = par_in
                fix_r = False
                pn = 2
            else:
                print_error("Invaild parameter name")
                return

            callfnc = test_n2logLf_EB
            if nonfid:
                callfnc = test_n2logLf_EB_nonfid

            if (specplt == True):
                res, res_h, res_m, cls_est = callfnc(As_in=As, tau_in=tau, r_in=r,
                                                     fix_As=fix_As, fix_tau=fix_tau, fix_r=fix_r, 
                                                     spec=True)
                cls_arr.append(cls_est)
            else:
                res, res_h, res_m = callfnc(As_in=As, tau_in=tau, r_in=r,
                                            fix_As=fix_As, fix_tau=fix_tau, fix_r=fix_r)

            val = res[1][pn]['value']
            err = res[1][pn]['error']
            errh = res_h[pn]['error']
            errl = res_m[pname]['lower']
            erru = res_m[pname]['upper']

            f.write("%d\t%e\t%e\t%e\t%e\t%e\n" % 
                    (i, val, err, errh, errl, erru))
            f.flush()

        except:
            e = sys.exc_info()[0]
            print_error(('Exception: %s occured at ' % e) + pname + ('_in=%2.3e' % (par_in)))
            print (sys.exc_info())
            continue

    if (specplt == True):
        plt.figure()
        ell = np.arange(lmax+1)
        cls_ana = cl2dl(get_spectrum_camb(lmax=lmax, tau=tau, As=As, r=r, isDl=False))

        for cls_est in cls_arr:
            cls_est = cl2dl(cls_est)
            plt.loglog(ell[2:], cls_est[:3].T[2:], linewidth=0.5)

        plt.loglog(ell[2:], cls_ana[:3].T[2:])
        plt.xlabel('Multipole moment, $l$')
        plt.ylabel('$D_l (K ^2)$')

        plt.show()
        plt.savefig(dir + '/' + pname + ('_%2.3e.png' % par_in))

    f.close()


def test_ensemble_write(pname='As', par_in=2.09e-9, ntest=1, dir='.'):
    f=open(dir + '/' + pname + ('_%2.3e.dat' % par_in), 'w')
    f.write("#\t" + pname + "_val\tmigrad_err\thesse_err\tminos_lower\tminos_upper\n")

    for i in range(ntest):
        f.write("%d\t%e\t%e\t%e\t%e\t%e\n" % 
                (1, 2, 3, 4, 5, 6))

    f.close()


#==================================================
#
# Main
#
#==================================================

def ens_As(pars=None, lmax_in=1535, ntest=10, specplt=False, nonfid=False, dir=None):
    global lmax 
    lmax = lmax_in

    if pars is None:
        parr = np.arange(1.0, 4.0, 0.5)*1e-9
    else:
        parr = np.array(pars)

    print ('As=', parr)

    if dir is None:
        dir = ('./ensemble_lmax%d_As_' % lmax) + time.strftime('%Y-%m-%d_%X')
    else:
        dir = dir + ('/ensemble_lmax%d_As_' % lmax) + time.strftime('%Y-%m-%d_%X')

    if not os.path.isdir(dir):
        os.mkdir(dir)
        print (dir, 'has been created.')

    for par_in in parr:
        test_ensemble(pname='As', par_in=par_in, ntest=ntest, dir=dir, specplt=specplt, nonfid=nonfid)


def ens_tau(pars=None, lmax_in=1535, ntest=10, specplt=False, nonfid=False, dir=None):
    global lmax 
    lmax = lmax_in

    if pars is None:
        parr = np.arange(0.01, 0.1, 0.01)
    else:
        parr = np.aray(pars)

    print ('tau=', parr)

    if dir is None:
        dir = ('./ensemble_lmax%d_tau_' % lmax) + time.strftime('%Y-%m-%d_%X')
    else:
        dir = dir + ('/ensemble_lmax%d_tau_' % lmax) + time.strftime('%Y-%m-%d_%X')


    if not os.path.isdir(dir):
        os.mkdir(dir)
        print (dir, 'has been created.')

    for par_in in parr:
        test_ensemble(pname='tau', par_in=par_in, ntest=ntest, dir=dir, specplt=specplt, nonfid=nonfid)


def ens_r(pars=None, lmax_in=1535, ntest=10, specplt=False, nonfid=False, dir=None):
    global lmax 
    lmax = lmax_in

    if pars is None:
        parr = np.arange(0.0, 0.1, 0.01)
    else:
        parr = np.array(pars)

    print ('r=', parr)

    if dir is None:
        dir = ('./ensemble_lmax%d_r_' % lmax) + time.strftime('%Y-%m-%d_%X')
    else:
        dir = dir + ('/ensemble_lmax%d_r_' % lmax) + time.strftime('%Y-%m-%d_%X')
    if not os.path.isdir(dir):
        os.mkdir(dir)
        print (dir, 'has been created.')

    for par_in in parr:
        print (par_in)
        test_ensemble(pname='r', par_in=par_in, ntest=ntest, dir=dir, specplt=specplt, nonfid=nonfid)


def main():
    ens_As()

if __name__=='__main__':
    main()

