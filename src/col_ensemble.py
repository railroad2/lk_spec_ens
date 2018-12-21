import numpy as np
import pylab as plt
import iminuit 
import sys
import kmlike.spectrum as sp

def linefnc(x, a, b):
    return a*x + b

def linefit(x, y, yerr, fix_b=False):
    a0 = 1.
    b0 = 0.
    def fnc(a, b): 
        chisq = sum((linefnc(x, a, b)-y)**2/yerr**2)
        return chisq

    mi = iminuit.Minuit(fnc, a=a0, b=b0, error_a=0.1, error_b=0.1, fix_b=fix_b)

    mi.migrad()

    a = mi.values['a']
    a_err = mi.errors['a']
    b = mi.values['b']
    b_err = mi.errors['b']

    return a, b, a_err, b_err

def plot_spec(parname, pars, lmax=1000):
    ell = np.arange(lmax+1) 

    if (parname=='As'):
        pars *= 1e-9
    
    plt.figure()
    kwargs = {}
    scaleEE = []
    scaleBB = []

    for par in pars:
        kwargs['As'] = par 
        kwargs['tau'] = 0.0522
        kwargs['r'] =0.01
        dls = sp.get_spectrum_camb(lmax, **kwargs)[:3].T
        scaleEE.append(dls[2,1])
        scaleBB.append(dls[2,2])
        print (scaleEE[-1]/scaleBB[-1])
        plt.loglog(ell[2:], dls[2:])

    plt.xlabel('Multipolt moment, $l$')
    plt.ylabel('$D_l (\mu K)$')

    return scaleEE, scaleBB

def col_ensemble(parname=None):
    if (parname==None):
        parname = 'As'

    fname = sys.argv[1]

    val_in, val_fit, err_gauss, err_like = np.genfromtxt(fname)[:-6].T

    plt.rcParams.update({'font.size':15})
    plt.errorbar(val_in, val_fit, xerr=0, yerr=err_like, 
                 marker='', linestyle='', capsize=5, label='Likelihood fit errors')
    plt.errorbar(val_in, val_fit, xerr=0, yerr=err_gauss, 
                 marker='', linestyle='', capsize=3, label='Gaussian fit error')
    plt.plot(val_in, val_fit,
                 marker='s', linestyle='', color='black', mfc='k', markersize=5) 

    a, b, a_err, b_err = linefit(val_in, val_fit, err_gauss, fix_b=False)
    plt.plot(val_in, a*val_in, label='(%f$\pm$%f)x+(%f$\pm$%f)' % (a, a_err, b, b_err))
    #plt.xlabel('Input $10^9 A_s$')
    #plt.ylabel('Fit $10^9 A_s$')
    plt.xlabel('As_in')
    plt.ylabel('As_fit')
    plt.legend()

    scaleEE, scaleBB = plot_spec(parname, val_fit)

    plt.figure()
    plt.plot(val_fit, err_gauss)
    plt.xlabel('As_fit')
    plt.ylabel('Gaussian fit error')

    plt.figure()
    plt.plot(val_fit, scaleEE)
    plt.xlabel('As_fit')
    plt.ylabel('$D_2^{EE}$')

    plt.figure()
    plt.plot(scaleEE, err_gauss)
    plt.xlabel('$D_2^{EE}$')
    plt.ylabel('Gaussian fit error')

    plt.figure()
    plt.plot(scaleBB, err_gauss)
    plt.xlabel('D_2^{BB}')
    plt.ylabel('Gaussian fit error')

    plt.show()

if __name__=='__main__':
    col_ensemble(parname='As')
