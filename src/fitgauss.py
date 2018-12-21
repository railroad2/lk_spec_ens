import sys
import numpy as np
import pylab as plt
import iminuit

def gauss(x, a, m, s):
    return a/(2*np.pi)**0.5/s*np.exp(-(x - m)**2/2/s**2)

def fitgauss(x, y):
    a0=np.mean(y)
    m0=np.mean(x)
    s0=np.std(x)

    print(a0, m0, s0)

    def fnc(amp, mean, sig):
        ls = sum((gauss(x, amp, mean, sig) - y)**2)

        return ls 

    mi = iminuit.Minuit(fnc, amp=a0, mean=m0, sig=s0, 
                        error_amp=a0/5, error_mean=m0/5, error_sig=s0/5)
    mi.migrad()

    a = mi.values['amp']
    m = mi.values['mean']
    s = mi.values['sig']

    return a, m, s

def varfit(dat, dat_sig, pname, path='./', par_in=None):
    plt.figure()
    N, bin_edge, __ = plt.hist(dat, bins=20) 
    plt.xlabel(pname)

    bins = (bin_edge[:-1] + bin_edge[1:])/2
    a, m, s = fitgauss(bins, N)

    x = np.linspace(bins[0], bins[-1], 100)
    plt.plot(x, gauss(x, a, m, s))
    if (par_in == None):
        m_in = m
    else:
        m_in = par_in

    plt.savefig(path+'/%s_%2.3e_gauss.png' % (pname, m_in))

    plt.figure()
    plt.hist(dat_sig, bins=200)
    plt.xlabel(pname+'_sigma')
    plt.savefig(path+'/%s_%2.3e_sigma.png' % (pname, m_in))

    print ('mean('+pname+'_sigma)=', np.mean(dat_sig))
    print ('std('+pname+'_sigma)=',  np.std(dat_sig))

    return m, s, np.mean(dat_sig)

def main():
    fn = sys.argv[1]
    data = np.genfromtxt(fn).T

    tau = data[0]
    tau_sig = data[1]
    varfit(tau, tau_sig, 'tau')
    
    As = data[2] * 1e9
    As_sig = data[3] * 1e9
    varfit(As, As_sig, 'As')

    r = data[4]
    r_sig = data[5]
    varfit(r, r_sig, 'r')

    plt.show()

def main_2():
    fn = sys.argv[1]
    data = np.genfromtxt(fn).T

    As = data[1] * 1e9
    As_sig = data[2] * 1e9
    As_sig_h = data[3] * 1e9
    As_sig_l = data[4] * 1e9
    As_sig_u = data[5] * 1e9
    varfit(As, As_sig, 'As')

    print ('mean(As_sigma)=', np.mean(As_sig))
    print ('std(As_sigma)=',  np.std(As_sig))

    print ('mean(As_sigma_hesse)=', np.mean(As_sig_h))
    print ('std(As_sigma_hesse)=',  np.std(As_sig_h))

    print ('mean(As_sigma_low)=', np.mean(As_sig_l))
    print ('std(As_sigma_low)=',  np.std(As_sig_l))

    print ('mean(As_sigma_up)=', np.mean(As_sig_u))
    print ('std(As_sigma_up)=',  np.std(As_sig_u))

    plt.show()

if __name__=='__main__':
    main_2()
