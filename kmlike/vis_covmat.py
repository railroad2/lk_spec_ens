import pylab as plt

def show_cls(cls, ofname=None, Dl=False, newfig=True):
    ell = np.arange(len(cls[0]))

    if (newfig):
        plt.figure()

    #plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

    if (Dl):
        plt.loglog(ell, cls[0])
        plt.loglog(ell, cls[1])
        plt.loglog(ell, cls[2])
    else:
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[0])
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[1])
        plt.loglog(ell, (ell*(ell+1)/2./np.pi)*cls[2])

    plt.xlabel(r'Multipole moment, $l$')
    plt.ylabel(r'$\frac{l(l+1)}{2\pi} C_l$', fontsize=12)
    plt.legend(['TT', 'EE', 'BB'])

    if (not ofname==None):
        plt.savefig(fname)

def show_cov(cov, ofname=None, title=None):
    plt.matshow(cov)
    plt.colorbar()
    if (title != None)
        plt.title = title
