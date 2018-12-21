import os, fnmatch
import sys
import numpy as np
from fitgauss import varfit

def analysis():
    try:
        path = sys.argv[1]
    except:
        path = '.'

    flist = sorted(fnmatch.filter(os.listdir(path), '*.dat'))

    val_list = []
    sig_list = []

    pname = flist[0].split('_')[0]

    f = open(path+'/ens_'+pname+'_gauss', 'w')
    f.write('#'+pname+'_in\t\tval\t\t\t\terr_gauss\t\terr_like\n')

    for fn in flist:
        tmp = fn[:-4]
        unp = tmp.split('_')
        print (unp)
        pname = unp[0]
        val_in = float(unp[1])

        n, val, err, errh, errl, erru = np.genfromtxt(path+'/'+fn).T
        if pname == 'As':
            val_in *= 1e9
            val *= 1e9
            err *= 1e9
        m, s, sl = varfit(val, err, pname, path, par_in=val_in)

        f.write('%f\t%e\t%e\t%e\n' % (val_in, m, s, sl))

    return

if __name__=='__main__':
    analysis()

