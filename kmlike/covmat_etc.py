import numpy as np

def getcov_diag(npix, var):
    cov = np.eye(npix) * var
    return cov


