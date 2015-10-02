"""Small module to compute photoionisation rates using the table from the Opacity Project: astro-ph/9601009"""

import numpy as np

#For hydrogen
class PIParams(object):
    """Class for computing photoionisation cross-sections from the Opacity Project.
    Values are from Table 1 of astro-ph/9601009"""
    def __init__(self, nuthr, nu0, sigma0, ya, P, yw, y0, y1):
        self.nuthr = nuthr
        self.nu0 = nu0
        #Convert from Mb to cm^2
        self.sigma0 = 1e-18*sigma0
        self.ya = ya
        self.Pp = P
        self.yw = yw
        self.y0 = y0
        self.y1 = y1

    def photo(self,nu):
        """Find photoionisation cross-section in cm^2 from Verner et al 1996"""
        cross = np.zeros_like(nu)
        x = nu / self.nu0 - self.y0
        y = np.sqrt(x**2 + self.y1**2)
        Ff = ((x-1)**2 + self.yw**2)*y**(0.5*self.Pp-5.5)*(1+np.sqrt(y/self.ya))**(-self.Pp)
        #Deal explicitly with a single number, as numpy gets confused
        if np.size(nu) == 1:
            return self.sigma0*Ff*(nu >= self.nuthr)
        ind = np.where(nu >= self.nuthr)
        if np.size(ind) > 0:
            cross[ind] = self.sigma0*Ff[ind]
        return cross

hyd = PIParams(nuthr = 1.36e+1, nu0 = 4.298e-1, sigma0 = 5.475e+4, ya = 3.288e+1, P = 2.963, yw = 0, y0 = 0, y1 = 0)

si = PIParams(nuthr = 1.635e+1, nu0 = 2.556, sigma0 = 4.140, ya = 1.337e+1, P = 1.191e+1, yw = 1.56, y0 = 6.634, y1 = 1.272e-1)


