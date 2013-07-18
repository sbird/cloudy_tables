# -*- coding: utf-8 -*-
"""Make a plot showing the fraction of SiII with density and also the fraction of HI with density"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import convert_cloudy as cc
import cold_gas as cg
import os.path as path
import numpy as np
from save_figure import save_figure

outdir = "testplots/"
print "Plots at: ",outdir

def plot_SivsHI(temp = 3e4, atten=1, elem="Si", ion=2):
    """
        Plot the SiII fraction as a function of density, for some temperature.
        temp is an array, in K.
    """
    if np.size(temp) == 1:
        temp = np.array([temp,])
    if atten == 1:
        tab = cc.CloudyTable(3)
    elif atten == 2:
        tab = cc.CloudyTable(3, "ion_out_fancy_atten")
    else:
        tab = cc.CloudyTable(3, "ion_out_no_atten")

    #The hydrogen density in atoms/cm^3
    dens = np.logspace(-5,2,100)

    #Roughly mean DLA metallicity

    tabHI = cg.RahmatiRT(3, 0.71)

    fracHI = tabHI.neutral_fraction(dens,temp[0])
    plt.semilogx(dens, fracHI, color="red",ls="--")

    ls = [":","-","-."]
    for tt in temp:
        ttSi = tt*np.ones_like(dens)
        fracSi = tab.ion(elem,ion,dens, ttSi)
        plt.semilogx(dens, fracSi, color="green",ls=ls.pop())

    plt.xlabel(r"$\rho_\mathrm{H}\; (\mathrm{amu}/\mathrm{cm}^3$)")
    plt.ylabel(r"$\mathrm{m}_\mathrm{SiII} / \mathrm{m}_\mathrm{Si}$")
    plt.show()
    if atten == 1:
        save_figure(path.join(outdir,elem+"_fracs"))
    elif atten == 2:
        save_figure(path.join(outdir,elem+"_fracs_fancy_atten"))
    else:
        save_figure(path.join(outdir,elem+"_fracs_no_atten"))
    plt.clf()

plot_SivsHI([1e4, 2e4, 3e4], 1)
plot_SivsHI([1e4, 2e4, 3e4], 0)
plot_SivsHI([1e4, 2e4, 3e4], 2)
plot_SivsHI([1e4, 2e4, 3e4], 1, "He", 1)
plot_SivsHI([1e4, 2e4, 3e4], 0, "He", 1)
plot_SivsHI([1e4, 2e4, 3e4], 2, "He", 1)
plot_SivsHI([1e4, 2e4, 3e4], 1, "H", 1)
plot_SivsHI([1e4, 2e4, 3e4], 0, "H", 1)
plot_SivsHI([1e4, 2e4, 3e4], 2, "H", 1)
