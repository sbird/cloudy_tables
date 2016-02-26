# -*- coding: utf-8 -*-
"""Make a plot showing the fraction of SiII with density and also the fraction of HI with density"""

from __future__ import print_function
import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import convert_cloudy as cc
import cold_gas as cg
import os.path as path
import numpy as np
from save_figure import save_figure

outdir = "testplots/"
print("Plots at: ",outdir)

def romanise_num(num):
    """Turn a number into a roman numeral (very badly)"""
    if num == 1:
        return "I"
    elif num == 2:
        return "II"
    elif num == 3:
        return "III"
    elif num == 4:
        return "IV"
    elif num == 5:
        return "V"
    elif num == 6:
        return "VI"
    else:
        return str(num)

def get_cloudy_table(ss_corr, redshift=3):
    """Helper function to load a table.
    ss_corr chooses which self-shielding correction to use.
    ss_corr == 3 is best. ss_corr == 4 disables self-shielding."""
    if ss_corr == 1:
        tab = cc.CloudyTable(redshift, "ion_out")
    elif ss_corr == 2:
        tab = cc.CloudyTable(redshift, "ion_out_fancy_atten")
    elif ss_corr == 3:
        tab = cc.CloudyTable(redshift, "ion_out_photo_atten")
    else:
        tab = cc.CloudyTable(redshift, "ion_out_no_atten")
    return tab

def save_plot(ss_corr, elem, ion, suffix):
    """Helper function to save a nicely named file"""
    filename = elem+str(ion)+"_"+suffix
    if ss_corr == 1:
        save_figure(path.join(outdir,filename))
    elif ss_corr == 2:
        save_figure(path.join(outdir,filename+"_fancy_atten"))
    elif ss_corr == 3:
        save_figure(path.join(outdir,filename+"_photo_atten"))
    else:
        save_figure(path.join(outdir,filename+"_no_atten"))

def plot_SivsHI(temp = 3e4, ss_corr=1, elem="Si", ion=2):
    """
        Plot the SiII fraction as a function of density, for some temperature.
        temp is an array, in K.
    """
    if np.size(temp) == 1:
        temp = np.array([temp,])
    tab = get_cloudy_table(ss_corr)
    #The hydrogen density in atoms/cm^3
    dens = np.logspace(-5,0,100)

    #Roughly mean DLA metallicity

    tabHI = cg.RahmatiRT(3, 0.71)

    fracHI = tabHI.neutral_fraction(dens,temp[0])
    plt.semilogx(dens, fracHI, color="red",ls="--", label="HI/H")

    ls = [":","-","-."]
    for tt in temp:
        ttSi = tt*np.ones_like(dens)
        fracSi = tab.ion(elem,ion,dens, ttSi)
        plt.semilogx(dens, fracSi, color="green",ls=ls.pop(), label=r"$"+str(int(tt/1e4))+r"\times 10^4$ K")

    plt.xlabel(r"$\rho_\mathrm{H}$ (cm$^{-3}$)")
    plt.ylabel(r"$\mathrm{m}_\mathrm{"+elem+romanise_num(ion)+r"} / \mathrm{m}_\mathrm{"+elem+r"}$")
    plt.ylim(0,1)
    plt.legend(loc=2)
    plt.show()
    save_plot(ss_corr, elem, ion,"fracs")
    plt.clf()


def plot_td_contour(ss_corr=1, elem="Si", ion=2, tlim=(3.5, 5.5), dlim=(-6,1), redshift=3):
    """
        Plot the ionic fraction as a function of density and temperature.
    """
    #Temperature
    nsamp = 400
    temp = np.logspace(tlim[0], tlim[1],nsamp)
    #The hydrogen density in atoms/cm^3
    dens = np.logspace(dlim[0], dlim[1],nsamp)

    tab = get_cloudy_table(ss_corr, redshift=redshift)

    dd, tt = np.meshgrid(dens, temp)
    ions = np.empty_like(dd)
    for i in range(nsamp):
        ions[:,i] = tab.ion(elem, ion, dd[:,i], tt[:,i])
    maxx = np.max(ions)
    levels = np.concatenate([[0.01,],  np.linspace(0.1, np.floor(10*maxx)/10., 4)])
    cont = plt.contour(dd, tt, ions,levels=levels)
    #Note this has to happen before the labels are drawn
    plt.yscale('log')
    plt.xscale('log')
    plt.clabel(cont,inline=True)

    plt.xlabel(r"$\rho_\mathrm{H}$ (cm$^{-3}$)")
    plt.ylabel(r"T (K)")
    plt.legend(loc=2)
    plt.show()
    save_plot(ss_corr, elem, ion, "contour_"+str(redshift))
    plt.clf()

if __name__ == "__main__":
    for atten in (3,4):
        plot_SivsHI([1e4, 2e4], atten, "Si", 2)
        plot_SivsHI([1e4, 2e4, 3e4], atten, "He", 1)
        plot_SivsHI([1e4, 2e4, 3e4], atten, "H", 1)
        plot_SivsHI([1e4, 2e4, 3e4], atten, "C", 2)
        plot_SivsHI([1e4, 2e4, 3e4], atten, "Mg", 2)
        plot_td_contour(atten, "C", 4)
        plot_td_contour(atten, "C", 3, dlim=(-5,1))
        plot_td_contour(atten, "O", 6, tlim=(3.5,6))
        plot_td_contour(atten, "Si", 4, dlim=(-5,1))
