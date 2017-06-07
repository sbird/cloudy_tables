# -*- coding: utf-8 -*-
"""Generate parameter files for cloudy, then run cloudy a number of times to generate a folder tree of files"""
import math
import subprocess
import os
import os.path as path
import multiprocessing as mp
import functools
import numpy as np
import photocs
import fake_spectra.gas_properties as gas_properties

def load_uvb_table(redshift, uvb_path="UVB_tables"):
    """Load one of Claude's UVB tables, and convert to Cloudy units"""
    uvb_file = path.join(uvb_path, "fg_uvb_dec11_z_"+str(float(redshift))+".dat")
    #Columns: nu (Ryd)        Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)
    uvb_table = np.loadtxt(uvb_file)

    #Cloudy continnum input 4 pi J_\nu [erg cm^-2 s^-1 Hz^-1]
    #so convert units
    uvb_table[:,1] = np.log10(4*math.pi*uvb_table[:,1])-21
    return uvb_table

class UVBAtten(gas_properties.GasProperties):
    """Attenuate the UVB using the self-shielding prescription of Rahmati 2013"""

    def atten(self,hden, temp):
        """Compute the reduction in the photoionisation rate at an energy of 13.6 eV
        at a given density and temperature, using the Rahmati fitting formula.
        Note the Rahmati formula is based on the FG09 UVB; if you use a different UVB,
        the self-shielding critical density will change somewhat.

        For z < 5 the UVB is probably known well enough that not much will change, but for z > 5
        the UVB is highly uncertain; any conclusions about cold gas absorbers at these redshifts
        need to marginalise over the UVB amplitude here.
        """
        return self._photo_rate(10**hden, 10**temp)/self.gamma_UVB

def gen_cloudy_uvb_shape_atten(uvb_table, redshift, hden,temp, uvb_factor=1.):
    """
    Generate the cloudy input string from a UVB table, reducing the UVB amplitude at energies >= 13.6eV
    by a self-shielding factor.

    The self-shielding factor at 13.6 eV is given by the Rahmati fitting formula.
    At higher energies the HI cross-section reduces like ν^-3.

    Account for this by noting that self-shielding happens when τ = 1, ie
    τ = n σ L = 1.
    Thus a lower cross-section requires higher densities.
    Assume then that HI self-shielding is really a function of τ, and thus at a frequency ν,
    the self-shielding factor can be computed by working out the optical depth for the
    equivalent density at 13.6 eV. ie, for Γ(n, T), account for frequency dependence with:

    Γ( n / (σ(13.6) / σ(ν) ), T).

    (so that a lower x-section leads to a lower effective density)

    Note Rydberg ~ 1/wavelength, and 1 Rydberg is the energy of a photon at the Lyman limit, ie,
    with wavelength 911.8 Angstrom.
    """
    UVB = UVBAtten(redshift,None)
    #Attenuate the UVB by an amount dependent on the hydrogen
    #photoionisation cross-section for hydrogen as a function of frequency.
    #This is zero for energies less than 13.6 eV = 1 Ryd, and then falls off like E^-3
    #Normalise the profile in terms of 1 Ryd, where the radiative transfer was calculated originally.
    hics = photocs.hyd.photo(13.6*uvb_table[:,0])/photocs.hyd.photo(13.6)
    #Compute adjusted UVB table
    ind = np.where(hics > 0)
    uvb_table[ind,1] += np.log10(UVB.atten(hden+np.log10(hics[ind]), temp))
    #First output very small background at low energies
    uvb_str = "interpolate ( 0.00000001001 , -35.0)\n"
    uvb_str+="continue ("+str(uvb_table[0,0]*0.99999)+", -35.0)\n"
    #Then output main body
    for xx in range(np.shape(uvb_table)[0]):
        uvb_str+="continue ("+str(uvb_table[xx,0])+" , "+str(uvb_table[xx,1])+" )\n"
    #Then output zero background at high energies
    uvb_str+="continue ("+str(uvb_table[-1,0]*1.0001)+" , -35.0 )\n"
    uvb_str+="continue ( 7354000.0 , -35.0 ) \n"
    #That was the UVB shape, now print the amplitude
    uvb_amp = uvb_table[0,1]+ np.log10(uvb_factor)
    uvb_str+="f(nu)="+str(uvb_amp)+" at "+str(uvb_table[0,0])+" Ryd\n"
    return uvb_str


def gen_cloudy_uvb(uvb_table, redshift, hden,temp, atten=True, uvb_factor=1.):
    """Generate the cloudy input string from a UVB table"""
    UVB = UVBAtten(redshift, None)
    #First output very small background at low energies
    uvb_str = "interpolate ( 0.00000001001 , -35.0)\n"
    uvb_str+="continue ("+str(uvb_table[0,0]*0.99999)+", -35.0)\n"
    #Then output main body
    for xx in range(np.shape(uvb_table)[0]):
        uvb_str+="continue ("+str(uvb_table[xx,0])+" , "+str(uvb_table[xx,1])+" )\n"
    #Then output zero background at high energies
    uvb_str+="continue ("+str(uvb_table[-1,0]*1.0001)+" , -35.0 )\n"
    uvb_str+="continue ( 7354000.0 , -35.0 ) \n"
    #That was the UVB shape, now print the amplitude
    uvb_amp = uvb_table[0,1]+ np.log10(uvb_factor)
    if atten:
        uvb_amp += np.log10(UVB.atten(hden, temp))
    uvb_str+="f(nu)="+str(uvb_amp)+" at "+str(uvb_table[0,0])+" Ryd\n"
    return uvb_str

def output_cloudy_config(redshift, hden, metals, temp, atten=True, tdir="ion_out_photo_atten",outfile="cloudy_param.in", uvb_factor=1.):
    """Generate a cloudy config file with the given options, in directory outdir/zz(redshift).
    If atten=True, reduce the UVb at high frequencies to account for self-shielding by neutral hydrogen.
    The overall UVB amplitude is multiplied by uvb_factor."""

    real_outdir = outdir(redshift, hden, temp, tdir)
    out = open(path.join(real_outdir,outfile),'w')
    #header with general cloudy parameters
    header="""no molecules                            #turn off all molecules, only atomic cooling
no induced processes                    #to follow Wiersma et al.
stop zone 1
iterate to convergence
abundances GASS10
#no free free
#no collisional ionization
#no Compton effect\n"""
    out.write(header)

    #Get the UVB table
    uvb_table = load_uvb_table(redshift)
    if atten:
        uvb_str = gen_cloudy_uvb_shape_atten(uvb_table, redshift, hden,temp, uvb_factor=uvb_factor)
    else:
        uvb_str = gen_cloudy_uvb(uvb_table, redshift, hden,temp,atten=False, uvb_factor=uvb_factor)

    #Print UVB
    out.write(uvb_str)
    #Print output options
    out.write("hden "+str(hden)+" log\n")
    out.write("metals "+str(metals)+" log\n")
    out.write("constant temperature "+str(temp)+" log\n")
    out.write("save last ionization means \"ionization.dat\"\n")
    return real_outdir


def cloudy_job(redshift, hden, outfile="mpi_submit_script"):
    """Start a cloudy jobfile"""
    #Cloudy appends the ".in"
    jobstring = """#!/bin/bash
#$ -N Cloudy-tables-sshort
#$ -l h_rt=1:00:00,exclusive=true
#$ -j y
#$ -cwd
#$ -pe orte 1
#$ -m bae
#$ -V
export OMP_NUM_THREADS=1
LOCAL=/home/spb/.local
MPI=/usr/local/openmpi/intel/x86_64
FFTW=$LOCAL/fftw
export PYTHONPATH=$HOME/.local/python
export LD_LIBRARY_PATH=${MPI}/lib:${LD_LIBRARY_PATH}:$FFTW/lib:/usr/lib64
export LIBRARY_PATH=${MPI}/lib:${LIBRARY_PATH}:$FFTW/lib:/usr/lib64
export PATH=${MPI}/bin:$PATH:$LOCAL/misc/bin
    """
    out=open(outfile,'w')
    out.write(jobstring)
    for temp in np.arange(3.5,8):
        real_outdir= outdir(redshift, hden, temp)
        jobstr="./cloudy.exe -r "
        paramfile = path.join(real_outdir, "cloudy_param")
        jobstr+=paramfile
        jobstr+="\n"
        out.write(jobstr)

def outdir(redshift, hden, temp, tdir="ion_out"):
    """Get the directory for output, and make sure it exists"""
    true_outdir= path.join(tdir,"zz"+str(redshift))
    if np.abs(hden) < 1.e-3:
        paramdir = path.join("h0.0","T"+str(temp))
    else:
        paramdir = path.join("h"+str(hden),"T"+str(temp))
    real_outdir = path.join(true_outdir, paramdir)
    try:
        os.makedirs(real_outdir)
    except OSError:
        pass
    return real_outdir

def gen_density(hhden, atten=True, tdir = "ion_out_photo_atten", rrange=None, uvb_factor=1.):
    """Generate tables at given density"""
    if rrange is None:
        rrange = np.arange(7,-1,-1)
    for rredshift in rrange:
        for ttemp in np.arange(3.,8.6,0.05):
            ooutdir = output_cloudy_config(rredshift, hhden, -1, ttemp,atten=atten,tdir=tdir, uvb_factor=uvb_factor)
            cloudy_exe = path.join(os.getcwd(),"cloudy.exe")
            if not path.exists(path.join(ooutdir, "ionization.dat")):
                subprocess.call([cloudy_exe, '-r', "cloudy_param"],cwd=ooutdir)

def make_tables(processes=32, atten=True, tdir="ion_out_photo_atten", rrange=None,uvb_factor=1.):
    """Make a table using a parallel multiprocessing pool."""
    pool = mp.Pool(processes=processes)
    f = functools.partial(gen_density,atten=atten, tdir=tdir, rrange=rrange,uvb_factor=uvb_factor)
    pool.map(f,np.arange(-7.,4.,0.2))

if __name__ == "__main__":
    #Make the default table
    make_tables(atten=True, tdir="ion_out_photo_atten")
    #Make the table without self-shielding
    make_tables(atten=False, tdir="ion_out_no_atten")
