# -*- coding: utf-8 -*-
"""Generate parameter files for cloudy, then run cloudy a number of times to generate a folder tree of files"""
import os.path as path
import numpy as np
import math
import subprocess
import os
import multiprocessing as mp
import cold_gas

def load_uvb_table(redshift, uvb_path="UVB_tables"):
    """Load one of Claude's UVB tables, and convert to Cloudy units"""
    uvb_file = path.join(uvb_path, "fg_uvb_dec11_z_"+str(float(redshift))+".dat")
    #Columns: nu (Ryd)        Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)
    uvb_table = np.loadtxt(uvb_file)

    #Cloudy continnum input 4 pi J_\nu [erg cm^-2 s^-1 Hz^-1]
    #so convert units
    uvb_table[:,1] = np.log10(4*math.pi*uvb_table[:,1])-21
    return uvb_table

class UVBAtten(cold_gas.RahmatiRT):
    """Attenuate the UVB using the self-shielding prescription of Rahmati 2013"""

    def atten(self,hden, temp):
        """Compute the UVB attenuation factor at high densities.
        Reduce the UVB intensity by the same factor that Rahmati
        reduces the photoionisation rate by."""
        #Note: the UVB is not attentuated at high hydrogen densities.
        #It should only be attenuated for a particular frequency band around 1217 A.
        #Doing this right is really too complicated.
        return self.photo_rate(10**hden, 10**temp)/self.gamma_UVB

def gen_cloudy_uvb_shape_atten(uvb_table, redshift, hden,temp):
    """
    Generate the cloudy input string from a UVB table, with slightly more physical attenuation.
    Attenuate uniformly (following the Rahmati prescription) all photons at higher energy than Lyman alpha.
    Note Rydberg ~ 1/wavelength, and 1 Rydberg is the energy of a photon at the Lyman limit, ie,
    with wavelength 911.8 Angstrom.
    """
    UVB = UVBAtten(redshift)
    waveuvb = 911.8/uvb_table[:,0]
    #Attenuate uniformly the part of the UVB above Lya
    profile = np.zeros_like(uvb_table[:,0])
    ind = np.where((waveuvb < 1280))
    profile[ind]=1.
    #Compute adjusted UVB table
    uvb_table[:,1] += np.log10(UVB.atten(hden, temp))*profile
    #First output very small background at low energies
    uvb_str = "interpolate ( 0.00000001001 , -35.0)\n"
    uvb_str+="continue ("+str(uvb_table[0,0]*0.99999)+", -35.0)\n"
    #Then output main body
    for xx in xrange(np.shape(uvb_table)[0]):
        uvb_str+="continue ("+str(uvb_table[xx,0])+" , "+str(uvb_table[xx,1])+" )\n"
    #Then output zero background at high energies
    uvb_str+="continue ("+str(uvb_table[-1,0]*1.0001)+" , -35.0 )\n"
    uvb_str+="continue ( 7354000.0 , -35.0 ) \n"
    #That was the UVB shape, now print the amplitude
    uvb_str+="f(nu)="+str(uvb_table[0,1])+" at "+str(uvb_table[0,0])+" Ryd\n"
    return uvb_str


def gen_cloudy_uvb(uvb_table, redshift, hden,temp, atten=True):
    """Generate the cloudy input string from a UVB table"""
    UVB = UVBAtten(redshift)
    #First output very small background at low energies
    uvb_str = "interpolate ( 0.00000001001 , -35.0)\n"
    uvb_str+="continue ("+str(uvb_table[0,0]*0.99999)+", -35.0)\n"
    #Then output main body
    for xx in xrange(np.shape(uvb_table)[0]):
        uvb_str+="continue ("+str(uvb_table[xx,0])+" , "+str(uvb_table[xx,1])+" )\n"
    #Then output zero background at high energies
    uvb_str+="continue ("+str(uvb_table[-1,0]*1.0001)+" , -35.0 )\n"
    uvb_str+="continue ( 7354000.0 , -35.0 ) \n"
    #That was the UVB shape, now print the amplitude
    if atten:
        uvb_str+="f(nu)="+str(uvb_table[0,1]+np.log10(UVB.atten(hden,temp)))+" at "+str(uvb_table[0,0])+" Ryd\n"
    else:
        uvb_str+="f(nu)="+str(uvb_table[0,1])+" at "+str(uvb_table[0,0])+" Ryd\n"
    return uvb_str

def output_cloudy_config(redshift, hden, metals, temp, atten=1, tdir="ion_out",outfile="cloudy_param.in"):
    """Generate a cloudy config file with the given options, in directory outdir/zz(redshift)"""

    real_outdir = outdir(redshift, hden, temp, tdir)
    out = open(path.join(real_outdir,outfile),'w')
    #header with general cloudy parameters
    header="""no molecules                            #turn off all molecules, only atomic cooling
no induced processes                    #to follow Wiersma et al.
stop zone 1
iterate to convergence
#no free free
#no collisional ionization
#no Compton effect\n"""
    out.write(header)

    #Get the UVB table
    uvb_table = load_uvb_table(redshift)
    if atten < 2:
        uvb_str = gen_cloudy_uvb(uvb_table, redshift, hden,temp,atten)
    else:
        uvb_str = gen_cloudy_uvb_shape_atten(uvb_table, redshift, hden,temp)

    #Print UVB
    out.write(uvb_str)
    #Print output options
    out.write("hden "+str(hden)+" log\n")
    out.write("metals "+str(metals)+" log\n")
    out.write("constant temperature "+str(temp)+" log\n")
    ionsave = path.join(real_outdir,"ionization.dat")
    out.write("save last ionization means \""+ionsave+"\"\n")
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

def gen_redshift(rredshift):
    """Function for generating tables at a particular redshift"""
    for hhden in np.arange(-7.,4.,0.2):
        for ttemp in np.arange(3.,8.6,0.05):
            ooutdir = output_cloudy_config(rredshift, hhden, -1, ttemp,1,"ion_out")
            infile = path.join(ooutdir, "cloudy_param")
            if not path.exists(path.join(ooutdir, "ionization.dat")):
                subprocess.call(['./cloudy.exe', '-r', infile])

if __name__ == "__main__":
    pool = mp.Pool(processes=3)
    pool.map(gen_redshift,[4,3,2,1,0])
