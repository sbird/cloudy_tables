# -*- coding: utf-8 -*-
"""Generate parameter files for cloudy, then run cloudy a number of times to generate a folder tree of files"""
import os.path as path
import numpy as np
import math
import subprocess
import os
import multiprocessing as mp
import cold_gas
from scipy.interpolate import interp1d

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

def broad(lambda_X, gamma_X):
    """Natural broadening parameter, a, for a line.
    This is computed by taking the Voigt profile in the limit that doppler broadening is zero,
    then multiplying by λ_D/λ_X, where λ_D is the Doppler broadening parameter.
    Thus we have a' = λ_X Γ_X/ 4π c
    Arguments:
    lambda_X - transitions wavelength in m
    gamma_X - transition probability in Hz
    """
    return (lambda_X * gamma_X)/(4*math.pi*2.9979e8)

def sigma_X(lambda_X, fosc_X):
    """The cross-section of the absorption transition, in m^2.
       This is the sqrt of the Thompson cross-section for the photon times the x-section of the transition.
    """
    sigma_T = 6.652458558e-29 #Thompson cross-section in m^2
    return np.sqrt(3.0*math.pi/8.0*sigma_T)*lambda_X*fosc_X

def profile(ll, lambda_X, gamma_X, fosc_X, hden):
    """The profile for the absorption we assume to be Lorentzian, in the limit where there is no
    thermal broadening of the line.

    To compute it, we use F(l) = F_0 (1 - exp(-τ)), where F_0 is chosen so that the profile
    is unity at the transition, its maximal point.

    The optical depth, τ is given by
    τ = (σ_X c N) / (sqrt{π} b) H(a,x)
    where H(a,x) is the Voigt profile.
    In the limit where T-> 0, the thermal parameter, b-> 0, and the Voigt profile becomes:

    H(a, x) = (b/c) α / (π ( ζ^2 + α^2))

    where α = λ_i Γ_i / (4 π c) and ζ = (λ - λ_i) / λ_i

    Then we have

    τ = σ_X N α / (π^(3/2) ( ζ^2 + α^2))

    """
    aa = broad(lambda_X, gamma_X)
    xx = (ll - lambda_X)/lambda_X
    ss = sigma_X(lambda_X, fosc_X)
    #Column density: characteristic length of each particle is assumed to be 1 kpc/h.
    #Note hden is given in log([1/cm^3])
    cdd = 10**hden*1e6*3.1e19/0.7

    tau = ss*cdd*aa/(xx**2+aa**2)/math.pi**1.5

    f0 = 1-np.exp(-ss*cdd/aa/math.pi**1.5)
    flux = (1-np.exp(-tau))/f0
    return flux

def gen_cloudy_uvb_shape_atten(uvb_table, redshift, hden,temp):
    """
    Generate the cloudy input string from a UVB table, with some more physical attenuation.
    Assume that the gas is optically thick only to HI and HeI, as all other species are a
    factor of 10^4 less abundant. Neglect H2 absorption.

    The attenuation is that calculated by Rahmati 2012. This was only calculated for H Lya, so for
    other lines we scale the attenuation with the optical depth. The optical depth goes like (for fixed columns):
    τ ~ σ_X n
    so for other transitions model the change in cross-section by adjusting the density to get the same τ.
    So compute attenuation at new density n' = n (σ_X'/σ_X) = n /(λ_X' / λ_X) (f^osc_X'/f^osc_X)
    This is not really right - it neglects the fact that the transition rates are also getting smaller,
    which will make for less self-shielding from these lines.

    Ly-beta is about a factor of 5 smaller in x-section, ly-gamma another.
    Higher order transition rates are also smaller, making the line less broad, and
    fewer relevant metal transitions are at higher energies.

    Then the effect is moved to densities rather near the star-forming regime, where we are less confident,
    and our quick approximation becomes increasingly bad.
    Because of CIE at these densities all we really want is the right number of free electrons, which we
    get by having the HI and HeI fractions right.

    So we just cut the UVB from Lya, lyb and He Lya, and assume everything else does not
    significantly attenuate the UVB.

    The profile around each shielded transition is Lorentzian, for a DLA in the naturally broadened limit.

    Note Rydberg ~ 1/wavelength, and 1 Rydberg is the energy of a photon at the Lyman limit, ie,
    with wavelength 91.18 nm.
    """
    UVB = UVBAtten(redshift)
    #Subsample the UVB so we have more points around the regions we care about
    intuvb = interp1d(uvb_table[:,0], uvb_table[:,1])
    newuvb = np.append(uvb_table[:,0], 91.18e-9/(1100+np.arange(0,200,10)))
    newuvb = np.append(newuvb, 91.18e-9/(520+np.arange(0,100,10)))
    newuvb = np.sort(newuvb)
    origuvb = 10**intuvb(newuvb)
    waveuvb = 91.18e-9/newuvb
    #Calculate the profile of absorption from HI Lya data from VPFIT.
    lya_prof = origuvb*(1-UVB.atten(hden, temp))*profile(waveuvb, 1215.6701*1e-10, 6.265e8, 0.4164, hden)
    #Do Ly-beta: cross-section
    #sbeta = (1025.7223*0.07912)/(1215.6701*0.4164)
    #lyb_prof = origuvb*(1-UVB.atten(hden*sbeta, temp))*profile(waveuvb, 1025.7223*1e-10, 1.897e8, 0.079129, hden)
    #Now calculate the profile of absorption from He Lya: data again from VPFIT.
    saHe = (584.334*0.285)/(1215.6701*0.4164)
    lyaHe_prof = origuvb*(1-UVB.atten(hden*saHe, temp))*profile(waveuvb, 584.334*1e-10, 2e8, 0.285, hden)
    #Compute adjusted UVB table
    subuvb = origuvb - (lya_prof+lyaHe_prof)
    subuvb[np.where(subuvb < 1.e-35)] = 1.e-35
    uvb_table[:,1] = np.log10(subuvb)
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
            ooutdir = output_cloudy_config(rredshift, hhden, -1, ttemp)
            infile = path.join(ooutdir, "cloudy_param")
            if not path.exists(path.join(ooutdir, "ionization.dat")):
                subprocess.call(['./cloudy.exe', '-r', infile])

if __name__ == "__main__":
    pool = mp.Pool(processes=3)
    pool.map(gen_redshift,[4,3,2])
