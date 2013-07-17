# -*- coding: utf-8 -*-
"""Short script to read cloudy table files and output numpy arrays."""

import numpy as np
import re
import os.path as path
import scipy.interpolate as intp
from scipy.ndimage import map_coordinates

#Max number of ion species to count
nions = 17
cloudy_dir = path.join(path.dirname(__file__),"ion_out/")

def handle_single_line(splitline):
    """Short function to handle a single cloudy line, split by whitespace, and deal with lack of spaces"""
    #Sometimes Cloudy fails to put a space between output columns.
    #Test for this by trying to create a float from the element
    #and seeing if it fails.
    ii = 0
    while ii < np.size(splitline):
        s = splitline[ii]
        try:
            s=float(s)
        except ValueError:
            #Cloudy elements are <= 0, so we can split on -.
            tmp = s.split("-")
            try:
                splitline[ii] = -1*float(tmp[-1])
                #Insert elements backwards: First element is ignored because it will be ""
                for q in range(-2,-np.size(tmp),-1):
                    splitline.insert(ii, -1*float(tmp[q]))
                    ii+=1
            except ValueError:
                pass
        ii+=1
    return splitline

def convert_single_file(ion_table):
    """Convert a single file to a numpy (text) format.
    Discard all metals except the 9 we follow in Arepo, which are:
    H, He, C, N, O, Ne, Mg, Si, Fe
    Output an 8 x 22 array, the number of ions of iron cloudy gives.
    Array elements greater than the physical ionisation of the metal will be set to -30
    which is what cloudy uses for log(0).
    """
    #Species names to search the cloudy table for
    species = ("Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron")
    #Next element
    elmt = 0
    num_table = -30*np.ones((np.size(species), nions))

    f=open(ion_table)
    line = f.readline()
    while line != "" and elmt < np.size(species):
        #This line contains a species we want
        if re.search(species[elmt],line):
            #Split with whitespace
            splitline=line.split()
            #Remove name column
            if splitline[0] == species[elmt]:
                splitline = splitline[1:]
            else:
                splitline[0] = re.sub(species[elmt], "", splitline[0])
            splitline = handle_single_line(splitline)
            #Ignore H2 formation
            if species[elmt] == "Hydrogen":
                num_table[elmt,0:2] = np.array(splitline)[:2]
            #Set table
            else:
                num_table[elmt, 0:np.size(splitline)] = np.array(splitline)
            #Iron will continue onto a second line.
            #Ignore ion species > 17. This only affects iron at high densities
            #if species[elmt] == "Iron":
            #    line2 = f.readline()
            #    splitline2 = line2.split()
            #    splitline2 = handle_single_line(splitline2)
            #    start = np.size(splitline)
            #    num_table[elmt, start:(start+np.size(splitline2))] = np.array(splitline2)
            elmt+=1
        line = f.readline()
    return num_table

def read_all_tables(red, dens, temp, directory):
    """
    Read a batch of tables on a grid of density (dens) and temperature (rho) from a directory.
    Subdirs are in the form: zz$red/h$DENS/T$TEMP/
    Returns a table with the indices:
    REDSHIFT, DENSITY, TEMPERATURE, SPECIES, ION
    """
    #Last two indices are num. species and n. ions
    nred = np.size(red)
    nrho = np.size(dens)
    ntemp = np.size(temp)
    tables = np.empty([nred, nrho, ntemp,9, nions])
    for zz in xrange(nred):
        zdir = "zz"+str(red[zz])
        for rr in xrange(nrho):
            for tt in xrange(ntemp):
                strnums = path.join("h"+str(dens[rr]), "T"+str(temp[tt]))
                if np.abs(dens[rr]) < 1.e-3:
                    strnums = path.join("h0.0", "T"+str(temp[tt]))
                ionfile = path.join(directory, zdir, strnums, "ionization.dat")
                tables[zz,rr,tt,:,:] = convert_single_file(ionfile)
    return tables

class CloudyTable:
    """Class to interpolate tables from cloudy for a desired redshift, density and temperature"""
    def __init__(self, redshift, directory=cloudy_dir):
        """Read a cloudy table from somewhere"""
        self.savefile = path.join(directory,"cloudy_table.npz")
        self.reds = np.arange(2,5)
        self.dens = np.arange(-6,4,0.2)
        self.temp = np.arange(3,8.0, 0.1)
        self.species = ("H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe")
        #Solar abundances from Hazy table 7.1 as Z = n/n(H)
        self.solar = np.array([1, 0.1, 2.45e-4, 8.51e-5, 4.9e-4, 1.e-4, 3.47e-5, 3.47e-5, 2.82e-5])
        try:
            datafile = np.load(self.savefile)
            self.table = datafile["table"]
        except (IOError, KeyError):
            self.table = read_all_tables(self.reds, self.dens, self.temp, directory)
            self.save_file()
        self.directory = directory
        #Set up interpolation objects
        #Redshift is first axis.
        if redshift < 2.0 and redshift > 1.9:
            redshift = 2.0
        #Floating point roundoff taking this out of the integration boundary
        if redshift < 4.1 and redshift > 4.0:
            redshift = 4.0
        red_ints = intp.interp1d(self.reds, self.table,axis = 0)
        self.red_table = red_ints(redshift)


    def ion(self,species, ion, rho, temp):
        """Interpolate a table onto given redshift, density and temperature for species
        Returns a log( ionisation fraction ).
        rho is the density of hydrogen in atoms / cm^3. Internally the log is taken
        temp is the temperature in K
        red is the redshift
        species is the element name of the metal species we want
        ion is the ionisation number, starting from 1, so CIV is ('C', 4).
        Returns the fraction of this ion in the species
        """
        cspe = self.species.index(species)
        #Grid coords.
        crho = (np.log10(rho)-self.dens[0])*(np.size(self.dens)-1)/(self.dens[-1]-self.dens[0])
        ctemp = (np.log10(temp)-self.temp[0])*(np.size(self.temp)-1)/(self.temp[-1]-self.temp[0])
        coords = np.vstack((crho, ctemp))
        #mode = nearest handles points outside the grid - they get put onto the nearest grid point
        if crho > np.max(self.dens)+0.2 or crho < np.min(self.dens)-0.2:
            raise ValueError("Density outside of allowed values")
        if ctemp > np.max(self.temp)+0.1 or ctemp < np.min(self.temp)-0.1:
            raise ValueError("Temperature outside of allowed values")
        ions = map_coordinates(self.red_table[:,:,cspe,ion-1],coords, mode='nearest')
        return 10**ions

    def get_solar(self,species):
        """Get the solar metallicity for a species"""
        cspe = self.species.index(species)
        return self.solar[cspe]

    def save_file(self):
        """
        Saves tables to a file, because they are slow to generate.
        File is hard-coded to be directory/cloudy_table.npz.
        """
        np.savez(self.savefile,table=self.table)



