# -*- coding: utf-8 -*-
"""Test the accuracy of the interpolation onto the cloudy grid"""
import os.path as path
import subprocess
import numpy as np
import convert_cloudy as cc
import make_cloudy_input as mc #hammer

#Test cases
class TestCase:
    """Test-case class, containing an exactly worked out table"""
    def __init__(self, dens, temp, met = -1, redshift=3):
        self.dens = dens
        self.temp = temp
        self.met = met
        self.redshift = redshift
        self.species = ("H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe")

    def make_table(self, atten, tdir="ion_out"):
        """Make the test cases"""
        ooutdir = mc.output_cloudy_config(self.redshift, self.dens, self.met, self.temp, atten, path.join(tdir,"test"))
        infile = path.join(ooutdir, "cloudy_param")
        if not path.exists(path.join(ooutdir, "ionization.dat")):
            subprocess.call(['./cloudy.exe', '-r', infile])
        self.table = cc.convert_single_file(path.join(ooutdir, "ionization.dat"))

    def get_ion(self, element, ionn):
        """Get the ion fraction direct from the file, using the same format as convert_cloudy uses"""
        cspe = self.species.index(element)
        return 10**self.table[cspe, ionn-1]

def do_tests(tests, atten=2,cdir="ion_out"):
    clou = cc.CloudyTable(3, cdir)
    ions = {"H":1, "Si":2, "C":4, "O":6, "He":1}
    maxd = 0.
    maxad = 0.
    for test in tests:
        print "== d = ",test.dens, "T=",np.round(10**test.temp,2),"=="
        test.make_table(atten, cdir)
        for (elem, ion) in ions.iteritems():
            iotab = np.round(np.log10(clou.ion(elem, ion, 10**test.dens, 10**test.temp))[0],2)
            abstab = np.log10(test.get_ion(elem, ion))
            #Find percent deviation in log space
            if abstab < 0:
                dev = np.abs(iotab/abstab-1)
            elif iotab < 0:
                dev = np.abs(abstab/iotab -1)
            else:
                dev = 0
            absdev = np.abs(iotab-abstab)
            if dev > maxd:
                maxd = dev
            if absdev > maxad:
                maxad = absdev
            print elem, ion, "ion=",abstab, iotab,"rel dev: ",np.round(dev,2), "abs dev: ",absdev

    print "Max dev: ",maxd, "max abs dev: ",maxad

if __name__ == "__main__":
    tests = [TestCase(-1.035,4.43), TestCase(-1.035, 4.4), TestCase(-2.43,5.77), TestCase(1.64,4.12), TestCase(0.52, 4.22), TestCase(-1.43, 4.26)]
    print "==Uniform UVB attenuation=="
    do_tests(tests,1)
    print "==No UVB attenuation=="
    do_tests(tests, 0, "ion_out_no_atten")
    print "==Fancy UVB attenuation around Lya=="
    do_tests(tests, 2, "ion_out_fancy_atten")
