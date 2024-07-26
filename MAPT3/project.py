# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    MAPT3 project
"""

# External dependencies:
import numpy as np
import sys, os

# Internal dependencies:
from .planetaryModels import Earth
from .generics import im
from .errors import ParFileExtError, ParFileNameError, ProjectCheckError


# ----------------- CLASS -----------------


class MAPT3Project:
    """Data structure for MAPT3 projects"""

    def __init__(self):
        # General
        self.path = './'                # path to the project
        self.nop  = 368120              # number of mesh point at the surface of the sphere (theoretically)
                                        # used for memory allocation: can be maximised if unknown
        self.planetaryModel = Earth     # model of planet
        self.modelRadius = 2.1988971841 # radius of the convection model
        self.polyminsize = 20           # minimum polygon area (in number of points): below the inversion of the velocity field is not computed and the polygon is not considered as a plate
        # Tessellation
        self.pmin = np.array([1000,\
                              2000])    # minimum persistence thresholds tested during the tessellation
                                        # be careful: have to be ordered from small values to large values 
        # Optimisation
        self.P1c = 0.90                 # critical (minimal) plateness of plates to be defined as rigid
        self.P2c = 0.80                 #   -> Definition of plateness after Janin et al., 2024, Alisic et al., 2012
                                        #   -> Values defined in Janin et al., 2024
        self.fragment_size = int(self.nop*0.8/100) # Area of the smallest spatial fragment for which the plateness have to be good to be considered as rigid:
                                        # (area in number of surface point (here ~0.8% total surface))
        self.check()

    def default(self):
        """Reset the parameters of the project"""
        self.__init__()
    
    def check(self):
        if np.count_nonzero(self.pmin == np.sort(self.pmin)) != self.pmin.shape[0]:
            raise ProjectCheckError('pmin have to be ordered')
        if self.P1c > 1 or self.P1c < 0:
            raise ProjectCheckError('P1c have to be in [0,1]')
        if self.P2c > 1 or self.P2c < 0:
            raise ProjectCheckError('P2c have to be in [0,1]')
        if self.fragment_size >= self.nop:
            raise ProjectCheckError('Inconsistency: fragment_size have to be < nop')
        
    
    def set(self,parfile,verbose=False):
        """
        Imports a parameter file.
        
        Args:
            parfile = str, path to reach the par file
                      (e.g.:  parfile = '../mypar.py')
                      parfile have to contain the extension '.py'
                      and not contain other '.'
        """
        pName = 'MAPT3Project'
        parfile = os.path.abspath(parfile)
        dir  = '/'.join(parfile.split('/')[:-1])+'/'
        file = parfile.split('/')[-1].split('.py')[0]
        if len(parfile.split('/')[-1].split('.py')) <= 1:
            raise ParFileExtError()
        if '.' in file:
            raise ParFileNameError()
        sys.path.append(dir)
        im('Parameter file: '+file,pName,verbose)
        mod = __import__(file)
        # ---
        try:
            self.path = os.path.abspath(mod.path)
            im('Importation of proj path:\n  -> '+self.path,pName,verbose)
        except:
            im('Importation of proj path:\n  -> default',pName,verbose)
        try:
            self.nop = mod.nop
            im('Importation of nop:\n  -> '+str(self.nop),pName,verbose)
        except:
            im('Importation of nop:\n  -> default',pName,verbose)
        try:
            self.planetaryModel = mod.planetaryModel
            im('Importation of a planetary model:\n  -> '+str(self.planetaryModel),pName,verbose)
        except:
            im('Importation of a planetary model:\n  -> default',pName,verbose)
        try:
            self.modelRadius = mod.modelRadius
            im('Importation of the radius:of the model surface:\n  -> '+str(self.modelRadius),pName,verbose)
        except:
            im('Importation of the radius:of the model surface:\n  -> default',pName,verbose)
        try:
            self.polyminsize = mod.polyminsize
            im('Importation of a minimum polygon size:\n  -> '+str(self.modelRadius),pName,verbose)
        except:
            im('Importation of a minimum polygon size:\n  -> default',pName,verbose)
        try:
            self.pmin = mod.pmin
            im('Importation of a list of minimum persistence thresholds for the tessellation:\n  -> '+str(self.pmin),pName,verbose)
        except:
            im('Importation of a list of minimum persistence thresholds for the tessellation:\n  -> default',pName,verbose)
        try:
            self.P1c = mod.P1c
            im('Importation of P1c:\n  -> '+str(self.P1c),pName,verbose)
        except:
            im('Importation of P1c:\n  -> default',pName,verbose)
        try:
            self.P2c = mod.P2c
            im('Importation of P2c:\n  -> '+str(self.P2c),pName,verbose)
        except:
            im('Importation of P2c:\n  -> default',pName,verbose)
        try:
            self.fragment_size = mod.fragment_size
            im('Importation of a minimum fragment size for the plate rigidity:\n  -> '+str(self.fragment_size),pName,verbose)
        except:
            im('Importation of a minimum fragment size for the plate rigidity:\n  -> default',pName,verbose)
        # control
        self.check()



# ----------------- INSTANCE -----------------


Project = MAPT3Project()
