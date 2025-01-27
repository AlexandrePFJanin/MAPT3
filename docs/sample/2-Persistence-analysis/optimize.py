# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import os

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project
from MAPT3.optimize import optimize

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================


# Define the list of pmin values that will be consider  during the optimization
pthreshold = Project.pmin # here, all the values used for the tessellation

# Tessellation that will be optimized
myfile = 'AGE467-555_llsvp_vp00071'

# Here, we can define a function to help locate the tessellation file
def path2h5file(pmin,file=myfile):
    path  = os.path.abspath('../1-Tessellation/TTK_outputs/'+'p'+intstringer(round(pmin),5))
    file  = myfile+'.h5'
    return path+'/'+file

# Output file name
ofilename  = myfile

# Optimization function
optimize(path2h5file, pthreshold, ofilename, output_path='./OPTIMIZED',\
        plot_missed = False, add_missedPlates = True, overlap_order = 0, geographic_search = 1, \
        edges_detection_method='paraview', verbose_edgesExtraction=False,\
        simplex_threshold = 0.01,hidden=True)



