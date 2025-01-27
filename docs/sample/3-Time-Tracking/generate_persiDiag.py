# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# MAPT3 importation
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================

# Example of script to generate fictive persistence
# diagrams from plate barycenters

# ---- 1. Locate the optimized tessellations

path  = '../2-Persistence-analysis/OPTIMIZED/'
files = ['AGE467-555_llsvp_vp00071_optimized',\
         'AGE467-555_llsvp_vp00073_optimized',\
         'AGE467-555_llsvp_vp00075_optimized']

# ---- 2. Creat the PlateGather object and export the output for each file

for file in files:
    pg = PlateGather()
    pg.load_from_h5(path+file+'.h5')
    pg.generate_persistenceDiag4tracking(file,path= './PERSISTENCE-DIAG/',plot=True)


