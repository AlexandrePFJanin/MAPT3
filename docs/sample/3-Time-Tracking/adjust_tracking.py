# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

# MAPT3 importation
from MAPT3.timetracking import PlateTracking
from MAPT3.project import Project

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================

# ---- 0. Prepare files

list2optimizedh5 = [Project.path+'/2-Persistence-analysis/OPTIMIZED/AGE467-555_llsvp_vp00071_optimized.h5',Project.path+'/2-Persistence-analysis/OPTIMIZED/AGE467-555_llsvp_vp00073_optimized.h5',Project.path+'/2-Persistence-analysis/OPTIMIZED/AGE467-555_llsvp_vp00075_optimized.h5']

path2optimizedh5 = lambda index_simu: list2optimizedh5[index_simu]

path2persistenceDiagram_h5 = [Project.path+'/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00071_optimized.h5',Project.path+'/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00073_optimized.h5',Project.path+'/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00075_optimized.h5']

time_indices = np.arange(len(list2optimizedh5))



# ---- 1. Create the PlateTracking object

pg = PlateTracking()

# --- 1.1
# Loading of the OPTIMIZED tessellations through a function 'path2file' and a list of time indices

pg.build(path2optimizedh5,time_indices)

# --- 1.2 
# Loading of the tracking file

path  = './TRACKING/'
file  = 'myTracking.csv'

pg.load_tracking(path+file,ftype='ttk')


# ---- 2. Adjustment of the tracking: Add polygons living only 1 time step by the analysis of precomputed persistence diagrams

# Compute the adjustement and export the result in './TRACKING-ADJUSTED/'

pg.adjstTracking(path2persistenceDiagram_h5,'myTracking_adjusted.csv',path='./TRACKING/',plot=False,detect_only=False)



    





