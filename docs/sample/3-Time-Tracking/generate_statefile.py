# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import os

# MAPT3 importation
from MAPT3.timetracking import generateState_TTKplateTimeTrackingFromPersistenceDiag


# -----------------------------------------------------------------------------
# DEFINE THE VARIABLE 'path2persistenceDiagram_xdmf' 
#  -> List of the paths to the pre-computed persistence diagrams (plate barycenters)

path = os.path.abspath('./PERSISTENCE-DIAG/')

files = ['AGE467-555_llsvp_vp00071_optimized',\
         'AGE467-555_llsvp_vp00073_optimized',\
         'AGE467-555_llsvp_vp00075_optimized']

path2persistenceDiagram_xdmf = [path + '/' + file +'.xdmf' for file in files]

# ----------- GENERATE STATE FILE ----------------------------------------
# Generate a paraview state file in python format where the persistence diagrams are open twince 

statefile = 'TTKPlateTimeTracking_statefile'

generateState_TTKplateTimeTrackingFromPersistenceDiag(path2persistenceDiagram_xdmf,statefile,path='./TRACKING/',\
                                                      logfile=True,verbose=True)



