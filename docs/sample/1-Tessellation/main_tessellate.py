# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
from os import listdir, path
from os.path import isfile, join
import numpy as np

# MAPT3 importation
from MAPT3.tessellate import get_TTKtessellation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project

# Load the project parameters
Project.set('../myparameters.py')


# -----------------------------------------------------------------------------
# DEFINE THE VARIABLE 'pthreshold'
#  -> the list of tested minimum persistence threshold

pthreshold = Project.pmin


# -----------------------------------------------------------------------------
# DEFINE THE VARIABLE 'projPath'
#  -> path of the project

projPath = Project.path


# -----------------------------------------------------------------------------
# DEFINE THE VARIABLE 'xdmffname'
#  -> corresponding to the VTK file for the tessellation

xdmffname = projPath+'/XDMF-H5/AGE467-555_llsvp_vp00071.xdmf'


# -----------------------------------------------------------------------------
# DEFINE THE VARIABLES 'vpFileName', 'directory_surfaces' and 'directory_edges'
#  -> defining the names of the different directories for the outputs of the tessellation

vpFileName           = xdmffname.split('/')[-1].split('.')[-2]
directory_surfaces   = 'Surfaces_'+vpFileName
directory_edges      = 'Edges_'+vpFileName


# -----------------------------------------------------------------------------
# DEFINE HERE BELOW THE VARIABLE 'NOD'
#  -> maximum number of points on a surface (for memory allocation). You can maximize this parameter if unknown

NOD = Project.nop


# -----------------------------------------------------------------------------
# COMPUTE THE TESSELLATION OF THE FILE: 'xdmffname'

for i in range(0,len(pthreshold)):

    path  = './TTK_outputs/'+'p'+intstringer(round(pthreshold[i]),5)+'/'
    
    print('='*20)
    print(path)
    print('Persistence threshold: '+str(pthreshold[i]))
    print('='*20)
    print(xdmffname)

   # ---- 1. Plate tessellation 
    get_TTKtessellation(xdmffname,suffix='',path=path,persistenceMin=pthreshold[i],persistenceMax=None,velocityField='Cartesian Velocity',\
                        directory_surfaces=directory_surfaces,directory_edges=directory_edges,\
                        pointArray=['Cartesian Velocity', 'Pressure', 'Spherical Velocity','PointID'],\
                        normalize_MagGrad=False,logfile=True,paraview_version='5.11')
    

    print()
    print('Compaction H5')



    dirSurfaces   = path + directory_surfaces + '/'
    dirEdges      = path + directory_edges    + '/'

    genName    = 'Surface'
    surffiles  = [f for f in listdir(dirSurfaces) if isfile(join(dirSurfaces,f)) and genName in f]

    genName    = 'Boundary'
    edgesfiles = [f for f in listdir(dirEdges) if isfile(join(dirEdges,f)) and genName in f]

    path2bound = path + '1SeparatriceGeom_'+vpFileName+'.csv'


    # ---- 2. Creat the PlateGather object

    pg = PlateGather()
    pg.load(NOD,dirSurfaces,surffiles,dirEdges,edgesfiles,path2bound)
    
    # ---- 3. Export in .h5 to have only one tessellation file per time step and pmin value
    pg.export2h5(path+vpFileName+'.h5')







