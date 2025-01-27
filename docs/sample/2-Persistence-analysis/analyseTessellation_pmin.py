# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# MAPT3 importation
from MAPT3.generics import intstringer
from MAPT3.tessellation import PlateGather
from MAPT3.project import Project

# Load the project parameters
Project.set('../myparameters.py')


# ==================================================

# Quick and simple visualisation of a tessellation

# ---- 1. Tessellation description

pmin = 2000

path  = '../1-Tessellation/TTK_outputs/'+'p'+intstringer(round(pmin),5)+'/'
file  = 'AGE467-555_llsvp_vp00071.h5'

# ---- 2. Creat the PlateGather object

pg = PlateGather()
pg.load_from_h5(path+file)

# ---- 3. Simple figure

# Visualisation of the plate boundaries

fig = plt.figure(figsize=(8,5))
ax  = fig.add_subplot(111, projection=ccrs.Robinson())
ax.scatter(pg.lonb,pg.latb,s=1,transform=ccrs.PlateCarree())
#fig.savefig('AGE467-555_llsvp_vp00073_p02000.png',dpi=200)
plt.show()
