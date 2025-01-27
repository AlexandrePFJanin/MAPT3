# -*- coding: utf-8 -*-
"""
@summary: Example of a MAPT3 workflow
@author:  Alexandre JANIN
"""

# Packages importation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
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

time_indices = np.arange(len(list2optimizedh5))

# ---- 1. Creat the PlateTracking object

pg = PlateTracking()

# --- 1.1
# Loading of the OPTIMIZED tessellation through a function 'path2file' and a list of time indices

pg.build(path2optimizedh5,time_indices)

# --- 1.2 
# Loading of the tracking file

path  = './TRACKING/'
file  = 'myTracking_adjusted.csv'

pg.load_tracking(path+file,ftype='bind')



# ===== 2. Exploration: example of data manipulation

# connected component ID
ccID = 17

mask = pg.connectedComponentID == ccID

print('ccID: '+str(ccID))
for i in range(np.count_nonzero(mask)):
    cti  = pg.ctime[mask][i]
    loni = pg.lon[mask][i]
    lati = pg.lat[mask][i]
    print('  ctime= %s: %s, %s'%(str(cti),str(loni),str(lati)))


# ===== 3. A simple tracking figure

track = True # plot time tracks

fig = plt.figure()
ax1 = fig.add_subplot(111,projection=ccrs.Robinson())
pg.ci = len(pg.indices)-1-1
pg.iterate()
ax1.scatter(pg.drop.lonb,pg.drop.latb,color='k',s=1,transform=ccrs.PlateCarree())
if track:
    ucc = np.unique(pg.connectedComponentID)
    id  = np.array(range(len(pg.x)))
    cNorm     = colors.Normalize(vmin=0, vmax=len(ucc))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='jet')
    for i in range(len(ucc)):
        mask = pg.connectedComponentID == ucc[i]
        mask = id[mask]
        if pg.ctype[mask][0] == 3:
            lon = pg.lon[mask]
            lat = pg.lat[mask]
            ax1.scatter(lon,lat,color=scalarMap.to_rgba(i),transform=ccrs.PlateCarree())
            if np.count_nonzero(mask) > 1:
                ax1.plot(lon,lat,color=scalarMap.to_rgba(i),transform=ccrs.Geodetic())
plt.show()
