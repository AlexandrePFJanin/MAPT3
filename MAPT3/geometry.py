# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Geometric toolkit
"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Internal dependencies:
from .generics import gaussian3D
from .geotransform import haversine
from .project import Project


# ----------------- FUNCTIONS -----------------


def distance3d(x1,y1,z1,x2,y2,z2):  
    """
    Distance function for the function MAPT3.geometry.areatriangle3d
    """  
    a=(x1-x2)**2+(y1-y2)**2 + (z1-z2)**2
    d= a ** 0.5  
    return d  


def areatriangle3d(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    """
    Function computing very efficiently the surface area of a 3D triangle.
    """
    a=distance3d(x1,y1,z1,x2,y2,z2)  
    b=distance3d(x2,y2,z2,x3,y3,z3)  
    c=distance3d(x3,y3,z3,x1,y1,z1)  
    return heron(a,b,c)


def get_barycenter(x,y,z,sigma=0.07):
    """
    Get barycenter of a plate from points belonging to the plate.
    Method: Gaussian convolution.

    Args:
        x (ndarray): x coordinates (flatten) of points
        y (ndarray): y coordinates (flatten) of points
        z (ndarray): z coordinates (flatten) of points
        sigma (float): std of the 3D Gaussian

    Returns:
        lonb (float): longitude of the barycenter
        latb (float): latitude of the barycenter
    """
    density = x.copy()*0
    for i in range(len(x)):
        density = density + gaussian3D(x,y,z,x[i],y[i],z[i],sigma)
    ind = np.where(density == np.amax(density))[0]
    if len(ind)>1:
        ind = ind[0]
    xb = x[ind]
    yb = y[ind]
    zb = z[ind]
    # get ENU
    rb     = np.sqrt(xb**2+yb**2+zb**2)
    latb   = np.arctan2(np.sqrt(xb**2+yb**2),zb)
    lonb   = np.arctan2(yb,xb)
    latb   = -(latb*180/np.pi-90)
    lonb   = lonb*180/np.pi
    return lonb,latb


def polygon_perimeter_and_ordening(lats,lons, radius=Project.planetaryModel.radius, plot=False):
    """
    Computes iteratively the perimeter of a polygon on the sphere
    and return the order edges points.

    Args:
        lats (np.ndarray): latitude of points belonging to the edges of the polygon (in radians)
        lons (np.ndarray): longitude of points belonging to the edges of the polygon (in radians)
        radius (int/float): radius of the planet (in km)

    Returns:
        peri (float): perimeter of the polygon (in km)
        order (np.ndarray): order of points along the edges of the polygon
        dist_list (np.ndarray)

    """
    nod = len(lons)

    undone = np.ones(nod,dtype=bool)
    order  = np.zeros(nod+1,dtype=np.int32)
    dist_list = np.zeros(nod+1)
    peri   = 0

    undone[0]  = False
    order[0]   = 0
    previousID = 0

    totID = np.arange(nod)
    
    i = 1
    ccontinue = True
    activation_first = False
    while ccontinue and i<=nod:
        remeningID = totID[undone]
        lonRef = lons[previousID]
        latRef = lats[previousID]
        lon = lons[undone]
        lat = lats[undone]
        d = haversine(lonRef,latRef,lon,lat,radius)
        #
        dmin =  np.amin(d)
        dist_list[i] = dmin
        peri += dmin
        next_min = remeningID[np.where(d == dmin)[0][0]]
        order[i] = next_min
        undone[next_min] = False
        previousID = next_min
        if i >= int(nod/2) and not activation_first:
            activation_first = True
            undone[0] = True
        if next_min == 0:
            ccontinue = False
            order = order[0:i+1]
            dist_list = dist_list[0:i+1]
        i = i+1
    # ending
    if plot:
        lons = lons*180/np.pi
        lats = lats*180/np.pi
        fig = plt.figure(figsize=(10,6))
        ax  = fig.add_subplot(111,projection=ccrs.Robinson())
        ax.set_global()
        ax.set_title('Number of pts: '+str(len(lons)))
        ax.plot(lons[order],lats[order],transform=ccrs.Geodetic())
        ax.scatter(lons[order],lats[order],c=np.arange(len(lons[order])),transform=ccrs.PlateCarree())
        ax.scatter(lons[order][0],lats[order][0],c='red',s=15,transform=ccrs.PlateCarree())
        plt.show()
    return peri, order, dist_list
