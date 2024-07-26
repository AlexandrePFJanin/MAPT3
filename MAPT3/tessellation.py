# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for Plate Processing and Analysis

"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cartopy.crs as ccrs
import h5py
import random
import shapefile
from tqdm import tqdm
from time import time
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

# Internal dependencies:
from .project import Project
from .generics import line_count, im
from .geotransform import *
from .geometry import areatriangle3d, polygon_perimeter_and_ordening
from .kinematics import *
from .io import write_PersistenceDiag_VTK, shell2VTK


# ----------------- FUNCTIONS -----------------


def compute_platecouple(cdist,xe,ye,ze,plateIDe,xs,ys,zs,plateIDs,xb,yb,zb):
    """
    Determines for each boundary points passed as an argument from which
    plate couple they belong from the analysis of data carried by edges
    and surface points.

    Args:
        cdist (float): critical distance between a boundary point
                       and an edge point to defined the boundary point
                       to be linked to the edge point
        xe,ye,ze (3x np.ndarray): cartesian coordinates of edge points
        plateIDe (np.ndarray): ID of the plate/polygon carried by the 
                        edge points.
        xs,ys,zs (3x np.ndarray): cartesian coordinates of the surface
                        points.
        plateIDs (np.ndarray): ID of the plate/polygon carried by the
                        surface points.
        xb,yb,zb (3X np.ndarray): cartesian coordinates of the boundary
                        points.
    
    Returns:
        platecouple (np.ndarray, shape=xb.shape): list of plate pairs
                        for each plate boundary points.
    """
    nod = len(xb)
    platecouple = np.ones((nod,2),dtype=np.int32)*99999
    for i in range(nod):
        xi,yi,zi = xb[i],yb[i],zb[i]
        cond1 = xe > xi-cdist
        cond2 = xe < xi+cdist
        cond  = cond1 * cond2           # condition on x
        cond1 = ye > yi-cdist
        cond2 = ye < yi+cdist
        cond  = cond * cond1 * cond2    # condition on y
        cond1 = ze > zi-cdist
        cond2 = ze < zi+cdist
        cond  = cond * cond1 * cond2    # condition on z
        # grid
        plate = plateIDe[cond]
        xgrid = xe[cond]
        ygrid = ye[cond]
        zgrid = ze[cond]
        # distances
        d      = np.sqrt((xgrid-xi)**2+(ygrid-yi)**2+(zgrid-zi)**2)
        sorder = np.argsort(d)
        splate = plate[sorder]
        # attribution
        allp = np.unique(splate[0:4])
        if len(allp) == 1:
            platecouple[i,0] = splate[0]
            platecouple[i,1] = splate[0]
        elif len(allp) >1:
            platecouple[i,0] = allp[0]
            platecouple[i,1] = allp[1]
        else:
            d2  = np.sqrt((xs-xi)**2+(ys-yi)**2+(zs-zi)**2)
            ids = np.where(d2 == np.amin(d2))[0][0]
            platecouple[i,0] = plateIDs[ids]
            platecouple[i,1] = plateIDs[ids]
    return platecouple






# ----------------- TessellatedPolygon -----------------

class TessellatedPolygon:
    """ Data structure for TessellatedPolygon objects"""
    def __init__(self):
        self.pName   = 'TessellatedPolygon' # program name
        self.verbose = True       # verbose output condition
        self.path2file = ''       # complete path to imported data
        # TTK outputs
        self.header  = ''   # textual header of the plate file
        self.header_imposed = []
        self.x       = []   # x coordinates of the grid
        self.y       = []   # y coordinates of the grid
        self.z       = []   # z coordinates of the grid
        self.vx      = []   # velocity component on the x direction
        self.vy      = []   # velocity component on the y direction
        self.vz      = []   # velocity component on the z direction
        self.vr      = []   # velocity component on the radial direction
        self.vtheta  = []   # velocity component on the theta direction
        self.vphi    = []   # velocity component on the phi direction
        self.pressure= []   # pressure grid
        self.plateID = []   # list of plate index
        self.magGSV  = []   # magnitude of the Gradient of Spherical Velocities
        self.pointID = []   # list of the point indices in the complete stagData object
                            # useful to get for exemple the vorticity on a specific point (open the vor
                            # stagData and take the point indicated by pointID!)
                            # Note: Indicies are correct only if you consider the same
                            #       resampling parameter, beginIndex and endIndex! check readme files!
        # computed
        self.lon     = []   # longitude of each point of the plate
        self.lat     = []   # latitude of each point of the plate
        self.r       = []   # distance to the center of the sphere for each point
        # Plate centroid
        self.centroidid = 0 # list id of the centroid of the figure
        self.xc = 0         # x coordinates of the centroid of the figure
        self.yc = 0         # y coordinates of the centroid of the figure
        self.zc = 0         # z coordinates of the centroid of the figure
        self.centrodist = []# list of the distance of each points to the centroid
        # Others
        self.bin = 0
        self.BIN = []
    


    def im(self,textMessage,error=False):
        """Print verbose internal message."""
        im(textMessage,self.pName,self.verbose,error=error)
    

    def memory_allocation(self,nol):
        """
        Allocate the memory for all fields in the current data structure.

        Args:
            nol (int): size of each vectors (e.g. self.x).
        """
        # The grid
        self.x = np.empty(nol)
        self.y = np.empty(nol)
        self.z = np.empty(nol)
        # Cartesian velocities
        self.vx = np.empty(nol)
        self.vy = np.empty(nol)
        self.vz = np.empty(nol)
        # Spehrical velocities
        self.vr     = np.empty(nol)
        self.vtheta = np.empty(nol)
        self.vphi   = np.empty(nol)
        # Presure
        self.pressure = np.empty(nol)
        # Gradient
        self.magGSV  = np.empty(nol)
        # Others: Plates and points IDs
        self.plateID = np.empty(nol)
        self.pointID = np.empty(nol)
    


    def load(self,path2file,separator=',',descendingManifold=True,\
             grid_name='Points',cartv_name='Cartesian Velocity',sphev_name='Spherical Velocity',\
             magG_name='magGSV',ptID_name='PointID',pressure_name='Pressure'):
        """Loads data from a .csv file generated with MAPT3.tessellate function
        into the current class instance.

        Args:
            path2file (str): Path to the .csv file
            separator (str, optional): Separator of the .csv file. Defaults to ','.
            descendingManifold (bool, optional): Select if your tessellation is based on the descendingManifold
                                                 or on the ascendingManifold. Defaults to True.
            
            # Key arguments:
            grid_name (str, optional):  Name of the grid description on the .csv file. Defaults to 'Point'.
            cartv_name (str, optional): Name of the cartesian velocity field on the original vtu/xdmf file (the same as in the .csv). 
                                        If unknown and you want to compute it from the spherical velocity field (if given)
                                        set cartv_name = 'compute'. (NOTE: cartv_name and sphev_name cannot be together set to 'compute')
                                        Defaults to 'Cartesian Velocity'.
            sphev_name (str, optional): Name of the spherical velocity field on the original vtu/xdmf file (the same as in the .csv). 
                                        If unknown and you want to compute it from the cartisian velocity field (if given)
                                        set sphev_name = 'compute'. (NOTE: cartv_name and sphev_name cannot be together set to 'compute')
                                        Defaults to 'Spherical Velocity'.
            magG_name (str, optional):  Name of the field containing the magnitude of the velocity gradient used to compute the tessellation.
                                        Defaults to 'magGSV'.
            ptID_name (str, optional):  Name of the field containing the ID of point. If unknow, set ptID_name = 'compute'
                                        Defaults to 'PointID'.
        """
        self.im("Extraction TTK outputs")
        self.header_imposed = [grid_name,cartv_name,sphev_name,magG_name,ptID_name]
        self.path2file = path2file
        self.im('TTK data, importation from file: '+path2file.split('/')[-1])
        nol = line_count(path2file)
        self.im('Number of line in the file: '+str(nol))
        # Memory allocation:
        self.im('Memory allocation')
        self.memory_allocation(nol-1) # -1: remove the header in the count
        self.im('Prepare header')
        with open(path2file,'r') as data:
            self.header = data.readline() #remove header
            if len(self.header) < 2:
                self.im('WARNING, empty plate detected')
            else:
                self.header = self.header.strip().split('"')
                self.header = [self.header[i] for i in range(len(self.header)) if self.header[i] != separator and self.header[i] != '']
                self.header = np.array(self.header)
                # find x,y and z
                xind     = np.where(self.header == grid_name+':0')[0][0]
                yind     = np.where(self.header == grid_name+':1')[0][0]
                zind     = np.where(self.header == grid_name+':2')[0][0]
                if cartv_name != 'compute':
                    cartv0ind  = np.where(self.header == cartv_name+':0')[0][0]
                    cartv1ind  = np.where(self.header == cartv_name+':1')[0][0]
                    cartv2ind  = np.where(self.header == cartv_name+':2')[0][0]
                if sphev_name != 'compute':
                    sphev0ind  = np.where(self.header == sphev_name+':0')[0][0]
                    sphev1ind  = np.where(self.header == sphev_name+':1')[0][0]
                    sphev2ind  = np.where(self.header == sphev_name+':2')[0][0]
                if descendingManifold:   # mean that plate cut have been made on descending manifold!
                    plateIDind = np.where(self.header == 'DescendingManifold')[0][0]
                else:
                    plateIDind = np.where(self.header == 'AscendingManifold')[0][0]
                pressureind = np.where(self.header == pressure_name)[0][0]
                maggsvind   = np.where(self.header == magG_name)[0][0]
                if ptID_name != 'compute':
                    pointIDind = np.where(self.header == ptID_name)[0][0]
                else:
                    pointIDind = plateIDind   # copy of the plateID e
                i = 0
                self.im('Reading...')
                if cartv_name != 'compute' and sphev_name != 'compute':
                    for line in data:
                        line = line.strip().split(separator)
                        self.x[i] = float(line[xind])
                        self.y[i] = float(line[yind])
                        self.z[i] = float(line[zind])
                        self.vx[i] = float(line[cartv0ind])
                        self.vy[i] = float(line[cartv1ind])
                        self.vz[i] = float(line[cartv2ind])
                        self.vr[i] = float(line[sphev0ind])
                        self.vtheta[i] = float(line[sphev1ind])
                        self.vphi[i] = float(line[sphev2ind])
                        self.pressure[i] = float(line[pressureind])
                        self.magGSV[i] = float(line[maggsvind])
                        self.plateID[i] = int(line[plateIDind])
                        self.pointID[i] = int(line[pointIDind])
                        i += 1
                elif cartv_name != 'compute' and sphev_name == 'compute':
                    for line in data:
                        line = line.strip().split(separator)
                        self.x[i] = float(line[xind])
                        self.y[i] = float(line[yind])
                        self.z[i] = float(line[zind])
                        self.vx[i] = float(line[cartv0ind])
                        self.vy[i] = float(line[cartv1ind])
                        self.vz[i] = float(line[cartv2ind])
                        self.pressure[i] = float(line[pressureind])
                        self.magGSV[i] = float(line[maggsvind])
                        self.plateID[i] = int(line[plateIDind])
                        self.pointID[i] = int(line[pointIDind])
                        i += 1
                    # compute spherical velocities
                    lat   = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
                    lon   = np.arctan2(self.y,self.x)
                    lat   = -(lat*180/np.pi-90)
                    lon   = lon*180/np.pi
                    #self.vphi,self.vtheta,self.vr = ecef2enu(lat*np.pi/180,lon*np.pi/180,self.vx,self.vy,self.vz)
                    self.vphi,self.vtheta,self.vr = ecef2enu_stagYY(self.x,self.y,self.z,self.vx,self.vy,self.vz)
                elif cartv_name == 'compute' and sphev_name != 'compute':
                    for line in data:
                        line = line.strip().split(separator)
                        self.x[i] = float(line[xind])
                        self.y[i] = float(line[yind])
                        self.z[i] = float(line[zind])
                        self.vr[i] = float(line[sphev0ind])
                        self.vtheta[i] = float(line[sphev1ind])
                        self.vphi[i] = float(line[sphev2ind])
                        self.pressure[i] = float(line[pressureind])
                        self.magGSV[i] = float(line[maggsvind])
                        self.plateID[i] = int(line[plateIDind])
                        self.pointID[i] = int(line[pointIDind])
                        i += 1
                    # compute topocentric velocities
                    lat   = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
                    lon   = np.arctan2(self.y,self.x)
                    lat   = -(lat*180/np.pi-90)
                    lon   = lon*180/np.pi
                    self.vx,self.vy,self.vz = enu2ecef(lat*np.pi/180,lon*np.pi/180,self.vphi,self.vtheta,self.vr)
                
        # ===============
        # check if you select good DescendingManifold or AscendingManifold
        self.bin = np.array(self.plateID)
        self.bin = np.unique(self.bin)     # if you chose good, you must have just one value
        if len(self.bin) > 1:
            self.im('-----------------------')
            self.im('    !!! WARNING !!!')
            self.im('-----------------------')
            self.im('You have several plateID number in this file!')
            self.im('Check if you set correctectly the DescendingManifold or AscendingManifold')
        # ===============
        # complete
        self.im('Reading complete.')
    
    

    def remove_edges_from_file(self,path2file,separator=',',remove_from_coordinates=True,minSize=50):
        """This function removes edges of a surface from an edge file based on the filed ptID
        or coordinates"""
        self.im("Remove Edges from input file")
        if len(self.x) <= minSize:
            self.im('   - WARNING, Plate too small to remove edges...')
            self.im('     Request aborded')
        else:
            self.im('   - TTK data, importation from file: '+path2file.split('/')[-1])
            self.im('      - Prepare header')
            id = np.array(range(len(self.x)))
            mask = np.ones(len(self.x),dtype=bool)
            with open(path2file,'r') as data:
                self.header = data.readline() #remove header
                if len(self.header) < 2:
                    self.im('WARNING, empty plate detected')
                else:
                    self.header = self.header.strip().split('"')
                    self.header = [self.header[i] for i in range(len(self.header)) if self.header[i] != separator and self.header[i] != '']
                    self.header = np.array(self.header)
                    i = 0
                    self.im('      - Reading...')
                    if remove_from_coordinates:
                        xind     = np.where(self.header == self.header_imposed[0]+':0')[0][0]
                        yind     = np.where(self.header == self.header_imposed[0]+':1')[0][0]
                        zind     = np.where(self.header == self.header_imposed[0]+':2')[0][0]
                        for line in data:
                            line = line.strip().split(separator)
                            xe = float(line[xind])
                            ye = float(line[yind])
                            ze = float(line[zind])
                            cond1 = self.x == xe
                            cond2 = self.y == ye
                            cond3 = self.z == ze
                            bind  = id[cond1*cond2*cond3]
                            mask[bind] = False
                            i += 1
                    else: # based on pointID
                        pointIDind = np.where(self.header == self.header_imposed[-1])[0][0]
                        for line in data:
                            line = line.strip().split(separator)
                            pointIDe = int(line[pointIDind])
                            bind = id[self.pointID == pointIDe]
                            mask[bind] = False
                            i += 1
            # ---------------
            self.im('      - Reading complete.')
            if np.count_nonzero(mask) < 20:
                self.im('WARNING, Edges removal aborded: not enough points in the plate (less than 20).')
            else:
                self.im('   - Edges removal...')
                self.x = self.x[mask]
                self.y = self.y[mask]
                self.z = self.z[mask]
                self.vx = self.vx[mask]
                self.vy = self.vy[mask]
                self.vz = self.vz[mask]
                self.vtheta = self.vtheta[mask]
                self.vphi = self.vphi[mask]
                self.vr = self.vr[mask]
                self.pressure = self.pressure[mask]
                self.plateID = self.plateID[mask]
                self.magGSV = self.magGSV[mask]
                self.pointID = self.pointID[mask]
                if len(self.lon) > 0:
                    self.lon = self.lon[mask]
                    self.lat = self.lat[mask]
                    self.r = self.r[mask]
                self.im('   - Remove edges complete!')
        
    

    def xyz2latlon(self):
        """
        Transforms x,y,z cartesian coordinates into geographical lon,lat,r
        coordinates (in degrees)
        """
        self.r     = np.sqrt(self.x**2+self.y**2+self.z**2)
        self.lat   = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lon   = np.arctan2(self.y,self.x)
        self.lat   = -(self.lat*180/np.pi-90)
        self.lon   = self.lon*180/np.pi
    
    

    def cut_with_polygon(self,polygon,plot=True):
        """This function cut the loaded plate to keep only points what are in
        the input polygon using matplotlib.path.

        Args:
            polygon (2D np.ndarray)
        """
        if len(self.lon) == 0:
            self.xyz2latlon()  
        # creat a list of all points
        points  = [[self.lon[i],self.lat[i]] for i in range(len(self.lon))]
        # Matplotlib mplPath
        path = mpltPath.Path(polygon)
        inside = path.contains_points(points)
        # Cut!
        self.x = self.x[inside]
        self.y = self.y[inside]
        self.z = self.z[inside]
        self.vx = self.vx[inside]
        self.vy = self.vy[inside]
        self.vz = self.vz[inside]
        self.vtheta = self.vtheta[inside]
        self.vphi = self.vphi[inside]
        self.vr = self.vr[inside]
        self.pressure = self.pressure[inside]
        self.plateID = self.plateID[inside]
        self.magGSV = self.magGSV[inside]
        self.pointID = self.pointID[inside]
        self.lon = self.lon[inside]
        self.lat = self.lat[inside]
        self.r = self.r[inside]
        # the plot
        if plot:
            lonpoly = [polygon[i][0] for i in range(len(polygon))]
            latpoly = [polygon[i][1] for i in range(len(polygon))]
            if lonpoly[-1] != lonpoly[0] or latpoly[-1] != latpoly[0]:
                lonpoly.append(lonpoly[0])
                latpoly.append(latpoly[0])
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
            ax.set_global()
            ax.set_title("Map of the plate cutting from a polygon")
            ax.plot(lonpoly,latpoly,'--',c='red',transform=ccrs.PlateCarree(),label='cutting polygon')
            ptsin  = np.array(points)[inside]
            ptsout = np.array(points)[~inside]
            ax.scatter(ptsin[:,0],ptsin[:,1],c='red',s=1,transform=ccrs.PlateCarree(),label='new plate (in)')
            ax.scatter(ptsout[:,0],ptsout[:,1],c='black',s=1,transform=ccrs.PlateCarree(),label='excluded points')
            ax.legend()
            plt.show()
    






# ----------------- PlateGather -----------------


class PlateGather:
    def __init__(self):
        """ Data structure for PlateGather objects """
        # --------- General
        self.pName   = 'PlateGather' # program name
        self.verbose = True          # verbose output condition
        self.path2file = ''          # complete path to imported data
        self.header = []             # list of .csv header name
        # Plates info
        self.nop    = 0          # number of plates
        self.persistence = None  # topological persistence associated to the current plate tessellation
        # --------- Surface Gather
        self.surfdir  = ''  # directory of the plate (surface) files
        self.surffile = []  # list of the plate files loaded
        self.pstart   = []  # list of starting index of plate:
                            # e.g. self.pstart[i] = first index of points in the plate number i
        self.pend     = []  # list of ending index of plate:
                            # e.g. self.pend[i-1] = last index of points in the plate number i
        self.surf     = []  # number of grid point that compose each plate
        self.surfdim  = []  # surface of each plates (size = self.nop) in ** KM^2 **
        # Plates
        self.x       = []   # x coordinates of the grid
        self.y       = []   # y coordinates of the grid
        self.z       = []   # z coordinates of the grid
        self.vx      = []   # velocity component on the x direction
        self.vy      = []   # velocity component on the y direction
        self.vz      = []   # velocity component on the z direction
        self.vr      = []   # velocity component on the radial direction
        self.vtheta  = []   # velocity component on the theta direction
        self.vphi    = []   # velocity component on the phi direction
        self.pressure= []   # pressure grid
        self.plateID = []   # list of plate index
        self.magGSV  = []   # magnitude of the Gradient of Spherical Velocities
        self.pointID = []   # list of the point indices in the complete stagData object
                            # useful to get for exemple the vorticity on a specific point (open the vor
                            # stagData and take the point indicated by pointID!)
                            # Note: Indicies are correct only if you consider the same
                            #       resampling parameter, beginIndex and endIndex! check readme files!
        self.pmin    = []   # minimum persistence from which the plate is rigid
        # Geographic grid
        self.lon = []       # list of longitudes
        self.lat = []       # list of latitudes
        self.r   = []       # list of radial coordinates
        # --------- Edge Gather
        self.edgedir   = ''  # directory of the boundary files
        self.edgefile  = []  # list of the boundary files loaded
        self.estart    = []  # list of starting index of plate:
                             # e.g. self.estart[i] = first index of points in the plate number i
        self.eend      = []  # list of ending index of plate:
                             # e.g. self.eend[i-1] = last index of points in the plate number i
        self.peri      = []  # number of grid point that compose the perimeters of each plate
        self.peridim   = []  # perimeter of each plates (size = self.nop) in ** KM **
        # Edges
        self.xe       = []   # x coordinates of the grid
        self.ye       = []   # y coordinates of the grid
        self.ze       = []   # z coordinates of the grid
        self.vxe      = []   # velocity component on the x direction
        self.vye      = []   # velocity component on the y direction
        self.vze      = []   # velocity component on the z direction
        self.vre      = []   # velocity component on the radial direction
        self.vthetae  = []   # velocity component on the theta direction
        self.vphie    = []   # velocity component on the phi direction
        self.pressuree= []   # pressure grid
        self.plateIDe = []   # list of plate index
        self.magGSVe  = []   # magnitude of the Gradient of Spherical Velocities
        self.pointIDe = []   # list of the point indices in the complete stagData object
                             # useful to get for exemple the vorticity on a specific point (open the vor
                             # stagData and take the point indicated by pointID!)
                             # Note: Indicies are correct only if you consider the same
                             #       resampling parameter, beginIndex and endIndex! check readme files!
        self.pmine    = []   # minimum persistence from which the plate is rigid
        # Geographic grid
        self.lone = []       # list of longitudes
        self.late = []       # list of latitudes
        self.re   = []       # list of radial coordinates
        # --------- Boudary Gather
        self.path2bound  = ''  # directory of the boundary files
        # Boundaries
        self.xb       = []   # x coordinates of the grid
        self.yb       = []   # y coordinates of the grid
        self.zb       = []   # z coordinates of the grid
        self.vxb      = []   # velocity component on the x direction
        self.vyb      = []   # velocity component on the y direction
        self.vzb      = []   # velocity component on the z direction
        self.vrb      = []   # velocity component on the radial direction
        self.vthetab  = []   # velocity component on the theta direction
        self.vphib    = []   # velocity component on the phi direction
        self.pressureb= []   # pressure grid
        self.magGSVb  = []   # magnitude of the Gradient of Spherical Velocities
        self.pminb    = []   # minimum persistence from which the boundary appears
        # Geographic grid
        self.lonb = []       # list of longitudes
        self.latb = []       # list of latitudes
        self.rb   = []       # list of radial coordinates
        # --------- Non-rigid Gather
        # non rigid
        self.xnr       = []   # x coordinates of the grid
        self.ynr       = []   # y coordinates of the grid
        self.znr       = []   # z coordinates of the grid
        self.vxnr      = []   # velocity component on the x direction
        self.vynr      = []   # velocity component on the y direction
        self.vznr      = []   # velocity component on the z direction
        self.vrnr      = []   # velocity component on the radial direction
        self.vthetanr  = []   # velocity component on the theta direction
        self.vphinr    = []   # velocity component on the phi direction
        self.pressurenr= []   # pressure grid
        self.magGSVnr  = []   # magnitude of the Gradient of Spherical Velocities
        self.pminnr    = []   # minimum persistence from which the boundary appears
        self.pointIDnr = []   # list of the point indices in the complete stagData object
        # Geographic grid
        self.lonnr = []       # list of longitudes
        self.latnr = []       # list of latitudes
        self.rnr   = []       # list of radial coordinates
        # --------- Computed
        self.platecouple   = []   # give for each boundary point the index of the two adjacent plates
        self.neighborhood  = []   # list (len = number of plates) of the list of all neighbors for each plate
        # barycenter
        self.baryLon = []   # longitudes of barycenters of all plates
        self.baryLat = []   # latitudes of barycenters of all plates
        # --------- Rotations
        # flags
        self.is_all_rotations = False
        self.plate1 = None
        self.wx1,self.wy1,self.wz1 = 0, 0, 0
        self.wx, self.wy, self.wz  = 0, 0, 0
        self.residual1         = 0
        self.misfit_xyz1     = []
        self.misfit_enu1     = []
        self.misfit_normcrossprod1 = []
        self.P11,self.P12    = 0, 0
        # all plates
        self.wx,self.wy,self.wz = [], [], []
        self.misfit = []
        self.misfit_normcrossprod = []
        self.rotmask = []
        self.P1 = []
        self.P2 = []
        # --------- Grid rotation
        self.grot_theta  = []
        self.grot_axis   = []
        self.grot_format = []
        # --------- Rotations
        # Plateness to define a plate as rigid
        self.P1_crit = Project.P1c # Default plateness P1
        self.P2_crit = Project.P2c # Default plateness P2
        # --------- From polygons
        self.poly_path2shp = ''         # path to reach the shape file of the loaded polygons
        self.poly_path2dbf = ''         # path to reach the .dbf file of the loaded polygons
        self.plateIDpoly = None         # plate ID number from an external tessellation shapefile
        self.nop_poly    = None         # number of plates from the external tessellation
        # --------- Others
        self.bin = 0
        self.BIN = []
    


    def im(self,textMessage,error=False):
        """Print verbose internal message."""
        im(textMessage,self.pName,self.verbose,error=error)
    

    def xyz2latlon(self,dtype):
        """
        """
        if dtype == 'Surfaces':
            self.r     = np.sqrt(self.x**2+self.y**2+self.z**2)
            self.lat   = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
            self.lon   = np.arctan2(self.y,self.x)
            self.lat   = -(self.lat*180/np.pi-90)
            self.lon   = self.lon*180/np.pi
        elif dtype == 'Edges':
            self.re    = np.sqrt(self.xe**2+self.ye**2+self.ze**2)
            self.late  = np.arctan2(np.sqrt(self.xe**2+self.ye**2),self.ze)
            self.lone  = np.arctan2(self.ye,self.xe)
            self.late  = -(self.late*180/np.pi-90)
            self.lone  = self.lone*180/np.pi
        elif dtype == 'Boundaries':
            self.rb    = np.sqrt(self.xb**2+self.yb**2+self.zb**2)
            self.latb  = np.arctan2(np.sqrt(self.xb**2+self.yb**2),self.zb)
            self.lonb  = np.arctan2(self.yb,self.xb)
            self.latb  = -(self.latb*180/np.pi-90)
            self.lonb  = self.lonb*180/np.pi



    def load(self,nod,surfdir,surflist,edgedir,edgelist,path2boundaries,persistence=np.nan,separator=',',internal_verbose=False,\
            minimum_import=True,rfactor=1,interpMethod='linear',\
            grid_name='Points',cartv_name='Cartesian Velocity',sphev_name='Spherical Velocity',\
            magG_name='magGSV',ptID_name='PointID',pressure_name='Pressure'):
        """
        Function loading together surface, edge and boundary data.
        Used to transform the collection of numerous .csv file coming from
        the plate tessellation (MAPT3.tessellate.get_TTKtessellation)
        into a unique .h5 tessellation file, coupled to the function self.export2h5
        """
        self.persistence = persistence
        #
        self.im('---- Loading plate!')
        self.im('Prepare and check file lists')
        self.header = [grid_name,cartv_name,sphev_name,magG_name,ptID_name]
        # surflist
        fidsurf   = [int(fname.split('_')[1].split('.')[0]) for fname in surflist]
        zipped    = zip(fidsurf,surflist)
        surflist  = list(zip(*sorted(zipped)))[1]
        fidsurf   = [int(fname.split('_')[1].split('.')[0]) for fname in surflist]
        # bounlist
        fidedge   = [int(fname.split('_')[1].split('.')[0]) for fname in edgelist]
        zipped    = zip(fidedge,edgelist)
        edgelist  = list(zip(*sorted(zipped)))[1]
        fidboun   = [int(fname.split('_')[1].split('.')[0]) for fname in edgelist]
        # check
        if fidboun == fidsurf:
            # load
            self.im('Input files conform')
            self.im('Loading operation:')
            self.load_surfaces(nod,surfdir,surflist,separator=separator,internal_verbose=internal_verbose,\
                               grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                               magG_name=magG_name,ptID_name=ptID_name,pressure_name=pressure_name)
            self.load_edges(nod,edgedir,edgelist,separator=separator,internal_verbose=internal_verbose,\
                               grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                               magG_name=magG_name,ptID_name=ptID_name,pressure_name=pressure_name)
            self.load_boundaries(path2boundaries,separator=separator,minimum_import=minimum_import,rfactor=rfactor,interpMethod=interpMethod)
            # ---
            self.im('Correction of the number of plate')
            self.nop = np.unique(self.plateID).shape[0]
            mask = self.surf > 0
            self.surf = self.surf[mask]
            self.peri = self.peri[mask]
            self.im('Loading done!')
        else:
            self.im('Wrong files in input, incoherent surface and boundary files')



    def load_surfaces(self,nod,directory,filelist,separator=',',internal_verbose=False,\
             grid_name='Points',cartv_name='Cartesian Velocity',sphev_name='Spherical Velocity',\
             magG_name='magGSV',ptID_name='PointID',pressure_name='Pressure'):
        """
        Load .csv surface files contained in 'directory'.
        nod = number of points in the entire surface
        """
        # Presorting
        fidsurf   = [int(fname.split('_')[1].split('.')[0]) for fname in filelist]
        zipped    = zip(fidsurf,filelist)
        filelist  = list(zip(*sorted(zipped)))[1]
        # General
        self.im(' -> Loading plate surfaces data')
        if self.nop == 0:
            self.nop = len(filelist)  #number of plates
            self.surf = np.zeros(self.nop,dtype=np.int32)
            self.peri = np.zeros(self.nop,dtype=np.int32)
        self.surffile = []
        # Surface (plates)
        self.im('Memory allocation')
        self.x       = np.zeros(nod)
        self.y       = np.zeros(nod)
        self.z       = np.zeros(nod)
        self.vx      = np.zeros(nod)
        self.vy      = np.zeros(nod)
        self.vz      = np.zeros(nod)
        self.vr      = np.zeros(nod)
        self.vtheta  = np.zeros(nod)
        self.vphi    = np.zeros(nod)
        self.pressure= np.zeros(nod)
        self.magGSV  = np.zeros(nod)
        self.plateID = np.zeros(nod,dtype=np.int32)
        self.pointID = np.zeros(nod,dtype=np.int32)
        # index
        self.pstart = np.zeros(self.nop,dtype=np.int32)  # plate i start index
        self.pend   = np.zeros(self.nop,np.int32)  # plate i end index
        #
        self.im('Extraction...')
        for i in range(self.nop):
            pttk = TessellatedPolygon()
            pttk.verbose = internal_verbose
            try:
                self.surfdir = directory
                self.surffile.append(filelist[i])
                pttk.load(directory+filelist[i],separator=separator,\
                          grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                          magG_name=magG_name,ptID_name=ptID_name,pressure_name=pressure_name)
                self.pend[i] = self.pstart[i]+int(len(pttk.x))
                m = self.pstart[i]
                n = self.pend[i]
                self.surf[i]      = int(n-m)
                self.x[m:n]       = pttk.x
                self.y[m:n]       = pttk.y
                self.z[m:n]       = pttk.z
                self.vx[m:n]      = pttk.vx
                self.vy[m:n]      = pttk.vy
                self.vz[m:n]      = pttk.vz
                self.vr[m:n]      = pttk.vr
                self.vtheta[m:n]  = pttk.vtheta
                self.vphi[m:n]    = pttk.vphi
                self.pressure[m:n]= pttk.pressure
                self.magGSV[m:n]  = pttk.magGSV
                self.plateID[m:n] = pttk.plateID
                self.pointID[m:n] = pttk.pointID
                if i < self.nop-1:
                    self.pstart[i+1] = self.pend[i]
            except:
                print('\n'+'-'*40)
                self.im('ERROR detected during the loading of plate file')
                self.im('  Diagnostic:')
                self.im('    - File affected: '+filelist[i])
                if n-m > len(self.x)-m:
                    self.im('    - Size of the file exceeds the save allocated')
                print('-'*40+'\n')
        # resize surfaces
        self.im('Resize the memory space')
        self.x = self.x[0:n]
        self.y = self.y[0:n]
        self.z = self.z[0:n]
        self.vx = self.vx[0:n]
        self.vy = self.vy[0:n]
        self.vz = self.vz[0:n]
        self.vtheta = self.vtheta[0:n]
        self.vphi = self.vphi[0:n]
        self.vr = self.vr[0:n]
        self.pressure = self.pressure[0:n]
        self.magGSV = self.magGSV[0:n]
        self.plateID = self.plateID[0:n]
        self.pointID = self.pointID[0:n]
        self.pmin = self.x.copy() * self.persistence
        # cret lon lat
        self.im('Compute geographical coordinates')
        self.xyz2latlon('Surfaces')
        self.im('Surfaces loaded!')
    


    def load_edges(self,nod,directory,filelist,separator=',',internal_verbose=False,\
             grid_name='Points',cartv_name='Cartesian Velocity',sphev_name='Spherical Velocity',\
             magG_name='magGSV',ptID_name='PointID',pressure_name='Pressure'):
        """
        Load .csv edges files contained in 'directory'.
        nod = number of points in the entire surface
        """
        # Presorting
        fidedge   = [int(fname.split('_')[1].split('.')[0]) for fname in filelist]
        zipped    = zip(fidedge,filelist)
        filelist  = list(zip(*sorted(zipped)))[1]
        # General
        self.im(' -> Loading plate edges data')
        if self.nop == 0:
            self.nop = len(filelist)  #number of plates
            self.surf = np.zeros(self.nop,dtype=np.int32)
            self.peri = np.zeros(self.nop,dtype=np.int32)
        self.edgefile = []
        # Surface (plates)
        self.im('Memory allocation')
        self.xe      = np.zeros(nod)
        self.ye      = np.zeros(nod)
        self.ze      = np.zeros(nod)
        self.vxe     = np.zeros(nod)
        self.vye     = np.zeros(nod)
        self.vze     = np.zeros(nod)
        self.vre     = np.zeros(nod)
        self.vthetae = np.zeros(nod)
        self.vphie   = np.zeros(nod)
        self.pressuree=np.zeros(nod)
        self.magGSVe = np.zeros(nod)
        self.plateIDe= np.zeros(nod,dtype=np.int32)
        self.pointIDe= np.zeros(nod,dtype=np.int32)
        # index
        self.estart = np.zeros(self.nop,dtype=np.int32)  # plate i start index
        self.eend   = np.zeros(self.nop,np.int32)  # plate i end index
        #
        self.im('Extraction...')
        for i in range(self.nop):
            pttk = TessellatedPolygon()
            pttk.verbose = internal_verbose
            try:
                self.edgedir = directory
                self.edgefile.append(filelist[i])
                pttk.load(directory+filelist[i],separator=separator,\
                          grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                          magG_name=magG_name,ptID_name=ptID_name,pressure_name=pressure_name)
                self.eend[i] = self.estart[i]+int(len(pttk.x))
                m = self.estart[i]
                n = self.eend[i]
                self.peri[i]       = int(n-m)
                self.xe[m:n]       = pttk.x
                self.ye[m:n]       = pttk.y
                self.ze[m:n]       = pttk.z
                self.vxe[m:n]      = pttk.vx
                self.vye[m:n]      = pttk.vy
                self.vze[m:n]      = pttk.vz
                self.vre[m:n]      = pttk.vr
                self.vthetae[m:n]  = pttk.vtheta
                self.vphie[m:n]    = pttk.vphi
                self.pressuree[m:n]= pttk.pressure
                self.magGSVe[m:n]  = pttk.magGSV
                self.plateIDe[m:n] = pttk.plateID
                self.pointIDe[m:n] = pttk.pointID
                if i < self.nop-1:
                    self.estart[i+1] = self.eend[i]
            except:
                print('-'*40)
                self.im("ERROR in the file loading")
                print(filelist[i])
                print('-'*40)
        # resize boundaries
        self.im('Resize the memory space')
        self.xe = self.xe[0:n]
        self.ye = self.ye[0:n]
        self.ze = self.ze[0:n]
        self.vxe = self.vxe[0:n]
        self.vye = self.vye[0:n]
        self.vze = self.vze[0:n]
        self.vthetae = self.vthetae[0:n]
        self.vphie = self.vphie[0:n]
        self.vre = self.vre[0:n]
        self.pressuree = self.pressuree[0:n]
        self.magGSVe = self.magGSVe[0:n]
        self.plateIDe = self.plateIDe[0:n]
        self.pointIDe = self.pointIDe[0:n]
        self.pmine = self.xe.copy() * self.persistence
        # cret lon lat
        self.im('Compute geographical coordinates')
        self.xyz2latlon('Edges')
        self.im('Edeges loaded!')
    


    def load_boundaries(self,path2file,separator=',',minimum_import=False,rfactor=1,interpMethod='linear'):
        """
        Load .csv boundary files contained in 'directory'.
        nod = number of points in the entire surface
        """
        # General
        self.im(' -> Loading plate bondaries data')
        self.boundfile = path2file
        # Read coordinates
        self.im('   - Get coordinates')
        self.xb = []
        self.yb = []
        self.zb = []
        with open(path2file,'r') as data:
            header = data.readline()
            for line in data:
                line = line.strip().split(separator)
                self.xb.append(float(line[-3]))
                self.yb.append(float(line[-2]))
                self.zb.append(float(line[-1]))
        self.xb = np.array(self.xb)[::rfactor]
        self.yb = np.array(self.yb)[::rfactor]
        self.zb = np.array(self.zb)[::rfactor]
        nod  = len(self.xb)
        # creat lon lat
        self.im('   - Compute geographical coordinates')
        self.xyz2latlon('Boundaries')
        # Get plateID
        self.im('   - Get plate ID')
        if np.unique(self.plateID).shape[0] == 1:
            # means that there is only one plate
            self.platecouple = np.zeros((nod,2),dtype=np.int32)
            # ----
            # interpolate from surface data
            self.im('   - Interpolation from Surface data')
            points = np.array([(self.lon[i],self.lat[i]) for i in range(len(self.lon))])
            self.im('      - Interpolation of vr')
            self.vrb     = griddata(points, self.vr, (self.lonb, self.latb), method=interpMethod)
            self.im('      - Interpolation of vtheta')
            self.vthetab = griddata(points, self.vtheta, (self.lonb, self.latb), method=interpMethod)
            self.im('      - Interpolation of vphi')
            self.vphib   = griddata(points, self.vphi, (self.lonb, self.latb), method=interpMethod)
            if not minimum_import:
                self.im('      - Interpolation of the pressure')
                self.pressureb = griddata(points, self.pressure, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vx')
                self.vxb     = griddata(points, self.vx, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vy')
                self.vyb     = griddata(points, self.vy, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vz')
                self.vzb     = griddata(points, self.vz, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of magGSV')
                self.magGSVb = griddata(points, self.magGSV, (self.lonb, self.latb), method=interpMethod)
                self.im('   - Remove NaN')
                mask = ~np.isnan(self.vrb) * ~np.isnan(self.vthetab) * ~np.isnan(self.vphib) *\
                    ~np.isnan(self.vxb) * ~np.isnan(self.vyb) * ~np.isnan(self.vzb) * ~np.isnan(self.magGSVb)
            else:
                self.im('   - Remove NaN')
                mask = ~np.isnan(self.vrb) * ~np.isnan(self.vthetab) * ~np.isnan(self.vphib)
        else:
            self.im('Attribution of plate couple')
            # --- auto compute the step based on the average (over 'nTests' tests) point spacing
            noTests = 50
            tested_step = np.zeros(noTests,dtype=np.float64)
            for i in range(noTests): # 5 test
                ptID = np.random.randint(0,len(self.x))
                mydist = np.sqrt((self.x[ptID]-self.x)**2+(self.y[ptID]-self.y)**2+(self.z[ptID]-self.z)**2)
                mydist = mydist[mydist>0]
                tested_step[i] = np.amin(mydist)
            distance_cluster_threshold = 2 * np.mean(tested_step)
            # --- call external function
            time0 = time()
            self.platecouple = compute_platecouple(distance_cluster_threshold,self.xe,self.ye,self.ze,self.plateIDe,self.x,self.y,self.z,self.plateID,self.xb,self.yb,self.zb)
            self.im('  -> Process done: '+str(time()-time0)+' sec')
            # ----
            # interpolate from edges data
            self.im('   - Interpolation from Edges data')
            points = np.array([(self.lone[i],self.late[i]) for i in range(len(self.lone))])
            self.im('      - Interpolation of vr')
            self.vrb     = griddata(points, self.vre, (self.lonb, self.latb), method=interpMethod)
            self.im('      - Interpolation of vtheta')
            self.vthetab = griddata(points, self.vthetae, (self.lonb, self.latb), method=interpMethod)
            self.im('      - Interpolation of vphi')
            self.vphib   = griddata(points, self.vphie, (self.lonb, self.latb), method=interpMethod)
            if not minimum_import:
                self.im('      - Interpolation of the pressure')
                self.pressureb     = griddata(points, self.pressuree, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vx')
                self.vxb     = griddata(points, self.vxe, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vy')
                self.vyb     = griddata(points, self.vye, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of vz')
                self.vzb     = griddata(points, self.vze, (self.lonb, self.latb), method=interpMethod)
                self.im('      - Interpolation of magGSV')
                self.magGSVb = griddata(points, self.magGSVe, (self.lonb, self.latb), method=interpMethod)
                self.im('   - Remove NaN')
                mask = ~np.isnan(self.vrb) * ~np.isnan(self.vthetab) * ~np.isnan(self.vphib) *\
                    ~np.isnan(self.vxb) * ~np.isnan(self.vyb) * ~np.isnan(self.vzb) * ~np.isnan(self.magGSVb)
            else:
                self.im('   - Remove NaN')
                mask = ~np.isnan(self.vrb) * ~np.isnan(self.vthetab) * ~np.isnan(self.vphib)
        # ---
        self.xb = self.xb[mask]
        self.yb = self.yb[mask]
        self.zb = self.zb[mask]
        self.rb = self.rb[mask]
        self.lonb = self.lonb[mask]
        self.latb = self.latb[mask]
        self.vrb = self.vrb[mask]
        self.vphib = self.vphib[mask]
        self.vthetab = self.vthetab[mask]
        self.platecouple = self.platecouple[mask,:]
        if not minimum_import:
            self.vxb = self.vxb[mask]
            self.vyb = self.vyb[mask]
            self.vzb = self.vzb[mask]
            self.magGSVb = self.magGSVb[mask]
            self.pressureb = self.pressureb[mask]
        self.pminb = self.xb.copy() * self.persistence
        # ---
        self.im('      - Number of Nan points removed: '+str(np.count_nonzero(~mask)))
        self.im('Boundaries loaded!')
    
    

    def compute_platecouple(self):
        """Compute automatically the pairing of plate at plate boundary
        and fill the field self.platecouple."""
        self.im('Attribution of plate couple')
        # --- auto compute the step based on the average (over 'nTests' tests) point spacing
        noTests = 50
        tested_step = np.zeros(noTests,dtype=np.float64)
        for i in range(noTests):
            ptID = np.random.randint(0,len(self.x))
            mydist = np.sqrt((self.x[ptID]-self.x)**2+(self.y[ptID]-self.y)**2+(self.z[ptID]-self.z)**2)
            mydist = mydist[mydist>0]
            tested_step[i] = np.amin(mydist)
        distance_cluster_threshold = 2 * np.mean(tested_step)
        # --- call numba
        time0 = time()
        self.platecouple = compute_platecouple(distance_cluster_threshold,self.xe,self.ye,self.ze,self.plateIDe,self.x,self.y,self.z,self.plateID,self.xb,self.yb,self.zb)
        self.im('  -> Process done: '+str(time()-time0)+' sec')


    
    def load_magGSVb(self,interpMethod='linear'):
        """
        Estimates the field magGSVb on the boundary points
        from interpolation on edge points.

        Args:
            interpMethod (str, optional): Method of scipy.interpolate.griddata
                                        to interpolate data (arg method).
                                        Defaults: interpMethod='linear'
        """
        self.im('Compute magGSV on plate boundaries by interpolation')
        self.im('  - interp method: '+interpMethod)
        points = np.concatenate((self.lone,self.late)).reshape((2,self.lone.shape[0])).T
        # interpolate from edges data
        self.im('      - Interpolation of magGSV')
        self.magGSVb = griddata(points, self.magGSVe, (self.lonb, self.latb), method=interpMethod)
        self.im('   - Remove NaN')
        mask =  ~np.isnan(self.magGSVb)
        # ---
        self.magGSVb = self.magGSVb[mask]
        self.im('      - Number of Nan points removed: '+str(np.count_nonzero(~mask)))
        self.im('Magnitude of the velocity gradient loaded on plate boundaries!')
    

    def load_from_h5(self,path2h5):
        """
        Function replacing the routine self.load when the dataset has been
        previously converted into h5 format.

        Args:
            path2h5 (str): Path to the .h5 data containing the plate tessellation.
                        This file can be generated from the inner funtion
                        self.export2h5 from a collection of .csv files read with
                        the function self.load, or with the function
                        MAPT3.optimize.optimize.
        """
        fid  = h5py.File(path2h5,'r')
        self.path2file = path2h5
        keys = fid.keys()
        self.im('Loading plate data from h5 file!')
        self.im('  - Source file: '+path2h5)
        self.im('Loading...')
        # Plate Surfaces data
        self.nop = np.array(fid['nop'])
        if 'persistence' in keys:
            self.persistence = np.array(fid['persistence'])
            self.pmin  = np.array(fid['pmin'])
            self.pmine = np.array(fid['pmine'])
            self.pminb = np.array(fid['pminb'])
        self.surf = np.array(fid['surf'])
        self.peri = np.array(fid['peri'])
        self.x = np.array(fid['x'])
        self.y = np.array(fid['y'])
        self.z = np.array(fid['z'])
        self.vx = np.array(fid['vx'])
        self.vy = np.array(fid['vy'])
        self.vz = np.array(fid['vz'])
        self.vtheta = np.array(fid['vtheta'])
        self.vphi = np.array(fid['vphi'])
        self.vr = np.array(fid['vr'])
        self.pressure = np.array(fid['pressure'])
        self.magGSV = np.array(fid['magGSV'])
        self.plateID = np.array(fid['plateID'])
        self.pointID = np.array(fid['pointID'])
        self.pstart = np.array(fid['pstart'])
        self.pend = np.array(fid['pend'])
        self.r = np.array(fid['r'])
        self.lat = np.array(fid['lat'])
        self.lon = np.array(fid['lon'])
        # Plate Edges data
        self.xe = np.array(fid['xe'])
        self.ye = np.array(fid['ye'])
        self.ze = np.array(fid['ze'])
        self.vxe = np.array(fid['vxe'])
        self.vye = np.array(fid['vye'])
        self.vze = np.array(fid['vze'])
        self.vthetae = np.array(fid['vthetae'])
        self.vphie = np.array(fid['vphie'])
        self.vre = np.array(fid['vre'])
        self.pressuree = np.array(fid['pressuree'])
        self.magGSVe = np.array(fid['magGSVe'])
        self.plateIDe = np.array(fid['plateIDe'])
        self.pointIDe = np.array(fid['pointIDe'])
        self.estart = np.array(fid['estart'])
        self.eend = np.array(fid['eend'])
        self.re = np.array(fid['re'])
        self.late = np.array(fid['late'])
        self.lone = np.array(fid['lone'])
        # Plate Boundary data
        self.xb = np.array(fid['xb'])
        self.yb = np.array(fid['yb'])
        self.zb = np.array(fid['zb'])
        self.rb = np.array(fid['rb'])
        self.lonb = np.array(fid['lonb'])
        self.latb = np.array(fid['latb'])
        self.vrb = np.array(fid['vrb'])
        self.vphib = np.array(fid['vphib'])
        self.vthetab = np.array(fid['vthetab'])
        self.platecouple = np.array(fid['platecouple'])
        self.vxb = np.array(fid['vxb'])
        self.vyb = np.array(fid['vyb'])
        self.vzb = np.array(fid['vzb'])
        self.magGSVb = np.array(fid['magGSVb'])
        self.pressureb = np.array(fid['pressureb'])
        # non rigid parts
        if 'xnr' in keys:
            self.xnr = np.array(fid['xnr'])
            self.ynr = np.array(fid['ynr'])
            self.znr = np.array(fid['znr'])
            self.vxnr = np.array(fid['vxnr'])
            self.vynr = np.array(fid['vynr'])
            self.vznr = np.array(fid['vznr'])
            self.vrnr = np.array(fid['vrnr'])
            self.vthetanr = np.array(fid['vthetanr'])
            self.vphinr = np.array(fid['vphinr'])
            self.pressurenr = np.array(fid['pressurenr'])
            self.magGSVnr = np.array(fid['magGSVnr'])
            self.pminnr = np.array(fid['pminnr'])
            self.pointIDnr = np.array(fid['pointIDnr'])
            # Geographic grid
            self.lonnr = np.array(fid['lonnr'])
            self.latnr = np.array(fid['latnr'])
            self.rnr = np.array(fid['rnr'])
        # Check number of plates:
        nop = np.unique(self.plateID).shape[0]
        if self.nop != nop:
            self.nop = nop
            self.im(' -> The number of plates has been corrected')
        if self.surf.shape[0] != self.nop:
            mask = self.surf > 0
            self.surf = self.surf[mask]
            self.peri = self.peri[mask]
            if self.surf.shape[0] == self.nop:
                self.im(' -> The plate surface/peri have been corrected')
            else:
                self.im('  -> WARNING. Uncompatible number of plates and length of plate surface and perimeter lists')
        # close
        fid.close()
        self.im('File loaded successfully!')

    

    def load_polygons(self,path2shp,path2dbf,plot=True):
        """
        Import another tessallation from a shapefile polygon file.
        this function will filled the internal field plateIDpoly.
        """
        # open and read shapefile
        self.im('Import another tessallation from a shapefile polygon file.')
        myshpR = open(path2shp, "rb")
        mydbfR = open(path2dbf, "rb")
        seg    = shapefile.Reader(shp=myshpR, dbf=mydbfR)
        self.poly_path2shp = path2shp
        self.poly_path2dbf = path2dbf

        lon     = []    # longitudes of points composing the polygons
        lat     = []    # latitudes of points composing the polygons
        plateID = []    # plateID for each points on each polygons
        k     = 0
        #Iterate through shapes in shapefile
        self.im('   - Iterate through shapes in shapefile')
        for shape in seg.shapeRecords():
            arr = np.asarray(shape.shape.points)
            loni     = arr[:,0]
            lati     = arr[:,1]
            j = 0
            sign = np.sign(loni[j])
            for i in range(1,len(loni)):
                if np.sign(loni[i]) != sign and abs(loni[i]-loni[i-1])>180:
                    lon.append(loni[j:i-1])
                    lat.append(lati[j:i-1])
                    plateID.append(np.ones(len(lati[j:i-1]))*k)
                    j = i
                sign = np.sign(loni[i])
            lon.append(loni[j:len(loni)])
            lat.append(lati[j:len(lati)])
            plateID.append(np.ones(len(lati[j:len(lati)]))*k)
            k += 1
        
        nopoly  = len(lon)  #number of polygons
        noplate = k         # number of plates
        self.im('   - Number of polygons: '+str(nopoly))
        self.im('   - Number of plates:   '+str(noplate))
        self.nop_poly = noplate
        
        self.plateIDpoly = np.empty(len(self.lon))*np.nan
        # iterate on the number of plates
        self.im('   - Iterative plate ID attribution.')
        for k in range(noplate):
            # points on all the PlateGather object
            points  = [[self.lon[i],self.lat[i]] for i in range(len(self.lon))]
            #stack of the polygons forming the same plate
            lonk = []
            latk = []
            for i in range(len(lon)):
                if plateID[i][0] == k:
                    lonk += list(lon[i])
                    latk += list(lat[i])
                    
            # correction for polar plates #### NEED TO BE GENERALIZED #####
            if k == 18: 
                zipped = zip(lonk,latk)
                slist  = list(zip(*sorted(zipped)))
                lonk = list(slist[0])
                latk = list(slist[1])
                if lonk[0] > -180:
                    lonk.insert(0,-180)
                    latk.insert(0,latk[0])
                if lonk[-1] < 180:
                    lonk.insert(len(lonk),180)
                    latk.insert(len(latk),latk[-1])
                lonk.insert(len(lonk),180)
                latk.insert(len(latk),-90)
                lonk.insert(0,-180)
                latk.insert(0,-90)
                
            lonk = np.array(lonk)
            latk = np.array(latk)
            # find the minimum angle theta to have just 1 polygon
            theta = 0
            ccontinue = True
            while theta < 360 and ccontinue:
                nlon = lonk + theta
                gind = nlon > 180
                nlon[gind] = -180+(nlon[gind]-180)
                if np.amax(nlon) <= 170:
                    ccontinue = False
                else:
                    theta += 1
            if theta == 360:
                nlon  = lonk
                theta = 0
            # add the theta correction on the points lon coordinates
            for i in range(len(points)):
                points[i][0] = points[i][0]+theta
                if points[i][0] > 180:
                    points[i][0] = -180+(points[i][0]-180)
            # the new polygon
            polygon = [[nlon[i],latk[i]] for i in range(len(nlon))]
            # Matplotlib mplPath
            path = mpltPath.Path(polygon)
            inside = path.contains_points(points)
            self.plateIDpoly[inside] = k
        
        # rescale self.plateIDpoly between 0 and self.nop_poly-1
        self.plateIDpoly -= 1
        
        # correction of 0 size plate
        for i in range(self.nop_poly):
            mask      = self.plateIDpoly == i
            maskupper = self.plateIDpoly > i
            if np.count_nonzero(mask) == 0 and np.count_nonzero(maskupper) > np.count_nonzero(np.isnan(self.plateIDpoly)):
                self.plateIDpoly[maskupper] -= 1
        nop_poly = int(np.amax(self.plateIDpoly[~np.isnan(self.plateIDpoly)])+1)
        self.im('Number of empty plates removes: '+str(int(self.nop_poly-nop_poly)))
        self.nop_poly = nop_poly
        self.im('Number of points without plate: '+str(np.count_nonzero(np.isnan(self.plateIDpoly))))
        self.im('New number of plate: '+str(self.nop_poly))
        
        if plot:
            self.im('   - Plot the resulting tessallation')
            # Create a colormap to pick up the color of the line
            cNorm  = colors.Normalize(vmin=0, vmax=noplate)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='viridis')
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
            ax.set_global()
            ax.set_title("Tessellation map from an input shapefile")
            ax.scatter(self.lon,self.lat,c=self.plateIDpoly,s=1,transform=ccrs.PlateCarree())
            for k in range(k):
                for i in range(len(lon)):
                    if plateID[i][0] == k:
                        ax.plot(lon[i],lat[i],'-',color='black',transform=ccrs.PlateCarree())
            plt.show()
    


    def get_plateID(self,lon,lat,on_poly=False):
        """
        Returns the plate ID near a point defined by the input lon/lat
        
        Args:
            lon (int/float): longitude of the point (in degree)
            lat (int/float): latidude of the point (in degree)
            on_poly (bool, optional): If True then the plate ID will be
                        search on the plateIDpoly field (external tessellation
                        from a polygon file).
                        Defaults: on_poly = False
        
        Returns:
            minind (ind): Indice on the field self.plateID or self.plateIDpoly
                        of the closet point to the input point (lon/lat).
                        The ID of the plate on which the point (lon/lat) is 
                        located is:
                        >> pid = self.plateID[minind] 
        """
        # First sort: isolate a 2deg x 2deg box around the point
        latmax = lat + 1
        latmin = lat - 1
        lonmax = lon + 1
        lonmin = lon - 1
        cond1 = self.lon >= lonmin
        cond2 = self.lon <= lonmax
        cond1 = cond1 * cond2
        cond2 = self.lat >= latmin
        cond1 = cond1 * cond2
        cond2 = self.lat <= latmax
        cond1 = cond1 * cond2
        ind = np.array(range(len(self.x)))
        ind = ind[cond1]
        dist = np.zeros(len(ind))
        for i in range(len(ind)):
            reflon = self.lon[ind[i]]*np.pi/180
            reflat = self.lat[ind[i]]*np.pi/180
            dist[i] = distance_2points(lon*np.pi/180,lat*np.pi/180,reflon,reflat)
        minind = np.where(dist == np.amin(dist))[0][0]
        minind = ind[minind]
        self.im('Index of the nearest grid point: '+str(minind))
        if not on_poly:
            self.im('Points located on plate '+str(self.plateID[minind]))
        else:
            self.im('Points located on plate '+str(self.plateIDpoly[minind]))
        return minind
    

    
    def grid_rotation(self,axis='x',theta=1*np.pi/180,R=None):
        """
        Function for the rotation of the grid defined as (1) a rotation
        around a given cartesian axis or (2) as an angular rotation vector.
        (Defaults: rotation around an axis).

        Args:
            axis (str, optional): Rotation around a cartesian axis.
                        Have to be in ['x','y','z']
                        Defaults: axis = 'x'
              theta (int/float, optional): angle of the rotation
                        in *RADIANS*.
                        Defaults: theta = 1*np.pi/180
              R (np.ndarray, shape = (3,3)): Rotation defined by a
                        3x3 rotation matrix.
                        Defaults: R=None
        """
        self.im('Rotation of the grid and the velocity vectors')
        if R is None:
            self.im('  -> axis:  '+axis)
            self.im('  -> theta: '+str(theta))
            R = rotation_matrix_3D(axis,theta)
            self.grot_format.append('axis')
            self.grot_axis.append(axis)
            self.grot_theta.append(theta)
        else:
            self.im('  -> Rotation matrix: '+str(R))
            self.grot_format.append('R')
            self.grot_axis.append(None)
            self.grot_theta.append(None)
        # ---
        self.im('  ----> Grid rotation:')
        self.im('     -> Rotate the surface points')
        x = R[0,0]*self.x+R[0,1]*self.y+R[0,2]*self.z
        y = R[1,0]*self.x+R[1,1]*self.y+R[1,2]*self.z
        z = R[2,0]*self.x+R[2,1]*self.y+R[2,2]*self.z
        self.x = x
        self.y = y
        self.z = z
        self.xyz2latlon('Surfaces')
        self.im('     -> Rotate the edges points')
        x = R[0,0]*self.xe+R[0,1]*self.ye+R[0,2]*self.ze
        y = R[1,0]*self.xe+R[1,1]*self.ye+R[1,2]*self.ze
        z = R[2,0]*self.xe+R[2,1]*self.ye+R[2,2]*self.ze
        self.xe = x
        self.ye = y
        self.ze = z
        self.xyz2latlon('Edges')
        self.im('     -> Rotate the boundaries points')
        x = R[0,0]*self.xb+R[0,1]*self.yb+R[0,2]*self.zb
        y = R[1,0]*self.xb+R[1,1]*self.yb+R[1,2]*self.zb
        z = R[2,0]*self.xb+R[2,1]*self.yb+R[2,2]*self.zb
        self.xb = x
        self.yb = y
        self.zb = z
        self.xyz2latlon('Boundaries')
        self.im('  ----> Vectors rotation:')
        self.im('     -> Rotate the surface points')
        vx = R[0,0]*self.vx+R[0,1]*self.vy+R[0,2]*self.vz
        vy = R[1,0]*self.vx+R[1,1]*self.vy+R[1,2]*self.vz
        vz = R[2,0]*self.vx+R[2,1]*self.vy+R[2,2]*self.vz
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.vphi,self.vtheta,self.vr = ecef2enu_stagYY(self.x,self.y,self.z,self.vx,self.vy,self.vz)
        self.im('     -> Rotate the edges points')
        vx = R[0,0]*self.vxe+R[0,1]*self.vye+R[0,2]*self.vze
        vy = R[1,0]*self.vxe+R[1,1]*self.vye+R[1,2]*self.vze
        vz = R[2,0]*self.vxe+R[2,1]*self.vye+R[2,2]*self.vze
        self.vxe = vx
        self.vye = vy
        self.vze = vz
        self.vphie,self.vthetae,self.vre = ecef2enu_stagYY(self.xe,self.ye,self.ze,self.vxe,self.vye,self.vze)
        self.im('     -> Rotate the boundaries points')
        vx = R[0,0]*self.vxb+R[0,1]*self.vyb+R[0,2]*self.vzb
        vy = R[1,0]*self.vxb+R[1,1]*self.vyb+R[1,2]*self.vzb
        vz = R[2,0]*self.vxb+R[2,1]*self.vyb+R[2,2]*self.vzb
        self.vxb = vx
        self.vyb = vy
        self.vzb = vz
        self.vphib,self.vthetab,self.vrb = ecef2enu_stagYY(self.xb,self.yb,self.zb,self.vxb,self.vyb,self.vzb)
        

    
    def grid_rotation_reset(self):
        """Function reseting the geometry of the grid after a rotation
        with the function self.grid_rotation()."""
        nod = len(self.grot_axis)
        self.im('Number of rotation to reset: '+str(nod))
        for i in range(nod):
            if 'R' in self.grot_format:
                self.im('You cannot reset a rotation define by a matrix this way yet. Take a coffee and call your developer!', error=True)
                return None
        for i in range(nod):
            self.im('  -> Reset rotation '+str(nod-1-i))
            axis  = self.grot_axis[nod-1-i]
            theta = -self.grot_theta[nod-1-i]
            R = rotation_matrix_3D(axis,theta)
            x = R[0,0]*self.x+R[0,1]*self.y+R[0,2]*self.z
            y = R[1,0]*self.x+R[1,1]*self.y+R[1,2]*self.z
            z = R[2,0]*self.x+R[2,1]*self.y+R[2,2]*self.z
            self.x = x
            self.y = y
            self.z = z
            self.xyz2latlon('Surfaces')
            x = R[0,0]*self.xe+R[0,1]*self.ye+R[0,2]*self.ze
            y = R[1,0]*self.xe+R[1,1]*self.ye+R[1,2]*self.ze
            z = R[2,0]*self.xe+R[2,1]*self.ye+R[2,2]*self.ze
            self.xe = x
            self.ye = y
            self.ze = z
            self.xyz2latlon('Edges')
            x = R[0,0]*self.xb+R[0,1]*self.yb+R[0,2]*self.zb
            y = R[1,0]*self.xb+R[1,1]*self.yb+R[1,2]*self.zb
            z = R[2,0]*self.xb+R[2,1]*self.yb+R[2,2]*self.zb
            self.xb = x
            self.yb = y
            self.zb = z
            self.xyz2latlon('Boundaries')
        self.im('The geometry of the grid has been reseted ')
    


    def get_plateboundary_membership_mask(self,plate1,plate2):
        """
        Returns the mask for boundary point according to a plate
        membership condition: mask the points belonging to the plate
        boundary between 'plate1' and 'plate2).
        (removes boundary points that are within a plate).

        Args:
            plate1 (int): plate ID 1
            plate2 (int): plate ID 2
        
        Returns:
            cond1 (np.ndarray, shape=self.xb.shape, dtype=bool):
                    Mask of membership condition.
        """
        cond1 = np.logical_and(self.platecouple[:,0] == plate1, self.platecouple[:,1] == plate2)
        cond2 = np.logical_and(self.platecouple[:,1] == plate1, self.platecouple[:,0] == plate2)
        cond1 = cond1 + cond2
        cond2 = np.logical_and(self.platecouple[:,1] == plate1, self.platecouple[:,0] == plate1)
        cond1 = cond1 * ~cond2
        cond2 = np.logical_and(self.platecouple[:,1] == plate2, self.platecouple[:,0] == plate2)
        cond1 = cond1 * ~cond2
        return cond1

    
    def get_neighborhood(self,on_poly=False):
        """
        Get for each plate it neighborhood (i.e. the ID of each plate/polygon
        surrounding each plate).
        The result is stored in the internal field 'self.neighborhood' (list
        of list).

        Args:
            on_poly (bool, optional): Option controling if you want to apply
                        the function on the plateIDpoly field (external
                        tessellation from a polygon file).
                        Defaults: on_poly = False
        """
        self.im('Get the neighborhood of all plates')
        if not on_poly:
            self.neighborhood = []
            for i in range(self.nop):
                left  = np.unique(self.platecouple[self.platecouple[:,0] == i][:,1])
                right = np.unique(self.platecouple[self.platecouple[:,1] == i][:,0])
                self.neighborhood.append(np.unique(np.concatenate((left,right))))
        else:
            self.neighborhood = [[] for i in range(self.nop_poly)]
            # define a distance metric
            distL2 = lambda x1,y1,z1,x2,y2,z2: np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
            # estimation of average min distance between 2 points:
            self.im('   - Estimation of average min distance between 2 points')
            mindist = 0
            for i in range(10):
                n = random.randint(0,len(self.x)-1) # take a random point
                x = np.delete(self.x,n)
                y = np.delete(self.y,n)
                z = np.delete(self.z,n)
                mindist += np.amin(distL2(self.x[n],self.y[n],self.z[n],x,y,z))
            mindist = mindist/10
            self.im('      -> average mindist = '+str(mindist))
            nop     = self.nop_poly
            plateID = self.plateIDpoly
            self.im('   - Iteration on all plateIDpoly')
            for i in tqdm(range(nop)):
                mask   = plateID == i
                for j in range(np.count_nonzero(mask)):
                    xj    = self.x[mask][j]
                    yj    = self.y[mask][j]
                    zj    = self.z[mask][j]
                    dist  = distL2(xj,yj,zj,self.x,self.y,self.z)
                    mask2 = dist <= 3*mindist
                    id = plateID[mask2]
                    if len(np.unique(id)) != 1:
                        for j in range(len(np.unique(id))):
                            if id[j] not in self.neighborhood[i] and id[j]!= i:
                                if ~np.isnan(id[j]):
                                    self.neighborhood[i].append(int(id[j]))
        self.im('   - Neighborhood complete!')
    


    def get_rotation(self,plate1,remove_edges=False,r=100,rplot=1,on_poly=False,plot=True,maxiter=100,verbose=False):
        """
        Get the best rotation fitting the velocity field of 'plate1'.

        Args:
            plate1 (int): ID of the plate/polygon whose velocity field you want to invert.
            r (int, optional): Parameter controling the resampling of the data
                        to inverse. The function called will resample randomly
                        the surface velocity field of the plate/polygon 'plate1'.
                        Otherwise, can be very long to large plate.
                        Defaults: r = 100
            
            remove_edges (bool, optional): Option controling if you want to remove
                        edege points from the data to invert.
                        Default: remove_edges = False
            maxiter (int):  maximum number of iteration to converge.
                        Defaults: maxiter=100
            on_poly (bool, optional): Option controling if you want to apply
                        the function on the plateIDpoly field (external
                        tessellation from a polygon file).
                        Defaults: on_poly = False
            plot (bool, optional): If true, generates a figure showing the result
                        of the inversion.
                        Defaults: plot = True
            rplot (int, optional): Only if plot is True. Controls the number of
                        data point showed according to the data set considered
                        for the inversion (so, including the resampling with the
                        parameter 'r').
                        Default: rplot = 1
            verbose (bool, optional): If True then, generate a verbose output
                        in the terminal.
                        Defaults: verbose = False
        """
        if remove_edges:
            # prepare header
            grid_name,cartv_name,sphev_name,magG_name,ptID_name = self.header
        self.invrot_rfactor = r
        self.plate1 = plate1
        self.im('Get Rotation pole of plate '+str(plate1))
        self.im('   - Build TessellatedPolygon object for the plate 1')
        self.im('       -> plate ID: '+str(plate1))
        self.im('       -> rfactor : '+str(r))
        pttk1 = TessellatedPolygon()
        if not on_poly:
            if remove_edges:
                pttk1.load(self.surfdir+self.surffile[plate1],\
                        grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                        magG_name=magG_name,ptID_name=ptID_name)
                pttk1.remove_edges_from_file(self.edgedir+self.edgefile[plate1])
            else:
                cond1 = self.plateID == self.plate1
                pttk1.x  = self.x[cond1]
                pttk1.y  = self.y[cond1]
                pttk1.z  = self.z[cond1]
                pttk1.vx = self.vx[cond1]
                pttk1.vy = self.vy[cond1]
                pttk1.vz = self.vz[cond1]
        else:
            # membership condition
            cond1 = self.plateIDpoly == self.plate1
            pttk1.x  = self.x[cond1]
            pttk1.y  = self.y[cond1]
            pttk1.z  = self.z[cond1]
            pttk1.vx = self.vx[cond1]
            pttk1.vy = self.vy[cond1]
            pttk1.vz = self.vz[cond1]
        self.im('   - Inverse for the rotation pole of PLATE1 in FIX-MANTLE')
        self.wx1,self.wy1,self.wz1,self.residual1,self.misfit_xyz1,self.misfit_enu1,self.misfit_normcrossprod1,self.P11,self.P12,self.P1loc,self.P2loc,self.rotmask = regEEPP(pttk1,r=r,rplot=rplot,verbose=verbose,plot=plot,maxiter=maxiter)
        self.im('      - plateness P1 for the plate1: '+str(self.P11))
        self.im('      - plateness P2 for the plate1: '+str(self.P12))
        latp1,lonp1,omega1 = wxwywz2latlonw(self.wx1,self.wy1,self.wz1)  # result already on degrees
        self.im('     Rotation pole of PLATE1:')
        self.im('      - Lat: '+str(int(100*latp1)/100)+' deg N')
        self.im('      - Lon: '+str(int(100*lonp1)/100)+' deg E')
        self.im('      - Omega: '+str(int(100*omega1)/100)+' deg/Ma')
        return self.wx1,self.wy1,self.wz1



    def get_all_rotations(self,r=None,remove_edges=False,rplot=1,on_poly=False,plot=False,verbose=False,ignore_small=True,smin=Project.polyminsize,maxiter=10):
        """
        Computes all the rotations fitting the displacement field of all plates/polygons.
        
        Args:
            r (None/int, optional): Parameter controling the resampling of the data
                        to inverse. The function called will resample randomly
                        the surface velocity field of the plate/polygon 'plate1'.
                        Otherwise, can be very long to large plate.
                        If r is None then, apply a resampling parameter r depending
                        on the number of points composing the plate surface.
                        Else, if r is an integer then, apply the same resampling
                        parameter to all the inverted plates/polygons.
                        Defaults: r = None
            ignore_small (bool, optional): Option to ignore the smallest plate
                        have less than 'smin' points in their surface.
                        Defaults: ignore_small = True
            smin (int, optional): minimum number of points composing the surface
                        of a plate/polygon to be considered during the inversion if
                        the option 'ignore_small' is set to True.
                        Defaults: MAPT3.project.Project.polyminsize
            remove_edges (bool, optional): Option controling if you want to remove
                        edege points from the data to invert.
                        Default: remove_edges = False
            maxiter (int):  maximum number of iteration to converge.
                        Defaults: maxiter=100
            on_poly (bool, optional): Option controling if you want to apply
                        the function on the plateIDpoly field (external
                        tessellation from a polygon file).
                        Defaults: on_poly = False
            plot (bool, optional): If true, generates a figure showing the result
                        of the inversion.
                        Defaults: plot = True
            rplot (int, optional): Only if plot is True. Controls the number of
                        data point showed according to the data set considered
                        for the inversion (so, including the resampling with the
                        parameter 'r').
                        Default: rplot = 1
            verbose (bool, optional): If True then, generate a verbose output
                        in the terminal.
                        Defaults: verbose = False
        """
        if r is None:
            save_r = None
        else:
            save_r = r
        if remove_edges:
            # prepare header
            grid_name,cartv_name,sphev_name,magG_name,ptID_name = self.header
        self.is_all_rotations = True
        if not on_poly:
            nop     = self.nop
            plateID = self.plateID
        else:
            nop     = self.nop_poly
            plateID = self.plateIDpoly
        nod         = len(self.x)
        self.P1     = np.zeros(nop)
        self.rotmask= np.zeros(nod,dtype=bool)
        self.P2     = np.zeros(nop)
        self.misfit = np.zeros(nod)*np.nan
        self.misfit_normcrossprod = np.zeros(nod)*np.nan
        self.wx     = np.zeros(nop)
        self.wy     = np.zeros(nop)
        self.wz     = np.zeros(nop)
        id = np.array(range(nod))
        upID = np.unique(self.plateID)
        for i in range(nop):
            pID = upID[i]
            surf  = np.count_nonzero(plateID==pID)
            large = surf > smin
            if large:
                run_condition = True
            elif not large and not ignore_small:
                run_condition = True
            else:
                run_condition = False
            
            if run_condition:
                # ----
                if save_r is None:
                    if surf < 100:
                        r = 1
                    elif surf >= 100 and surf < 1000:
                        r = 3
                    elif surf >= 1000 and surf < 100000:
                        r = lambda surf: 45/61000*surf+4.26229508196721
                        r = int(r(surf))
                    else:
                        r = 100
                else:
                    r = save_r
                # ----
                mask = plateID == pID
                mask = id[mask]
                
                #self.plate1 = pID
                if self.verbose:
                    print('-'*50)
                self.im('Get Rotation pole of plate '+str(i)+'/'+str(nop-1))
                self.im('   - Number of point: '+str(surf)+'  rfactor: '+str(r))
                self.im('   - Build TessellatedPolygon object for the plate 1')
                pttk1 = TessellatedPolygon()
                pttk1.verbose = verbose
                if not on_poly and remove_edges:
                    pttk1.load(self.surfdir+self.surffile[i],\
                    grid_name=grid_name,cartv_name=cartv_name,sphev_name=sphev_name,\
                    magG_name=magG_name,ptID_name=ptID_name)
                    pttk1.remove_edges_from_file(self.edgedir+self.edgefile[i])
                else:
                    # membership condition
                    cond1 = plateID == pID
                    pttk1.x  = self.x[cond1]
                    pttk1.y  = self.y[cond1]
                    pttk1.z  = self.z[cond1]
                    pttk1.vx = self.vx[cond1]
                    pttk1.vy = self.vy[cond1]
                    pttk1.vz = self.vz[cond1]
                self.im('   - Inverse for the rotation pole of PLATE1 in FIX-MANTLE')
                self.wx[i],self.wy[i],self.wz[i],residual,misfit_xyz,misfit_enu,misfit_normcrossprod,self.P1[i],self.P2[i],P1loc,P2loc,IDmask = regEEPP(pttk1,r=r,rplot=rplot,verbose=verbose,plot=plot,maxiter=maxiter)
                mask = mask[IDmask]
                self.rotmask[mask] = True
                self.misfit[mask]               = misfit_xyz
                self.misfit_normcrossprod[mask] = np.arcsin(misfit_normcrossprod)
                self.im('      - plateness P1 for the plate1: '+str(self.P1[i]))
                self.im('      - plateness P2 for the plate1: '+str(self.P2[i]))
            else:
                self.im('Ignore a very small plate')

    
    def compute_dim_perimeter_area(self,plot=False):
        """
        This function computes the dimensioned surface and perimeter of each
        plates -> size: self.nop, data save in self.surfdim and self.peridim.
        """
        self.im('Computation of the dimensioned plate area (in km^2) and perimeter (in km)')
        uPID = np.unique(self.plateID)
        self.surfdim = np.zeros(self.nop)
        self.peridim = np.zeros(self.nop)
        # Pre-compute the trianguation of the surface of the computation of the surface of plates
        nod = len(self.x)
        points = np.zeros((nod,3))
        points[:,0] = self.x
        points[:,1] = self.y
        points[:,2] = self.z
        tri = ConvexHull(points)
        simplices = tri.simplices
        ptID = self.pointID[simplices]
        # -------
        # Iterate on all plates
        for i in tqdm(range(len(uPID))):
            # Perimeter
            me = self.plateIDe == uPID[i]
            lone = self.lone[me].copy()*np.pi/180
            late = self.late[me].copy()*np.pi/180
            temp1, temp2, temp3 = polygon_perimeter_and_ordening(late,lone,plot=plot)
            self.peridim[i] = temp1
            # Area
            m = self.plateID == uPID[i]
            maskin = np.isin(ptID,self.pointID[m])
            maskinNb = np.zeros(maskin.shape[0])
            maskinNb[maskin[:,0]] += 1
            maskinNb[maskin[:,1]] += 1
            maskinNb[maskin[:,2]] += 1
            maskinWithin = maskinNb >= 2 # Consider that all the triangles having at least 2 points within the plate are forming the plate
            simplicesPlate = simplices[maskinWithin,:]
            area = 0
            for j in range(simplicesPlate.shape[0]):
                pt1,pt2,pt3 = simplicesPlate[j,:]
                x1 = points[pt1,0]
                y1 = points[pt1,1]
                z1 = points[pt1,2]
                x2 = points[pt2,0]
                y2 = points[pt2,1]
                z2 = points[pt2,2]
                x3 = points[pt3,0]
                y3 = points[pt3,1]
                z3 = points[pt3,2]
                area += areatriangle3d(x1,y1,z1,x2,y2,z2,x3,y3,z3)
            self.surfdim[i] = area #adim
        self.surfdim = self.surfdim * (4*np.pi*Project.planetaryModel.radius**2)/tri.area # dim
        
          

    def expand_plate2surface(self,arr,on_poly=False):
        """
        Returns an array having the lengh of the total number of surface points
        and that contains the values of an input array (which has the lenght of
        the total number of plates/polygons) whose its values are distributed 
        according to the field self.plateID.
        
        Args:
            arr (np.ndarray, shape = self.nop): array that will be expand on the
                    entire surface.
            on_poly (bool, optional): If True then, expand the result using the field
                    'self.plateIDpoly' (external tessellation from a polygon file).
                    Defaults: on_poly = False
        
        Returns:
            output (np.ndarray, shape=self.x.shape): expended array.
        """
        if not on_poly:
            nop     = self.nop
            plateID = self.plateID
        else:
            nop     = self.nop_poly
            plateID = self.plateIDpoly
        output = np.zeros(len(self.x))
        upID = np.unique(self.plateID)
        if len(arr) == nop:
            for i in range(nop):
                mask = plateID == upID[i]
                output[mask] = arr[i]
            return output
        else:
            self.im('Your input filed has not the size of the total number of plates',error=True)
    
    

    def get_barycenter_from_edges(self,plate):
        """
        Returns the coordinates of the barycenter of the plate/polygon identified
        with the ID 'plate'.

        Args:
            plate (int): ID of the plate (value of the field self.plateID).
        
        Returns:
            lonb (float): longitude of the barycenter (degree)
            latb (float): latitude of the barycenter (degree)
            dmin (float): distance used to defined the point as the barycenter
            baryID (bool): index of the barycenter on the surface data
                        (e.g. on self.lon).
        """
        mask_surface = self.plateID  == plate
        mask_edges   = self.plateIDe == plate
        ids = list(np.array(range(len(self.x)))[mask_surface])
        ide = list(np.array(range(len(self.xe)))[mask_edges])
        # iterate on all selected points on the plates for the selected points on the edges
        dmin = np.zeros(len(ids))
        for i in range(len(ids)):
            loni = self.lon[ids[i]] * np.pi/180
            lati = self.lat[ids[i]] * np.pi/180
            lone = self.lone[ide]   * np.pi/180
            late = self.late[ide]   * np.pi/180
            dist = haversine(loni,lati,lone,late)
            dmin[i] = np.amin(dist)
        # seach the farest point
        gind = np.where(dmin == np.amax(dmin))[0]
        if len(gind) > 1 or isinstance(gind,np.ndarray):
            gind = gind[0]
        # barycenter
        lonb = self.lon[ids[gind]]
        latb = self.lat[ids[gind]]
        baryID = ids[gind]
        return lonb,latb,dmin,baryID
    
    
    
    def generate_persistenceDiag4tracking(self,fname,path='./',plot=False):
        """
        Generates a VTK (XDMF+H5) file containing a fictive persistence diagram (embeded in the domain)
        for the time tracking of plate barycenters.
        Paring between barycenters (maxima) and the most distant edge point (relative to a plate and a barycenter)
        The persistence of the pairs correspond to the haversin distance between barycenters and saddle points.
        Uses the function MAPT3.io.write_PersistenceDiag_VTK.

        Args:
            fname (str): File name for the export (without format extension).
            path (str, optional): path to the directory to store the output file
                        Defaults: path = './'
            plot (bool, optional): If True then, generates a figure to visualize
                        the detected barycenters.
                        Defaults: plot = False
        """
        self.im('Generates a fictive PersistenceDiagram VTK object for Paraview TTK plate time tracking')
        self.im('  - Computation of barycenters of all plates')
        plates       = np.unique(self.plateID)
        nop          = len(plates)
        self.baryLon = np.zeros(nop)
        self.baryLat = np.zeros(nop)
        xm,ym,zm,xs,ys,zs,p = np.zeros(nop), np.zeros(nop), np.zeros(nop), np.zeros(nop), np.zeros(nop), np.zeros(nop), np.zeros(nop)
        for i in tqdm(range(nop)):
            pi = plates[i]
            mask_edges = self.plateIDe == pi
            self.baryLon[i],self.baryLat[i],dmin,IDbary = self.get_barycenter_from_edges(pi)
            dist = haversine(self.baryLon[i]*np.pi/180,self.baryLat[i]*np.pi/180,self.lone[mask_edges]*np.pi/180,self.late[mask_edges]*np.pi/180)
            xm[i] = self.x[IDbary]
            ym[i] = self.y[IDbary]
            zm[i] = self.z[IDbary]
            p[i]  = np.amin(dist)
            saddleID = np.where(dist==np.amin(dist))[0]
            #print(saddleID)
            xs[i] = self.xe[mask_edges][saddleID]
            ys[i] = self.ye[mask_edges][saddleID]
            zs[i] = self.ze[mask_edges][saddleID]
        self.bin = p
        # ---
        if plot:
            fig = plt.figure()
            ax  = fig.add_subplot(111,projection=ccrs.Robinson())
            ax.scatter(self.lonb,self.latb,s=1,color='k',transform=ccrs.PlateCarree())
            ax.scatter(self.baryLon,self.baryLat,s=10,color='red',transform=ccrs.PlateCarree(),label='Barycenters')
            plt.legend()
            plt.show()
        # ---
        if path[-1] != '/':
            path += '/'
        self.im('  - Exportation to VTK (XDMF+HDF5) file: '+path+fname+'.xdmf/.h5')
        write_PersistenceDiag_VTK(xm,ym,zm,xs,ys,zs,p,fname,path=path)
    


    def generate_trackingField(self,fname,path='./',plot=False,verbose=True):
        """
        [obsolete method]
        Generates a VTK (XDMF+H5) file containing a field equal to the number of points
        composing a plate i at its barycenters and zeros everywhere else.
        Uses the function MAPT3.io.shell2VTK.

        Args:
            fname (str): name of the file (without extension) to export the VTK
            path (str, optional): path to the directory to save the file.
                            Defaults: './'
            plot (bool, optional): If Truen then, display a map of plate barycenters.
                            Defaults: False
            verbose (bool, optional): Option controling the verbose output on the
                            terminal. Defaults: True
        """
        self.im('Generates a fictive PersistenceDiagram VTK object for Paraview TTK plate time tracking')
        self.im('  - Computation of barycenters of all plates')
        plates       = np.unique(self.plateID)
        nop          = len(plates)
        self.baryLon = np.zeros(nop)
        self.baryLat = np.zeros(nop)
        vm = np.zeros(self.x.shape[0])
        for i in tqdm(range(nop)):
            pi = plates[i]
            self.baryLon[i],self.baryLat[i],dmin,IDbary = self.get_barycenter_from_edges(pi)
            vm[IDbary] = self.surf[i]
        # ---
        if plot:
            fig = plt.figure()
            ax  = fig.add_subplot(111,projection=ccrs.Robinson())
            ax.scatter(self.lonb,self.latb,s=1,color='k',transform=ccrs.PlateCarree())
            ax.scatter(self.baryLon,self.baryLat,s=10,color='red',transform=ccrs.PlateCarree(),label='Barycenters')
            plt.legend()
            plt.show()
        # ---
        if path[-1] != '/':
            path += '/'
        self.im('  - Exportation to VTK (XDMF+HDF5) file: '+path+fname+'.xdmf/.h5')
        shell2VTK(self.x,self.y,self.z,vm,fname,'topo',path=path,verbose=verbose)
    


    def export2h5(self,path2h5):
        """
        Function exporting the current data structure in a h5 file
        in order to make the loading process faster and memory friendly.
        The .h5 file thus created can be loaded into the current class instance
        using the self.load_from_h5 function.

        Args:
            path2h5 (str): complete path to the .h5 data file.
        """
        fid  = h5py.File(path2h5,'w')
        self.im('Export the current data structure into a  h5 file.')
        self.im('Export into file: '+path2h5)
        self.im('Exportation...')
        # Plate Surfaces data
        dset = fid.create_dataset('nop', data = self.nop)
        dset = fid.create_dataset('persistence', data = self.persistence)
        dset = fid.create_dataset('pmin', data = self.pmin)
        dset = fid.create_dataset('pmine', data = self.pmine)
        dset = fid.create_dataset('pminb', data = self.pminb)
        dset = fid.create_dataset('surf', data = self.surf)
        dset = fid.create_dataset('peri', data = self.peri)
        dset = fid.create_dataset('x', data = self.x)
        dset = fid.create_dataset('y', data = self.y)
        dset = fid.create_dataset('z', data = self.z)
        dset = fid.create_dataset('vx', data = self.vx)
        dset = fid.create_dataset('vy', data = self.vy)
        dset = fid.create_dataset('vz', data= self.vz)
        dset = fid.create_dataset('vtheta', data = self.vtheta)
        dset = fid.create_dataset('vphi', data = self.vphi)
        dset = fid.create_dataset('vr', data = self.vr)
        dset = fid.create_dataset('pressure', data = self.pressure)
        dset = fid.create_dataset('magGSV', data = self.magGSV)
        dset = fid.create_dataset('plateID', data = self.plateID)
        dset = fid.create_dataset('pointID', data = self.pointID)
        dset = fid.create_dataset('pstart', data = self.pstart)
        dset = fid.create_dataset('pend', data = self.pend)
        dset = fid.create_dataset('r', data = self.r)
        dset = fid.create_dataset('lat', data = self.lat)
        dset = fid.create_dataset('lon', data = self.lon)
        # Plate Edges data
        dset = fid.create_dataset('xe', data = self.xe)
        dset = fid.create_dataset('ye', data = self.ye)
        dset = fid.create_dataset('ze', data = self.ze)
        dset = fid.create_dataset('vxe', data = self.vxe)
        dset = fid.create_dataset('vye', data = self.vye)
        dset = fid.create_dataset('vze', data = self.vze)
        dset = fid.create_dataset('vthetae', data = self.vthetae)
        dset = fid.create_dataset('vphie', data = self.vphie)
        dset = fid.create_dataset('vre', data = self.vre)
        dset = fid.create_dataset('pressuree', data = self.pressuree)
        dset = fid.create_dataset('magGSVe', data = self.magGSVe)
        dset = fid.create_dataset('plateIDe', data = self.plateIDe)
        dset = fid.create_dataset('pointIDe', data = self.pointIDe)
        dset = fid.create_dataset('estart', data = self.estart)
        dset = fid.create_dataset('eend', data = self.eend)
        dset = fid.create_dataset('re', data = self.re)
        dset = fid.create_dataset('late', data = self.late)
        dset = fid.create_dataset('lone', data = self.lone)
        # Plate Boundary data
        dset = fid.create_dataset('xb', data = self.xb)
        dset = fid.create_dataset('yb', data = self.yb)
        dset = fid.create_dataset('zb', data = self.zb)
        dset = fid.create_dataset('rb', data = self.rb)
        dset = fid.create_dataset('lonb', data = self.lonb)
        dset = fid.create_dataset('latb', data = self.latb)
        dset = fid.create_dataset('vrb', data = self.vrb)
        dset = fid.create_dataset('vphib', data = self.vphib)
        dset = fid.create_dataset('vthetab', data = self.vthetab)
        dset = fid.create_dataset('platecouple', data = self.platecouple)
        dset = fid.create_dataset('vxb', data = self.vxb)
        dset = fid.create_dataset('vyb', data = self.vyb)
        dset = fid.create_dataset('vzb', data = self.vzb)
        dset = fid.create_dataset('magGSVb', data = self.magGSVb)
        dset = fid.create_dataset('pressureb', data = self.pressureb)
        # non rigid parts
        dset = fid.create_dataset('xnr', data = self.xnr)
        dset = fid.create_dataset('ynr', data = self.ynr)
        dset = fid.create_dataset('znr', data = self.znr)
        dset = fid.create_dataset('rnr', data = self.rnr)
        dset = fid.create_dataset('lonnr', data = self.lonnr)
        dset = fid.create_dataset('latnr', data = self.latnr)
        dset = fid.create_dataset('vrnr', data = self.vrnr)
        dset = fid.create_dataset('vphinr', data = self.vphinr)
        dset = fid.create_dataset('vthetanr', data = self.vthetanr)
        dset = fid.create_dataset('pointIDnr', data = self.pointIDnr)
        dset = fid.create_dataset('vxnr', data = self.vxnr)
        dset = fid.create_dataset('vynr', data = self.vynr)
        dset = fid.create_dataset('vznr', data = self.vznr)
        dset = fid.create_dataset('magGSVnr', data = self.magGSVnr)
        dset = fid.create_dataset('pressurenr', data = self.pressurenr)
        dset = fid.create_dataset('pminnr', data = self.pminnr)
        # close
        fid.close()
        self.im('Exportation done successfully!')
