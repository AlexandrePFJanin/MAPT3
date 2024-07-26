# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Geometric transformations
"""

# External dependencies:
import numpy as np
from math import sin, cos, sqrt, atan2

# Internal dependencies:
from .project import Project


# ----------------- FUNCTIONS -----------------


def xyz2latlon(x,y,z):
    """
    Transforms x,y,z cartesian coordinates into geographical lon,lat,r
    coordinates (in radians)
    """
    r     = np.sqrt(x**2+y**2+z**2)
    lat   = np.arctan2(np.sqrt(x**2+y**2),z)
    lon   = np.arctan2(y,x)
    return lat,lon,r


def latlon2xyz(lat,lon,R=Project.modelRadius):
    """
    Return the X,Y,Z cartesian coordinates in the ECEF reference frame
    of a point described by a geodetic coordinates (lat,lon) and the radius
    of the sphere R (default, R=MAPT3.Project.modelRadius)

    lat,lon in RADIANS
    """
    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)
    return x,y,z


def distance_2points(lon1,lat1,lon2,lat2,R=Project.modelRadius):
    """
    Computes the distance (in km) between two points defined by their
    lon/lat coordinates in RADIANS !!.
    R = average radius of the sphere (default, R=MAPT3.Project.modelRadius)
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def haversine(lon1,lat1,lon2,lat2,R=Project.modelRadius):
    """
    Computes the haversine distance between points on a sphere 
    defined by their lon/lat coordinates in RADIANS !!.
    R = average radius of the sphere (default, R=MAPT3.Project.modelRadius)
    """
    dlon = abs(lon1-lon2)
    dlat = abs(lat1-lat2)
    return 2*R*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + (1-np.sin(dlat/2)**2-np.sin((lat1+lat2)/2)**2)*np.sin(dlon/2)**2))



def latlon2UTM(lat,lon,E0=0,N0=0,lon0=0,lat0=45*np.pi/180):
    """ UTM ref for perfectly spherical earth 
    lat and lon (and lat0 and lon0) must be in RADIANS.
    Used in optimize
    """
    k0 = 0.9996
    B = np.cos(lat)*np.sin(lon-lon0)
    E = E0+k0/2*np.log((1+B)/(1-B))
    N = N0+k0*(np.arctan2(np.tan(lat),np.cos(lon))-lat0)
    return E,N


def sweeplat2south(lat):
    "Used in optimize"
    return lat/2-45*np.pi/180


def sweeplat2north(lat):
    "Used in optimize"
    return lat/2+45*np.pi/180



def rotation_matrix_3D(axis,theta):
    """
    Computes the rotation matrix R in cartesian 3D geometry for a rotation
    on x, y or z axis and a rotation angle theta
    <i> axis = str, 'x' for a rotation around the x axis
                    'y' for a rotation around the y axis
                    'z' for a rotation around the z axis
        theta = int/float, rotation angle in *RADIANS*
    NOTE: Application:
    If you have a vector A that you want to rotate aroud the X axis
    with a rotation angle theta = 45 deg, write:

    >> R     = rotation_matrix_3D('x',45*np.pi/180)
    >> A_rot = np.dot(R,A)
    """
    if axis == 'x':
        R = np.array([[             1,             0,            0],\
                      [             0, np.cos(theta),-np.sin(theta)],\
                      [             0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        R = np.array([[ np.cos(theta),             0, np.sin(theta)],\
                      [             0,             1,            0],\
                      [-np.sin(theta),             0, np.cos(theta)]])
    elif axis == 'z':
        R = np.array([[ np.cos(theta),-np.sin(theta),            0],\
                      [ np.sin(theta), np.cos(theta),            0],\
                      [             0,             0,            1]])
    return R


def Rgt(lat,lon):
    """
    Rgt =  geocentric to topocentric rotation matrix

    For Venu = (Ve,Vn,Vz) and Vxyz = (Vx,Vy,Vz)

    Venu = np.dot(Rgt,Vxyz)
    Vxyz = np.dot(Rgt_inv,Venu)

    ** Lat, Lon coordinates in RADIANS **
    """
    return np.array([[-np.sin(lon),np.cos(lon),0],\
                     [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],\
                     [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])



def ecef2enu(lat,lon,vx,vy,vz):
    """
    Transforms velocities from ECEF to ENU.
    ** Lat, Lon coordinates in RADIANS **
    """
    nod  = len(vx)
    vlon = np.zeros(nod)
    vlat = np.zeros(nod)
    vr   = np.zeros(nod)
    for i in range(nod):
        v_i = np.array([vx[i],vy[i],vz[i]])
        Rmatrix = Rgt(lat[i],lon[i])
        vlon[i],vlat[i],vr[i] = np.dot(Rmatrix,v_i)
    return vlon,-vlat,vr



def ecef2enu_stagYY(x,y,z,vx,vy,vz):
    """
    Transforms the ECEF vectors (vx,vy,vz) into ENU vectors (Vphi,vtheta,vr)
    Here, vtheta is -vlat (as computed in stagData)
    """
    lat = np.arctan2(np.sqrt(x**2+y**2),z)
    lon = np.arctan2(y,x)
    vtheta =  vx*(np.cos(lon)*np.cos(lat)) + vy*(np.sin(lon)*np.cos(lat)) - vz*(np.sin(lat))
    vphi   = -vx*(np.sin(lon))             + vy*(np.cos(lon))
    vr     = -vx*(np.cos(lon)*np.sin(lat)) - vy*(np.sin(lon)*np.sin(lat)) - vz*(np.cos(lat))
    vr = -vr
    return vphi,vtheta,vr



def enu2ecef(lat,lon,vlon,vlat,vr):
    """
    Transforms velocities from ENU to ECEF.
    ** Lat, Lon coordinates in RADIANS **
    """
    nod  = len(vlon)
    vx   = np.zeros(nod)
    vy   = np.zeros(nod)
    vz   = np.zeros(nod)
    for i in range(nod):
        v_i = np.array([vlon[i],-vlat[i],vr[i]])
        Rmatrix = Rgt(lat[i],lon[i])
        vx[i],vy[i],vz[i] = np.dot(np.linalg.inv(Rmatrix),v_i)
    return vx,vy,vz



def rot(x,y,z,poleLon,poleLat,poleangle):
    """This function rotates in space a cloud of points defined by 3 matrices
    x,y and z according to corrdinates of rotation pole and a rotation angle
    <i> : x = list/np.ndarray, matrix of the cartesian x coordinates defining
              points to be rotate
          y = list/np.ndarray, matrix of the cartesian y coordinates defining
              points to be rotate
          z = list/np.ndarray, matrix of the cartesian z coordinates defining
              points to be rotate
          poleLon = int/float, Longitude <! in degree !> of the rotation pole
          poleLat = int/float, Latitude <! in degree !> of the rotation pole
          poleangle = int/float, angle for the rotation <! in degree !>
    <o> : Return the rotated grid (x_rot,y_rot,z_rot)
    """
    plat = 90 - poleLat #colatitude of the pole
    #Parameters of the pole
    poleangle = poleangle*np.pi/180
    poleLon   = poleLon*np.pi/180
    plat      = plat*np.pi/180
    # ===============
    # Rotation matrix
    #detail
    cps = np.cos(poleangle)
    sps = np.sin(poleangle)
    ct  = np.cos(plat)
    st  = np.sin(plat)
    cp  = np.cos(poleLon)
    sp  = np.sin(poleLon)
    #Matrix
    rotm      = np.zeros((3,3))
    rotm[0,0] = (cp*cp*ct*ct+sp*sp)*cps+cp*cp*st*st
    rotm[0,1] = -sp*cp*st*st*(cps-1.)-ct*sps
    rotm[0,2] = -cp*st*ct*(cps-1.)+sp*st*sps
    rotm[1,0] = -sp*cp*st*st*(cps-1.)+ct*sps
    rotm[1,1] = (sp*sp*ct*ct+cp*cp)*cps+sp*sp*st*st
    rotm[1,2] = -sp*st*ct*(cps-1.)-cp*st*sps
    rotm[2,0] = -cp*st*ct*(cps-1.)-sp*st*sps
    rotm[2,1] = -sp*st*ct*(cps-1.)+cp*st*sps
    rotm[2,2] = st*st*cps+ct*ct
    #Creation of 
    nop = len(x)
    coord     = [[x[i],y[i],z[i]] for i in range(nop)]
    coord     = np.array(coord)
    coord_rot = np.array([np.matmul(rotm,coord[i]) for i in range(nop)])
    #Reconstruction of grid from coord
    x_rot = coord_rot[:,0]
    y_rot = coord_rot[:,1]
    z_rot = coord_rot[:,2]
    return (x_rot,y_rot,z_rot)







