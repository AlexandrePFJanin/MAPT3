# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Plate kinematics toolkit
"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random

# Internal dependencies:
from .geotransform import xyz2latlon, Rgt




# ----------------- FONCTIONS -----------------


def wxwywz2latlonw(wx,wy,wz):
    """
    Input: wx,wy,wz in *RADIANS/Myr*
    Return lat,lon in *DEGREES* and omega in *RADIANS/Myr*
    """
    omega = np.sqrt(wx**2+wy**2+wz**2)
    latP = np.arcsin(wz/omega)*180/np.pi
    if wx<0:
        signe = -1
    elif wx>0:
        signe = 1
    elif wx ==0:
        signe=0
    lonP = np.arctan(wy/wx)*180/np.pi+180*(1-signe)/2
    return (latP,lonP,omega)
    


def latlonw2wxwywz(lat,lon,omega):
    """
    lat,lon in RAD and omega in radian/Myr
    To convert omega from deg/Ma -> rad/Ma : omega*np.pi/180
    """
    wx = np.cos(lat)*np.cos(lon)*omega   #in rad/Ma
    wy = np.cos(lat)*np.sin(lon)*omega   #in rad/Ma
    wz = np.sin(lat)*omega               #in rad/Ma
    return (wx,wy,wz)



def cartfipole(x,y,z,wx,wy,wz):
    """
    wx,wy,wz in rad/Myrs
    returns vx,vy,vz
    """
    vx =  0*wx +  z*wy + -y*wz
    vy = -z*wx +  0*wy +  x*wz
    vz =  y*wx + -x*wy +  0*wz
    return vx,vy,vz



def regEEPP(pttk,r=1,mp=(1,1,1),maxiter=100,errmax=0.01,convmax=1e-3,tarantola=False,kappa=1,ponderated=False,plot=True,rplot=100,\
    scale=6e4,width=5e-4,verbose=True):
    """
    regEEPP: Regularized Estimation of Euler Pole Parameters for a tessellated polygon

    Inverses the surface velocity field of a plate/block and returns the rotation vector.
    
    Args:
        pttk = MAPT3.tessellation.TessellatedPolygon object containing position of points and velocity vectors.
         <optional>: r       = int, resampling parameter to do the inversion just on a part of the dataset [default, r=1]
                     mp      = tuple, prior model for (Wx,Wy,Wz) [default, mp=(1,1,1)]
                     maxiter = int,  maximum number of iteration [default, maxiter=100]
                     errmax  = float, maximum residual [default, errmax=0.01]
                     convmax = minimum residual difference allowed between two consecutive iterations
                               [default, convmax=1e-3]
                     tarantola = bool, to adopte the Generalized Least Square formulation (Tarantola 2005)
                                 otherwise, the numerical sheme will be tykhonov regularization
                                 [default, tarantola=False]
                     kappa   = int, tykhonov regularization parameter used only if tarantola == False
                               [default, kappa=1]
                     plot    = bool, option to display the solution [default, plot = True]
                     rplot   = int, resampling parameter to enable to plot all the data
                     verbose = bool, if True then produce a verbose output
    Returns:
        wx,wy,wz : the rotation vector in * RADIANS/Mys *
                   (the appriated format for the function wxwywz2latlonw.)
                   -> to transform the vector in rotation pole with :
                      lon_pole [deg], lat_pole [deg] and omega [deg/Myr] do:
                      lon,lat,omega = wxwywz2latlonw(wx,wy,wz) # latp,lonp directly in deg but omega in rad/Myr
                      omega = omega * 180/np.pi
    """
    N    = int(len(pttk.x)/r)
    id   = list(range(len(pttk.x)))
    mask = random.sample(id,N)
    
    x = pttk.x[mask]
    y = pttk.y[mask]
    z = pttk.z[mask]

    vx = pttk.vx[mask]
    vy = pttk.vy[mask]
    vz = pttk.vz[mask]
    
    N = len(x)

    # compute errors
    sigmax = 1/100 * abs(np.mean(pttk.vx))
    sigmay = 1/100 * abs(np.mean(pttk.vy))
    sigmaz = 1/100 * abs(np.mean(pttk.vz))
    
    dvx = np.ones(N)*sigmax
    dvy = np.ones(N)*sigmay
    dvz = np.ones(N)*sigmaz

    # -------- ITERATIVE SHEME -----------
    n            = 0                        # init iteration counter
    residual     = [1e10]                    # init the residual
    while  n < maxiter:
        if residual[-1] < errmax:           # break if reached the residual convergence criterion
            break                                        
        # ----
        A      = np.zeros((3*N,3))          # coefficient A matrix
        errors = np.zeros(3*N)              # cov error on measurements
        tykhonov = np.zeros((3*N,3))        # tykhonov matrix
        cartv    = np.zeros(3*N)            # d = cart velo vector
        for i in range(N):
            errors[i*3:(i+1)*3]     = np.array([dvx[i],dvy[i],dvz[i]])
            if ponderated:                  # considered a ponderation of data error according to the distance of each points and the centroid of the figure
                pttk.get_centroid(plot=False)
                errors[i*3:(i+1)*3] = errors[i*3:(i+1)*3] * (pttk.centrodist[i]/np.amax(pttk.centrodist))*50
            xi,yi,zi                = x[i],y[i],z[i]
            A[i*3:(i+1)*3,:]        = np.array([[0, zi, -yi],\
                                                [-zi, 0, xi],\
                                                [yi, -xi, 0]])
            cartv[i*3:(i+1)*3]      = np.array([vx[i],vy[i],vz[i]])

        Cd    = np.diag(errors)             # cov matrix data
        small = np.eye(len(Cd))*1e-5
        Cd = Cd + small # avoid singularites
        Cdinv = np.linalg.inv(Cd)           # inv cov matrix
        tykhonov = np.linalg.inv(np.dot(A.T,np.dot(Cdinv,A))) # tykhonov matrix
        W = np.dot(A,mp)-cartv   
        m = mp - np.dot(np.linalg.inv(np.dot(A.T,np.dot(Cdinv,A)) + kappa**2*tykhonov),np.dot(A.T,np.dot(Cdinv,W)))
        # --------------------------- other formulation : least quare
        if tarantola:
            Cm     = np.eye(3,3)
            W      = cartv - np.dot(A,mp)
            a      = np.dot(np.linalg.inv(np.dot(A,np.dot(Cm,A.T))+Cd),W)
            m      = mp + np.dot(Cm,np.dot(A.T,a))
        # ------------------------ Tarantola page 72
        if not tarantola:
            Cm = tykhonov       # a posteriori covariance matrix of the model
        # -------------------
        # -- Operations on the iterative sheme:
        vxfit, vyfit, vzfit = cartfipole(x,y,z,m[0],m[1],m[2])
        residual.append(np.sum(np.sqrt((vxfit-vx)**2+(vyfit-vy)**2+(vzfit-vz)**2)))
        # Iteration number
        n += 1
        # Convergence condition
        if residual[-2]-residual[-1] <= convmax:
            break
        # Model update
        mp = m
    if verbose:
        print('---- Inversion diagnostic:')
        if n == maxiter:
            print('- StopCondition: Max iteration number reached')
            print('     | N = ',n)
            print('     | R = ',residual[-1])
        if residual[-1] < errmax:
            print('- StopCondition: Residual condition reached')
            print('     | N = ',n)
            print('     | R = ',residual[-1])
        if residual[-2]-residual[-1] <= convmax:
            print('- StopCondition: Convergence reached')
            print('     | N = ',n)
            print('     | R = ',residual[-1])
        print()

    # -------- OUTPUT ------------
    
    # My resulting model: My inverted rotations
    wx,wy,wz = m

    if verbose:
        print('My solution: (Wx,Wy,Wz) =',wx,wy,wz)
        print('Solution norm:',np.linalg.norm(np.array([wx,wy,wz])))
        print('My Residual:  ',residual[-1])

    # Get the velocities
    vxfit, vyfit, vzfit = cartfipole(x,y,z,wx,wy,wz)

    # ------
    # Prepare data for the plot and output
    latp,lonp,omega = wxwywz2latlonw(wx,wy,wz)
    lat,lon,r = xyz2latlon(x,y,z)
    nod2 = len(lon)
    dvlon  = np.zeros(nod2)     # lon misfit
    dvlat  = np.zeros(nod2)     # lat misfit
    dvr    = np.zeros(nod2)     # r misfit
    vlonfit = np.zeros(nod2)    # lon v fit
    vlatfit = np.zeros(nod2)    # lat v fit
    rfit   = np.zeros(nod2)     # r v fit
    londv  = np.zeros(nod2)     # lon v data
    latdv  = np.zeros(nod2)     # lat v data
    rdv    = np.zeros(nod2)     # r v data
    # misfits
    misfit_xyz          = np.zeros((nod2,3))  # XYZ (dvx,dvy,dvz) misfit on each grid point
    misfit_enu          = np.zeros((nod2,3))  # ENU (dvlon,dvlat,dvr) misfit on each grid point
    misfit_normcossprod = np.zeros(nod2)      # normalized cross product Vfit x Vobs
    # plateness P1 and P2
    P1loc  = np.zeros(nod2)
    P2loc  = np.zeros(nod2)
    P1 = 0
    P2 = 0
    # iterate
    for i in range(nod2):
        lon_i = lon[i]*np.pi/180
        lat_i = lat[i]*np.pi/180
        dvxi   = vxfit[i]-vx[i]
        dvyi   = vyfit[i]-vy[i]
        dvzi   = vzfit[i]-vz[i]
        vxifit = vxfit[i]
        vyifit = vyfit[i]
        vzifit = vzfit[i]
        vxd = vx[i]
        vyd = vy[i]
        vzd = vz[i]
        Rmatrix = Rgt(lat_i,lon_i)
        dVxyz     = np.array([dvxi,dvyi,dvzi])          # residual
        misfit_xyz[i,:] = dVxyz
        Vxyz_fit  = np.array([vxifit,vyifit,vzifit])    # fit
        Vxyz_d    = np.array([vxd,vyd,vzd])             # data
        dvlon[i],dvlat[i],dvr[i]      = np.dot(Rmatrix,dVxyz)
        misfit_enu[i,:] = np.array([dvlon[i],dvlat[i],dvr[i]])
        vlonfit[i],vlatfit[i],rfit[i] = np.dot(Rmatrix,Vxyz_fit)
        londv[i],latdv[i],rdv[i]      = np.dot(Rmatrix,Vxyz_d)
        # normalized cross product
        crossp  = np.cross(Vxyz_fit,Vxyz_d)
        normVf  = np.linalg.norm(Vxyz_fit)
        normVd  = np.linalg.norm(Vxyz_d)
        normdot = np.linalg.norm(crossp)
        misfit_normcossprod[i] = normdot/(normVd*normVf)
        # plateness
        P1loc[i] = np.dot(Vxyz_fit,Vxyz_d)/(normVd*normVf)
        P2loc[i] = np.sqrt(dvxi**2+dvyi**2+dvzi**2)/normVd
        P1 += P1loc[i]
        P2 += P2loc[i]
    
    misfit_xyz = np.linalg.norm(misfit_xyz,axis=1)/normVd
    misfit_enu = np.linalg.norm(misfit_enu,axis=1)/normVd
    
    P1 = P1/nod2
    P2 = 1-P2/nod2
    
    if verbose:
        print("P1 plateness: "+str(P1))
        print("P2 plateness: "+str(P2))
    
    if plot:
        
        lon = lon*180/np.pi
        lat = -(lat*180/np.pi-90)
        
        lon = lon[::rplot]
        lat = lat[::rplot]
        
        plt.figure()
        plt.title('Resudial plot')
        xresidual = list(range(0,len(residual)))
        xresidualbis = xresidual[1:len(residual)]
        plt.plot(xresidual,residual,'or',label='residual')
        plt.plot(xresidual,residual,'--',color='red')
        plt.plot(xresidualbis,residual[1:len(residual)],'-',color='red')
        plt.xlabel('iteration number')
        plt.ylabel('residual')
        minr,maxr = np.amin(residual[1:len(residual)]),np.amax(residual[1:len(residual)])
        plt.ylim(minr-0.1*(maxr-minr),maxr+0.1*(maxr-minr))
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(131)
        fig.suptitle('Result of the best fit in the velocity space')
        ax1.plot(vx[::rplot],vy[::rplot],'o',label='synthetic data')
        ax1.plot(vxfit[::rplot],vyfit[::rplot],'+',label='fit')
        ax1.set_xlabel('cartesian vx')
        ax1.set_ylabel('cartesian vy')
        ax1.legend()
        #
        ax2 = fig.add_subplot(132)
        ax2.plot(vx[::rplot],vz[::rplot],'o',label='synthetic data')
        ax2.plot(vxfit[::rplot],vzfit[::rplot],'+',label='fit')
        ax2.set_xlabel('cartesian vx')
        ax2.set_ylabel('cartesian vz')
        ax2.legend()
        #
        ax3 = fig.add_subplot(133)
        ax3.plot(vy[::rplot],vz[::rplot],'o',label='synthetic data')
        ax3.plot(vyfit[::rplot],vzfit[::rplot],'+',label='fit')
        ax3.set_xlabel('cartesian vy')
        ax3.set_ylabel('cartesian vz')
        ax3.legend()
        plt.show()

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        ax.set_global()
        ax.set_title("Map of the inverted velocity field")
        ax.scatter(lon,lat,c='darkblue',s=2,alpha=0.5,transform=ccrs.PlateCarree())
        ax.quiver(lon,lat,dvlon[::rplot],dvlat[::rplot],scale=scale,color='orange',width=width,transform=ccrs.PlateCarree(),label='residual')
        ax.quiver(lon,lat,vlonfit[::rplot],vlatfit[::rplot],scale=scale,color='blue',width=width,transform=ccrs.PlateCarree(),label='best fit')
        ax.quiver(lon,lat,londv[::rplot],latdv[::rplot],scale=scale,color='black',width=width,transform=ccrs.PlateCarree(),label='data')
        ax.scatter(lonp,latp,marker='D',c='red',s=20,alpha=1,transform=ccrs.PlateCarree(),label='rotation pole')
        ax.legend(loc=2)
        plt.show()
    
    return wx,wy,wz,residual,misfit_xyz,misfit_enu,misfit_normcossprod,P1,P2,P1loc,P2loc,mask
