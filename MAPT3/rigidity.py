# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for Plate Processing and Analysis

"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata

# Internal dependencies:
from .kinematics import cartfipole
from .generics import im
from .compute import clustering
from .project import Project


# ----------------- FUNCTIONS -----------------


def rigid(pgi,pID,wx,wy,wz,P1,P2,P1c=Project.P1c,P2c=Project.P2c,fragment_size=Project.fragment_size,verbose=False,plot=False):
    """
    Returns if a plate/block is rigid or not.

    WARNING, function different from MAPT3.optimize.rigid

    Args:
        pgi (MAPT3.tessellation.PlateGather): contains the plate/block to test
        pID (int): ID of the plate/block to test (in the pgi.plateID field)
        wx,wy,wz (3x np.ndarray): angular rotation of the plate/block to test.
        P1 (float): plateness P1 of the plate/block to test
        P2 (float): plateness P2 of the plate/block to test
        P1c (float, optional): critical plateness P1 to defined a plate/block as rigid
                    Defaults: P1c = MAPT3.project.Project.P1c
        P2c (float, optional): critical plateness P2 to defined a plate/block as rigid
                    Defaults: P2c = MAPT3.project.Project.P2c
        fragment_size (int, optional): Fragment size to defined a plate/block as rigid
                    Defaults: fragment_size = MAPT3.project.Project.fragment_size
        plot (bool, optional): If set to True generates a figure displaying the 
                    result of the test.
                    Defaults: plot = False
        verbose (bool, optional): Option to display a verbose output in the terminal
                    Defaults: verbose = False
    
    Returns:
        rigid (bool): True if the plate/block is rigid according to the criteria
                    pass in arguments.
    """
    pName = 'rigid'
    im('Test the rigidity of a polygon',pName,verbose)
    # 1. Test the plateness at the scale of the plate
    im('Test global',pName,verbose)
    if P1 >= P1c and P2 >= P2c:
        # 2. Test the plateness at the scale of the plate fragment
        # --- Prepare the plate
        maskInvPts = np.arange(len(pgi.x))[pgi.plateID==pID]    # Plate mask
        # --- Cut the input mosaic
        xp = pgi.x[maskInvPts]
        yp = pgi.y[maskInvPts]
        zp = pgi.z[maskInvPts]
        # --- Adjust the global draco mosaic for the plate
        dracoMosaic_new = dracoParcellationFragmenter(xp,yp,zp,fragment_size,verbose=verbose)
        # --- Prepare the data for the computation of P1 and P2 on each fragments
        vx = pgi.vx[maskInvPts]     # 'Data' velocities
        vy = pgi.vy[maskInvPts]
        vz = pgi.vz[maskInvPts]
        # --- Result of the inversion
        vxfit, vyfit, vzfit = cartfipole(xp,yp,zp,wx,wy,wz)
        # --- Build matrices
        Vxyz_f = np.zeros((len(vx),3))
        Vxyz_d = np.zeros((len(vx),3))
        Vxyz_f[:,0] = vxfit
        Vxyz_f[:,1] = vyfit
        Vxyz_f[:,2] = vzfit
        Vxyz_d[:,0] = vx
        Vxyz_d[:,1] = vy
        Vxyz_d[:,2] = vz
        # --- Compute the norms
        normVd  = np.linalg.norm(Vxyz_d,axis=1)
        normVf  = np.linalg.norm(Vxyz_f,axis=1)
        # --- Compute P1 and P2 for all points
        P1loc = (Vxyz_f[:,0]*Vxyz_d[:,0]+Vxyz_f[:,1]*Vxyz_d[:,1]+Vxyz_f[:,2]*Vxyz_d[:,2])/(normVd*normVf)
        P2loc = 1-np.sqrt((Vxyz_f[:,0]-Vxyz_d[:,0])**2+(Vxyz_f[:,1]-Vxyz_d[:,1])**2+(Vxyz_f[:,2]-Vxyz_d[:,2])**2)/normVd
        # --- Compute for each fragments
        im('Test for each fragment',pName,verbose)
        um  = np.unique(dracoMosaic_new)
        nop = len(um)
        P1f = np.zeros(nop)     # P1 for each fragment
        P2f = np.zeros(nop)     # P2 for each fragment
        for i in range(nop):
            mmi = dracoMosaic_new == um[i]
            P1f[i] = np.mean(P1loc[mmi])
            P2f[i] = np.mean(P2loc[mmi])
        # --- Test the plateness P1 and P2 for each fragments
        mask_good_P1f = P1f >= P1c
        mask_good_P2f = P2f >= P2c
        mask_good_plateness = mask_good_P1f * mask_good_P2f
        # --- plot
        im('Plot',pName,verbose)
        if plot:
            myP1 = np.zeros(len(maskInvPts))
            myP2 = np.zeros(len(maskInvPts))
            for i in range(nop):
                mmi = dracoMosaic_new == um[i]
                myP1[mmi] = P1f[i]
                myP2[mmi] = P2f[i]
            print('P1f') ; print('-'*80) ; print(P1f)
            print('P2f') ; print('-'*80) ; print(P2f)
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(121,projection=ccrs.Robinson())
            ax2 = fig.add_subplot(122,projection=ccrs.Robinson())
            ax1.set_title('P1')
            ax2.set_title('P2')
            if np.count_nonzero(~mask_good_plateness) == 0:
                fig.suptitle('Rigid')
            else:
                fig.suptitle('Non-rigid')
            ax1.set_global()
            ax2.set_global()
            cmap1 = ax1.scatter(pgi.lon[maskInvPts],pgi.lat[maskInvPts],c=myP1,vmin=None,vmax=1,transform=ccrs.PlateCarree())
            cmap2 = ax2.scatter(pgi.lon[maskInvPts],pgi.lat[maskInvPts],c=myP2,vmin=None,vmax=1,transform=ccrs.PlateCarree())
            fig.colorbar(cmap1,ax=ax1,shrink=0.5)
            fig.colorbar(cmap2,ax=ax2,shrink=0.5)
            fig.tight_layout()
            plt.show()
            
        if np.count_nonzero(~mask_good_plateness) == 0:
            return True
        else:
            return False
    else:
        return False



def dracoParcellationFragmenter(xp,yp,zp,fragment_size,verbose=True):
    """ Function computing the parcellation of a cloud of points
    representing a 3D surface (plate/block) into a set of fragments
    having a 'dragon scale' shape.

    Args:
        xp (np.ndarray): x coordinates
        yp (np.ndarray): y coordinates
        zp (np.ndarray): z coordinates
        fragment_size (int): average size of each fragment [in number of points]
    
    Returns:
        mosaic (np.ndarray): contains the index of the mosaic (i.e. fragment)
                        for each input points resulting from the parcellation.
    """
    pName = 'DracoFrag'
    im('Compute the Draco Parcellation of the input surface',pName,verbose)
    
    nod    = len(xp)    # number of data
    
    if nod > 2*fragment_size:
    
        done   = np.zeros(nod,dtype=bool)
        mosaic = np.zeros(nod,dtype=np.int32)
        jvalue = []
        endval = []
        ptIDs  = np.arange(nod)

        # ----------------------
        # auto compute step

        im('  -> Auto compute the spatial search step',pName,verbose)
        noTests = 100
        tested_step = np.zeros(noTests,dtype=np.float64)
        for i in range(noTests): # 5 test
            ptID = np.random.randint(0,len(xp))
            mydist = np.sqrt((xp[0]-xp)**2+(yp[0]-yp)**2+(zp[0]-zp)**2)
            mydist = mydist[mydist>0]
            tested_step[i] = np.amin(mydist)
        step = 2 * np.mean(tested_step)
        im('     step = '+str(step),pName,verbose)

        # ----------------------
        # auto compute radius_max

        im('  -> Auto compute the maximum search radius',pName,verbose)
        noTests = 100
        tested_radius = np.zeros(noTests,dtype=np.float64)
        for i in range(noTests): # 5 test
            ptID = np.random.randint(0,len(xp))
            refx, refy, refz = xp[ptID], yp[ptID], zp[ptID]
            dist = np.sqrt((refx-xp)**2+(refy-yp)**2+(refz-zp)**2)
            j = 0 ; radius = step ; noNeighbours = False
            ptin = 1
            while ptin < fragment_size:
                j += 1
                radius = step*j
                maskin = dist < radius
                ptin = np.count_nonzero(maskin)
            tested_radius[i] = radius
        radius_max = np.mean(tested_radius) + 1*np.mean(tested_radius)
        im('     max radius = '+str(radius_max),pName,verbose)

        # ----------------------
        # prepare the grid

        parcell = 0

        # Draco Parcellation Fragmenter
        n = 0 ; progress_bar_step = 10
        im('  -> Compute the Draco Parcellation',pName,verbose)
        while np.count_nonzero(done) < len(done):
            undone = ~done
            if n == 0:
                ptID = ptIDs[~done][0]
            else:
                masknext = dist >= radius
                masknext = masknext * undone
                ptID = ptIDs[masknext][np.where(dist[masknext] == np.amin(dist[masknext]))[0][0]]
            progress_percentage = int(np.count_nonzero(done)/len(done)*10000)/100
            if progress_percentage > progress_bar_step:
                im('        '+str(progress_percentage)+'%',pName,verbose)
                progress_bar_step += 10
            
            # ---
            refx, refy, refz = xp[ptID], yp[ptID], zp[ptID]
            dist = np.sqrt((refx-xp)**2+(refy-yp)**2+(refz-zp)**2)
            j = 0 ; radius = step ; noNeighbours = False
            ptin = 1
            undone = ~done
            while ptin < fragment_size and radius < radius_max and not noNeighbours:
                j += 1
                radius = step*j
                maskin = dist < radius
                ptin = np.count_nonzero(maskin*undone)
                if np.count_nonzero(maskin) >= 5:
                    if np.count_nonzero(done[maskin]) == len(done[maskin])-1:
                        noNeighbours = True
            # ---
            jvalue.append(j)
            m  = maskin*undone
            if radius >= radius_max and ptin < fragment_size :
                endval.append(1)
            elif noNeighbours and ptin < fragment_size:
                endval.append(2)
            else:
                endval.append(0)
            # ---
            n += 1
            mosaic[m] = parcell
            parcell += 1
            done[m] = True
            
        jvalue = np.array(jvalue)
        endval = np.array(endval)
        
        im('        '+str(100)+'%',pName,verbose)

        # ----------------------
        # Remove artifacts with clustering

        im('  -> Remove artifacts step 1:',pName,verbose)
        im('       -> Detect fragmented draco scales',pName,verbose)
        ptIDs  = np.arange(nod)
        max_mosaic_ID = np.amax(mosaic) +1
        um = np.unique(mosaic)

        for i in range(len(um)):
            mi = mosaic == um[i]
            clusters, coutput = clustering(step,xp[mi],yp[mi],zp[mi])
            if len(coutput) > 1:
                for i in range(1,len(coutput)): #pass the first: keep the same ID
                    mci = ptIDs[mi][np.array(coutput[i],dtype=np.int32)]
                    mosaic[mci] = max_mosaic_ID
                    max_mosaic_ID += 1

        # ----------------------
        # Merge the points of the small cells with the closest largest cell
        # to ensure the have fragment sizes >= fragment_size

        im('  -> Remove artifacts step 2:',pName,verbose)
        im('       -> Merge the too small fragments with the good draco scales',pName,verbose)
        
        mosaicUniqueID, mcounts = np.unique(mosaic,return_counts=True)
        
        maskSmallMosaic = mcounts < fragment_size/1.5
        maskLargeMosaic = ~maskSmallMosaic
        if np.count_nonzero(maskLargeMosaic) >= 2: # 2 is the min number of large plates to obtain results in the remapping
            maskLargeMosaicPts = np.zeros(xp.shape[0],dtype=bool)
            for i in range(np.count_nonzero(maskLargeMosaic)):
                mmi = mosaic == mosaicUniqueID[maskLargeMosaic][i]
                maskLargeMosaicPts[mmi] = True
            nodi = np.count_nonzero(maskSmallMosaic)
            points = np.zeros((np.count_nonzero(maskLargeMosaicPts),3))
            points[:,0] = xp[maskLargeMosaicPts]
            points[:,1] = yp[maskLargeMosaicPts]
            points[:,2] = zp[maskLargeMosaicPts]
            mosaic2 = mosaic.copy()
            for i in range(nodi):
                mmi = mosaic == mosaicUniqueID[maskSmallMosaic][i]
                mosaic2[mmi] = griddata(points, mosaic[maskLargeMosaicPts], (xp[mmi],yp[mmi],zp[mmi]), method='nearest')
            
            # ----------------------
            # Rename the mosaic ID
            
            im('  -> Re-indexing of the mosaic IDs',pName,verbose)
            um  = np.unique(mosaic2)
            for i in range(len(um)):
                mmi = mosaic2 == um[i]
                mosaic[mmi] = i
        
        else:
            mosaic = mosaic.copy()*0  # return only 1 plate
        
        im('Parcellation completed!',pName,verbose)
        a,b = np.unique(mosaic,return_counts=True)
        im('  -> Size of individual fragments: '+str(b),pName,verbose)
    else:
        mosaic = np.zeros(nod)
    return mosaic


