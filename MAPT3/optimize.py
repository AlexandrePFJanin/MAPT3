# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tessellation optimization toolkit
"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.figure as mfigure
import matplotlib.cm as cmx
import cartopy.crs as ccrs
import random
import h5py
from tqdm import tqdm
from time import time
import os
import alphashape
from paraview.simple import XDMFReader, ExtractSurface, FeatureEdges, SaveData, Delete, ResetSession
import matplotlib.path as mpltPath

# Internal dendencies:
from .tessellation import PlateGather
from .generics import im, line_count
from .kinematics import cartfipole
from .geotransform import xyz2latlon, latlon2xyz, latlon2UTM, sweeplat2north, sweeplat2south
from .errors import OptimizationError, OptimizationSettingsError
from .rigidity import dracoParcellationFragmenter
from .compute import clustering
from .project import Project
from .io import surface2VTK


# ----------------- FONCTIONS -----------------


def is_point_on_grid(point,pgi):
    """
    Tests if a point defined by the tuple (x,y,z) is in PlaterGather instance (pgi)
    """
    x,y,z = point
    dist  = np.sqrt((pgi.x-x)**2 + (pgi.y-y)**2 + (pgi.z-z)**2)
    distmin = np.amin(dist)
    if distmin == 0:
        return True
    else:
        return False



def find_points_opti(pts,pgi):
    """
    Find multiple points in a PlateGather instance based on a grid point ID
    and return their ID pts is a list.
    """
    ptID  = np.zeros(len(pts),dtype=np.int32)
    found = np.zeros(len(pts),dtype=bool)
    for i in range(len(pts)):
        whereAmI = np.where(pgi.pointID == pts[i])[0]
        if len(whereAmI)>0:
            ptID[i]  = whereAmI
            found[i] = True
    return ptID, found



def resampling_param(surf):
    """
    Function returning a reasonable resampling parameter for the inversion of
    the surface velocity field of a plate.
    It may be necessary to modify it to adjust it to your data (optimization problem).
    """
    if surf < 100:
        r = 1
    elif surf >= 100 and surf < 1000:
        r = 3
    elif surf >= 1000 and surf < 100000:
        r = lambda surf: 45/61000*surf+4.26229508196721
        r = int(r(surf))
    else:
        r = 100
    return r



def rigid(pgi,fragment_size,verbose=False,plot=False):
    """
    Returns if a plate/block is rigid or not.

    Args:
        pgi (MAPT3.tessellation.PlateGather): contains the plate/block to test
        fragment_size (int/float):  minimum fragment size for the plate rigidity
                    (see par file of the project)
        plot (bool, optional): If set to True generates a figure displaying the 
                    result of the test.
                    Defaults: plot = False
        verbose (bool, optional): Option to display a verbose output in the terminal
                    Defaults: verbose = False
    """
    # 1. Test the plateness at the scale of the plate
    if pgi.P11 >= Project.P1c and pgi.P12 >= Project.P2c:
        # 2. Test the plateness at the scale of the plate fragment
        # --- Prepare the plate
        pID = pgi.plate1                            # Get the ID of the plate on which the inversion has been computed
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
        wx,wy,wz = pgi.wx1,pgi.wy1,pgi.wz1
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
        um  = np.unique(dracoMosaic_new)
        nop = len(um)
        P1f = np.zeros(nop)     # P1 for each fragment
        P2f = np.zeros(nop)     # P2 for each fragment
        for i in range(nop):
            mmi = dracoMosaic_new == um[i]
            P1f[i] = np.mean(P1loc[mmi])
            P2f[i] = np.mean(P2loc[mmi])
        # --- Test the plateness P1 and P2 for each fragments
        mask_good_P1f = P1f >= Project.P1c
        mask_good_P2f = P2f >= Project.P2c
        mask_good_plateness = mask_good_P1f * mask_good_P2f
        
        if plot:
            myP1 = np.zeros(len(maskInvPts))
            myP2 = np.zeros(len(maskInvPts))
            for i in range(nop):
                mmi = dracoMosaic_new == um[i]
                myP1[mmi] = P1f[i]
                myP2[mmi] = P2f[i]
            print('P1f') ; print('-'*80) ; print(P1f)
            print('P2f') ; print('-'*80) ; print(P2f)
            fig = plt.figure(figsize=(10,6))
            ax  = fig.add_subplot(111,projection=ccrs.Robinson())
            if np.count_nonzero(~mask_good_plateness) == 0:
                ax.set_title('Rigid')
            else:
                ax.set_title('Non-rigid')
            ax.set_global()
            cmap = ax.scatter(pgi.lon[maskInvPts],pgi.lat[maskInvPts],c=myP1,vmin=None,vmax=1,transform=ccrs.PlateCarree())
            plt.colorbar(cmap)
            plt.show()
            
        if np.count_nonzero(~mask_good_plateness) == 0:
            return True
        else:
            return False
    else:
        return False



def overlap(todo,ids):
    """
    Tests the todo list (array) and the new points to add.
    Return True if overlaps are detected
    """
    if np.count_nonzero(todo[ids]) == len(ids):
        # Means no overlapping
        return False
    else:
        return True



def find_overlap(todo,ids):
    """
    Returns the list of overlapping points and non-overlapping points
    """
    return ids[todo[ids]],ids[~todo[ids]]




def is_points_out_edges(points,points_edges,resolution=0.000001):
    """ Tests if in a list of points, there are points located out of
    a shape (a closed polygon) defined by edges points. Return True is
    there is points out of the polygon defined by edges and False otherwise.
    
               ** This function works only in 2D **
               
    Args:
        points (ndarray): Tested points. Shape, 2D array.
                            points[:,0] = list of x coordinates
                            points[:,1] = list of y coordinates
        points_edges (ndarray): Points forming the edges of the tested polygon. Shape, 2D array.
                            points_edges[:,0] = list of x coordinates for edges
                            points_edges[:,1] = list of y coordinates for edges
        resolution (float, optional): Precision to say that two coordinates defined by
                                      floating numbers are identical. Defaults to 0.000001.
    """
    # --- grid
    N = len(points)
    xg = points[:,0]
    yg = points[:,1]
    gridID = np.arange(N)
    
    # --- edges
    xb = points_edges[:,0]
    yb = points_edges[:,1]
    
    xe = points_edges[:,0]
    ye = points_edges[:,1]
    
    #--- find edges points on the grid according to a given accuraty (resolution)
    eID    = np.zeros(len(xb),dtype=np.int32)-1
    for j in range(len(xb)):
        mx1 = xg >= xb[j] - resolution
        mx2 = xg <= xb[j] + resolution
        my1 = yg >= yb[j] - resolution
        my2 = yg <= yb[j] + resolution
        m   = mx1*mx2*my1*my2
        if np.count_nonzero(m) == 0:
            print('not found')
        elif np.count_nonzero(m) > 1:
            eID[j] = gridID[m][0]
            print('found multiple points')
        elif np.count_nonzero(m) == 1:
            eID[j] = gridID[m]

    # --- remap edges
    xb = xg[eID]
    yb = yg[eID]
    points_edges[:,0] = xb
    points_edges[:,1] = yb
    
    edgesIDmask = np.zeros(N,dtype=bool)
    edgesIDmask[eID] = True  
    
    # --- find points inside edges
    path = mpltPath.Path(points_edges)
    inside = path.contains_points(points)
    
    # --- points not inside and not in the edges
    m_not_edges  = ~edgesIDmask
    m_not_inside = inside == False
    mm  = m_not_edges * m_not_inside
    
    if np.count_nonzero(mm) == 0:
        return False
    else:
        return True



def get_indices_of_edges(points,points_edges,resolution=0.000001):
    """ Function that find in a list of points (x,y coordinates), the points
    defined by their coordinates as edges.
    This function returns a boolean mask which has the same length as the number
    of points. True if the point has been detected in the matrix points_edges
    and False otherwise.

    Args:
        points (ndarray): Tested points. Shape, 2D array.
                            points[:,0] = list of x coordinates
                            points[:,1] = list of y coordinates
        points_edges (ndarray): Points forming the edges of the tested polygon. Shape, 2D array.
                            points_edges[:,0] = list of x coordinates for edges
                            points_edges[:,1] = list of y coordinates for edges
        resolution (float, optional): Precision to say that two coordinates defined by
                                      floating numbers are identical. Defaults to 0.000001.
    Returns:
        ndarray: boolean mask
    """
    # --- grid
    N = len(points)
    xg = points[:,0]
    yg = points[:,1]
    gridID = np.arange(N)
    
    # --- edges
    xb = points_edges[:,0]
    yb = points_edges[:,1]
    
    #--- find edges points on the grid according to a given accuraty (resolution)
    eID    = np.zeros(len(xb),dtype=np.int32)-1
    for j in range(len(xb)):
        mx1 = xg >= xb[j] - resolution
        mx2 = xg <= xb[j] + resolution
        my1 = yg >= yb[j] - resolution
        my2 = yg <= yb[j] + resolution
        m   = mx1*mx2*my1*my2
        if np.count_nonzero(m) == 0:
            print('not found')
        elif np.count_nonzero(m) > 1:
            print('found multiple points')
        elif np.count_nonzero(m) == 1:
            eID[j] = gridID[m]

    # --- remap edges
    xb = xg[eID]
    yb = yg[eID]
    points_edges[:,0] = xb
    points_edges[:,1] = yb

    edgesIDmask = np.zeros(N,dtype=bool)
    edgesIDmask[eID] = True 
    
    return edgesIDmask
        


def get_edges(points,alpha_init=80,alpha_search_step=10,convergeThreshold=3,verbose=True):
    """ Function extracting points forming the edges of a 2D cloud of points based on
    the computation of a concave hull (alpha shape hull) and an iterative search of an
    optimal alpha parameter.
    
    This function deals with the following edges extraction optimization:
        - maximization of the number of points forming the polygon edges
        - no points out of the shape (closed polygon) defined by the edges
        
    This function stops also when the number of points forming the edges is stagnant over
    several iterations defined by the integer input parameter 'convergeThreshold'. N.B. 
    Can also depend of the alpha search step you enter (alpha_search_step).

    Args:
        points (ndarray): 2D coordinates of the cloud of points.
                            points[:,0] = list of x coordinates
                            points[:,1] = list of y coordinates
        alpha_init (int/float, optional): Initial value of the alpha parameter for the
                                          iterative search. Defaults to 80.
        alpha_search_step (int/float, optional): variation (step) of the value of alpha at
                                                 each iteration. Defaults to 10.
        convergeThreshold (int, optional): Maximum iteration number where the number of
                                           points forming edges is stagnant. Defaults to 3.
        verbose (bool, optional): Verbose output switch. Defaults to True.

    Returns:
        ndarray: 2D matrix containing the coordinates of edge points.
    """
    
    pName = 'get_edges'
    
    # Define alpha parameter
    alpha = alpha_init
    
    im('Get edges from points using alphashape hull',pName,verbose)

    # Generate the alpha shape
    alpha_run = True
    past_nod = 0
    stagnant = 0
    edges_coord = None
    while alpha_run and stagnant < convergeThreshold and alpha>0:
        im('  -> Try alpha: '+str(alpha),pName,verbose)
        try:
            alpha_shape = alphashape.alphashape(points, alpha)
            pnts = np.array(alpha_shape.boundary.coords[:])
            if stagnant == -1:
                if not is_points_out_edges(points,pnts):
                    alpha_run = False # comes from a depreciated alpha value
                    edges_coord = alpha_shape.boundary.coords
                    im('  -> Return edges from alpha: '+str(alpha),pName,verbose)
                else:
                    im('     -> Loose points',pName,verbose)
                    alpha -= alpha_search_step
                    alpha_init = alpha
                    stagnant = -1
                    alpha_run = True
                    im('     -> Depreciate alpha to '+str(alpha),pName,verbose)
            else:
                if len(pnts) > past_nod:
                    if not is_points_out_edges(points,pnts):
                        past_nod  = len(pnts)
                        stagnant  = 0
                        alpha += alpha_search_step
                        im('     -> Succes',pName,verbose)
                        im('        - nop: '+str(len(pnts)),pName,verbose)
                        im('     -> Stagnant solution: '+str(stagnant)+'/'+str(convergeThreshold-1),pName,verbose)
                        edges_coord = alpha_shape.boundary.coords
                    else:
                        im('     -> Loose points',pName,verbose)
                        if alpha != alpha_init:
                            alpha_run = False
                            alpha -= alpha_search_step
                            im('     -> Return edges from alpha: '+str(alpha),pName,verbose)
                        else:
                            alpha -= alpha_search_step
                            alpha_init = alpha
                            stagnant = -1
                            alpha_run = True
                            im('     -> Depreciate alpha to '+str(alpha),pName,verbose)
                elif len(pnts) == past_nod:
                    if not is_points_out_edges(points,pnts):
                        stagnant += 1
                        alpha += alpha_search_step
                        im('     -> Succes',pName,verbose)
                        im('        - nop: '+str(len(pnts)),pName,verbose)
                        im('     -> Stagnant solution: '+str(stagnant)+'/'+str(convergeThreshold-1),pName,verbose)
                        edges_coord = alpha_shape.boundary.coords
                    else:
                        im('     -> Loose points',pName,verbose)
                        if alpha != alpha_init:
                            alpha_run = False
                            alpha -= alpha_search_step
                            im('     -> Return edges from alpha: '+str(alpha),pName,verbose)
                        else:
                            alpha -= alpha_search_step
                            alpha_init = alpha
                            stagnant = -1
                            alpha_run = True
                            im('     -> Depreciate alpha to '+str(alpha),pName,verbose)
                elif len(pnts) < past_nod:
                    alpha_run = False
                    alpha -= alpha_search_step
                    im('     -> Depreciate alpha to '+str(alpha),pName,verbose)
        
        except:
            if alpha != alpha_init:
                alpha -= alpha_search_step
                im('     -> Crashed: Depreciate alpha to '+str(alpha),pName,verbose)
                im('     -> Return edges from alpha: '+str(alpha),pName,verbose)
                alpha_run = False
            else:
                alpha -= alpha_search_step
                alpha_init = alpha
                stagnant = -1
                im('     -> Crashed: Depreciate alpha to '+str(alpha),pName,verbose)
                alpha_run = True
    if alpha <= 0:
        im('Failed to solve edges: Return input points as edges',pName,verbose)
        edges_coord = points
    else:
        edges_coord = np.array(edges_coord[:])
    return edges_coord



def optimize(path2h5file, pthreshold, ofile, output_path='./', small_plate = Project.polyminsize, \
             overlap_order = 0, add_missedPlates = True, geographic_search = 0,\
             edges_detection_method = 'paraview',\
             simplex_threshold = 0.01, hidden = True,\
             alpha_init = 90, alpha_search_step = 10, convergeThreshold = 3,\
             plot_missed = False, plot = True, plot_rigidity = False,\
             verbose=False, verbose_rigidity=False, verbose_edgesExtraction=False):
    """
    Optimization of the tessellation at a given time step. Combine multiple
    tessellations computed with different value of pmin (minimum persistence
    threshold) using geodynamics considerations. See Janin et al., 2024 for
    more details.
    Export the result of the optimization in a .h5 file with the same format
    as the .h5 files built at a given pmin value.

    overlap_order can be 'forceFlexible' or 'skip'   

    Args:
        path2h5file (function): function returning the path to the different .h5 files
                                built with MAPT3.tessellation at different pmin and
                                containing the tessellation data. path2h5file have to
                                be a function taking only one argument: pmin.
        pthreshold (np.ndarray): array containing the values of pmin that will be
                                considered for the optimization. Its values will be the
                                input of path2h5file. If you want to used all the values
                                you tested in the project, you can called your 
                                MAPT3.project instance if you used it.
        ofile (str): prefix of the file name used for the output .h5 file (ofile has no
                    file extension). The final output file will have '_optimized' at its
                    end.
        output_path (str, optional): path to directory on which the optimized tessellation
                                    will be exported. Defaults to './'.
        small_plate (int/float, optional): minimum polygon size. Below, the polygon will be
                                considered as non rigid and its solid body rotation will
                                not be computed.
                                Advice: used a MAPT3.Project instance (Project):
                                        small_plate = Project.polyminsize
                                Defaults: MAPT3.project.Project.polyminsize
        overlap_order (int, optional): Method to apply on the management of overlapping points.
                            Have to be either 1 or 0.
                            If overlap_order == 0:
                                Treatment of overlapping points:  Skip, i.e. when a polygon 
                                is detected as rigid but have overlapping with an already saved
                                polygon, save only the part of the second polygon that is
                                not overlapping on the first polygon.
                            If overlap_order == 1:
                                Treatment of overlapping points:  Force the flexible mode
                                i.e.  when a polygon is detected as rigid but have overlapping
                                with an already saved polygon, continu to decrease of the
                                minimum persistence until reaching a rigid plate without overlapping.
                            Defaults: overlap_order = 0
        geographic_search (int, optional): Method to determined the next anchor point.
                            Have to be either 1 or 0.
                            If geographic_search == 0:
                                Geographic search: Deterministic. The next anchor is choose
                                as being the next one in the input sequence (reproductible)
                            If geographic_search == 1:
                                Geographic search: Randomized. The next anchor is choose
                                randomly.
                            Defaults: geographic_search = 0
        add_missedPlates (bool, optional): If set to True: Add the missed points as plates with
                            clustering. Else, these points will be simply ignored.
                            Defaults to True.
        edges_detection_method (str, optional): Method used to deted edges of the new polygons.
                            Have to be in ['paraview','alphashape'].
                            Defaults: edges_detection_method = 'paraview'
        simplex_threshold (float, optional): Only if edges_detection_method = 'paraview'. 
                                            Maximum length of the simplices you authorize
                                            for the meshing of the surface. Above, the
                                            simplices are remove. May depend on the spacing of the
                                            mesh. Defaults to 0.01.
        hidden (bool, optional): Only if edges_detection_method = 'paraview'.
                            Option controling if you want to hide the construction file
                            needed to compute the edges detection with paraview.
                            The file will start with '.' and thus will be insisible for 
                            linux and Mac default file listing.
        verbose_edgesExtraction (bool, optional): Only if edges_detection_method = 'paraview'.
                            Option controling if you want to display a verbose output in the
                            terminal during the extraction of polygon edges.
        alpha_init (int/float, optional): Only if edges_detection_method = 'alphashape'.
                            Parameter for the detection of edges. See get_edges for more details.
                            Defaults: alpha_init = 90
        alpha_search_step (int/float, optional): Only if edges_detection_method = 'alphashape'.
                            Parameter for the detection of edges. See get_edges for more details.
                            Defaults: alpha_search_step = 10
        convergeThreshold (int, optional): Only if edges_detection_method = 'alphashape'.
                            Parameter for the detection of edges. See get_edges for more details.
                            Defaults: convergeThreshold = 3
        verbose_rigidity (bool, optional): Option controling the verbose output during the
                            determination of polygons rigidity. 
                            Defaults: verbose_rigidity = False
        verbose (bool, optional): Option controling the verbose output for inner functions of
                            MAPT3.tessellation.PlateGather object.
                            Defaults: verbose = False
        plot_missed (bool, optional): Option controling the construction of a figure to show
                            on a global surface map the location of points missed during the
                            fisrt steps of the optimization and that will be recovered with
                            point clustering.
                            Defaults: plot_missed = False
        plot_rigidity (bool, optional): If set to True generates a figure displaying the 
                            result of the rigidity test, for every tests.
                            Defaults: plot_rigidity = False
        plot (bool, optional): If set to True, makes a map of the result of the optimization.
                            Defaults: plot = True
    """
    # reference value of pmin
    pmin_ref_ID = -1 # allways starts from  the highest value
    
    # compute the time need for the optimization
    time0 = time()
    
    if pmin_ref_ID == -1:
        pmin_ref_ID = len(pthreshold) - 1
    pmin_ref    = pthreshold[pmin_ref_ID]
    
    print('-------------------------------')
    print('Optimize the plate tessellation')
    print('-------------------------------')
    print()
    print('  -> Optimize with the reference: '+path2h5file(pmin_ref))
    print('  -> Critical plateness: P1='+str(Project.P1c)+', P2='+str(Project.P2c))
    print('  -> Fragment size: '+str(Project.fragment_size)+' pts')
    print()
    print('Iterative computation:\n')
    
    if geographic_search == 0:
        print('\n'+'Geographic search: Deterministic')
    elif geographic_search == 1:
        print('\n'+'Geographic search: Randomized')
    else:
        raise OptimizationSettingsError('Unrecognized value for the argument geographic_search')

    if overlap_order == 0:
        print('\n'+'Treatment of overlapping points:  Skip')
        print('  -> i.e. Save separately and immediatly the non-overlapping points for rigid plate when encountered')
    elif overlap_order == 1:
        print('\n'+'Treatment of overlapping points:  Force the flexible mode')
        print('  -> i.e. Decrease of the minimum persistence until reaching a rigid plate without overlapping')
    else:
        raise OptimizationSettingsError('Unrecognized treatment of overlapping points')

    # --- Reference -----------

    print()
    print('  -> Prepare the reference tessellation')
    ref = PlateGather()
    ref.verbose = verbose
    ref.load_from_h5(path2h5file(pmin_ref))
    # -- reference grid -----------
    lon, lat = ref.lon, ref.lat
    

    # --- optimized data -----------
    todo        = np.ones(len(lon),dtype=bool)          # points remaining to do
    plateID     = np.zeros(len(lon),dtype=np.int32)-1   # ID of the plate kept in the optimized model (before the re-indexing)
    persistence = np.zeros(len(lon),dtype=np.float64)-1 # highest persistence to have a rigid plate 
    rigidity    = np.zeros(len(lon),dtype=bool)         # True if rigid
    nPlateID    = np.zeros(len(lon),dtype=np.int32)     # New plate ID
    missedPlates= np.zeros(len(lon),dtype=np.int32)     # Map of the points that are missed by the optimization due to overlapping issues
    trackedFF   = np.zeros(len(lon),dtype=bool)         # Map of points affected by an overlapping and affected by a forced flexible
    missedPlates_pmin = np.zeros(len(lon),dtype=np.float64)-1 # Map of the persistence of missed plates
    
    # --- Dictonary of plate (sorted by pmin) for which the rotation pole has already been computed
    rotPoleCompute = {}   # init the dictionary containing the info if yes or no the rot pole of a given plate has been computed and if yes or no the pole was good.
                          # this dic avoid to compute several time the same inversion.
    
    # --- Restart the persistence search from the top
    restart_at_pmin  = None

    # ---- Start -----------
    n  = 0      # track the index of plate
    nn = 1      # track the index of missed cluster of points due to overlapping issues
    while np.count_nonzero(todo) != 0:

        if geographic_search == 0:
            ptID = 0                    # take the first point of the todo list
        elif geographic_search == 1:
            ptID = np.random.randint(0,np.count_nonzero(todo))
        
        loni, lati = lon[todo][ptID], lat[todo][ptID]   # ENU coordinates of the current point
        point = [ref.x[todo][ptID],ref.y[todo][ptID],ref.z[todo][ptID]]
        pID   = ref.plateID[ref.get_plateID(loni,lati)]
        surf  = np.count_nonzero(ref.plateID == pID)
        
        # set pID in the dictornary of already computed rot pole
        my_loc_pmin = pmin_ref
        my_loc_obj  = ref
        if 'p'+str(my_loc_pmin) not in rotPoleCompute.keys():
            rotPoleCompute['p'+str(my_loc_pmin)] = {}
            all_plates_IDs = np.unique(my_loc_obj.plateID) # list of all platesID for this pmin value
            for kk in range(len(all_plates_IDs)):
                rotPoleCompute['p'+str(my_loc_pmin)][str(all_plates_IDs[kk])] = 0  # 0: undone, 1: rigid(pg)=False, 2: rigid(pg)=True
        else:
            pass
        
        forced_flexible = False   # To force the flexible mode is overlapping is detected on a rigid plate
        
        print()
        print('-------')
        print('Remaining points: '+str(np.count_nonzero(todo))+'/'+str(len(todo))+': '+str(int(100*np.count_nonzero(todo)/len(todo)))+'%')
        print('Plate ID: '+str(pID)+', \t Surface: '+str(surf))
        
        #  Define here what is a small plate 
        if surf > small_plate:
            largePlate = True
        else:
            largePlate = False
        
        if largePlate:
            # if large enough: search for a rotation
            r   = resampling_param(surf)
            
            # Test if the plate velocoity field has already been inverted
            if rotPoleCompute['p'+str(my_loc_pmin)][str(pID)] == 0:     # rot pole inversion undone yet
                ref.get_rotation(pID,plot=False,r=r)
                local_rigidity = rigid(ref,Project.fragment_size,verbose=verbose_rigidity,plot=plot_rigidity)
                if local_rigidity:
                    rotPoleCompute['p'+str(my_loc_pmin)][str(pID)] = 2
                else:
                    rotPoleCompute['p'+str(my_loc_pmin)][str(pID)] = 1
            elif rotPoleCompute['p'+str(my_loc_pmin)][str(pID)] == 1:   # bad rot pole
                local_rigidity = False
            elif rotPoleCompute['p'+str(my_loc_pmin)][str(pID)] == 2:   # good rot pole
                local_rigidity = True

            # -----
            if local_rigidity and restart_at_pmin is None:
                # --- Rigid plate -----------
                ids = np.arange(len(ref.x))[ref.plateID == pID]
                if not overlap(todo,ids):
                    todo[ids]        = False
                    plateID[ids]     = pID
                    persistence[ids] = pmin_ref
                    rigidity[ids]    = True
                    nPlateID[ids]    = n
                    # --- Restart the persistence search from the top
                    restart_at_pmin  = None
                    
                    print('   -----> Succed at pmin: '+str(pmin_ref))
                
                else:
                    if overlap_order == 1:
                        trackedFF[ids] = True
                        forced_flexible = True
                        restart_at_pmin  = None
                        noids, oids = find_overlap(todo,ids)
                        print('   -----> Failed success at pmin: '+str(pmin_ref))
                        print('       -> Overlap detected on a rigid plate: Restart on flexible mode'+', overlap = '+str(len(oids))+' points')
                    elif overlap_order == 0:
                        noids, oids = find_overlap(todo,ids)
                        print('   -----> Failed success at pmin: '+str(pmin_ref))
                        print('   -> Overlap detected --> Save non overlapping: '+str(len(noids))+' points')
                        missedPlates_pmin[noids] = pmin_ref
                        missedPlates[noids] = nn
                        todo[noids]        = False
                        plateID[noids]     = 999999   # will be not found in a PlateGather object
                        persistence[noids] = 999999   # easily tracked in persistence
                        rigidity[noids]    = False
                        nPlateID[noids]    = n
                        # --- Restart the persistence search from the top
                        restart_at_pmin  = None
                        nn += 1
                    
                
            if not local_rigidity or restart_at_pmin is not None or forced_flexible:
                
                # --- Non rigid plate .OR. Overlapping issues -----------
                
                if restart_at_pmin is None:
                    if not forced_flexible:
                        print('   -> Failed at pmin: '+str(pmin_ref))
                    k = 0
                else:
                    k = pmin_ref_ID - (restart_at_pmin+1)
                    
                flexible = True
                
                
                # --- Flexible mode -----------
                
                while flexible:
                    
                    if restart_at_pmin is None:
                        # Decrease the persistence threshold
                        pmin_ref_ID_loc = pmin_ref_ID - (k+1)
                    else:
                        pmin_ref_ID_loc = restart_at_pmin
                    
                    if pmin_ref_ID_loc >= 0: # it remains ids
                        
                        pmin = pthreshold[pmin_ref_ID_loc]
                        
                        # --- Creation of a PlateGather instance at a lower persistence
                        pg = PlateGather()
                        pg.verbose = verbose
                        pg.load_from_h5(path2h5file(pmin))
                        
                        
                        # Check if need to add to dic
                        my_loc_pmin = pmin
                        my_loc_obj  = pg
                        if 'p'+str(my_loc_pmin) not in rotPoleCompute.keys():
                            rotPoleCompute['p'+str(my_loc_pmin)] = {}
                            all_plates_IDs = np.unique(my_loc_obj.plateID) # list of all platesID for this pmin value
                            for kk in range(len(all_plates_IDs)):
                                rotPoleCompute['p'+str(my_loc_pmin)][str(all_plates_IDs[kk])] = 0
                            

                        if is_point_on_grid(point,pg):
                            
                            pIDloc  = pg.plateID[pg.get_plateID(loni,lati)]
                            surfloc = np.count_nonzero(pg.plateID == pIDloc)
                            
                            if surfloc > small_plate:
            
                                r = resampling_param(np.count_nonzero(pg.plateID == pIDloc))
                                
                                # Test if the plate velocoity field has already been inverted
                                if rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 0:     # rot pole inversion undone yet
                                    pg.get_rotation(pIDloc,plot=False,r=r)
                                    local_rigidity_loc = rigid(pg,Project.fragment_size,verbose=verbose_rigidity,plot=plot_rigidity)
                                    if local_rigidity_loc:
                                        rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] = 2
                                    else:
                                        rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] = 1
                                elif rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 1:   # bad rot pole
                                    local_rigidity_loc = False
                                elif rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 2:   # good rot pole
                                    local_rigidity_loc = True


                                if local_rigidity_loc:
                                    #
                                    flexible = False
                                    #
                                    surfloc = np.count_nonzero(pg.plateID == pIDloc)
                                    mask = pg.plateID == pIDloc
                                    ids, found  = find_points_opti(pg.pointID[mask],ref)
                                    ids         = ids[found]
                                    
                                    # -- test the overlapping
                                    if not overlap(todo,ids):
                                        print('   -----> Succed at pmin: '+str(pmin)+', \t surface: '+str(surfloc)+': '+str(int(100*surfloc/surf))+'%'+' surf init')
                                        todo[ids]        = False
                                        plateID[ids]     = pIDloc
                                        persistence[ids] = pmin
                                        rigidity[ids]    = True
                                        nPlateID[ids]    = n
                                        # --- Restart the persistence search from the top
                                        restart_at_pmin  = None
                                    else:
                                        if overlap_order == 1:
                                            trackedFF[ids] = True
                                            flexible = True # restart the flexible mode
                                            noids, oids = find_overlap(todo,ids)
                                            print('   -----> Failed success at pmin: '+str(pmin)+', \t surface: '+str(surfloc)+': '+str(int(100*surfloc/surf))+'%'+' surf init')
                                            print('       -> Overlap detected on a rigid plate: Restart on flexible mode'+', overlap = '+str(len(oids))+' points')
                                            # --- Restart the persistence search from the top
                                            restart_at_pmin  = None
                                        elif overlap_order == 0:
                                            noids, oids = find_overlap(todo,ids)
                                            print('   -----> Failed success at pmin: '+str(pmin)+', \t surface: '+str(surfloc)+': '+str(int(100*surfloc/surf))+'%'+' surf init')
                                            print('   -> Overlap detected --> Save non overlapping: '+str(len(noids))+' points')
                                            missedPlates_pmin[noids] = pmin
                                            missedPlates[noids] = nn
                                            todo[noids]        = False
                                            plateID[noids]     = 999999   # will be not found in a PlateGather object
                                            persistence[noids] = 999999   # easily tracked in persistence
                                            rigidity[noids]    = False
                                            nPlateID[noids]    = n
                                            # --- Restart the persistence search from the top
                                            restart_at_pmin  = None
                                            nn += 1

                                else:
                                    #
                                    flexible = True
                                    print('   -> Failed at pmin: '+str(pmin))
                                    # --- Restart the persistence search from the top
                                    restart_at_pmin  = None
                                    #
                            
                            else:
                                flexible = False
                                print('   -----> Failed at pmin: '+str(pmin)+', too small: \t surface: '+str(surfloc))
                                mask = pg.plateID == pIDloc
                                ids, found  = find_points_opti(pg.pointID[mask],ref)
                                ids         = ids[found]
                                
                                # -- test the overlapping
                                if not overlap(todo,ids):
                                    todo[ids]        = False
                                    plateID[ids]     = pIDloc
                                    persistence[ids] = pmin
                                    rigidity[ids]    = False
                                    nPlateID[ids]    = n
                                    # --- Restart the persistence search from the top
                                    restart_at_pmin  = None
                                else:
                                    noids, oids = find_overlap(todo,ids)
                                    print('   -> Overlap detected on a small plate --> Save non overlapping: '+str(len(noids))+' points')
                                    missedPlates_pmin[noids] = pmin
                                    missedPlates[noids] = nn
                                    todo[noids]         = False
                                    plateID[noids]      = 999999    # will be not found in a PlateGather object
                                    persistence[noids]  = 999999    # easily tracked in persistence
                                    rigidity[noids]     = False
                                    nPlateID[noids]     = n
                                    # --- Restart the persistence search from the top
                                    restart_at_pmin  = None
                                    nn += 1
                            
                        else:
                            # Means that reducing the persistence, the ref point is now out of a plate...
                            flexible = False
                            print('   -> Ref point out of the domain at pmin: '+str(pmin))
                            ids = np.arange(len(todo))[todo][ptID]
                            todo[ids]        = False
                            plateID[ids]     = 999999   # will be not found in a PlateGather object
                            persistence[ids] = pmin
                            rigidity[ids]    = False
                            nPlateID[ids]    = n
                            # --- as it failed because the ref point does not exist at this persistence, restart at the same persistence but with the next point
                            restart_at_pmin  = pmin_ref_ID_loc # ID on the list pthreshold
                    else:
                        #
                        flexible = False
                        #
                        pmin = pthreshold[0] # Take the lowest persistence
                        
                        # ---- Creat a PlateGather instance at the lower persistence
                        pg = PlateGather()
                        pg.verbose = verbose
                        pg.load_from_h5(path2h5file(pmin))
                        
                        pIDloc = pg.plateID[pg.get_plateID(loni,lati)]
                        r = resampling_param(np.count_nonzero(pg.plateID == pIDloc))
                        
                        
                        # Test if the plate velocoity field has already been inverted
                        if rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 0:     # rot pole inversion undone yet
                            pg.get_rotation(pIDloc,plot=False,r=r)
                            local_rigidity_loc = rigid(pg,Project.fragment_size,verbose=verbose_rigidity,plot=plot_rigidity)
                            if local_rigidity_loc:
                                rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] = 2
                            else:
                                rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] = 1
                        elif rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 1:   # bad rot pole
                            local_rigidity_loc = False
                        elif rotPoleCompute['p'+str(my_loc_pmin)][str(pIDloc)] == 2:   # good rot pole
                            local_rigidity_loc = True
                        
                        
                        surfloc = np.count_nonzero(pg.plateID == pIDloc)
                        
                        print('   -----> Reloaded [Failed] at pmin: '+str(pmin)+', \t surface: '+str(surfloc)+': '+str(int(100*surfloc/surf))+'%'+' surf init')
                        mask = pg.plateID == pIDloc
                        ids, found  = find_points_opti(pg.pointID[mask],ref)
                        ids         = ids[found]
                        
                        # -- test the overlapping
                        if not overlap(todo,ids):
                            todo[ids]        = False
                            plateID[ids]     = pIDloc
                            persistence[ids] = pmin
                            rigidity[ids]    = False
                            nPlateID[ids]    = n
                            # --- Restart the persistence search from the top
                            restart_at_pmin  = None
                        else:
                            noids, oids = find_overlap(todo,ids)
                            print('       -> Overlap detected --> Save non overlapping: '+str(len(noids))+' points')
                            missedPlates_pmin[noids] = pmin
                            missedPlates[noids] = nn
                            todo[noids]        = False
                            plateID[noids]     = 999999   # will be not found in a PlateGather object
                            persistence[noids] = 999999   # easily tracked in persistence
                            rigidity[noids]    = False
                            nPlateID[noids]    = n
                            # --- Restart the persistence search from the top
                            restart_at_pmin  = None
                            nn += 1
                            
                    # ------
                    k += 1
            
        else:
            # in the case of a very small polygon -> ignore the computation of a rotation pole
            
            print('  -----> too small: \t surface: '+str(surf))
            
            ids = np.arange(len(ref.x))[ref.plateID == pID]
            
            # -- test the overlapping
            if not overlap(todo,ids):
                todo[ids]        = False
                plateID[ids]     = pID
                persistence[ids] = pmin_ref
                rigidity[ids]    = False
                nPlateID[ids]    = n
                # --- Restart the persistence search from the top
                restart_at_pmin  = None
            else:
                noids, oids = find_overlap(todo,ids)
                print('   -> Overlap detected on a small plate --> Save non overlapping: '+str(len(noids))+' points')
                missedPlates_pmin[noids] = pmin
                missedPlates[noids] = nn
                todo[noids]        = False
                plateID[noids]     = 999999   # will be not found in a PlateGather object
                persistence[noids] = 999999   # easily tracked in persistence
                rigidity[noids]    = False
                nPlateID[noids]    = n
                # --- Restart the persistence search from the top
                restart_at_pmin  = None
                nn += 1
                        
        # --- iterate -----------
        n += 1

    # --- rescale the new plate ID -----------

    nod   = len(nPlateID)
    unPID, surf = np.unique(nPlateID,return_counts=True)
    nop   = len(unPID)

    nPlateID_rescaled = nPlateID.copy()
    for i in range(nop):
        m = nPlateID == unPID[i]
        nPlateID_rescaled[m] = i
        

    # --- END ----
    time1 = time()

    print()
    print('-'*50)
    print('Time taken by the optimization: '+str(int(time1-time0))+' seconds')


    # --- Export to h5 ---------------------------------

    print()
    print('-'*50)
    print('Exportation of the optimization parameters')

    if output_path[-1] != '/':
        output_path += '/'
    print(' -> Export here: '+output_path)

    path2h5 = output_path+ofile+'_optimized_param.h5'
    fid  = h5py.File(path2h5,'w')
    dset = fid.create_dataset('P1P2_crit',    data = np.array([Project.P1c,Project.P2c]))
    dset = fid.create_dataset('fragment_size',data = np.array([Project.fragment_size])) 
    dset = fid.create_dataset('pthreshold',   data = pthreshold)
    dset = fid.create_dataset('pmin_ref_ID',  data = pmin_ref_ID)
    dset = fid.create_dataset('pointID',      data = ref.pointID)
    dset = fid.create_dataset('plateID',      data = plateID)
    dset = fid.create_dataset('nPlateID',     data = nPlateID_rescaled)
    dset = fid.create_dataset('persistence',  data = persistence)
    dset = fid.create_dataset('rigidity',     data = rigidity)
    dset = fid.create_dataset('missedPlates', data = missedPlates)
    dset = fid.create_dataset('trackedFF',    data = trackedFF)
    dset = fid.create_dataset('missedPlates_pmin', data = missedPlates_pmin)
    fid.close()

    # --- Make a map of the optimization result  ---------------------------------
    if plot:
        
        print()
        print('-------')
        print('Map of the optimization result')
    
        nPlateID_rescaled_randomized = list(range(nop))
        random.shuffle(nPlateID_rescaled_randomized)
        nPlateID_rescaled_randomized = np.array(nPlateID_rescaled_randomized)
        zs = np.zeros(nod,dtype=np.int32)
        for j in range(nod):
            zs[j] = nPlateID_rescaled_randomized[nPlateID_rescaled[j]]

        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,3,1, projection=ccrs.Robinson())
        ax2 = fig.add_subplot(1,3,2, projection=ccrs.Robinson())
        ax3 = fig.add_subplot(1,3,3, projection=ccrs.Robinson())
        ax1.set_title('Tessellation')
        ax1.set_global()
        cmap1  = ax1.scatter(lon,lat,c=zs,s=1,cmap=plt.cm.magma,transform=ccrs.PlateCarree())
        cbar1 = fig.colorbar(cmap1,ax=ax1,orientation='horizontal')
        # ---
        ax2.set_title('Persistence')
        ax2.set_global()
        cmap2 = ax2.scatter(lon,lat,c=persistence,s=1,vmin=pthreshold[0],vmax=pthreshold[pmin_ref_ID],cmap=plt.cm.magma,transform=ccrs.PlateCarree())
        cbar2 = fig.colorbar(cmap2,ax=ax2,orientation='horizontal')
        # ---
        ax3.set_title('Rigidity')
        ax3.set_global()
        cmap3 = ax3.scatter(lon,lat,c=rigidity,s=1,cmap=plt.cm.magma,transform=ccrs.PlateCarree())
        cbar3 = fig.colorbar(cmap3,ax=ax3,orientation='horizontal')
        plt.tight_layout()
        plt.show()

    # --- Build the optimal PlateGather instance ---------------------------------

    optimize_from_param(path2h5,path2h5file,ofile,output_path=output_path,verbose=verbose,\
                        plot_missed = plot_missed, add_missedPlates = add_missedPlates,\
                        alpha_init = alpha_init, alpha_search_step = alpha_search_step,\
                        edges_detection_method = edges_detection_method, simplex_threshold = simplex_threshold, hidden = hidden,\
                        convergeThreshold = convergeThreshold, verbose_edgesExtraction = verbose_edgesExtraction)
    
    



# =====================================================



def optimize_from_param(path2param, path2h5file, ofile, output_path='./', \
                        add_missedPlates = True,\
                        edges_detection_method = 'paraview',\
                        simplex_threshold = 0.01, hidden = True,\
                        alpha_init = 90, alpha_search_step = 10, convergeThreshold = 3,\
                        plot_missed = False,\
                        verbose=False, verbose_edgesExtraction=False):
    """ Function that builds a MAPT3.tessellation.PlateGather instance from the
    file _optimized_param.h5 generated from the function MAPT3.optimize.optimize.
    The PlateGather instance is then save in an .h5 file ending by _optimized.h5.

    Args:
        path2param (str): Path to optimizated tessellation parameter file
                          ending by '_optimized_param.h5' and generated by
                          the function MAPT3.optimize.optimize.

        For other arguments: See the function MAPT3.optimize.optimize for details.

    """
    
    # --- Import to h5 ---------------------------------

    fid  = h5py.File(path2param,'r')
    pthreshold        = np.array(fid['pthreshold'])
    pmin_ref_ID       = np.array(fid['pmin_ref_ID'])
    plateID           = np.array(fid['plateID'])
    nPlateID_rescaled = np.array(fid['nPlateID'])
    persistence       = np.array(fid['persistence'])
    rigidity          = np.array(fid['rigidity'])
    missedPlates      = np.array(fid['missedPlates'])
    missedPlates_pmin = np.array(fid['missedPlates_pmin'])
    fid.close()
    
    # --- Reference -----------

    pmin_ref = pthreshold[pmin_ref_ID]
    
    ref = PlateGather()
    ref.verbose = verbose
    ref.load_from_h5(path2h5file(pmin_ref))

    nod   = len(nPlateID_rescaled)
    unPID, surf = np.unique(nPlateID_rescaled,return_counts=True)
    nop   = len(unPID)


    # --- Build the optimal PlateGather instance ---------------------------------
    
    opti = PlateGather()

    # Plates info
    opti.nop = nop
    opti.persistence = np.zeros(nop,dtype=np.float64)
    # --------- Surface Gather

    opti.surf    = np.zeros(nop,dtype=np.int32) # length is here nop and not nod
    done_surfaces= np.zeros(int(1.5*nod),dtype=bool)
    # Plates
    opti.x       = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.y       = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.z       = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vx      = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vy      = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vz      = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vr      = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vtheta  = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.vphi    = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.pressure= np.zeros(int(1.5*nod),dtype=np.float64)
    opti.plateID = np.zeros(int(1.5*nod),dtype=np.int32)
    opti.magGSV  = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.pointID = np.zeros(int(1.5*nod),dtype=np.int32)
    opti.pmin    = np.zeros(int(1.5*nod),dtype=np.float64)
    # Geographic grid
    opti.lon     = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.lat     = np.zeros(int(1.5*nod),dtype=np.float64)
    opti.r       = np.zeros(int(1.5*nod),dtype=np.float64)

    # --------- Edge Gather
    opti.peri     = np.zeros(nop,dtype=np.int32) # length is here nop and not nod
    done_edges    = np.zeros(nod,dtype=bool)
    # Plates
    opti.xe       = np.zeros(nod,dtype=np.float64)
    opti.ye       = np.zeros(nod,dtype=np.float64)
    opti.ze       = np.zeros(nod,dtype=np.float64)
    opti.vxe      = np.zeros(nod,dtype=np.float64)
    opti.vye      = np.zeros(nod,dtype=np.float64)
    opti.vze      = np.zeros(nod,dtype=np.float64)
    opti.vre      = np.zeros(nod,dtype=np.float64)
    opti.vthetae  = np.zeros(nod,dtype=np.float64)
    opti.vphie    = np.zeros(nod,dtype=np.float64)
    opti.pressuree= np.zeros(nod,dtype=np.float64)
    opti.plateIDe = np.zeros(nod,dtype=np.int32)
    opti.magGSVe  = np.zeros(nod,dtype=np.float64)
    opti.pointIDe = np.zeros(nod,dtype=np.int32)
    opti.pmine    = np.zeros(nod,dtype=np.float64)
    # Geographic grid
    opti.lone     = np.zeros(nod,dtype=np.float64)
    opti.late     = np.zeros(nod,dtype=np.float64)
    opti.re       = np.zeros(nod,dtype=np.float64)

    # --------- Boundary Gather
    done_boundaries = np.zeros(nod,dtype=bool)
    # Boundaries
    opti.xb       = np.zeros(nod,dtype=np.float64)
    opti.yb       = np.zeros(nod,dtype=np.float64)
    opti.zb       = np.zeros(nod,dtype=np.float64)
    opti.vxb      = np.zeros(nod,dtype=np.float64)
    opti.vyb      = np.zeros(nod,dtype=np.float64)
    opti.vzb      = np.zeros(nod,dtype=np.float64)
    opti.vrb      = np.zeros(nod,dtype=np.float64)
    opti.vthetab  = np.zeros(nod,dtype=np.float64)
    opti.vphib    = np.zeros(nod,dtype=np.float64)
    opti.pressureb= np.zeros(nod,dtype=np.float64)
    opti.magGSVb  = np.zeros(nod,dtype=np.float64)
    # Geographic grid
    opti.lonb     = np.zeros(nod,dtype=np.float64)
    opti.latb     = np.zeros(nod,dtype=np.float64)
    opti.rb       = np.zeros(nod,dtype=np.float64)

    # N.B. platecouple and pminb are initialized and defined below

    # --- Extract data from plates iteratively 

    ids0 = 0    # surfaces
    ids1 = 0

    ide0 = 0    # edges
    ide1 = 0

    idb0 = 0    # boundaries
    idb1 = 0

    print()
    print('-------')
    print('Building of an optimized PlateGather instance')
    print('  -> Iteration on all the plates:')

    current_plateID = 0

    for i in tqdm(range(nop)):
        
        # iterate on all plates
        m = nPlateID_rescaled == i

        if len(np.unique(persistence[m])) != 1 or len(np.unique(plateID[m])) != 1:
            raise OptimizationError('The current plate i is apparently linked to several persistence values or different source plateID values')
        
        pmin   = persistence[m][0]
        refpid = plateID[m][0]
        
        if pmin < 99999:
            # else: it is a missed plate
            
            # --- load source h5 file
            pg = PlateGather()
            pg.verbose = verbose
            pg.load_from_h5(path2h5file(pmin))

            # creat local masks for edges and boundaries
            ms  = pg.plateID  == refpid
            me  = pg.plateIDe == refpid
            mb1 = pg.platecouple[:,0] == refpid
            mb2 = pg.platecouple[:,1] == refpid
            mb  = mb1 +mb2
            

            if np.count_nonzero(ms) == 0:
                # means that the plate is just an isolated point
                pass
            else:
                
                opti.persistence[current_plateID] = pmin
                
                # --------- Surfaces Gather
                ids1 = ids0 + np.count_nonzero(ms)
                
                surf = np.count_nonzero(ms)
                opti.surf[current_plateID] = surf
                opti.x[ids0:ids1]       = pg.x[ms]
                opti.y[ids0:ids1]       = pg.y[ms]
                opti.z[ids0:ids1]       = pg.z[ms]
                opti.vx[ids0:ids1]      = pg.vx[ms]
                opti.vy[ids0:ids1]      = pg.vy[ms]
                opti.vz[ids0:ids1]      = pg.vz[ms]
                opti.vr[ids0:ids1]      = pg.vr[ms]
                opti.vtheta[ids0:ids1]  = pg.vtheta[ms]
                opti.vphi[ids0:ids1]    = pg.vphi[ms]
                opti.pressure[ids0:ids1]= pg.pressure[ms]
                opti.plateID[ids0:ids1] = current_plateID
                opti.magGSV[ids0:ids1]  = pg.magGSV[ms]
                opti.pointID[ids0:ids1] = pg.pointID[ms]
                opti.pmin[ids0:ids1]    = pmin
                # Geographic grid
                opti.lon[ids0:ids1]     = pg.lon[ms]
                opti.lat[ids0:ids1]     = pg.lat[ms]
                opti.r[ids0:ids1]       = pg.r[ms]

                # --- iterative conditions and mask
                done_surfaces[ids0:ids1] = True
                ids0 = ids1
                
                # --------- Edges Gather
                ide1 = ide0 + np.count_nonzero(me)
                
                peri = np.count_nonzero(me)
                opti.peri[current_plateID] = peri
                opti.xe[ide0:ide1]       = pg.xe[me]
                opti.ye[ide0:ide1]       = pg.ye[me]
                opti.ze[ide0:ide1]       = pg.ze[me]
                opti.vxe[ide0:ide1]      = pg.vxe[me]
                opti.vye[ide0:ide1]      = pg.vye[me]
                opti.vze[ide0:ide1]      = pg.vze[me]
                opti.vre[ide0:ide1]      = pg.vre[me]
                opti.vthetae[ide0:ide1]  = pg.vthetae[me]
                opti.vphie[ide0:ide1]    = pg.vphie[me]
                opti.pressuree[ide0:ide1]= pg.pressuree[me]
                opti.plateIDe[ide0:ide1] = current_plateID
                opti.magGSVe[ide0:ide1]  = pg.magGSVe[me]
                opti.pointIDe[ide0:ide1] = pg.pointIDe[me]
                opti.pmine[ide0:ide1]    = pmin
                # Geographic grid
                opti.lone[ide0:ide1]     = pg.lone[me]
                opti.late[ide0:ide1]     = pg.late[me]
                opti.re[ide0:ide1]       = pg.re[me]
                

                # --- iterative conditions and mask
                done_edges[ide0:ide1] = True
                ide0 = ide1
                
                # --------- Boundary Gather
                idb1 = idb0 + np.count_nonzero(mb)
                                
                opti.xb[idb0:idb1]       = pg.xb[mb]
                opti.yb[idb0:idb1]       = pg.yb[mb]
                opti.zb[idb0:idb1]       = pg.zb[mb]
                if len(pg.vxb) > 0:
                    opti.vxb[idb0:idb1]      = pg.vxb[mb]
                    opti.vyb[idb0:idb1]      = pg.vyb[mb]
                    opti.vzb[idb0:idb1]      = pg.vzb[mb]
                    opti.vrb[idb0:idb1]      = pg.vrb[mb]
                    opti.vthetab[idb0:idb1]  = pg.vthetab[mb]
                    opti.vphib[idb0:idb1]    = pg.vphib[mb]
                    opti.pressureb[idb0:idb1]= pg.pressureb[mb]
                    opti.magGSVb[idb0:idb1]  = pg.magGSVb[mb]
                # Geographic grid
                opti.lonb[idb0:idb1]     = pg.lonb[mb]
                opti.latb[idb0:idb1]     = pg.latb[mb]
                opti.rb[idb0:idb1]       = pg.rb[mb]
                
                # --- iterative conditions and mask
                done_boundaries[idb0:idb1] = True
                idb0 = idb1
                
                # --------------------------------------
                current_plateID += 1


    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
    # --- Add the missed plates one by one ------------

    print()
    print('-------')
    print('Number of missed points: '+str(np.count_nonzero(missedPlates>0)))
    if add_missedPlates:
        my_order = 'Add the missed points as plates with clustering'
    else:
        my_order = 'Ignore'
    print('  -> What to do about these points: '+my_order)

    if add_missedPlates:

        print()
        print('-------')
        print('Add the missed plates one by one')

        uMPID = np.unique(missedPlates)
        uMPID = uMPID[uMPID > 0]

        print('  -> Clustering analysis')
        
        uMPID = np.unique(missedPlates)
        uMPID = uMPID[uMPID > 0]

        missedPlates_new = np.zeros(missedPlates.shape[0],np.int32)
        
        # --- auto compute the step based on the average (over 'nTests' tests) point spacing
        noTests = 100
        tested_step = np.zeros(noTests,dtype=np.float64)
        for i in range(noTests): # 5 test
            ptID = np.random.randint(0,len(pg.x))
            mydist = np.sqrt((pg.x[ptID]-pg.x)**2+(pg.y[ptID]-pg.y)**2+(pg.z[ptID]-pg.z)**2)
            mydist = mydist[mydist>0]
            tested_step[i] = np.amin(mydist)
        distance_cluster_threshold = 2 * np.mean(tested_step)
        print('      -> Use the average point spacing: '+str(distance_cluster_threshold))
        
        for i in range(len(uMPID)):
            print('     - Iterate on missed plate ID: '+str(uMPID[i]))
            m = missedPlates == uMPID[i]
            clusterIDoutput, clusterloc = clustering(distance_cluster_threshold,ref.x[m],ref.y[m],ref.z[m])
            missedPlates_new[m] = clusterIDoutput + np.amax(missedPlates_new)  
        
        uMPIDn = np.unique(missedPlates_new)
        uMPIDn = uMPIDn[uMPIDn > 0]
        noc = len(uMPIDn) # number of clusters
        print('  -> Number of missed plates: '+str(noc))
        
        print('  -> Prepare plate boundaries')
        listOfFiles = [path2h5file(pthreshold[i]) for i in range(len(pthreshold))]
        tp = TopoPlates()
        tp.load(listOfFiles,pthreshold)

        for i in range(noc):

            ms = missedPlates_new == uMPIDn[i]

            if np.count_nonzero(ms) <= 1:
                # means that the plate is just an isolated point
                pass
            else:
                

                print('     | -> add plate: '+str(current_plateID))
                print('     |    nb of pts: '+str(len(missedPlates_pmin[ms])))
                print('     |    pmin:      '+str(np.unique(missedPlates_pmin[ms])))
                
                opti.persistence[current_plateID] = np.unique(missedPlates_pmin[ms])[0]
            
                # --------- Surfaces Gather
                ids1 = ids0 + np.count_nonzero(ms)
                
                surf = np.count_nonzero(ms)
                opti.surf[current_plateID] = surf
                opti.x[ids0:ids1]       = ref.x[ms]
                opti.y[ids0:ids1]       = ref.y[ms]
                opti.z[ids0:ids1]       = ref.z[ms]
                opti.vx[ids0:ids1]      = ref.vx[ms]
                opti.vy[ids0:ids1]      = ref.vy[ms]
                opti.vz[ids0:ids1]      = ref.vz[ms]
                opti.vr[ids0:ids1]      = ref.vr[ms]
                opti.vtheta[ids0:ids1]  = ref.vtheta[ms]
                opti.vphi[ids0:ids1]    = ref.vphi[ms]
                opti.pressure[ids0:ids1]= ref.pressure[ms]
                opti.plateID[ids0:ids1] = current_plateID
                opti.magGSV[ids0:ids1]  = ref.magGSV[ms]
                opti.pointID[ids0:ids1] = ref.pointID[ms]
                opti.pmin[ids0:ids1]    = missedPlates_pmin[ms]
                # Geographic grid
                opti.lon[ids0:ids1]     = ref.lon[ms]
                opti.lat[ids0:ids1]     = ref.lat[ms]
                opti.r[ids0:ids1]       = ref.r[ms]

                # --- iterative conditions and mask
                done_surfaces[ids0:ids1] = True
                ids0 = ids1
                
                # --------- Edges Gather
                xi = ref.x[ms]
                yi = ref.y[ms]
                zi = ref.z[ms]


                if np.count_nonzero(ms) <= 3:
                    # means that all the points of the plate are edges:
                    edges_mask = np.zeros(len(xi),dtype=bool)
                    
                else:
                    # Two different algorithms for the edges detection
                    
                    if edges_detection_method == 'alphashape':

                        # Define 2D x,y coordinates based on the distance (haversine) to a ref point
                        lati, loni, ri = xyz2latlon(xi,yi,zi)
                        lati = -(lati*180/np.pi -90)
                        lati = lati*np.pi/180
                        
                        # -------------- EDGES ---------------------------
                        LAT = lati
                        print('     |    Find plate edges')
                        sweepS = True
                        sweepN = True
                        if np.amax(LAT) > +89*np.pi/180:
                            sweepS = False
                        if np.amin(LAT) < -89*np.pi/180:
                            sweepN = False
                        # ----
                        if sweepN:
                            print('            -> Sweep to North: North pole UTM projection')
                            lat = sweeplat2north(LAT)
                            E,N = latlon2UTM(lat,loni)
                            # --- Edges detection
                            points = np.zeros((E.shape[0],2))
                            points[:,0] = E
                            points[:,1] = N
                            edges = get_edges(points,alpha_init=alpha_init,alpha_search_step=alpha_search_step,convergeThreshold=convergeThreshold)
                            mask_edges_N = get_indices_of_edges(points,edges)
                        # ----
                        if sweepS:
                            print('            -> Sweep to South: South pole UTM projection')
                            lat = sweeplat2south(LAT)
                            E,N = latlon2UTM(lat,loni)
                            # --- Edges detection
                            points = np.zeros((E.shape[0],2))
                            points[:,0] = E
                            points[:,1] = N
                            edges = get_edges(points,alpha_init=alpha_init,alpha_search_step=alpha_search_step,convergeThreshold=convergeThreshold)
                            mask_edges_S = get_indices_of_edges(points,edges)
                        # ---
                        # Merge edges obtain from the northern and the southern hemisphere projection
                        if not sweepS and sweepN:
                            edges_mask = mask_edges_N
                        elif not sweepN and sweepS:
                            edges_mask = mask_edges_S
                        else:
                            edges_mask = mask_edges_S + mask_edges_N
                        # ---                
                        if plot_missed:
                            fig = plt.figure(figsize=(10,6))
                            ax  = fig.add_subplot(111)
                            ax.set_title('Edges detection for the plate '+str(current_plateID))
                            ax.scatter(E,N,color='blue',alpha=0.5,s=1,label='surface')
                            ax.scatter(E[edges_mask],N[edges_mask],color='red',label='edges')
                            ax.legend()
                            plt.show()
                        # ------------------------------------------------

                    elif edges_detection_method == 'paraview':
                    
                        # Define 2D x,y coordinates based on the distance (haversine) to a ref point
                        lati, loni, ri = xyz2latlon(xi,yi,zi)
                        lati = -(lati*180/np.pi -90)
                        lati = lati*np.pi/180
                        
                        LAT = lati
                        print('     |    Find plate edges')
                        sweepS = True
                        sweepN = True
                        if np.amax(LAT) > +89*np.pi/180:
                            sweepS = False
                        if np.amin(LAT) < -89*np.pi/180:
                            sweepN = False
                            
                        if sweepN:
                            print('            -> Sweep to North: North pole UTM projection')
                            lat = sweeplat2north(LAT)
                            xi,yi,zi = latlon2xyz(lat,loni)
                        # ----
                        if sweepS:
                            print('            -> Sweep to South: South pole UTM projection')
                            lat = sweeplat2south(LAT)
                            xi,yi,zi = latlon2xyz(lat,loni)
                    
                        edgesID = extract_edges_paraview(xi,yi,zi,simplex_threshold=simplex_threshold,\
                                                        verbose=verbose_edgesExtraction,plot=plot_missed,\
                                                        hidden=hidden)
                        edges_mask = np.zeros(len(xi),dtype=bool)
                        edges_mask[edgesID] = True
                    
                    else:
                        print('  ** ERROR **')
                        print('Unrecognized value for edges_detection_method')
                        print('Process aborded')
                        return 0
                
                # ---- Finally
                if np.count_nonzero(edges_mask) == 0:
                    # means that their is only edges...
                    edges_mask = np.ones(len(xi),dtype=bool)
                
                # --- add
                ide1 = ide0 + np.count_nonzero(edges_mask)
                
                peri = np.count_nonzero(edges_mask)
                opti.peri[current_plateID] = peri
                opti.xe[ide0:ide1]       = ref.x[ms][edges_mask]
                opti.ye[ide0:ide1]       = ref.y[ms][edges_mask]
                opti.ze[ide0:ide1]       = ref.z[ms][edges_mask]
                opti.vxe[ide0:ide1]      = ref.vx[ms][edges_mask]
                opti.vye[ide0:ide1]      = ref.vy[ms][edges_mask]
                opti.vze[ide0:ide1]      = ref.vz[ms][edges_mask]
                opti.vre[ide0:ide1]      = ref.vr[ms][edges_mask]
                opti.vthetae[ide0:ide1]  = ref.vtheta[ms][edges_mask]
                opti.vphie[ide0:ide1]    = ref.vphi[ms][edges_mask]
                opti.pressuree[ide0:ide1]= ref.pressure[ms][edges_mask]
                opti.plateIDe[ide0:ide1] = current_plateID
                opti.magGSVe[ide0:ide1]  = ref.magGSV[ms][edges_mask]
                opti.pointIDe[ide0:ide1] = ref.pointID[ms][edges_mask]
                opti.pmine[ide0:ide1]    = missedPlates_pmin[ms][edges_mask]
                # Geographic grid
                opti.lone[ide0:ide1]     = ref.lon[ms][edges_mask]
                opti.late[ide0:ide1]     = ref.lat[ms][edges_mask]
                opti.re[ide0:ide1]       = ref.r[ms][edges_mask]
                

                # --- iterative conditions and mask
                done_edges[ide0:ide1] = True
                ide0 = ide1
                
                
                # --------- Boundaries Gather
                
                gIDb = []
                xei = ref.x[ms][edges_mask]
                yei = ref.y[ms][edges_mask]
                zei = ref.z[ms][edges_mask]
                idb = np.arange(len(tp.xb))
                for j in range(len(xei)):
                    dei = np.sqrt((tp.xb-xei[j])**2+(tp.yb-yei[j])**2+(tp.zb-zei[j])**2)
                    mloc_dei = dei < 0.1 # limit the search
                    if np.count_nonzero(mloc_dei) > 0:
                        zipped  = zip(dei[mloc_dei],idb[mloc_dei])
                        deis,idbs = list(zip(*sorted(zipped)))
                        a,b = np.unique(deis,return_counts=True)
                        for jj in range(7):
                            gIDb += idbs[np.sum(b[0:jj]):np.sum(b[0:jj+1])]
                gIDb = np.array(gIDb)
                
                # --- remove multiple
                
                nodb  = tp.lonb[gIDb].shape[0]
                todob = np.ones(nodb,dtype=bool)
                idb   = np.arange(nodb)
                for kk in range(nodb):
                    if todob[kk]:
                        mx = tp.xb[gIDb] == tp.xb[gIDb][kk]
                        my = tp.yb[gIDb] == tp.yb[gIDb][kk]
                        mz = tp.zb[gIDb] == tp.zb[gIDb][kk]
                        m  = mx*my*mz
                        locID = idb[m][1:] # keep only one points
                        todob[locID] = False
                
                # --- add
                
                idb1 = idb0 + np.count_nonzero(todob)
                    
                opti.xb[idb0:idb1]       = tp.xb[gIDb][todob]
                opti.yb[idb0:idb1]       = tp.yb[gIDb][todob]
                opti.zb[idb0:idb1]       = tp.zb[gIDb][todob]
                if len(tp.vxb) > 0:
                    opti.vxb[idb0:idb1]      = tp.vxb[gIDb][todob]
                    opti.vyb[idb0:idb1]      = tp.vyb[gIDb][todob]
                    opti.vzb[idb0:idb1]      = tp.vzb[gIDb][todob]
                    opti.vrb[idb0:idb1]      = tp.vrb[gIDb][todob]
                    opti.vthetab[idb0:idb1]  = tp.vthetab[gIDb][todob]
                    opti.vphib[idb0:idb1]    = tp.vphib[gIDb][todob]
                    opti.pressureb[idb0:idb1]= tp.pressureb[gIDb][todob]
                    opti.magGSVb[idb0:idb1]  = tp.magGSVb[gIDb][todob]
                # Geographic grid
                opti.lonb[idb0:idb1]     = tp.lonb[gIDb][todob]
                opti.latb[idb0:idb1]     = tp.latb[gIDb][todob]
                opti.rb[idb0:idb1]       = tp.rb[gIDb][todob]
                
                # --- iterative conditions and mask
                done_boundaries[idb0:idb1] = True
                idb0 = idb1
            
                # --------------------------------------
                current_plateID += 1
                

        # Plot missed plate clusters
        if plot_missed:
            ms = missedPlates > 0
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
            ax.set_global()
            ax.set_title('Map of all the missed plates')
            cmap = ax.scatter(ref.lon[ms],ref.lat[ms],c=missedPlates_pmin[ms],\
                              vmin=pthreshold[0],vmax=pthreshold[pmin_ref_ID],\
                              cmap=plt.cm.magma,\
                              transform=ccrs.PlateCarree())
            fig.colorbar(cmap,label='pmin',orientation='horizontal',pad=0.05,shrink=0.5)
            plt.show()


    # --- Add non rigid areas
    
    print()
    print('-------')
    print('Surface of non-rigid area: '+str(int(10000*(rigidity.shape[0]-np.count_nonzero(rigidity))/rigidity.shape[0])/100)+'%')
    print('  -> Add non-rigid areas')
    mnr = rigidity == False
    
    opti.xnr = ref.x[mnr]
    opti.ynr = ref.y[mnr]
    opti.znr = ref.z[mnr]
    opti.vxnr = ref.vx[mnr]
    opti.vynr = ref.vy[mnr]
    opti.vznr = ref.vz[mnr]
    opti.vrnr = ref.vr[mnr]
    opti.vthetanr = ref.vtheta[mnr]
    opti.vphinr = ref.vphi[mnr]
    opti.pressurenr = ref.pressure[mnr]
    opti.magGSVnr = ref.magGSV[mnr]
    opti.pminnr = np.zeros(np.count_nonzero(mnr))  # default, pmin non-rigid areas = 0
    opti.pointIDnr = ref.pointID[mnr]
    # Geographic grid
    opti.lonnr = ref.lon[mnr]
    opti.latnr = ref.lat[mnr]
    opti.rnr = ref.r[mnr]
        
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

    print()
    print('-------')
    print('Deallocate the unused memory')

    # --- Clean plate info

    opti.nop            = np.unique(opti.plateID).shape[0]
    mask_plate          = np.zeros(opti.surf.shape[0],dtype=bool)
    mask_plate[0:opti.nop] = True
    opti.surf           = opti.surf[mask_plate]
    opti.peri           = opti.peri[mask_plate]
    opti.persistence    = opti.persistence[mask_plate]

    # --- Clean surface data

    opti.x       = opti.x[done_surfaces]
    opti.y       = opti.y[done_surfaces]
    opti.z       = opti.z[done_surfaces]
    opti.vx      = opti.vx[done_surfaces]
    opti.vy      = opti.vy[done_surfaces]
    opti.vz      = opti.vz[done_surfaces]
    opti.vr      = opti.vr[done_surfaces]
    opti.vtheta  = opti.vtheta[done_surfaces]
    opti.vphi    = opti.vphi[done_surfaces]
    opti.pressure= opti.pressure[done_surfaces]
    opti.plateID = opti.plateID[done_surfaces]
    opti.magGSV  = opti.magGSV[done_surfaces]
    opti.pointID = opti.pointID[done_surfaces]
    opti.pmin    = opti.pmin[done_surfaces]
    # Geographic grid
    opti.lon     = opti.lon[done_surfaces]
    opti.lat     = opti.lat[done_surfaces]
    opti.r       = opti.r[done_surfaces]
        
    # --- Clean edges data

    opti.xe       = opti.xe[done_edges]
    opti.ye       = opti.ye[done_edges]
    opti.ze       = opti.ze[done_edges]
    opti.vxe      = opti.vxe[done_edges]
    opti.vye      = opti.vye[done_edges]
    opti.vze      = opti.vze[done_edges]
    opti.vre      = opti.vre[done_edges]
    opti.vthetae  = opti.vthetae[done_edges]
    opti.vphie    = opti.vphie[done_edges]
    opti.pressuree= opti.pressuree[done_edges]
    opti.plateIDe = opti.plateIDe[done_edges]
    opti.magGSVe  = opti.magGSVe[done_edges]
    opti.pointIDe = opti.pointIDe[done_edges]
    opti.pmine    = opti.pmine[done_edges]
    # Geographic grid
    opti.lone     = opti.lone[done_edges]
    opti.late     = opti.late[done_edges]
    opti.re       = opti.re[done_edges]

    # --- Clean boundaries data

    # 1. Remove undone points

    opti.xb       = opti.xb[done_boundaries]
    opti.yb       = opti.yb[done_boundaries]
    opti.zb       = opti.zb[done_boundaries]
    if len(opti.vxb) > 0:
        opti.vxb      = opti.vxb[done_boundaries]
        opti.vyb      = opti.vyb[done_boundaries]
        opti.vzb      = opti.vzb[done_boundaries]
        opti.vrb      = opti.vrb[done_boundaries]
        opti.vthetab  = opti.vthetab[done_boundaries]
        opti.vphib    = opti.vphib[done_boundaries]
        opti.pressureb= opti.pressureb[done_boundaries]
        opti.magGSVb  = opti.magGSVb[done_boundaries]
    # Geographic grid
    opti.lonb     = opti.lonb[done_boundaries]
    opti.latb     = opti.latb[done_boundaries]
    opti.rb       = opti.rb[done_boundaries]

    # 2. Remove multiples

    nodb  = opti.lonb.shape[0]
    todob = np.ones(nodb,dtype=bool)
    idb   = np.arange(nodb)
    for i in tqdm(range(nodb)):
        if todob[i]:
            mx = opti.xb == opti.xb[i]
            my = opti.yb == opti.yb[i]
            mz = opti.zb == opti.zb[i]
            m  = mx*my*mz
            locID = idb[m][1:]
            todob[locID] = False

    opti.xb       = opti.xb[todob]
    opti.yb       = opti.yb[todob]
    opti.zb       = opti.zb[todob]
    if len(opti.vxb) > 0:
        opti.vxb      = opti.vxb[todob]
        opti.vyb      = opti.vyb[todob]
        opti.vzb      = opti.vzb[todob]
        opti.vrb      = opti.vrb[todob]
        opti.vthetab  = opti.vthetab[todob]
        opti.vphib    = opti.vphib[todob]
        opti.pressureb= opti.pressureb[todob]
        opti.magGSVb  = opti.magGSVb[todob]
    # Geographic grid
    opti.lonb     = opti.lonb[todob]
    opti.latb     = opti.latb[todob]
    opti.rb       = opti.rb[todob]


    # --- Compute the remaining fields on boundaries

    nod_boundaries = len(opti.xb)

    opti.platecouple = np.zeros((nod_boundaries,2),dtype=np.int32)
    opti.pminb       = np.zeros(nod_boundaries,dtype=np.float64)
    
    print()
    print('-------')
    print('  -> Attribution of the couple of plates that bounds each plate boundary point:')
    opti.compute_platecouple()
        
    print()
    print('-------')
    print('  -> Compute the persistence on the plate boundaries:')
    for i in tqdm(range(nod_boundaries)):
        plate1 = opti.platecouple[i,0]
        plate2 = opti.platecouple[i,1]
        persi1 = opti.persistence[plate1]
        persi2 = opti.persistence[plate2]
        # Attribution:
        opti.pminb[i] = max(persi1,persi2)


    # --- Export the optimized plate tessellation in an hdf5 file
    if output_path[-1] != '/':
        output_path += '/'
    print(' -> Export here: '+output_path)
    
    print()
    print('-'*50)
    path2h5 = output_path+ofile+'_optimized.h5'
    opti.export2h5(path2h5)








# ------------------------------------------------

    
class TopoPlates:
    def __init__(self):
        """
        Main constructor
        """
        # --------- General
        self.pName   = 'TopoPlates'  # program name
        self.verbose = False         # verbose output condition
        self.path2file = ''          # complete path to imported data
        self.pmin = []
        # ----------
        self.drop  = []
        self.ci    = -1
        #
        self.pminb = []
        self.xb    = []
        self.yb    = []
        self.zb    = []
        self.rb    = []
        self.lonb  = []
        self.latb  = []
        self.vxb   = []
        self.vyb   = []
        self.vzb   = []
        self.vrb   = []
        self.vthetab = []
        self.vphib   = []
        self.pressureb = []
        self.magGSVb = []
        # --------- Other
        self.BIN = None
        self.bin = None
    
    
    def im(self,textMessage,error=False):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
            pName = str, name of the subprogram
            verbose = bool, condition for the verbose output
        """
        if self.verbose and not error:
            print('>> '+self.pName+'| '+textMessage)
        if error:
            #print error message
            print('>> '+self.pName+'| --- ----- ---')
            print('>> '+self.pName+'| --- ERROR ---')
            print('>> '+self.pName+'| --- ----- ---')
            print('>> '+self.pName+'| '+textMessage)


    def alloc(self,memory_alloc,add_velocities=False):
        self.im('Memory allocation')
        self.pminb = np.zeros(memory_alloc)-1 # negative persistence
        self.xb    = np.zeros(memory_alloc)
        self.yb    = np.zeros(memory_alloc)
        self.zb    = np.zeros(memory_alloc)
        self.rb    = np.zeros(memory_alloc)
        self.lonb  = np.zeros(memory_alloc)
        self.latb  = np.zeros(memory_alloc)
        if add_velocities:
            self.vxb        = np.zeros(memory_alloc)
            self.vyb        = np.zeros(memory_alloc)
            self.vzb        = np.zeros(memory_alloc)
            self.vrb        = np.zeros(memory_alloc)
            self.vthetab    = np.zeros(memory_alloc)
            self.vphib      = np.zeros(memory_alloc)
            self.pressureb  = np.zeros(memory_alloc)
            self.magGSVb    = np.zeros(memory_alloc)
    
    
    def load(self,listOfFiles,listOfpmin,memory_alloc=int(1e6)):
        """
        Load from .h5 file
        """
        # --- test if load velocities
        pg = PlateGather()
        pg.verbose = self.verbose
        pg.load_from_h5(listOfFiles[0])
        if len(pg.vxb) > 0:
            add_velocities = True
        else:
            add_velocities = False
        # ---
        self.path2file = listOfFiles
        self.pmin      = listOfpmin
        self.im('Importation of all files')
        self.im('  -> Importation of boundaries')
        nod = len(listOfFiles)
        k   = 1
        compute = True
        while compute:
            id0 = 0 ; past_id1 = 0
            self.alloc(memory_alloc*k,add_velocities=add_velocities)
            compute = False
            for i in range(nod):
                if past_id1<memory_alloc*k:
                    pg = PlateGather()
                    pg.verbose = self.verbose
                    pg.load_from_h5(listOfFiles[i])
                    id1 = id0+len(pg.xb)
                    past_id1 = id1
                if id1<memory_alloc*k:
                    self.xb[id0:id1]    = pg.xb
                    self.yb[id0:id1]    = pg.yb
                    self.zb[id0:id1]    = pg.zb
                    self.rb[id0:id1]    = pg.rb
                    self.lonb[id0:id1]  = pg.lonb
                    self.latb[id0:id1]  = pg.latb
                    self.pminb[id0:id1] = np.ones(len(pg.xb))*listOfpmin[i]
                    if add_velocities:
                        self.vxb[id0:id1]    = pg.vxb
                        self.vyb[id0:id1]    = pg.vyb
                        self.vzb[id0:id1]    = pg.vzb
                        self.vrb[id0:id1]    = pg.vrb
                        self.vthetab[id0:id1]  = pg.vthetab
                        self.vphib[id0:id1]  = pg.vphib
                        self.pressureb[id0:id1] = pg.pressureb
                        self.magGSVb[id0:id1] = pg.magGSVb
                    id0 = id1
                else:
                    compute = True
            if compute:
                self.im('Failed to load data with a memory allocation of '+str(memory_alloc*k)+' float points')
                k = k+1
                self.im(' -> Re-initialization with in a flexible mode with now '+str(memory_alloc*k)+' float points')
        self.im('Empty unused memory')
        mask = self.pminb >= 0
        self.pminb = self.pminb[mask]
        self.xb   = self.xb[mask]
        self.yb   = self.yb[mask]
        self.zb   = self.zb[mask]
        self.rb   = self.rb[mask]
        self.lonb = self.lonb[mask]
        self.latb = self.latb[mask]
        if add_velocities:
            self.vxb        = self.vxb[mask]
            self.vyb        = self.vyb[mask]
            self.vzb        = self.vzb[mask]
            self.vrb        = self.vrb[mask]
            self.vthetab    = self.vthetab[mask]
            self.vphib      = self.vphib[mask]
            self.pressureb  = self.vpressureb[mask]
            self.magGSVb    = self.vmagGSVb[mask]

    
    def reset(self):
        """
        reset the value of self.ci
        """
        self.ci = -1
    
    
    def scatter_map(self,s=5,cmap=plt.cm.magma,\
        projection=ccrs.Robinson(),transform=ccrs.PlateCarree(),\
        ticks=False,gridline=False,cbar=True,\
        ifigure=None,save=False,plot=True,dpi=300):
        """
        """
        # Prepare the map
        if ifigure is not None and isinstance(ifigure,mfigure.Figure):
            get = True
            fig = ifigure
        else:
            get = False
            fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1, projection=projection)
        if projection is not None:
            ax.set_global()
        # ticks
        if ticks or gridline:
            if projection is not None:
                if gridline:
                    gllinewidth = 1
                    glalpha = 0.5
                else:
                    gllinewidth = 0
                    glalpha = 0
                gl = ax.gridlines(crs=transform, draw_labels=ticks,
                    linewidth=gllinewidth, color='gray', alpha=glalpha, linestyle='--')
                gl.rotate_labels = False
        # ---
        pmin = self.pmin
        cNorm     = colors.Normalize(vmin=np.amin(pmin), vmax=np.amax(pmin))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        for i in range(len(pmin)):
            mask = self.pminb == pmin[i]
            ax.scatter(self.lonb[mask],self.latb[mask],s=s,color=scalarMap.to_rgba(pmin[i]),transform=transform)
        if cbar:
            cbaxes = fig.add_axes([0.3, 0.10, 0.4, 0.01]) # [left, bottom, width, height]
            cbar = plt.colorbar(scalarMap,cax=cbaxes,orientation='horizontal')
            cbar.set_label('pmin')
        # ---
        if save:
            self.im('Save the scatter plot')
            if fname is None:
                fname = './autoSave.png'
            self.im('   - Exportation on file: '+fname)
            plt.savefig(fname,dpi=dpi)
        if plot:
            plt.show()
        else:
            plt.close()
        if get:
            return fig,ax
    
    
    def mask_unique(self):
        """
        """
        nodb  = self.xb.shape[0]
        todob = np.ones(nodb,dtype=bool)
        idb   = np.arange(nodb)
        for i in tqdm(range(nodb)):
            if todob[i]:
                mx = self.xb == self.xb[i]
                my = self.yb == self.yb[i]
                mz = self.zb == self.zb[i]
                m  = mx*my*mz
                locID = idb[m][1:] # keep only one points
                todob[locID] = False
        return todob







# --- EDGES EXTRACTION ------------------------
# Functions for the Edges extraction using Paraview python and the filter FeatureEdges



def extract_edges_paraview(x,y,z,simplex_threshold=0.01,hidden=True,plot=False,verbose=False):
    """ Function returning the IDs of points forming the edges of a 3D surface
    using paraview and the filter FeatureEdges.

    Args:
        x (np.ndarray): x coordinates of the surface
        y (np.ndarray): y coordinates of the surface
        z (np.ndarray): z coordinates of the surface
        simplex_threshold (float, optional): Maximum length of the simplices you authorize
                                             for the meshing of the surface. Above, the
                                             simplices are remove. Defaults to 0.01.
        hidden (bool, optional): Option controling if you want to hide the construction file
                                 needed to compute the edges detection with paraview.
                                 The file will start with '.' and thus will be insisible for 
                                 linux and Mac default file listing.
        plot (bool, optional): Option if you want to display the result. Defaults to False.
        verbose (bool, optional): Option controlling the verbose output. Defaults to False.
    """
    if hidden:
        preffix = '.'
    else:
        preffix = ''
    
    tempFile_ofname = preffix + 'temp_edgesDetection' # without extension
    tempFile_opath  = './' # projPath + 'pTeTrack/tempFiles/'

    #if not os.path.exists(tempFile_opath):
    #    os.makedirs(tempFile_opath)

    # ----------------
    # Creation of a temp XDMF+H5 file containing the surface
    pointsID = np.arange(len(x))
    surface2VTK(x,y,z,pointsID,tempFile_ofname,'pointsID',path=tempFile_opath,\
                simplex_threshold=simplex_threshold,verbose=verbose)

    # ----------------
    # Call paraview

    # create a new 'XDMF Reader'
    #print(tempFile_opath+tempFile_ofname+'.xdmf')
    test_edgesDetectionxdmf = XDMFReader(registrationName='test_edgesDetection.xdmf', FileNames=[tempFile_opath+tempFile_ofname+'.xdmf'])
    test_edgesDetectionxdmf.PointArrayStatus = ['pointsID']
    #test_edgesDetectionxdmf.GridStatus = ['Grid_2']

    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=test_edgesDetectionxdmf)

    # create a new 'Feature Edges'
    featureEdges1 = FeatureEdges(registrationName='FeatureEdges1', Input=extractSurface1)

    path_csvData   = tempFile_opath
    ofname_csvData = tempFile_ofname + '.csv'
    SaveData(path_csvData + ofname_csvData, featureEdges1,Precision=10)
    Delete(featureEdges1)
    ResetSession()

    # ----------------
    # Call paraview
    
    separator = ','
    path2file = path_csvData + ofname_csvData
    nol = line_count(path2file)
    edgesID = np.zeros(nol,dtype=np.int32)
    with open(path2file,'r') as data:
        header = data.readline()
        i = 0
        for line in data:
            line = line.strip().split(separator)
            edgesID[i] = int(float(line[0]))
            i += 1
    
    if plot:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title('Result of the automoatic edges detection')
        ax.scatter(x, y, z, color='blue', s=10)
        ax.scatter(x[edgesID], y[edgesID], z[edgesID], color='red', s=20,label='edges')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()
        plt.show()
    
    return edgesID

