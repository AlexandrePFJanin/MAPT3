# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for Plate tracking and Analysis

"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import h5py
from tqdm import tqdm
from paraview.simple import *
import sys

# Internal dependencies:
from .tessellation import PlateGather
from .geotransform import xyz2latlon, haversine
from .generics import line_count, im
from .errors import LoadingFormatError


# ----------------- FUNCTIONS -----------------


def generateState_TTKplateTimeTrackingFromPersistenceDiag(path2persiDiag,fname,path='./',\
    verbose=True,logfile=True):
    """
    Generates a python state file for paraview loading each 'fictive'
    persistence diagram twice.
    Then, you will be able to open it in paraview and apply the filter
    'TTKplateTimeTrackingFromPersistenceDiag' directly and save the
    result in .csv with the 'SaveData' filter.

    Args:
        path2persiDiag (list of str): list of path to reach each persistence
                    diagram.
        fname (str): name of the state file (without extension)
        path (str, optional): path to the directory in which you want to save
                    the state file.
                    Defaults: path = './'
        logfile (bool, optional): Option controling the automatic generation
                    of a log file.
                    Defaults: logfile = True
        verbose (bool, optional): Option controling the display of a verbose
                    output on the terminal.
                    Defaults: verbose = True
    """
    
    # adjust the path if needed
    if path[-1] != '/':
            path += '/'

    # number of timestep
    nod = len(path2persiDiag)
    
    if verbose:
        print('Working on '+str(nod)+' persistence diagrams')
    
    # Prepare logfile
    if logfile:
        logfile_name = path + 'TTKplateTracking_' + fname + '.log'
        flog = open(logfile_name,'w')
        flog.write('-'*24+'\nTTKplateTracking log file'+'\n'+'-'*24+'\n\n')
        flog.write('Number of time steps: '+str(nod)+'\n\n')
        flog.write('Working files:\n')
    
    # Write State instruction:
    
    f = open(path + 'TTKplateTracking_' + fname + '.py','w')
    f.write("from paraview.simple import *"+'\n')
    f.write("#### disable automatic camera reset on 'Show'"+'\n')
    f.write("paraview.simple._DisableFirstRenderCameraReset()"+'\n')
    f.write(""+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write("# setup views used in the visualization"+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write(""+'\n')
    f.write("# Create a new 'Render View'"+'\n')
    f.write("renderView1 = CreateView('RenderView')"+'\n')
    f.write("renderView1.ViewSize = [1492, 802]"+'\n')
    f.write("renderView1.AxesGrid = 'GridAxes3DActor'"+'\n')
    f.write("renderView1.CenterOfRotation = [1e-20, 0.0, 0.0]"+'\n')
    f.write("renderView1.StereoType = 'Crystal Eyes'"+'\n')
    f.write("renderView1.CameraPosition = [-14.715165944823049, 0.0, 0.0]"+'\n')
    f.write("renderView1.CameraFocalPoint = [1e-20, 0.0, 0.0]"+'\n')
    f.write("renderView1.CameraViewUp = [0.0, 0.0, 1.0]"+'\n')
    f.write("renderView1.CameraFocalDisk = 1.0"+'\n')
    f.write("renderView1.CameraParallelScale = 3.808565198364234"+'\n')
    f.write(""+'\n')
    f.write("SetActiveView(None)"+'\n')
    f.write(""+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write("# setup view layouts"+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write(""+'\n')
    f.write("# create new layout object 'Layout #1'"+'\n')
    f.write("layout1 = CreateLayout(name='Layout #1')"+'\n')
    f.write("layout1.AssignView(0, renderView1)"+'\n')
    f.write("layout1.SetSize(1492, 802)"+'\n')
    f.write(""+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write("# restore active view"+'\n')
    f.write("SetActiveView(renderView1)"+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write(""+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    f.write("# setup the data processing pipelines"+'\n')
    f.write("# ----------------------------------------------------------------"+'\n')
    
    # iterative loading of all the pre-computed persistence Diagram
    for i in range(nod):
        
        # fill the log file
        if logfile:
            flog.write('   - '+path2persiDiag[i]+'\n')
        
        j = i*2
        f.write(""+'\n')
        f.write("# create a new 'XDMF Reader'"+'\n')
        f.write(""+'\n')
        f.write('persistenceDiag_%s'%int(j) + " = XDMFReader(registrationName='" + 'persistenceDiagram_%s.xdmf'%i + "', FileNames=['" + path2persiDiag[i] + "'])"+'\n')
        f.write('persistenceDiag_%s'%int(j) + ".PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']"+'\n')
        f.write('persistenceDiag_%s'%int(j) + ".CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']"+'\n')
        #f.write('persistenceDiag_%s'%int(j) + ".GridStatus = ['Grid_%s'"%int(j) + "]"+'\n')
        
        j = i*2+1
        f.write(""+'\n')
        f.write("# create a new 'XDMF Reader'"+'\n')
        f.write(""+'\n')
        f.write('persistenceDiag_%s'%int(j) + " = XDMFReader(registrationName='" + 'persistenceDiagram_%s.xdmf'%i + "', FileNames=['" + path2persiDiag[i] + "'])"+'\n')
        f.write('persistenceDiag_%s'%int(j) + ".PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']"+'\n')
        f.write('persistenceDiag_%s'%int(j) + ".CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']"+'\n')
        #f.write('persistenceDiag_%s'%int(j) + ".GridStatus = ['Grid_%s'"%int(j) + "]"+'\n')
    
    f.close()
    
    if verbose:
        print('Exportation of the state file')
        print('  - path: '+path)
        print('  - file: '+fname+'.py')
    
    if logfile:
        flog.write('\n')
        flog.write('Exportation of the state file:\n')
        flog.write('   - path: '+path+'\n')
        flog.write('   - file: '+fname+'.py\n')
    
    if verbose:
        print('Process complete!')
        
    if logfile:
        flog.write('\n')
        flog.write('Process complete')




def point_inside(point,points,precision=1e-5):
    """
    Tests if the point 'point' is in the point collection 'points'
    """
    xi,yi,zi = point
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    m1 = x >= xi - precision
    m2 = x <= xi + precision
    m3 = y >= yi - precision
    m4 = y <= yi + precision
    m5 = z >= zi - precision
    m6 = z <= zi + precision
    m  = m1*m2*m3*m4*m5*m6
    if np.count_nonzero(m) == 0:
        return False
    elif np.count_nonzero(m) == 1:
        return True
    else:
        return True 




# ----------------- PlateTracking -----------------


class PlateTracking:
    def __init__(self):
        """ Data structure for PlateTracking objects.
        Works as a cloud data: Data not loaded all together. """
        # --------- General
        self.pName   = 'PlateTracking'  # program name
        self.verbose = True           # verbose output condition
        # --------- Critical points tracking info
        self.path2file = ''           # complete path to imported tracking data
        self.header = []              # list of .csv header name
        self.ftype = 'ttk'            # file type for the tracking:
        # Data
        self.nt_tracking = 0      # Detected number of timesteps in the tracking file
        self.nt_tessellation = 0  # Detected number of timesteps in the tessellation files
        self.notp  = 0            # Number of tracked points
        # Cartesian XYZ coordinates
        self.x    = []     # x cartesian coordinates of all critical points that are tacked in time
        self.y    = []     # y cartesian coordinates of all critical points
        self.z    = []     # z cartesian coordinates of all critical points
        # Geographic ENU coordinates
        self.r    = []     # r (radial) coordinates of all critical points
        self.lon  = []     # longitudes of all critical points
        self.lat  = []     # latitudes of all critical points
        # Tracking info
        self.ctype = []     # type of all critical points (0 = local minima, 3 = local maxima)
        self.ctime = []     # time associated to tracking data (in index of time, from 0 to N)
        self.connectedComponentID = []  # Index of each critical identified, unique for a tracked points
        # Computed
        self.matching = []  # Matching between the connected component IDs and the plate IDs of the current drop
        # --------- Generic: PlateGather construction
        # For H5 files
        self.path2h5 = None        # function returning the path to a tessellation at a given time step
        # indices
        self.indices  = None            #List of all indices
        # drop state
        self.ci       = -1              # current drop index
        # --------- PlateGather
        self.drop = None    #An instance of cloud (like a rain drop)
                            #self.drop will have a type derived from PlateGather
        # --------- Other
        self.BIN = None
        self.bin = None


    def im(self,textMessage,error=False):
        """Print verbose internal message."""
        im(textMessage,self.pName,self.verbose,error=error)
    

    def xyz2latlon_trackingPTS(self):
        """
        Computes automatically the ENU coordinates of tracked critical
        pts from their xyz coordinates.
        """
        self.r     = np.sqrt(self.x**2+self.y**2+self.z**2)
        self.lat   = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lon   = np.arctan2(self.y,self.x)
        self.lat   = -(self.lat*180/np.pi-90)
        self.lon   = self.lon*180/np.pi
    
    
    def __load_tracking_bind(self,path2file,separator=',',headerLen=1):
        """
        Function loading the tracking data from a .csv file in a 'bind' format
        -> adjusted tracking data

        Args:
            path2file (str): path to the .csv file containing the tracking
                        data (in the 'bind' format: see the function
                        MAPT3.timetracking.PlateTracking.adjstTracking).
            separator (str): file separator for the .csv file.
                        Defaults: separator = ','
            headerLen (int): number of header line to be remove.
                        Defaults: headerLen = 1
        """
        self.im('Load TimeTracking data from file:')
        self.im('   -> '+path2file)
        nol = line_count(path2file)
        nol -= headerLen
        self.im('   -> Number of Lines: '+str(nol))
        self.im('Memory allocation')
        # memory allocation
        self.ctime = np.zeros(nol,dtype=np.int32)
        self.connectedComponentID = np.zeros(nol,dtype=np.int32)
        self.ctype = np.zeros(nol,dtype=np.int32)
        self.x = np.zeros(nol,dtype=np.float64)
        self.y = np.zeros(nol,dtype=np.float64)
        self.z = np.zeros(nol,dtype=np.float64)
        # iterative reading
        self.im('Iterative reading')
        with open(path2file,'r') as data:
            for i in range(headerLen):
                self.header = data.readline()
            i = 0
            for line in data:
                line = line.strip().split(separator)
                self.connectedComponentID[i] = int(line[1])
                self.ctime[i] = int(line[0])
                self.ctype[i] = int(line[2])
                self.x[i] = float(line[3])
                self.y[i] = float(line[4])
                self.z[i] = float(line[5])
                i += 1
        self.im('Compute the ENU coordinates')
        self.xyz2latlon_trackingPTS()
        # number of tracked points
        self.notp = len(np.unique(self.connectedComponentID))
        self.im('   - Number of tracked points: '+str(self.notp))
        self.nt_tracking = len(np.unique(self.ctime))
        self.im('   - Number of time steps in the tracking file: '+str(self.nt_tracking))
        print()
    
    
    def __load_tracking_TTK(self,path2file,separator=',',headerLen=1,time_repeated=True):
        """
        Function loading the tracking data from a .csv file in a 'ttk' format.
        This format is the native format of the data exported by the filter 'SaveData'
        from filter 'TTKtrackingFromPersistenceDiagram'.
        
        The idea of the reading is the following: The writing of the tracking data
        introduces multiples and ghost points that come from the representation of the tracking
        by a collection of VTK lines in Paraview. This function read iteratively the input file
        and keep only the relevant information.
        
                                   * WARNING *
                                   
        This function has been written for an input tracking file where all the time steps
        has been repeated twice to insure saving and detecting the points living at least
        over two time steps (otherwise, it is three). Be sure that your input .csv file has
        been produce with this trick *OR* set the optional argument 'time_repeated' to False
        (but obsolete, prefere the first option).

        Advice: Generate the data from the state file build from the function
                'MAPT3.timetracking.generateState_TTKplateTimeTrackingFromPersistenceDiag'

        Args:
            path2file (str): Complete path to the tracking file in .csv (with the extension)
            separator (str, optional): Row separator in the tracking file. Defaults to ','.
            headerLen (int, optional): Length of the header in number of lines. Defaults to 1.
            time_repeated (bool, optional): Option specifying if the tracking file has been obtain
                                            with a time (persistence diag) repeated twice.
                                            Defaults to True.
        """

        self.im('Load TimeTracking data from file:')
        self.im('   -> '+path2file)
        nol = line_count(path2file)
        nol -= headerLen
        self.im('   -> Number of Lines: '+str(nol))
        self.im('Memory allocation')
        # ---
        # memory allocation for global variables (current class instance)
        self.ctime = np.zeros(nol,dtype=np.int32)
        self.connectedComponentID = np.zeros(nol,dtype=np.int32)
        self.ctype = np.zeros(nol,dtype=np.int32)
        self.x = np.zeros(nol,dtype=np.float64)
        self.y = np.zeros(nol,dtype=np.float64)
        self.z = np.zeros(nol,dtype=np.float64)
        # memory allocation for local variables (for a given connected component ID)
        if time_repeated:
            nodi = int(len(self.indices)*4)  # *4 due to ghots points + the time repetition
        else:
            nodi = int(len(self.indices)*2)  # *2 due to ghost points
        ctimei = np.zeros(nodi,dtype=np.int32)
        connectedComponentIDi = np.zeros(nodi,dtype=np.int32)
        ctypei = np.zeros(nodi,dtype=np.int32)
        xi    = np.zeros(nodi,dtype=np.float64)
        yi   = np.zeros(nodi,dtype=np.float64)
        zi   = np.zeros(nodi,dtype=np.float64)
        keep = np.zeros(nodi,dtype=bool)        # a mask that precise which elements of
        # ---                                   # of the local variable has been filled
        # iterative reading                     # during the loop i
        self.im('Iterative reading')
        with open(path2file,'r') as data:
            for i in range(headerLen):
                self.header = data.readline()   # save te header
            # --- read the line
            line = data.readline().strip().split(separator)
            # --- init
            ctimei[0] = int(line[0])
            connectedComponentIDi[0] = int(line[1])
            ctypei[0] = int(line[2])
            xi[0] = float(line[3])
            yi[0] = float(line[4])
            zi[0] = float(line[5])
            keep[0] = True
            # --- init counters
            cccID = self.connectedComponentID[0]
            ci = 0 ; ce = 0
            i  = 0
            # --- iteration
            for line in data:
                line = line.strip().split(separator)
                connectedComponentID = int(line[1])
                if connectedComponentID == cccID:
                    i += 1
                else:
                    cccID = connectedComponentID
                    # ---
                    keep[np.arange(0,int(nodi),2)] = False # remove the odd indices
                    keep[0] = True
                    loctime = ctimei[keep]
                    loctime[1:len(loctime)] = loctime[1:len(loctime)] + 1
                    ce = ci + np.count_nonzero(keep)
                    self.connectedComponentID[ci:ce] = connectedComponentIDi[keep]
                    self.ctime[ci:ce] = loctime
                    self.ctype[ci:ce] = ctypei[keep]
                    self.x[ci:ce] = xi[keep]
                    self.y[ci:ce] = yi[keep]
                    self.z[ci:ce] = zi[keep]
                    ci = ce
                    # ---
                    keep = np.zeros(nodi,dtype=bool) # re init the mask to False
                    i = 0 # re-init the counter to 0
                # Fill the local variable now since the 'i' index has been updated
                connectedComponentIDi[i] = int(line[1])
                ctimei[i] = int(line[0])
                ctypei[i] = int(line[2])
                xi[i] = float(line[3])
                yi[i] = float(line[4])
                zi[i] = float(line[5])
                keep[i] = True
            # ------
            # Save the last point
            keep[np.arange(0,int(nodi),2)] = False # remove the odd indices
            keep[0] = True
            loctime = ctimei[keep]
            loctime[1:len(loctime)] = loctime[1:len(loctime)] + 1
            ce = ci + np.count_nonzero(keep)
            self.connectedComponentID[ci:ce] = connectedComponentIDi[keep]
            self.ctime[ci:ce] = loctime#ctimei[keep]
            self.ctype[ci:ce] = ctypei[keep]
            self.x[ci:ce] = xi[keep]
            self.y[ci:ce] = yi[keep]
            self.z[ci:ce] = zi[keep]
            ci = ce
        # --- Deallocate
        self.im('Deallocate the unused memory')
        processed = np.zeros(nol,dtype=bool)
        processed[0:ce] = True
        self.connectedComponentID = self.connectedComponentID[processed]
        self.ctime = self.ctime[processed]
        self.ctype = self.ctype[processed]
        self.x = self.x[processed]
        self.y = self.y[processed]
        self.z = self.z[processed]
        # --- Remove multiple
        if time_repeated:
            self.im('Remove multiples due the time repetition x2')
            self.im('  -> Allocate memory')
            # memory allocation for global variables
            ctime = np.zeros(nol,dtype=np.int32)
            connectedComponentID = np.zeros(nol,dtype=np.int32)
            ctype = np.zeros(nol,dtype=np.int32)
            x = np.zeros(nol,dtype=np.float64)
            y = np.zeros(nol,dtype=np.float64)
            z = np.zeros(nol,dtype=np.float64)
            processed = np.zeros(nol,dtype=bool)
            self.im('  -> Iterative cleaning')
            uPID = np.unique(self.connectedComponentID)
            ci = 0 ; ce = 0
            for i in range(len(uPID)):
                m = self.connectedComponentID == uPID[i]
                nod = np.count_nonzero(m)
                keep = np.ones(nod,dtype=bool) ; keep[np.arange(1,nod,2)] = False   # odd index to False
                ce = ci + np.count_nonzero(keep)
                connectedComponentID[ci:ce] = self.connectedComponentID[m][keep]
                ctime[ci:ce] = self.ctime[m][keep]/2
                ctype[ci:ce] = self.ctype[m][keep]
                x[ci:ce] = self.x[m][keep]
                y[ci:ce] = self.y[m][keep]
                z[ci:ce] = self.z[m][keep]
                processed[ci:ce] = True
                ci = ce
            self.connectedComponentID = connectedComponentID[processed]
            self.ctime = ctime[processed]
            self.ctype = ctype[processed]
            self.x = x[processed]
            self.y = y[processed]
            self.z = z[processed]
        # --- Compute ENU
        self.im('Compute the ENU coordinates')
        self.xyz2latlon_trackingPTS()
        # number of tracked points
        self.notp = len(np.unique(self.connectedComponentID))
        self.im('   - Number of tracked points: '+str(self.notp))
        self.nt_tracking = len(np.unique(self.ctime))
        self.im('   - Number of time steps in the tracking file: '+str(self.nt_tracking))
            

    
    def load_tracking(self,path2file,separator=',',headerLen=1,ftype='ttk',time_repeated=True):
        """
        Function calling the correct reading function depending on the argument ftype in input.
        For each format, have a look in the description of each specified reading function
        for more details (self.__load_tracking_TTK() and self.__load_tracking_bind()).

        Args:
            path2file (str): Complete path to the tracking file in .csv (with the extension)
            separator (str, optional): Row separator in the tracking file. Defaults to ','.
            headerLen (int, optional): Length of the header in number of lines. Defaults to 1.
            ftype (str): Description of the data format. Have to be in ['ttk','bind'].
                        Defaults: ftype = 'ttk'
            time_repeated (bool, optional): Option specifying if the tracking file has been obtain
                        with a time (persistence diag) repeated twice. Option ignored if ftype = 'bind'.
                        See the documentation of the function MAPT3.timetracking.load_tracking for
                        more details. Defaults to True.
        """
        self.path2file = path2file
        if ftype == 'ttk':
            self.__load_tracking_TTK(path2file,separator=separator,headerLen=headerLen,time_repeated=time_repeated)
            self.ftype = 'ttk'
        elif ftype == 'bind':
            self.__load_tracking_bind(path2file,separator=separator,headerLen=headerLen)
            self.ftype = 'bind'
        else:
            LoadingFormatError("['ttk','bind']")
    
    
    
    def build(self,path2h5,indices,verbose=True):
        """ Build the Cloud data with optimized tessellations.
        Makes the link between the current class instance and the collection of pre-computed
        optimized tessellation according to the arguments path2h5 and indices.

        Args:
            path2h5 (function): Function that takes in argument the elements of 'indices' and that
                                returns the path to the corresponding optimized tessellation .h5 file.
            indices (np.ndarray/list): indices of tessellations available as argument of the function
                                'path2h5'.
            verbose (bool, optional): Option to switch ON the verbose output. Defaults to True.

        Example:
            >>> path2h5 = lambda i: ['/path/to/file/1.h5',\
                                     '/path/to/file/2.h5',\
                                     '/path/to/file/3.h5'][i]
            >>> indices = np.arange(3)
        """
        self.im('Prepare the cloud data')
        # -- Path and file
        self.path2h5 = path2h5
        # -- indices
        self.indices = indices
        self.indices = np.array(self.indices)
        self.nt_tessellation = len(self.indices)
        self.nt_tracking     = len(self.indices)
        self.verbose = verbose
        self.im(' -> Number of time steps (tessellation files): '+str(self.nt_tessellation))
        
    
    def iterate(self,backward=False,verbose=None):
        """
        Iterates on the next (forward or backward) cloud index to build a drop.
        
        Args:
            backward (bool,optional): To iterate backward. Defaults to False.
            verbose (bool/None, optional): Option to switch ON the verbose output.
                        If verbose is set to None, take the value of self.verbose.
                        Defaults to None.
        """
        mySaveVerbose = None
        if verbose is None:
            verbose = self.verbose
        elif verbose != self.verbose:
            mySaveVerbose = self.verbose
            self.verbose = verbose
        # --- Prepare index
        if not backward:
            self.ci += 1
            if self.ci < len(self.indices):
                ind = self.indices[self.ci]
                self.im('Iteration on drop: '+str(self.ci))
                self.im('   -> File index: '+str(self.indices[self.ci]))
                self.im(' ---')
            else:
                self.ci -= 1
                self.im('You reach the end of the indices list!',error=False)
                self.im('   -> Current drop index: '+str(self.ci))
                return 0
        else:
            self.ci -= 1
            self.im('Backward itration')
            if self.ci >= 0:
                ind = self.indices[self.ci]
                self.im('Iteration on drop: '+str(self.ci))
                self.im('   -> File index: '+str(self.indices[self.ci]))
                self.im(' ---')
            else:
                self.ci += 1
                self.im('You reach the end of the indices list!',error=False)
                self.im('   -> Current drop index: '+str(self.ci))
                return 0
        # --- Build the drop
        self.drop = PlateGather()
        self.drop.verbose = verbose
        self.drop.load_from_h5(self.path2h5(self.indices[self.ci]))
        if mySaveVerbose is not None:
            self.verbose = mySaveVerbose

    

    def reset(self):
        """
        Resets the value of current drop index (self.ci)
        i.e. tessellation file (time step)
        """
        self.ci = -1
    
    
    def get_connectedComponentID(self,lon,lat,timestep=None,ctype=None):
        """
        Get the connected component ID of a critical point view on a map
        at a given time step (drop index). You can precise if you want to get
        the ID of the nearest minimum of maxmimum by indicating its ctype
        attribute.
        
        Args:
            lon (int/float): longitude in degree N of the target point.
            lat (int/float): latidude in degree E of the target point.
            timestep (int/None): value of self.ci (drop index, i.e. tessellation
                        file) on which you want to search the critical point.
                        If set to None, the time step will be ignored.
                        Defaults: timestep = None
            ctype (int/None): Type of the critical point.
                        If set to None then, the type of the critical point
                        will be ignored during the search.
                        Defaults: ctype = None.
        
        Returns:
            minind (int): Index of the closest critical point in the current
                        class instance fields like self.ccID, self.ctype,
                        self.ctime...
        """
        if timestep is not None:
            mask0 = self.ctime == timestep
        mask0 = np.ones(self.ctime.shape[0],dtype=bool)
        if ctype is not None:
            self.connectedComponentID
            mask1 = self.ctype == ctype
        else:
            mask1 = np.ones(len(self.x),dtype=bool)
        mask1 = mask1 * mask0
        #
        dist = haversine(lon*np.pi/180,lat*np.pi/180,self.lon[mask1]*np.pi/180,self.lat[mask1]*np.pi/180)
        id = np.array(range(len(self.x)))
        minind = np.where(dist == np.amin(dist))[0][0]
        minind = id[mask1][minind]
        self.im('Index of the nearest point: '+str(minind))
        self.im('   -> connectedComponentID: '+str(self.connectedComponentID[minind]))
        self.im('   -> ctype:     '+str(self.ctype[minind])+'  (0=min,3=max)')
        self.im('   -> longitude: '+str(self.lon[minind]))
        self.im('   -> latitude:  '+str(self.lat[minind]))
        return minind
    
    
    def compute_cc2plateID_matching(self):
        """
        Computes the list of the matching ID between the Connected Component IDs and the plate IDs
        of the current loaded drop.
        Save the result in the field 'self.matching'.
        """
        self.im('Compute Plate-CC matching:')
        old_verbose = self.drop.verbose
        self.drop.verbose = False
        plateID   = np.unique(self.drop.plateID)
        tmask     = self.ctime == self.indices[self.ci]
        ccID      = self.connectedComponentID[tmask]
        sizeIssue = False
        if len(plateID) != len(ccID):
            print('-'*40+'\n'+'  * IMPORTANT WARNING (not fatal) * \n'+'-'*40)
            sizeIssue = True
        if not sizeIssue:
            nod = len(plateID)
            self.matching = np.zeros((nod,2),dtype=np.int32)
            for i in tqdm(range(nod)):
                self.matching[i,0] = ccID[i]
                loni,lati = self.lon[tmask][i], self.lat[tmask][i]
                self.matching[i,1] = self.drop.plateID[self.drop.get_plateID(loni,lati)]
        else:
            nod = len(plateID)
            matching = []
            for i in tqdm(range(nod)):
                matching_i = [ccID[i]]
                loni,lati = self.lon[tmask][i], self.lat[tmask][i]
                matching_i.append(self.drop.plateID[self.drop.get_plateID(loni,lati)])
                matching.append(matching_i)
            self.matching = np.array(matching)
        self.drop.verbose = old_verbose
    

            
    def adjstTracking(self,path2persiDiag,ofname,path='./',separator=',',detect_only=False,plot=False,\
                         auto_reload=True):
        """ Function adjusting the currently loaded tracking data to had the missing polygon
        barycenters (missing because the TTK time tracking does not account critical points
        living only one (and two if you dont repeated the time) time steps.)
        To due so, this function read the pre-computed persistence diagrams that were used
        to build the tracking .csv file.
        This function exports the new tracking data in a .csv file defined by 'path'+'ofname'
        is the argument 'detect_only' is False. the new tracking data will be formated as a
        'bind' file. So to read it with the function self.load_tracking() you must specified
        ftype='bind'. 
        Furthermore, this function can refresh the current class instance with the new
        tracking data if the argument 'auto_reload' is True.

        Args:
            path2persiDiag (list): List of paths to access to the persistence diagrams
                                        the .h5 files, not the .xdmf) used to compute the 
                                        the initial tracking data.
            ofname (str): Output file name (.csv file) to save the adjusted tracking data.
            path (str, optional): path of the directory to export the resulting adjusted tracking.
                                        Defaults to './'.
            separator (str, optional): Row separator of the output file. Defaults to ','.
            detect_only (bool, optional): Option to control the writting of an output file.
                                        If True, will only return a verbose output.
                                        Defaults to False.
            auto_reload (bool, optional): Option controlling if Yes (True) or No (False) you
                                        want to refresh the current class instance with the
                                        adjusted tracking data.
                                        Note: The option detect_only must be False.
                                        Defaults to True.
            plot (bool, optional): Option to display for each time step a diagnostic map.
                                        Defaults to False.
        """
        
        self.im('Adjustment of the current plate tracking file from pre-computed persistence diagram')
        self.im('  -> Add the polygons living only 1 time step')
        
        # --- Prepare file
        if ofname.split('.')[-1] != 'csv':
            ofname += '.csv'
        
        # adjust the path if needed
        if path[-1] != '/':
            path += '/'
        
        self.im('')
        if not detect_only:
            self.im('Open the output .csv file')
            self.im('   - Output file: '+ofname)
            self.im('   - Output path: '+path)
            f = open(path+ofname,'w')
            self.im('')
            self.im('Save the current tracking state')
            self.im("  -> Output format: 'bind'")
            f.write(self.header)
            for i in range(len(self.x)):
                f.write(str(self.ctime[i])); f.write(separator)
                f.write(str(self.connectedComponentID[i])); f.write(separator)
                f.write(str(self.ctype[i])); f.write(separator)
                f.write(str(self.x[i])); f.write(separator)
                f.write(str(self.y[i])); f.write(separator)
                f.write(str(self.z[i])); f.write('\n')
        else:
            self.im('Detection only mode: ON')
        
        self.im('')
        self.im('Iterative search of missing barycenters')
        maxID = np.amax(self.connectedComponentID)
        self.im('  -> New connected component ID start from: '+str(maxID+1))
        # --- loop on the time
        self.im('')
        if plot:
            self.reset()
        for n in range(len(self.indices)):
            
            missing   = 0
            retrieved = 0
            
            ctime = n
            mt = self.ctime == ctime
            
            fid = h5py.File(path2persiDiag[n],'r')
            points = np.array(fid['Geometry/Points'])
            mask_maxima = np.array(fid['Node/CriticalType'])
            mask_maxima = mask_maxima == 3
            points = points[mask_maxima,:]
            loadedPoints = np.zeros((np.count_nonzero(mt),3))
            loadedPoints[:,0] = self.x[mt]
            loadedPoints[:,1] = self.y[mt]
            loadedPoints[:,2] = self.z[mt]
            
            lon1,lat1 = [],[]
            lon2,lat2 = [],[]
            nonretrieved = np.ones(points.shape[0],dtype=bool)
            for i in range(points.shape[0]):
                if not point_inside(points[i,:],loadedPoints,precision=1e-5):
                    
                    missing += 1
                    
                    if not detect_only:
                        # write missing data
                        ct = 3
                        t  = ctime
                        cc = maxID + 1
                        maxID += 1
                        # --- WRITE ---
                        baryx,baryy,baryz = points[i,:]
                        f.write(str(t)); f.write(separator)
                        f.write(str(cc)); f.write(separator)
                        f.write(str(ct)); f.write(separator)
                        f.write(str(baryx)); f.write(separator)
                        f.write(str(baryy)); f.write(separator)
                        f.write(str(baryz))#; f.write(separator)
                        f.write('\n')
                        
                    myx,myy,myz = points[i,:]
                    lat1i,lon1i,r = xyz2latlon(myx,myy,myz)
                    lon1.append(lon1i*180/np.pi)
                    lat1.append(-(lat1i*180/np.pi-90))
                else:
                    retrieved += 1
                    nonretrieved[i] = False
                    myx,myy,myz = points[i,:]
                    lat2i,lon2i,r = xyz2latlon(myx,myy,myz)
                    lon2.append(lon2i*180/np.pi)
                    lat2.append(-(lat2i*180/np.pi-90))            
            
            if plot:
                self.iterate()
                fig = plt.figure()
                ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
                ax.scatter(self.drop.lonb,self.drop.latb,color='k',s=1,transform=ccrs.PlateCarree())
                ax.scatter(lon2,lat2,color='red',transform=ccrs.PlateCarree(),label='Retrieved points')
                ax.scatter(lon1,lat1,color='blue',marker='+',transform=ccrs.PlateCarree(),label='Missing points')
                ax.legend()
                plt.show()
                
            self.im('')
            self.im('   --- Diagnostic step '+str(n)+'/'+str(len(self.indices)-1))
            self.im('   >> Time '+str(self.indices[n]))
            self.im('   >> missing points   = '+str(missing)+'/'+str(len(self.lon[mt])))
            self.im('   >> retrieved points = '+str(retrieved)+'/'+str(len(self.lon[mt])))
        
        if not detect_only:
            f.close()
            self.im('')
            self.im('Process complete!')
            self.im('  -> Output file created')
        
        if not detect_only and auto_reload:
            self.im('Auto-reload the adjusted time tracking data from the exported file.')
            self.im('')
            self.__load_tracking_bind(path+ofname,separator=separator,headerLen=1)
