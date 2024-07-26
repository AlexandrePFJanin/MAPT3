# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Inputs/Outputs management
"""

# External dependencies:
import numpy as np
import h5py
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Internal dependencies:
from .generics import im


# ----------------- FONCTIONS -----------------


def write_PersistenceDiag_VTK(xm,ym,zm,xs,ys,zs,p,fname,path='./',ttk_version=1.1):
    """
    Writes a VTK (XDMF+HDF5) file corresponding to a virtual (fictive) persistence
    diagram pairing barycenter (maxima) and the most distance edge point of each
    plates according to a series of input arguments:

    Args:
        xm, ym, zm (3x np.ndarray): Cartesian coordinates of plate barycenters
                                    (will be maxima)
        xs, ys, zs (3x np.ndarray): Cartesian coordinates of the most distant
                                    edge point (relative to a plate and its barycenter,
                                    (will be minima).
        p (np.ndarray): Persistence of the pairs
        fname (str): name of the file (without extension) to export the VTK
        path (str, optional): path to the directory to save the file.
                                    Defaults: './'
        ttk_version (float, optional): version of TTK that will be used to read the 
                                    exported persistence diagram.
                                    Defaults: ttk_version = 1.1
    """

    # prepare the path

    if path[-1] != '/':
        path += '/'
        
    xdmf_file = fname+'.xdmf'
    h5_file   = fname+'.h5'

    # --- prepare data
    
    nop = len(xm)   # number of pairs
    
    x = np.concatenate((xs,xm)).reshape(2,nop).reshape(2*nop,order='F')  # alternate: saddle-max saddle-max saddle-max etc
    y = np.concatenate((ys,ym)).reshape(2,nop).reshape(2*nop,order='F')
    z = np.concatenate((zs,zm)).reshape(2,nop).reshape(2*nop,order='F')
    
    points = np.zeros((nop*2,3))
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z
    
    pairIdentifier = np.arange(nop)
    pairType = np.ones(nop)
    ttkVertexScalarField = np.arange(2*nop)
    criticalType_max    = np.ones(nop)*3
    criticalType_saddle = np.ones(nop)*2 # saddle type 1
    criticalType = np.concatenate((criticalType_saddle,criticalType_max)).reshape(2,nop).reshape(2*nop,order='F')
    
    if ttk_version < 1.2:
        topology = np.arange(2*nop).reshape(nop,2)
        birth = np.zeros(2*nop)
        death = np.zeros(2*nop)
    else:
        topoa    = np.arange(2*nop).reshape(nop,2)
        topob    = np.ones(2*nop,dtype=np.int32)*2
        topob    = topob.reshape(nop,2)
        topology = np.concatenate((topob,topoa),axis=1).flatten()
        birth    = np.zeros(nop)
        isfinite = np.ones(nop,dtype=np.int32)
    
    # --- Write the XDMF file
    
    if ttk_version < 1.2:
        fid = open(path+xdmf_file,'w')
        fid.write('<?xml version="1.0" ?>'+'\n')
        fid.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+'\n')
        fid.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">'+'\n')
        fid.write('  <Domain>'+'\n')
        fid.write('    <Grid GridType="Uniform">'+'\n')
        fid.write('      <Topology TopologyType="Polyline" Dimensions="%s">'%int(nop/2)+'\n')
        fid.write('        <DataItem Dimensions="%s 2" NumberType="Int" Precision="8" Format="HDF">'%int(nop)+h5_file+':/Topology</DataItem>'+'\n')
        fid.write('      </Topology>'+'\n')
        fid.write('      <Geometry GeometryType="XYZ">'+'\n')
        fid.write('        <DataItem Dimensions="%s 3" NumberType="Float" Precision="4" Format="HDF">'%int(nop*2)+h5_file+':/Geometry/Points</DataItem>'+'\n')
        fid.write('      </Geometry>'+'\n')
        fid.write('      <Attribute Name="PairIdentifier" Active="1" AttributeType="None" Center="Cell">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Int" Precision="4" Format="HDF">'%int(nop)+h5_file+':/Cell/PairIdentifier</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="PairType" Active="1" AttributeType="None" Center="Cell">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Int" Precision="4" Format="HDF">'%int(nop)+h5_file+':/Cell/PairType</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="Persistence" Active="1" AttributeType="None" Center="Cell">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="8" Format="HDF">'%int(nop)+h5_file+':/Cell/Persistence</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="ttkVertexScalarField" Active="1" AttributeType="None" Center="Node">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Int" Precision="4" Format="HDF">'%int(nop*2)+h5_file+':Node/ttkVertexScalarField</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="CriticalType" Active="1" AttributeType="None" Center="Node">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Int" Precision="4" Format="HDF">'%int(nop*2)+h5_file+':Node/CriticalType</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="Birth" Active="1" AttributeType="None" Center="Node">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="8" Format="HDF">'%int(nop*2)+h5_file+':Node/Birth</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Name="Death" Active="1" AttributeType="None" Center="Node">'+'\n')
        fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="8" Format="HDF">'%int(nop*2)+h5_file+':Node/Death</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('    </Grid>'+'\n')
        fid.write('  </Domain>'+'\n')
        fid.write('</Xdmf>')
        fid.close()
    else:
        fid = open(path+xdmf_file,'w')
        fid.write('<?xml version="1.0" encoding="utf-8"?>'+'\n')
        fid.write('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">'+'\n')
        fid.write('  <Domain>'+'\n')
        fid.write('    <Grid Name="Grid">'+'\n')
        fid.write('      <Geometry Origin="" Type="XYZ">'+'\n')
        fid.write('        <DataItem DataType="Float" Dimensions="%s 3" Format="HDF" Precision="4">'%int(nop*2)+h5_file+':/Geometry/Points</DataItem>'+'\n')
        fid.write('      </Geometry>'+'\n')
        fid.write('      <Topology Dimensions="%s" Type="Mixed">'%int(nop)+'\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">'%int(nop*4)+h5_file+':/Topology</DataItem>'+'\n')
        fid.write('      </Topology>'+'\n')
        fid.write('      <Attribute Center="Node" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="ttkVertexScalarField" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="4">'%int(nop*2)+h5_file+':Node/ttkVertexScalarField</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Node" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="CriticalType" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="4">'%int(nop*2)+h5_file+':Node/CriticalType</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="PairIdentifier" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="4">'%int(nop)+h5_file+':/Cell/PairIdentifier</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="PairType" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="4">'%int(nop)+h5_file+':/Cell/PairType</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="Persistence" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">'%int(nop)+h5_file+':/Cell/Persistence</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="Birth" Type="None">'+'\n')
        fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">'%int(nop)+h5_file+':Cell/Birth</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('      <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="IsFinite" Type="None">'+'\n')
        fid.write('        <DataItem DataType="UChar" Dimensions="%s" Format="HDF" Precision="1">'%int(nop)+h5_file+':Cell/IsFinite</DataItem>'+'\n')
        fid.write('      </Attribute>'+'\n')
        fid.write('    </Grid>'+'\n')
        fid.write('  </Domain>'+'\n')
        fid.write('</Xdmf>')
        fid.close()

    # --- Write the HDF5 file

    # For N pair of points
    fid = h5py.File(path+h5_file,'w')
    # --- Points, shape = 2*N, 3 : xyz
    dset = fid.create_dataset('/Geometry/Points',           data = points)
    # --- PairIdentifier, shape = N, id of the pair = np.arange(N)
    dset = fid.create_dataset('/Cell/PairIdentifier',       data = pairIdentifier)
    # ---- PairType, shape = N : -1 for the pair maximum(maxima)-minimum(minima), 0 for pairs min-saddle type 1, 1 for pairs linking max-saddle type 2
    dset = fid.create_dataset('/Cell/PairType',             data = pairType)
    # --- Persistence, shape = N
    dset = fid.create_dataset('/Cell/Persistence',          data = p)
    # --- ttkVertexScalarField, shape = 2*N : id of the point on which the Persistence filter have been computed
    dset = fid.create_dataset('/Node/ttkVertexScalarField', data = ttkVertexScalarField)
    # --- CriticalType, shape = 2*N : 0 global min, 1 and 2 are saddles and 3 is the global max
    dset = fid.create_dataset('/Node/CriticalType',         data = criticalType)
    if ttk_version < 1.2:
        # --- Topology, shape = N, 2 : indices of points that are paired
        dset = fid.create_dataset('Topology',                   data = topology)
        # --- Birth, shape = 2*N
        dset = fid.create_dataset('/Node/Birth',                data = birth)
        # --- Death, shape = 2*N
        dset = fid.create_dataset('/Node/Death',                data = death)
    else:
        # --- Topology, shape = 4*N, : indices of points that are paired
        dset = fid.create_dataset('Topology',                   data = topology)
        # --- Birth, shape = N
        dset = fid.create_dataset('/Cell/Birth',                data = birth)
        # --- Death, shape = N
        dset = fid.create_dataset('/Cell/IsFinite',             data = isfinite)
    fid.close()



def surface2VTK(x,y,z,v,fname,fieldName,path='./',verbose=True,simplex_threshold=0.01):
    """ Writes a Paraview VTK (XDMF+H5) file for a surface described by
    3D (flatten) cartesian coordinates.

    Args:
        x (np.ndarray): x coordinates. Shape: flatten
        y (np.ndarray): y coordinates. Shape: flatten
        z (np.ndarray): z coordinates. Shape: flatten
        v (np.ndarray): field data. Shape: flatten
        fname (str): file name without format extension
        fieldName (str): name of the field 'v' that will be displayed
                    in Paraview.
        simplex_threshold (int/float): threshold on the length of simplices.
                    Remove simplices longer than simplex_threshold.
                    Defaults: simplex_threshold = 0.01
        path (str, optional): path to the output directory.
                    Defaults to './'.
        verbose (str, optional): Verbose option.
                    Defaults to True.
    """
    pName = 'surface2VTK'
    im('Exportation of 3D surface data points to meshed data for Paraview',pName,verbose)
    
    if path[-1] != '/':
        path += '/'

    xdmf_file = fname + '.xdmf'
    h5_file   = fname + '.h5'

    nod = len(x)
    points = np.zeros((nod,3))
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z

    im('  -> Delaunay Triangulation',pName,verbose)
    tri = Delaunay(points[:,0:2])
    
    # Simplification of the Delaunay triangulation: Remove long connections
    simplices = tri.simplices
    
    ccontinue = True
    simplex_threshold_init = simplex_threshold
    while ccontinue:
        im('   - Remove long simplex: threshold = '+str(simplex_threshold),pName,verbose)
        dist = np.zeros(simplices.shape)
        mask = np.ones(simplices.shape[0],dtype=bool)
        for i in range(simplices.shape[0]):
            a,b,c = simplices[i,:]
            dist[i,0] = np.sqrt((x[a]-x[b])**2+(y[a]-y[b])**2+(z[a]-z[b])**2)
            dist[i,1] = np.sqrt((x[b]-x[c])**2+(y[b]-y[c])**2+(z[b]-z[c])**2)
            dist[i,2] = np.sqrt((x[c]-x[a])**2+(y[c]-y[a])**2+(z[c]-z[a])**2)
            if np.count_nonzero(dist[i,:]>simplex_threshold) > 0:
                mask[i] = False

        if np.unique(simplices[mask,:]).shape[0] == nod:
            im('No point missed',pName,verbose)
            simplices = simplices[mask,:]
            ccontinue = False
        else:
            im('   * ERROR *',pName,verbose)
            im('Process aborded!',pName,verbose)
            im('Point(s) missed in the simplification of the Delaunay triangulation',pName,verbose)
            im('  -> Increase the argument simplex_threshold',pName,verbose)
            ccontinue = True
            simplex_threshold = simplex_threshold + simplex_threshold_init/2

    im('  -> Writing the instruction file (XDMF)',pName,verbose)
    fid = open(path+xdmf_file,'w')
    fid.write('<?xml version="1.0" ?>'+'\n')
    fid.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+'\n')
    fid.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">'+'\n')
    fid.write('  <Domain>'+'\n')
    fid.write('    <Grid GridType="Uniform">'+'\n')
    fid.write('      <Topology TopologyType="Triangle" Dimensions="%s">'%int(simplices.shape[0])+'\n')
    fid.write('        <DataItem Dimensions="%s'%int(simplices.shape[0])+' 3" NumberType="Int" Precision="8" Format="HDF">'+h5_file+':/Topology</DataItem>'+'\n')
    fid.write('      </Topology>'+'\n')
    fid.write('      <Geometry GeometryType="XYZ">'+'\n')
    fid.write('        <DataItem Dimensions="%s 3" NumberType="Float" Precision="4" Format="HDF">'%x.shape[0]+h5_file+':/Geometry/Points</DataItem>'+'\n')
    fid.write('      </Geometry>'+'\n')
    fid.write('      <Attribute Name="'+fieldName+'" Active="1" AttributeType="None" Center="Node">'+'\n')
    fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%x.shape[0]+h5_file+':/Node/field</DataItem>'+'\n')
    fid.write('      </Attribute>'+'\n')
    fid.write('    </Grid>'+'\n')
    fid.write('  </Domain>'+'\n')
    fid.write('</Xdmf>')
    fid.close()

    im('  -> Writing the data file (HDF5)',pName,verbose)
    fid = h5py.File(path+h5_file,'w')
    dset = fid.create_dataset('Topology', data = simplices)
    dset = fid.create_dataset('/Geometry/Points', data = points)
    dset = fid.create_dataset('/Node/field', data = v)
    fid.close()

    im('Process complete',pName,verbose)
    im('Files generated:',pName,verbose)
    im('  -> '+path+xdmf_file,pName,verbose)
    im('  -> '+path+h5_file,pName,verbose)



def shell2VTK(X,Y,Z,V,fname,fieldName,path='./',verbose=True,creat_pointID=True):
    """
    Exports a 3D shell defined by set of N points (X,Y,Z) and a field V.

    Args:
        X (np.ndarray, shape: (N)): X coordinates of points compising the shell
        Y (np.ndarray, shape: (N)): Y coordinates of points compising the shell
        Z (np.ndarray, shape: (N)): Z coordinates of points compising the shell
        V (np.ndarray, shape: (N)): cartesian field to be display on the mesh
        fname (str): name of the file (without extension) to export the VTK
        fieldName (str): name of the field V
        path (str, optional): path of the directory to save the file.
                            Defaults: './'
        create_pointID (bool, optional): If True then, create another field
                            containing the ID of points. Defaults: True
        verbose (bool, optional): Option controling the verbose output on the
                            terminal. Defaults: True
    """
    pName = 'shell2VTK'
    im('Exportation of 3D shell data points to meshed data for Paraview',pName,verbose)
    
    nod = len(X)

    if path[-1] != '/':
        path += '/'

    xdmf_file = fname + '.xdmf'
    h5_file   = fname + '.h5'

    # ---
    if creat_pointID:
        im('  -> Creat pointID',pName,verbose)
        pointID = np.arange(nod)
    # ---
    points = np.zeros((nod,3))
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = Z
    # ----
    im('  -> Convex Hull Triangulation',pName,verbose)    
    tri = ConvexHull(points)
    simplices = tri.simplices
    # --- Write the XDMF file
    im('  -> Writing the instruction file (XDMF)',pName,verbose)
    fid = open(path+xdmf_file,'w')
    fid.write('<?xml version="1.0" ?>'+'\n')
    fid.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+'\n')
    fid.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">'+'\n')
    fid.write('  <Domain>'+'\n')
    fid.write('    <Grid GridType="Uniform">'+'\n')
    fid.write('      <Topology TopologyType="Triangle" Dimensions="%s">'%int(simplices.shape[0])+'\n')
    fid.write('        <DataItem Dimensions="%s'%int(simplices.shape[0])+' 3" NumberType="Int" Precision="8" Format="HDF">'+h5_file+':/Topology</DataItem>'+'\n')
    fid.write('      </Topology>'+'\n')
    fid.write('      <Geometry GeometryType="XYZ">'+'\n')
    fid.write('        <DataItem Dimensions="%s 3" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Geometry/Points</DataItem>'+'\n')
    fid.write('      </Geometry>'+'\n')
    # --- Field
    fid.write('      <Attribute Name="'+fieldName+'" Active="1" AttributeType="Scalar" Center="Node">'+'\n')
    fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Node/field_scalar</DataItem>'+'\n')
    fid.write('      </Attribute>'+'\n')
    # --- pointID
    if creat_pointID:
        fid.write('      <Attribute AttributeType="Scalar" Center="Node" Name="PointID">\n')
        fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">\n'%pointID.shape[0])
        fid.write('            '+h5_file+':/Node/pointID\n')
        fid.write('        </DataItem>\n')
        fid.write('      </Attribute>\n\n')
    # --- close
    fid.write('    </Grid>'+'\n')
    fid.write('  </Domain>'+'\n')
    fid.write('</Xdmf>')
    fid.close()
    # --- write the H5 file
    im('  -> Writing the data file (HDF5)',pName,verbose)
    fid = h5py.File(path+h5_file,'w')
    dset = fid.create_dataset('Topology', data = simplices)
    dset = fid.create_dataset('/Geometry/Points', data = points)
    dset = fid.create_dataset('/Node/field_scalar', data = V)
    if creat_pointID:
        dset = fid.create_dataset('/Node/pointID', data = pointID)
    fid.close()
    # --- Finish!
    im('Process complete',pName,verbose)
    im('Files generated:',pName,verbose)
    im('  -> '+path+xdmf_file,pName,verbose)
    im('  -> '+path+h5_file,pName,verbose)
