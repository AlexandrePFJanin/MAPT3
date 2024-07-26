# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for the TTK plate tessellation

"""

# External dependencies:

from paraview.simple import *
import os


# ----------------- FUNCTIONS -----------------


def get_TTKtessellation(fname,suffix='',path='./',persistenceMin=None,persistenceMax=None,\
    velocityField='Velocity Cartesian',directory_surfaces=None,directory_edges=None,\
    pointArray=['Velocity Cartesian', 'Pressure', 'Spherical Velocity','PointID'],\
    slice_surface_radius= None,normalize_MagGrad=False,logfile=True,verbose=True,\
    MorseComplex='ascending',creat_pointID=False,paraview_version='5.10'):
    """
    Computes the tessellation of the model surface from a .xdmf file according to 
    a given min and max persistence threshold.
    Export the output in a set of .csv files with the following organization:
        path/directory_surfaces/Surface_*.csv      (surface files)
        path/directory_edges/Edge_*.csv            (edge files)
        path/1SeparatriceGeom_ + suffix            (plate boundary file)

    Args:
        fname (str): name of the .xdmf file that will be tessellated
        surfix (str, optional): suffix used to determined a directory_edges or directory_surfaces
                        if None as well as for the plate boundary file.
                        Defaults: suffix = ''
        path (str, optional): path to export the data set.
                        Defaults: path = './'
        persistenceMin (float/None, optional): minimum persistence threshold.
                        If set to None, will be determined automatically as
                        the minimum persistence threshold of the data set.
                        Defaults: persistenceMin = None
        persistenceMax (float/None, optional): maximum persistence threshold.
                        If set to None, will be determined automatically as
                        the maximum persistence threshold of the data set.
                        Defaults: persistenceMax = None
        pointArray (list of str, optional): list of fields that need to be loaded
                        Defaults: ['Velocity Cartesian', 'Pressure', 'Spherical Velocity','PointID']
        velocityField (str, optional): Indicate here which field in the list of
                        fields you loaded with the argument 'pointArray'
                        corresponds to the velocity field (cartesian) on which
                        you want to compute the tessellation.
                        Defaults: velocityField = 'Velocity Cartesian'
        slice_surface_radius (float/None, optional): If not None, you can specified here
                        at which radius of the sphere you want to compute the
                        tessellation. Otherwise, will be automatically the surface
                        of the model.
                        Defaults: slice_surface_radius = None
        MorseComplex (str, optional): Specified if the tessellation if based on
                        the ascending or descending Morse complexes.
                        Have to be in ['ascending','descending'].
                        Defaults: MorseComplex = 'ascending'
        normalize_MagGrad (bool, optional): Option controling if you want to
                        normalize the magnitude of the gradient of velocities.
                        (Not recommended).
                        Defaults: normalize_MagGrad = False
        create_pointID (bool, optional): If True then, create another field
                        containing the ID of points.
                        [obsolete method]
                        Defaults: create_pointID = False
        directory_surfaces (str/None, optional): name of the directory in which the set of .csv
                        files containing surface data will be exported.
                        If None then, will be generated automatically according to the
                        argument 'surffix'.
                        Defaults: directory_surfaces = None
        directory_edges (str/None, optional): name of the directory in which the set of .csv
                        files containing edge data will be exported.
                        If None then, will be generated automatically according to the
                        argument 'surffix'.
                        Defaults: directory_edges = None
        logfile (bool, optional): Option controling the automatic generation of a log file.
                        Defaults: logfile = True
        verbose (bool, optional): Option controling the display of a verbose output on
                        the terminal.
                        Defaults: verbose = True
        paraview_version (str, optional): String describing the version of paraview
                        you are using.
                        Defaults: paraview_version = '5.10'
    """
    # Prepare path
    if path[-1] != '/':
        path += '/'
    if path[0:2] == './':
        path = os.path.dirname(os.path.realpath('__file__')) + '/' + path[2:len(path)]
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Prepare file suffix
    if suffix == '':
        #suffix = fname.split('/')[-1].split('.')[0]
        suffix = '.'.join(fname.split('/')[-1].split('.')[0:-1])

    # Reset the paraview session to clean up flags
    ResetSession()
    
    # load file
    dtype = fname.split('.')[-1]
    if dtype == 'xdmf':
        # create a new 'XDMF Reader'
        input_data = XDMFReader(FileNames=[fname])
        input_data.PointArrayStatus = pointArray
        input_data.GridStatus = ['Grid']
    elif dtype == 'vtu':
        # create a new 'XML Unstructured Grid Reader'
        input_data = XMLUnstructuredGridReader(FileName=[fname])
        input_data.PointArrayStatus = pointArray
    elif dtype == 'vtk':
        # create a new 'Legacy VTK Reader'
        input_data = LegacyVTKReader(registrationName='input_vtk',FileNames=[fname])
    else:
        print('*'*50+'\n'+'WARNING\n\nProcess aborded: Unknown file extension\n'+'*'*50)
        return
    
    # Prepare logfile
    if logfile:
        logfile_name = path + 'TTKtessellation_' + suffix + '.log'
        flog = open(logfile_name,'w')
        flog.write('-'*24+'\nTTKtessellation log file'+'\n'+'-'*24+'\n\n'+' --- INPUT ---\n\n')
        flog.write('Working file:       '+fname+'\n')
        flog.write('File dtype:         '+dtype+'\n')
        flog.write('pointArray loaded:  '+', '.join(pointArray)+'\n')
        flog.write('velocityField name: '+velocityField+'\n')
        flog.write('paraview version declared : '+paraview_version+'\n')
    print('paraview version declared : '+paraview_version)
    
    if slice_surface_radius is not None:
        # create a new 'Slice'
        slice1 = Slice(registrationName='Slice1', Input=input_data)
        slice1.SliceType = 'Sphere'
        slice1.HyperTreeGridSlicer = 'Plane'
        slice1.SliceOffsetValues = [0.0]
        # init the 'Sphere' selected for 'SliceType'
        slice1.SliceType.Radius = slice_surface_radius
        input_data = slice1

    
    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(Input=input_data)

    # create a new 'Connectivity'
    connectivity1 = Connectivity(Input=extractSurface1)

    # create a new 'Clean to Grid'
    cleantoGrid1 = CleantoGrid(Input=connectivity1)

    # create a new 'Tetrahedralize'
    tetrahedralize1 = Tetrahedralize(Input=cleantoGrid1)

    # create a new 'Gradient Of Unstructured DataSet'
    if paraview_version >= '5.11':
        gradientOfUnstructuredDataSet1 = Gradient(registrationName='Gradient1', Input=tetrahedralize1)
        gradientOfUnstructuredDataSet1.ScalarArray = ['POINTS', velocityField]
        gradName = 'GSV'
        gradientOfUnstructuredDataSet1.ResultArrayName = gradName
    else:
        gradientOfUnstructuredDataSet1 = GradientOfUnstructuredDataSet(Input=tetrahedralize1)
        gradientOfUnstructuredDataSet1.ScalarArray = ['POINTS', velocityField]
        gradName = 'GSV'
        gradientOfUnstructuredDataSet1.ResultArrayName = gradName
    
    
    # create a new 'Calculator'
    calculator1 = Calculator(Input=gradientOfUnstructuredDataSet1)
    maggradName = 'magGSV'
    calculator1.ResultArrayName = maggradName
    calculator1.Function = 'sqrt(GSV_0^2+GSV_1^2+GSV_2^2+GSV_3^2+GSV_4^2+GSV_5^2+GSV_6^2+GSV_7^2+GSV_8^2)'
    
    # Normalize the magnitude of the gradient
    if normalize_MagGrad:
        # create a new 'Python Calculator'
        pythonCalculator1 = PythonCalculator(Input=calculator1)
        pythonCalculator1.Expression = "inputs[0].PointData['magGSV']/max(inputs[0].PointData['magGSV'])"
        pythonCalculator1.ArrayName = 'magGSV'
        calculator = pythonCalculator1
    else:
        calculator = calculator1
        
    
    # Complete logfile
    if logfile:
        flog.write('\n --- PROCESSING ---\n\n')
        if slice_surface_radius is not None:
            flog.write('Spherical slice:        R = '+str(slice_surface_radius)+'\n')
        flog.write('Gradient name:           '+gradName+'\n')
        flog.write('Mag Gradient name:       '+maggradName+'\n')
        if normalize_MagGrad:
            flog.write('normalize Mag Gradient:  '+'True'+'\n')
        else:
            flog.write('normalize Mag Gradient:  '+'False'+'\n')
    
    
    # create a new 'Threshold'
    threshold1 = Threshold(Input=calculator)
    threshold1.Scalars = ['POINTS', 'RegionId']
    if paraview_version >= '5.10':
        if slice_surface_radius is None:
            threshold1.LowerThreshold = 1.0
            threshold1.UpperThreshold = 1.0
        else:
            threshold1.LowerThreshold = 0
            threshold1.UpperThreshold = 0
    else:
        if slice_surface_radius is None:
            threshold1.ThresholdRange = [1.0, 1.0]
        else:
            threshold1.ThresholdRange = [0, 0]

    # create a new 'TTK PersistenceDiagram'
    tTKPersistenceDiagram1 = TTKPersistenceDiagram(Input=threshold1)
    tTKPersistenceDiagram1.ScalarField = ['POINTS', 'magGSV']
    tTKPersistenceDiagram1.InputOffsetField = ['POINTS', 'magGSV']
    tTKPersistenceDiagram1.EmbedinDomain = 1

    # create a new 'Threshold'
    threshold2 = Threshold(Input=tTKPersistenceDiagram1)
    threshold2.Scalars = ['CELLS', 'Persistence']
    pmin,pmax =  tTKPersistenceDiagram1.CellData.GetArray('Persistence').GetRange()
    if logfile:
        flog.write('Detected persistenceMin: '+str(pmin)+'\n')
        flog.write('Detected persistenceMax: '+str(pmax)+'\n')
    if persistenceMin is not None:
        pmin = persistenceMin
        if logfile:
            flog.write('Imposed persistenceMin:  '+str(persistenceMin)+'\n')
    else:
        if logfile:
            flog.write('Imposed persistenceMin:  None'+'\n')
    if persistenceMax is not None:
        pmax = persistenceMax
        if logfile:
            flog.write('Imposed persistenceMax:  '+str(persistenceMax))
    else:
        if logfile:
            flog.write('Imposed persistenceMax:  None'+'\n')
    if paraview_version >= '5.10':
        threshold2.LowerThreshold = pmin
        threshold2.UpperThreshold = pmax
    else:
        threshold2.ThresholdRange = [pmin,pmax]

    # create a new 'TTK TopologicalSimplification'
    tTKTopologicalSimplification1 = TTKTopologicalSimplification(Domain=threshold1,
        Constraints=threshold2)
    tTKTopologicalSimplification1.ScalarField = ['POINTS', 'magGSV']
    tTKTopologicalSimplification1.InputOffsetField = ['POINTS', 'magGSV']
    tTKTopologicalSimplification1.VertexIdentifierField = ['POINTS', 'Birth']

    # create a new 'TTK MorseSmaleComplex'
    tTKMorseSmaleComplex1 = TTKMorseSmaleComplex(Input=tTKTopologicalSimplification1)
    tTKMorseSmaleComplex1.ScalarField = ['POINTS', 'magGSV']
    tTKMorseSmaleComplex1.OffsetField = ['POINTS', 'magGSV']

    # find source
    tTKMorseSmaleComplex1_1 = FindSource('TTKMorseSmaleComplex1')

    # ascending or desending Morse complexes
    if MorseComplex == 'ascending':
        complexes_threshold = 1.0
        if logfile:
            flog.write('Morse complex:           '+MorseComplex+'\n')
    elif MorseComplex == 'descending':
        complexes_threshold = 0
        if logfile:
            flog.write('Morse complex:           '+MorseComplex+'\n')
    else:
        print('ERROR: unknown Morse complex: '+MorseComplex)
        print("  -> Please choose MorseComplex in ['ascending','descending']")
        return 0
    
    # create a new 'Threshold'
    threshold3 = Threshold(Input=OutputPort(tTKMorseSmaleComplex1_1,1))
    threshold3.Scalars = ['CELLS', 'SeparatrixType']
    if paraview_version >= '5.10':
        threshold3.LowerThreshold = complexes_threshold
        threshold3.UpperThreshold = complexes_threshold
    else:
        threshold3.ThresholdRange = [complexes_threshold, complexes_threshold]

    # find source
    tTKMorseSmaleComplex1_2 = FindSource('TTKMorseSmaleComplex1')

    # create a new 'TTK IdentifierRandomizer'
    tTKIdentifierRandomizer1 = TTKIdentifierRandomizer(Input=OutputPort(tTKMorseSmaleComplex1_2,3))
    tTKIdentifierRandomizer1.ScalarField = ['POINTS', 'DescendingManifold']
    tTKIdentifierRandomizer1.RandomSeed = 1
    
    if creat_pointID:
        # create a new 'Generate Global Ids'
        generateIds1 = GenerateGlobalIds(registrationName='GenerateGlobalIds1', Input=tTKIdentifierRandomizer1)
        # ccreat a calculator filter to put the field 'GlobalPointIds' on the points and give it the name 'PointID'
        calculator_extraction = Calculator(Input=generateIds1)
        calculator_extraction.ResultArrayName = 'PointID'
        calculator_extraction.Function = 'GlobalPointIds'
        tTKIdentifierRandomizer1 = calculator_extraction

    # ---------------------------------------------------
    # prepare the extraction of Surfaces and Edges of each plates
    identifierVTUobject = servermanager.Fetch(tTKIdentifierRandomizer1)
    idRange = identifierVTUobject.GetPointData().GetArray("DescendingManifold").GetRange()
    if verbose:
        print('-'*20)
        print('Tessellation diagnostic:')
        print('   -> Number of plates: '+str(int(idRange[1])+1))
        print('-'*20)
    if logfile:
        flog.write('Number of plates:        '+str(int(idRange[1])+1)+'\n')
    
    # Complete the logfile
    if logfile:
        flog.write('\n --- OUTPUT ---\n\n')
        flog.write('Working path:    '+path+'\n')
        flog.write('Output data available on path:'+'\n')
        
    # ---------------------------------------------------
    # prepare output directories
    if directory_surfaces is None:
        directory_surfaces = 'extracted_Surfaces_'+suffix
    if verbose:
        print('Your path for surfaces files: ' + path + directory_surfaces)
    if logfile:
        flog.write('    - Surfaces directory: '+directory_surfaces+'\n')   
    if not os.path.exists(path + directory_surfaces):
        if verbose:
            print('   -> Creation of a new directory')
            flog.write('       -> Creation of a new directory'+'\n')
        os.makedirs(path + directory_surfaces)
    else:
        if verbose:
            print('   -> The directory for surfaces files already exist: cleaning')
        if logfile:
            flog.write('       -> The directory for surfaces files already exist: cleaning'+'\n')
        old_files = [f for f in os.listdir(path + directory_surfaces)]
        for f in old_files:
            os.remove(path + directory_surfaces+'/'+f)
    
    if directory_edges is None:
        directory_edges = 'extracted_Edges_'+suffix
    if verbose:
        print('Your path for edges files:    ' + path + directory_edges)
    if logfile:
        flog.write('    - Edges directory: '+directory_edges+'\n')
    if not os.path.exists(path + directory_edges):
        if verbose:
            print('   -> Creation of a new directory')
        if logfile:
            flog.write('       -> Creation of a new directory'+'\n')
        os.makedirs(path + directory_edges)
    else:
        if verbose:
            print('   -> The directory for edges files already exist: cleaning')
        if logfile:
            flog.write('       -> The directory for edges files already exist: cleaning'+'\n')
        old_files = [f for f in os.listdir(path + directory_edges)]
        for f in old_files:
            os.remove(path + directory_edges+'/'+f)
    
    if verbose:
        print('-'*20)
    # ---------------------------------------------------
    # export the 1separatrice DescendingManifold
    SaveData(path + "1SeparatriceGeom" + '_' + suffix + ".csv", threshold3)
    if logfile:
        flog.write('    - Boundary file:   '+"1SeparatriceGeom" + '_' + suffix + ".csv"+'\n')
        flog.write('    - Log file:        '+logfile_name+'\n')
        flog.close()
        
    # iterate on all plates
    for i in range(int(idRange[1])+1):
    
        # create a new 'Threshold'
        threshold_i = Threshold(registrationName='Thresholdi', Input=tTKIdentifierRandomizer1)
        threshold_i.Scalars = ['POINTS', 'DescendingManifold']
        if paraview_version >= '5.10':
            threshold_i.LowerThreshold = i
            threshold_i.UpperThreshold = i
        else:
            threshold_i.ThresholdRange = [i, i]

        if verbose:
            print("Extracting region #%s out of "%i + str(int(idRange[1])))

        # create a new 'Extract Surface'
        extractSurface_i = ExtractSurface(registrationName='ExtractSurface_%s'%int(i), Input=threshold_i)

        SaveData(path + directory_surfaces + "/Surface_%s.csv"%int(i), extractSurface_i)

        # create a new 'Feature Edges'
        featureEdges_i = FeatureEdges(registrationName='FeatureEdges_%s'%int(i), Input=extractSurface_i)
        featureEdges_i.FeatureEdges = 0
        featureEdges_i.NonManifoldEdges = 0
        
        SaveData(path + directory_edges + "/Boundary_%s.csv"%int(i), featureEdges_i)
        
        Delete(threshold_i)
        Delete(extractSurface_i)
        Delete(featureEdges_i)
    
    if verbose and logfile:
        print('Logfile create!')