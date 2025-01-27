from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1492, 802]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [1e-20, 0.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-14.715165944823049, 0.0, 0.0]
renderView1.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 3.808565198364234

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1492, 802)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XDMF Reader'

persistenceDiag_0 = XDMFReader(registrationName='persistenceDiagram_0.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00071_optimized.xdmf'])
persistenceDiag_0.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_0.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']

# create a new 'XDMF Reader'

persistenceDiag_1 = XDMFReader(registrationName='persistenceDiagram_0.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00071_optimized.xdmf'])
persistenceDiag_1.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_1.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']

# create a new 'XDMF Reader'

persistenceDiag_2 = XDMFReader(registrationName='persistenceDiagram_1.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00073_optimized.xdmf'])
persistenceDiag_2.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_2.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']

# create a new 'XDMF Reader'

persistenceDiag_3 = XDMFReader(registrationName='persistenceDiagram_1.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00073_optimized.xdmf'])
persistenceDiag_3.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_3.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']

# create a new 'XDMF Reader'

persistenceDiag_4 = XDMFReader(registrationName='persistenceDiagram_2.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00075_optimized.xdmf'])
persistenceDiag_4.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_4.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']

# create a new 'XDMF Reader'

persistenceDiag_5 = XDMFReader(registrationName='persistenceDiagram_2.xdmf', FileNames=['/home/alexandre/Documents/sample/3-Time-Tracking/PERSISTENCE-DIAG/AGE467-555_llsvp_vp00075_optimized.xdmf'])
persistenceDiag_5.PointArrayStatus = ['Birth', 'CriticalType', 'Death', 'ttkVertexScalarField']
persistenceDiag_5.CellArrayStatus = ['PairIdentifier', 'PairType', 'Persistence']
