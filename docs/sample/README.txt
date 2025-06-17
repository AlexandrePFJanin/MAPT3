
        - MAPT3 workflow sample -

This directory contains an example of workflow
using the python package MAPT3 to automatically
tessellate the surface of a 3D spherical numerical
model in a set of tectonic plates that can be
then tracked in time.

In order to use the scripts and functions discussed
in this workflow example, you need to install MAPT3.
Installation instructions can be found here:
https://github.com/AlexandrePFJanin/MAPT3.git

This file and more generally the content of this
directory have the goal to show through a pratical
example what a MAPT3 pipeline can look like.
The data manipulated here as an example are the
data analysed in the Coltice et al., 2019 and
Janin et al., 2025 papers.
This directory is organized in subdirectories to
emphasized the different steps of the workflow.
More details can be find in the subdirectories
and the scripts they contained as well as in the
method section of Janin et al., 2025.


Important:

This example of workflow is an example. Scripts
provided here can and should be adapted to
better answer the problematics and the
specificities of each project.


Pipeline example:

The content of this simple workflow step by step:

    - 'XDMF-H5/' (directory)
    
    This directory contains two examples of velocity
    outputs that are tessellated here. Outputs have been
    transformed in files readable by the software
    Paraview (here, .xdmf and .h5 files).
    Notice that one file corresponds to one time step.
    
    - 'myparameters.py' (python file)
    
    This file contains all the generic parameters 
    of the project such as the number of points 
    expected at the model (for memory allocation),
    a planetary model or the list of values for
    the minimum topological persistence threshold
    (hereafter called 'pmin').
    
    - '1-Tessellation/' (directory)
    
    This directory contains a script: 'main_tessellate.py'
    to tessellate the surface of a 3D spherical
    numerical model from a file readable in paraview
    (here a .xdmf + .h5 in the directory 'XDMF-H5/').
    The result of the tessellation is stored
    by value of pmin. According to the structure
    of the script 'main_tessellate.py' presented here:
        - '1SeparatriceGeom_*.csv' contains plate
          boundary data at a given value of pmin and
          per file.
        - 'Edges_*' are directories containing
          plate edges for each plates for each files
          for a given value of pmin.
        - 'Surfaces_*' are directories containing
          plate surfaces for each plates for each files
          for a given value of pmin.

    - '2-Persistence-analysis/' (directory)
    
    This directory contains several script to 
    visualize plate tessellation and compute
    the optimization of the plate tessellation
    (i.e. the merging of plate tessellation at
    different values of pmin to get only one
    plate tessellation per time step).
        - The script 'analyseTessellation_pmin.py'
        loads a plate tessellation computed at a
        given pmin.
        - The script 'optimize.py' computes the
        automatic optimization of the plate
        tessellation.
        Here, the optimized tessellation (.h5)
        is stored in the directory 'OPTIMIZED/'.
        - The script 'analyseTessellation_opti.py
        loads an optmized plate tessellation and
        generate a simple map.

    - '3-Time-Tracking/' (directory)
    
    This directory contains examples of scripts
    to time track plate from already computed
    tessellations with the previous steps.
    According to the last stable release of 
    TTK, the topological toolkit, the main idea
    of the time tracking if the following:
        - 1. generate automatically a paraview
             state file (save time if your project
             covers a large time period).
        - 2. open in paraview the state file
             -> will generate the tracking file
        - 3. adjust the time tracking (auto
             check if plates are missing).
    Step by step with the provided scripts:
        - 1. 'generate_persiDiag.py' generates a
          file readable by paraview (.xdfm+.h5)
          containing a description of the
          barycenters of each plate on the
          input tessellation.
          The output files are here stored
          in the directory 'PERSISTENCE-DIAG/'
        - 2. 'generate_statefile.py' generates
          a paraview .py state file, loading
          in the good order the fictive
          persistence diagram generated before.
        - 3. Open the state file in Paraview,
          select all the loaded file in the
          pipeline browser and apply on them
          the TTK filter
          'TTKTrackingFromPersistenceDiagrams'.
          We recommend the following input
          options for this filter (available
          in the 'Properties' of the filter):
            - Persistence threshold = 0
              (no cleaning)
            - Extremum weight = 0
            - Saddle weight   = 0
              (0 weight on the persistence space)
            - X weight = 1
            - Y weight = 1
            - Z weight = 1
              (weight only on position)
          The tracking will appear as lines
          connecting the barycenters.
          NOTE: functions tested with
          Paraview 5.10.0
        - 4. Export in the tracking in .csv
          (here, the output file is called
          'mytracking.csv').
          In the Paraview pipeline browser,
          select your filter
          "TTKTrackingFromPersistenceDiagrams1"
          Then, on the Menu bar, click on
          File > Save Data.
          Give a path and a file name for the
          output file. Select ".csv" for the 
          type of file.
          In this example the file is:
          '3-Time-Tracking/TRACKING/myTracking.csv'
        - 5. 'adjust_tracking.py' adjusts the
          time tracking file (reformatting
          and check for missing plates i.e.
          plates on a single time step).
          Reads the tracking '.csv' file generated
          during the last step. Make sure to
          give the good path and file name.
          In this example, export the adjusted
          tracking file 'myTracking_adjusted.csv'
          in the directory 'TRACKING/'.
        - 6. 'analyseAdjTracking.py' shows a
          simple example of how to use MAPT3
          to manipulate the time tracking data.
          Generates a figure showing the plate
          barycenter tracks over time.
    
    
References:

  - Ahrens, J., Geveci, B., and Law, C.
    Paraview: An end-user tool for large data visualization.
    The visualization handbook 717(8) (2005).
    Material: https://www.paraview.org/

  - Coltice, N., Husson, L., Faccenna, C., and Arnould, M.
    What drives tectonic plates?
    Science advances 5(10), eaax4295 (2019).

  - Janin, A., Coltice, N., Chamot-Rooke, N., Tierny, J.
    Topological data analysis reveals mantle-lithosphere
    dynamical interactions through global plate reorganisations.
    Nature Geoscience. (accepted, 2025)

  - Masood, T. B., Budin, J., Falk, M., Favelier, G., Garth, C.,
    Gueunet, C., Guillou, P., Hofmann, L., Hristov, P.,
    Kamakshidasan, A., et al.
    An overview of the topology toolkit. In TopoInVis
    2019-Topological Methods in Data Analysis and Visualization, (2019).

  - Tierny, J., Favelier, G., Levine, J. A., Gueunet, C., and Michaux, M.
    The topology toolkit. IEEE transactions on visualization and
    computer graphics 24(1), 832–842 (2017).
    Material: https://topology-tool-kit.github.io/




