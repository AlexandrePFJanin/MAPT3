# MAPT$^3$
**M**ulti-disciplinary and **A**utomatic **P**late **T**essellation and **T**ime tracking **T**oolkit.

MAPT$^3$ is a python package providing a set of tools to (1) automatically tessellate the surface of a 3D spherical model using [TTK](https://topology-tool-kit.github.io/) and [paraview](https://www.paraview.org/) with a geodynamical analysis, (2) automatically track over time the detected plates and (3) manipulate the outputs.


## Install MAPT$^3:

### 1. Download

Download **MAPT$^3$** from this page and unzip the archive. Then, move into the main directory of the package:
```
cd MAPT3-main/
```

### 2. Creation of a new environnement

We strongly recommend installing MAPT$^3$ in a python environment other than the base using [conda](https://conda.io/projects/conda/en/latest/index.html):

```
conda create --name mapt3
```

Then activate the new environnement:

```
conda activate mapt3
```

### 3. Installation of TTK - topology toolkit

Before installing **MAPT$^3$**, installation of [TTK](https://topology-tool-kit.github.io/index.html):

```
conda install conda-forge::topologytoolkit
```

### 4. Linking MAPT$^3$

To link in a user module directory, use [pip](https://pip.pypa.io/en/stable/) and run:

```
python -m pip install .
```

## Uninstall MAPT$^3$ in the environnement:

As you used [pip](https://pip.pypa.io/en/stable/) for the installation, use it to uninstall the package. In a terminal, run:

```
pip uninstall MAPT3
```

%## Getting started:
%
%Several tutorials are provided with **MAPT$^3$** to learn step by step how to use the package. These tutorials take the form of Jupyter notebooks in the directory [docs](/docs).


