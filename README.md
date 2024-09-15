<p align="center">
    <img src="./icons/MAPT3-logo-full.png#gh-light-mode-only" height="120" width="120"/>
    <img src="./icons/MAPT3-logo-mini-circle-alpha.png#gh-dark-mode-only" height="120" width="120"/>
</p>

# MAPT<sup>3</sup>
**M**ulti-disciplinary and **A**utomatic **P**late **T**essellation and **T**ime tracking **T**oolkit.

MAPT<sup>3</sup> is a python package providing a set of tools to (1) automatically tessellate the surface of a 3D spherical model using [TTK](https://topology-tool-kit.github.io/) and [paraview](https://www.paraview.org/) with a geodynamical analysis, (2) automatically track over time the detected plates and (3) manipulate the outputs.


## Install MAPT<sup>3</sup>:

### 1. Download

Download **MAPT<sup>3</sup>** from this page and unzip the archive. Then, move into the main directory of the package:
```
cd MAPT3-main/
```

### 2. Creation of a new environment

We strongly recommend installing MAPT<sup>3</sup> in a python environment different from the base using [conda](https://conda.io/projects/conda/en/latest/index.html):

```
conda create --name mapt3
```

Activate the new environment:

```
conda activate mapt3
```

### 3. Installation of TTK - topology toolkit

Before installing **MAPT<sup>3</sup>**, installation of [TTK](https://topology-tool-kit.github.io/index.html):

```
conda install conda-forge::topologytoolkit
```

### 4. Linking MAPT<sup>3</sup>

To link in a user module directory, use [pip](https://pip.pypa.io/en/stable/) and run:

```
python -m pip install .
```

## Uninstall MAPT<sup>3</sup> in the environment:

As you used [pip](https://pip.pypa.io/en/stable/) for the installation, use it to uninstall the package. In a terminal, run:

```
pip uninstall MAPT3
```



