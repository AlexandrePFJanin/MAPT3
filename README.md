<p align="center">
    <img src="./icons/MAPT3-logo_white-bg.png#gh-light-mode-only" height="150" width="327"/>
    <img src="./icons/MAPT3-logo_dark-bg.png#gh-dark-mode-only" height="150" width="327"/>
</p>

# MAPT<sup>3</sup>
### **M**ulti-disciplinary and **A**utomatic **P**late **T**essellation and **T**ime tracking **T**oolkit.

MAPT<sup>3</sup> is a Python package developed by [Janin et al., (2025)](https://www.nature.com/articles/s41561-025-01772-7) for the analysis of numerical geodynamic models. It enables automatic tessellation of 3D spherical models, tracking of the detected tectonic plates through time, and efficient manipulation of outputs. It leverages [TTK](https://topology-tool-kit.github.io/) and [ParaView](https://www.paraview.org/), coupled to a geodynamical analysis it provides researchers with a powerful toolkit for geodynamical analysis.


For a detailed description of the methods and scientific background, we encourage you to read the corresponding paper: [Janin et al., (2025)](https://www.nature.com/articles/s41561-025-01772-7)

MAPT<sup>3</sup> is actively maintained and regularly updated to provide the best possible experience. Stay tuned for new features and improvements!


## Install MAPT<sup>3</sup>

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

## Uninstall MAPT<sup>3</sup> in the environment

As you used [pip](https://pip.pypa.io/en/stable/) for the installation, use it to uninstall the package. In a terminal, run:

```
pip uninstall MAPT3
```

## Use MAPT<sup>3</sup>

An example of a simple workflow using MAPT<sup>3</sup> is provided in [docs/sample](./docs/sample/) with scripts and data. Read the associated [README](./docs/sample/README.txt) file carefully and follow the step-by-step instructions.

**Tip:** For users of the mantle convection code [StagYY](https://doi.org/10.1016/j.pepi.2008.08.005), [pypStag](https://github.com/AlexandrePFJanin/pypStag) provides an easy and efficient way to generate input files for ParaView and MAPT<sup>3</sup> from the binary output files of StagYY.


## Cite MAPT<sup>3</sup>

If you use MAPT<sup>3</sup> in your research, we kindly ask that you cite the following paper that describe the package and the underlying method.

You can use the following APA style citation:

> Janin, A., Coltice, N., Chamot-Rooke, N., Tierny, J. Geodynamics of a global plate reorganization from topological data analysis. Nature Geoscience (2025). https://doi.org/10.1038/s41561-025-01772-7

Or the following bibTeX citation:

```bibtex
@article{janin2025geodynamics,
    author = {Janin, Alexandre and Coltice, Nicolas and Chamot-Rooke, Nicolas and Tierny, Julien},
    title = {Geodynamics of a global plate reorganization from topological data analysis},
    journal = {Nature Geoscience},
    year = {2025},
    month = {8},
    day = {20},
    issn = {1752-0908},
    doi = {10.1038/s41561-025-01772-7},
    url = {https://www.nature.com/articles/s41561-025-01772-7},
}
```
Citing our work helps us continue to improve and maintain this project. Thank you for your support!
