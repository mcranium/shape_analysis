# Shape analysis

Tools for a landmark free shape analysis of 3D meshes (work in progress)

## Attribution
This code was adapted from the supplementary data files of 
Fukunaga & Burns (2020) Metrics of Coral Reef Structural Complexity Extracted from 3D Mesh Models and Digital Elevation Models. *Remote Sensing* 12, 2676. https://doi.org/10.3390/rs12172676


## Functionality

Added functionality: Object oriented programming style instead of Jupyter notebooks & Multi-core processing.

What works so far: Calculation of vector dispersion and cube counting fractal dimension calculation.

What is still missing: Volumetric fractal dimension calculation, ...

Warning: This code will only work with one of the wavefront obj standards (faces coded as "f 15351 15206 15536", not "f 378//837 393//837 464//837" or similar)! When exporting from [Blender](https://www.blender.org), untick the "UV Coordinates" and "Normals" boxes under "Geometry" in the export menu. When in doubt, check the example meshes in this repository.

Dependencies: Python 3.9.2 (exact version is likely not important), for python packages see [shape_analysis.py](shape_analysis.py).


## Licensing
The article is available under the CC-BY 4.0 license.

The license of this code is GPL v.3


## Citing
Please cite Fukunaga & Burns (2020) if you are using this code for academic purposes.




