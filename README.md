# GSEGUtils

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3-yellow.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/gsegutils/badge/)](https://gsegutils.readthedocs.io/)

GSEGUtils provides some tools and functionality that are used across other development or research projects.
These are particularly focussed in the direction of research within the group.

The main goal if this repo is to help software development efforts and allow more focus on algorithm 
development, data processing and research rather than code debugging and maintenance.

Currently this module includes:

| Package             | Description                                                                         |
|---------------------|-------------------------------------------------------------------------------------|
| **BaseArrays**      | Sub-classable array objects with automated validation on type, shape and attributes |
| **Lazy Disk Cache** | Automated data offloading for memory management                                     |
| **Util**            | Includes utility functions such as angle unit conversions                           |
| **Validators**      | Helper functions for performing validation and normalisation                        | 

## Installation

This can be done directly from the repository directory:

```commandline
pip install git+ssh://git@github.com/gseg-ethz/GSEGUtils.git
```

or by cloning it

```commandline
git clone https://github.com/gseg-ethz/GSEGUtils.git
cd GSEGUtils
pip intstall .
 ```

## Usage
Accessing the different classes can then be easily done by importing from `GSEGUtils`:

```python
>>> from GSEGUtils.base_arrays import BaseArray, ArrayNx3, BaseVector
>>> import numpy as np
>>> data = np.random.rand(10,20,3)
>>> array = BaseArray(data)

>>> array.shape
(10, 20, 3)

>>> np.all(array == data)
np.True_

>>> a = ArrayNx3(np.random.rand(10, 3))
>>> b = a + 10
>>> np.all(b == a + 10)
np.True_

coords3D = Array_Nx3(np.random.rand(20, 3))
coords3d.arr = np.random.rand(100) 
# RAISES ERROR -> invalid shape not of (N, 3), expected (N,)

coords3d.arr = np.random.rand(100, 3) 
# Valid as the array is of shape Nx3
```