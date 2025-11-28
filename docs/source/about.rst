About
=====

|license-bsd-3| |documentation-status|

.. |license-bsd-3| image:: https://img.shields.io/badge/License-BSD_3-yellow.svg
    :alt: License: BSD-3
    :target: ./LICENSE.txt

.. |documentation-status| image:: https://readthedocs.org/projects/gsegutils/badge/?version=latest
    :alt: Documentation Status
    :target: https://readthedocs.org/projects/gsegutils/badge/

GSEGUtils provides some tools and functionality that could be used across other development or research projects.
These are particularly focussed in the direction of research within the group.

The main goal if this repo is to help software processing efforts and allow more focus on algorithm
development and research.

Currently this repository includes:

+-------------------+-------------------------------------------------------------------------------------+
| Package           | Description                                                                         |
+===================+=====================================================================================+
| `BaseArrays`      | Sub-classable array objects with automated validation on type, shape and attributes |
+-------------------+-------------------------------------------------------------------------------------+
| `Lazy Disk Cache` | Automated data offloading for memory management                                     |
+-------------------+-------------------------------------------------------------------------------------+
| `Util`            | Includes utility functions such as angle unit conversions                           |
+-------------------+-------------------------------------------------------------------------------------+
| `Validators`      | Helper functions for performing validation and normalisation                        |
+-------------------+-------------------------------------------------------------------------------------+

Installation
------------

This can be done directly from the repository directory:

::

    pip install "GSEGUtils @ git+https://github.com/gseg-ethz/GSEGUtils.git"


or by cloning it

::

    git clone https://github.com/gseg-ethz/GSEGUtils.git
    cd GSEGUtils
    pip intstall .

Usage
-----
Accessing the different classes can then be easily done by importing from `GSEGUtils`::


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

::

    coords3D = Array_Nx3(np.random.rand(20))
    coords3d.arr = np.random.rand(100) # Valid
    coords3d.arr = np.random.rand(100, 3) # RAISES ERROR -> invalid shape not of (N, 3), expected (N,)



