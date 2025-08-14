GSEGUtils API
=============
.. automodule:: GSEGUtils


.. toctree::
   :maxdepth: 3

   GSEGUtils.BaseArrays
   GSEGUtils.util
   GSEGUtils.validators

Help on Base Arrays.

NAME
    base_arrays

DESCRIPTION
    Subclassable base array module for better handling of array data
    ================================================================

    The base_arrays module is designed to create sub-classable array like objects that:

    * **Looks like a Numpy array. Sounds like a Numpy array. ACTS like a Numpy array!**
    * **Automated data validation (including array shape and dtype) with Pydantic**
    * **Support for extra attribute definition and access like a dataclass**
    * **Easy to subclass whilst getting the same functionality**

    This module contains the following base array types which can be subclassed as needed:

    :BaseArray: Base class for all other array types
    :NumericMixins: Extends numpy like magic method behaviour for equality and numeric operations
    :FixedLengthArray: Treats the array as a list of row vectors
    :BaseVector: 1D Vector
    :HomogeneousArray: Arrays that can be converted to homogeneous coordinates
    :ArrayNx2: Nx2 sized arrays (E.g. 2D coordinates)
    :ArrayNx3: Nx3 sized arrays (E.g. 3D coordinates)