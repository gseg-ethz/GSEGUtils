.. _BaseArraysDescription:

BaseArrays
**********

Description
===========

A subclassable array object for easier data handling and management

The main problem it aims to solve was the coupling of numeric array data with other attributes that are automatically
validated. The main benefits being:

* Type errors are caught early (fail-fast) and time spent debugging is reduced
* Code foot print is smaller and in turn more readable and algorithm logic is clearer
* Still has easy interaction like a Numpy array or dataclasses
* Can also be subclassed to perform array shape and data type validation

It can be subclassed easily, acts like a Numpy NDArray object and performs automatic attribute validation using
Pydantic. It has been designed largely around the `PCHandler` library for the extension to point cloud data but can be
used for other common array objects.

Included in the package are a number of predefined classes to create your custom classes with:

.. currentmodule:: GSEGUtils.base_arrays

.. autosummary::
   BaseArray
   NumericMixins
   FixedLengthArray
   BaseVector
   HomogeneousArray
   ArrayNx2
   ArrayNx3

This is broken down into two main files.:

* :py:mod:`GSEGUtils.base_arrays` contains all the major class definitions above
* :py:mod:`GSEGUtils.base_types` contains some pre-existing Numpydantic shape and dtype definitions for reuse in typehints or validation

For example, this class will automaticall validate the array data to be in the shape of [N, 2] and check the dtype is
np.Int32 when a new object is initialized::

    from GSEGUtils.base_arrays import NumericMixins
    from GSEGUtils.base_types import Array_Nx2_Int32_T

    class ValidatedArray(NumericMixins):
        arr: Array_Nx2_Int32_T


Motivation
==========

Numpy-like Behavior
-------------------

*Looks* like a Numpy array, *sounds* like a numpy array, it **acts** like a Numpy array! ::

    >>> a = BaseArray([[0, 1, 2], [3, 3, 3]])
    >>> np.add(a, 1)
    array([[1, 2, 3],
       [4, 4, 4]])


and with the NumericMixIns class for built in operators::

    >>> a = NumericMixins([[0, 1, 2], [3, 3, 3]])
    >>> a + 1
    array([[1, 2, 3],
       [4, 4, 4]])

    >>> a = NumericMixins([[0, 1, 2], [3, 3, 3]])
    >>> a += 1
    >>> a
    NumericMixins(arr=array([[1, 2, 3],
            [4, 4, 4]]))


Extra Attribute Definition and Validation
-----------------------------------------

It natively supports additional attribute information being assigned to the class. Much like python's dataclasses
module. ::

    @dataclass
    class DataclassBased:
        array: np.ndarray
        id: int
        name: str

    class CustomArray(NumericMixins):   #No need to define array
        id: int
        name: str


    data = np.random.rand(100,100)

    a = DataclassBased(data, 13, 'old_dataclasses_object')
    b = CustomArray(data, id=13, name='New object')


But importantly, it performs type validation unlike dataclasses using
`Pydantic <https://docs.pydantic.dev/latest/concepts/models/>`_ ::

    # No error is thrown here
    DataclassBased('not an array', 'not an int', 24)

    # Throw errors
    CustomArray('not an array', id=13, name='New object')
    CustomArray(data, id='string passed', name='Invalid ID')
    CustomArray(data, id=13, name=[1, 2, 3])


This leverages `Numpydantic <https://numpydantic.readthedocs.io/en/latest/index.html>`_ for shape and dtype validation ::

    class Array4x4Uint8(BaseArray):
        arr: NDArray[Shape['4, 4'], dtype=np.uint8]     # arr is the base attribute for the class

    data = np.ones((4,4), dtype=np.uint8)
    invalid_shape = np.ones((5,5), dtype=np.uint8)
    invalid_dtype = np.ones((4,4), dtype=np.float32)

    Array4x4Uint8(data) # This is ok
    Array4x4Uint8(invalid_shape) # Validation error on array shape
    Array4x4Uint8(invalid_dtype) # Validation error on dtype


.. note::
  You may see class names as `ArrayNx3` and `Array_Nx3_T`. *ArrayNx3* is designed to be a usable class whereas
  *Array_Nx3_T* with the *_T* at the end indicates it's a type for validation purposes.


Modules
=======

.. toctree::
   :maxdepth: 1

   GSEGUtils.base_arrays
   GSEGUtils.base_types
