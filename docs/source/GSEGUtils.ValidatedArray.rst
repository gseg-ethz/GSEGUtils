************************
GSEGUtils.ValidatedArray
************************

Description
===========

A subclassable array object for easier data handling and management in research projects.

The main problem it aims to solve was the coupling of numeric array data with other attributes that are automatically
validated. The main benefits being:

* Type errors are caught early (fail-fast) and time spent debugging is reduced
* Code foot print is smaller and in term the logic is clearer
* Still has easy interaction like a Numpy array or dataclasses
* Can also be subclassed to perform array shape and dtype validation

In essence, it can be subclassed easily like one makes a dataclass, acts like a Numpy NDArray object and performs
automatic attribute validation using Pydantic. It has been designed largely around the `PCHandler` library for handling
point cloud data.

Included in the package are a number of predefined classes to create your custom classes with:

:BaseArray: Base class supporting all array shapes with integer or floating datatypes
:NumericMixins: Adds support for python numeric and logical operators (`a+b`, `a != b`)
:FixedLengthArray: Supports data that can be sampled, reduced or extracted by row indexation
:BaseVector: Validates array shape is a 1D array / vector
:HomogeneousArray: Easy method to add a column of 1's (e.g. Homogeneous coordinates)
:ArrayNx2: Validates array to be of shape [N, 2]
:ArrayNx3: Validates array to be of shape [N, 3]


Motivation
==========

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

But importantly, it performs type validation unlike dataclasses::

    # No error is thrown here
    DataclassBased('not an array', 'not an int', 24)

    # Throw errors
    CustomArray('not an array', id=13, name='New object')
    CustomArray(data, id='string passed', name='Invalid ID')
    CustomArray(data, id=13, name=[1, 2, 3])

This is also leveraging `Numpydantic` for shape and dtype validation ::

    class Array4x4Uint8(BaseArray):
        arr: NDArray[Shape['4, 4'], dtype=np.uint8]     # arr is the base attribute for the class

    data = np.ones((4,4), dtype=np.uint8)
    invalid_shape = np.ones((5,5), dtype=np.uint8)
    invalid_dtype = np.ones((4,4), dtype=np.float32)

    Array4x4Uint8(data) # This is ok
    Array4x4Uint8(invalid_shape) # Validation error on array shape
    Array4x4Uint8(invalid_dtype) # Validation error on dtype


In essence, it is an amalgamated class combining
`Pydantic BaseModels <https://docs.pydantic.dev/latest/concepts/models/>`_ ,
`Numpy Ndarrays <https://numpy.org/devdocs/reference/generated/numpy.ndarray.html>`_ and
`Numpydantic <https://numpydantic.readthedocs.io/en/latest/index.html>`_ type definitions.

This module is broken down into two main files. `base_arrays` contains all the major class definitions above.
`base_types` contains some pre-existing NDArray shape and dtype definitions from Pydantic for re-use in defining custom
classes and type hinting your own code. ::

For example, this class will validate the array data to be in the shape of [N, 2] and check the dtype is np.Int32 ::

    from GSEGUtils.base_array import NumericMixin
    from GSEGUtils.base_types import Array_Nx2_Int32_T

    class ValidatedArray(NumericMixin):
        arr: Array_Nx2_Int32_T

Modules
=======
.. toctree::
   :maxdepth: 1

   GSEGUtils.base_arrays
   GSEGUtils.base_types