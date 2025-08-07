GSEGUtils.ValidatedArray
========================

DESCRIPTION
-----------

This ValidatedArray module provides a subclassable array class for data handling and management in research projects.

It's aim is to:

* . *Looks* like a Numpy array,
* \.. *sound* like a numpy array
* ... but most importantly, *ACTS* like a Numpy array!
* Provide per attribute automatic validation for early error detection
* Extend validation to array shapes and dtypes
* Easily extensible base class with extra attributes to act like Python dataclasses

Essentially it was to amalgamated the `Pydantic BaseModel <https://docs.pydantic.dev/latest/concepts/models/>`_ and
`Numpy ndarray <https://numpy.org/devdocs/reference/generated/numpy.ndarray.html>`_ into a single, extensible class.

**But Why?!?!**
    Python dynamic typing makes it super easy to use... sometimes too easy.
    This easily leads to bugs caused by incorrect data types being passed or automatic conversion of types.

    Thus breaking the fail-fast principle. ::

        a = "3"
        b = a * 4       # Doesn't fail, returns "3333"
        assert b == 12  # Fails

    For those leveraging Python's typing module, built-in type checkers like `mypy <https://github.com/python/mypy>`_,
    can help a lot. But this only shows a warning in the IDE and doesn't throw an error when run. If a type hint is
    missing, then this won't be easily detected. ::

        a: int = "3"    # Shows as a warning / error in IDEs
        b = a * 4
        assert b == "3333"  # True as python doesn't perform runtime type checking

    This then leads to programmers putting type validation all throughout their code. For example::

        def add(a, b):
            return a + b

        # Type hinted -> Concise and clear to read
        def add(a: int, b: int) -> int:
            return a + b

        # Type safe...
        def add(a: int, b:int) -> int:
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            else:
                raise TypeError(f"Input variables 'a' and/or 'b' are not of type int.")

    Thus adding significant bloat to the simplest of functions which can detract away from the fundamental logic or
    coded algorithms, particularly when sharing code with others. In this case ``a + b``

    Dataclasses or custom python classes are good for storing and access and have often been used for storing point cloud
    data with it's corresponding scalar fields::

        from dataclasses import dataclass

        @dataclass
        class PointCloud:
            xyz: np.ndarray
            intensity: np.ndarray

        # Creation of an object is easy.
        pcd = PointCloud(xyz=np.random.rand(100, 3), intensity=np.random.rand(100))

    Creation of objects is easy. But the code slowl

Python is a very easy to use language, with lots of flexibility and great libra
ries, making it a go to for the
scientific community. For academics and researchers, as algorithms get larger, data gets bigger, and researcher code
gets more complex, this often leads to long debug times due to errors caused by Python's dynamic typing.


When considering libraries for working with array data,

* **Looks like a Numpy array. Sounds like a Numpy array. ACTS like a Numpy array!**
* **Automated data validation (including array shape and dtype) with Pydantic**
* **Support for extra attribute definition and access like a dataclass**
* **Easy to subclass whilst getting the same functionality**

This module contains the following base array types which can be subclassed as needed:

* *BaseArray*: Base class for all other array types
* *NumericMixins*: Extends numpy like magic method behaviour for equality and numeric operations
* *FixedLengthArray*: Treats the array as a list of row vectors
* *BaseVector*: 1D Vector
* *HomogeneousArray*: Arrays that can be converted to homogeneous coordinates
* *ArrayNx2*: Nx2 sized arrays (E.g. 2D coordinates)
* *ArrayNx3*: Nx3 sized arrays (E.g. 3D coordinates)

PACKAGES
--------
.. toctree::
   :maxdepth: 1

   GSEGUtils.base_arrays
   GSEGUtils.base_types