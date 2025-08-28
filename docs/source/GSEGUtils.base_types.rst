GSEGUtils.base\_types
=====================

Provides predefined `Numpydantic`_ type hints to be used with Pydantic for automatic shape and dtype validation. In some
cases it can also be used for type-hinting, providing easier insight into expected shape and dtype of an array.

Please read the `docs <https://numpydantic.readthedocs.io/en/latest/index.html>`_ for more information on usage and
how you can create your own dtypes.

Use with Pydantic Models::

    from pydantic import BaseModel
    from numpydantic import NDArray, Shape

    class ValidatedArray(BaseModel):
        arr: NDArray[Shape['*, 1'], int]    # This is the numpydantic component and will validate tha array

Use in instance validation::

    Array_Nx3_Float32_T = NDArray[Shape['*, 3'], np.float32]

    isinstance(np.ones((10, 3), dtype=np.float32), Array_Nx3_Float32_T)
    # True

    isinstance(np.ones((10, 5), dtype=np.float32), Array_Nx3_Float32_T)
    # False (incorrect shape)

    isinstance(np.ones((10, 3), dtype=np.uint8), Array_Nx3_Float32_T)
    # False (incorrect dtype)


Use as a callable validator::

    >>> Array_Nx3_Float32_T(np.random.rand(2,3).astype(np.float32))
    array([[0.9211791 , 0.89427036, 0.80592966],
           [0.341839  , 0.8369464 , 0.7697314 ]], dtype=float32)

    # Raises errors
    >>> Array_Nx3_Float32_T(np.random.rand(2,5).astype(np.float32))
    numpydantic.exceptions.ShapeError: Invalid shape! expected shape ['*', '3'], got shape (2, 5)

    >>> Array_Nx3_Float32_T(np.random.randint(0, 255, (2,3), dtype=np.uint8))
    numpydantic.exceptions.DtypeError: Invalid dtype! expected <class 'numpy.float32'>, got uint8



.. automodule:: GSEGUtils.base_types
   :members:
   :undoc-members:
   :show-inheritance:

.. _Numpydantic: https://numpydantic.readthedocs.io/en/latest/index.html