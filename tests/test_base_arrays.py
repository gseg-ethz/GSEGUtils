from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpydantic import NDArray, Shape # type: ignore[import-untyped]
from numpydantic.exceptions import DtypeError, ShapeError, NoMatchError # type: ignore[import-untyped]
from pydantic import ValidationError

from GSEGUtils.base_arrays import (     # type: ignore[import-untyped]
    ArrayNx2,
    ArrayNx3,
    BaseArray,
    NumericMixins,
    BaseVector,
    FixedLengthArray,
    HomogeneousArray,
)
from GSEGUtils.base_types import make_ndarray_type  # type: ignore[import-untyped]


class TestNpydanticType:
    """
    Tests to support the make_ndarray_type function as a simple helper function
    """
    def test_shape_definitions(self) -> None:
        """
        Tests the use of the make_ndarray_type function for creating numpydantic type objects
        Returns
        -------

        """
        assert NDArray[Shape["*"], Any] == make_ndarray_type(None)
        assert NDArray[Shape["*, *"], Any] == make_ndarray_type(None, None)
        assert NDArray[Shape["*, *, *"], Any] == make_ndarray_type(None, None, None)
        assert NDArray[Shape["8"], Any] == make_ndarray_type(8)
        assert NDArray[Shape["8"], Any] == make_ndarray_type("8")
        assert NDArray[Shape["2, 4, 7, 9"], Any] == make_ndarray_type(2, 4, 7, 9)
        assert NDArray[Shape["*, ..."], Any] == make_ndarray_type()

    def test_type_definitions(self) -> None:
        assert NDArray[Shape["*"], np.float32] == make_ndarray_type(None, dtype=np.float32)
        assert NDArray[Shape["*"], np.float64] == make_ndarray_type(None, dtype=np.float64)
        assert NDArray[Shape["*"], np.uint8] == make_ndarray_type(None, dtype=np.uint8)
        assert NDArray[Shape["*"], np.uint16] == make_ndarray_type(None, dtype=np.uint16)
        assert NDArray[Shape["*"], np.int32] == make_ndarray_type(None, dtype=np.int32)
        assert NDArray[Shape["*"], np.bool_] == make_ndarray_type(None, dtype=np.bool_)
        assert NDArray[Shape["*, ..."], np.bool_] == make_ndarray_type(dtype=np.bool_)

    def test_numpydantic_type_usage(self) -> None:
        # Test type validation
        float_64_validator = make_ndarray_type(None, dtype=np.float64)

        with pytest.raises(DtypeError):
            float_64_validator(np.random.rand(10,3).astype(np.float32))

        # Test shape validation
        shape_10_3 = make_ndarray_type(10, 3)
        with pytest.raises(ShapeError):
            shape_10_3(np.random.rand(10, 4))
        with pytest.raises(ShapeError):
            shape_10_3(np.random.rand(3, 3))


class TestBaseArray:
    cls = BaseArray
    base_shape: tuple[int, ...] = (10, 3)

    def rand_32(self) -> npt.NDArray[np.float32]:
        return np.random.rand(*self.base_shape).astype(np.float32)

    def rand_64(self) -> npt.NDArray[np.float64]:
        return np.random.rand(*self.base_shape).astype(np.float64)

    def test_initialisation(self) -> None:
        self.check_initialisation_options(self.rand_32())

    def test_positional_init(self) -> None:
        a = self.rand_32()
        array = self.cls(a)
        assert isinstance(array, self.cls)
        assert np.all(a == array)

    @staticmethod
    def check_init_from_reference(data: npt.NDArray[Any], cls: type) -> Any:
        a = cls(arr=data)
        assert isinstance(a, cls)
        assert a is not data                    # base id does not match
        assert a.arr is data                    # array id matches
        assert np.all(a == data)                # Values are identical
        assert a.arr.base is None               # ensure it's not a view
        return a

    def check_initialisation_options(self, data: npt.NDArray[np.floating]) -> None:
        a = self.check_init_from_reference(data, self.cls)
        b = self.check_init_from_copy(data, self.cls, a)
        self.check_init_from_view(data, self.cls, a)
        self.check_init_from_array_like(b, self.cls, a)

    @staticmethod
    def check_init_from_copy(data: npt.NDArray[Any], cls: type, ref_array: npt.ArrayLike) -> Any:
        # Tests on a copy that is passed
        a_copied = cls(arr=data.copy())
        assert ref_array is not a_copied        # New object
        assert a_copied is not data             # Data differs to new array
        assert a_copied.arr is not data         # Array is a new object, not a copy
        assert a_copied.arr.base is None        # Ensure it's a copy and not a view
        assert np.all(a_copied == data)         # Values are identical
        return a_copied

    @staticmethod
    def check_init_from_view(data:npt.NDArray[Any], cls: type, ref_array: npt.ArrayLike) -> Any:
        # Tests on a view that is passed
        a_view = cls(arr=data.view())
        assert ref_array is not a_view          # New object
        assert a_view is not data               # Also a new object
        assert a_view.arr is not data           # View creates a new object
        assert a_view.arr.base is data          # But it does have a base which is the data object
        assert np.all(a_view == data)           # Values are identical

    @staticmethod
    def check_init_from_array_like(data: Any, cls: type, ref_array: npt.ArrayLike) -> Any:
        # Tests on a view that is passed
        a_base_array = cls(arr=data)
        assert ref_array is not a_base_array          # New object
        assert a_base_array is not data               # Also a new object
        assert a_base_array.arr is not data           # View creates a new object
        assert a_base_array.arr is data.arr           # But the array is a reference
        assert a_base_array.arr.base is None          # Check it's not a view
        assert np.all(a_base_array == data)           # Values are identical

    @pytest.mark.parametrize("input_values", ({"dict": "Should fail"}, "String is bad", {1, 2, 3}, None))
    def test_invalid_input_types(self, input_values: Any) -> None:
        with pytest.raises(ValidationError):
            self.cls(arr=input_values)

    def test_general_base_model(self) -> None:
        assert "arr" in self.cls.model_fields
        assert hasattr(self.cls, "model_fields")  # ensure we stick with the 'arr' attribute naming convention
        assert hasattr(self.cls, "model_config")
        assert isinstance(self.cls.model_fields["arr"].annotation, NDArray)

    def test_model_config(self) -> None:
        assert self.cls.model_config["arbitrary_types_allowed"] == True
        assert self.cls.model_config["validate_assignment"] == True
        assert self.cls.model_config["revalidate_instances"] == "never"
        assert self.cls.model_config["validate_default"] == True
        assert self.cls.model_config["strict"] == True
        assert self.cls.model_config["frozen"] == False
        assert self.cls.model_config["extra"] == "ignore"
        assert self.cls.model_config["serialize_by_alias"] == False
        assert self.cls.model_config["populate_by_name"] == False

        # Test validate assignment
        # noinspection PyArgumentList
        a = self.cls(arr=self.rand_32())
        with pytest.raises(ValidationError):
            # noinspection PyTypeChecker
            a.arr = "False"

        # Test frozen is false
        b = self.rand_32()
        a.arr = b.copy()
        assert np.all(a.arr == b)

        # Test extra fields ignore
        a = self.cls(arr=self.rand_32(), dummy_field=2)
        assert not hasattr(a, "dummy_field")

        # Without the appropriate fields, cannot test for:
        # - revalidate instances
        # - strict
        # - frozen (frozen assignment)
        # - serialize_by_alias
        # - populate_by_name

    def test_coerce_to_numpy_valid(self) -> None:
        # Case value is True / 1
        # noinspection PyTypeChecker
        a = self.cls(arr=True)
        assert a.shape == (1, )
        assert a.dtype == np.bool_
        assert a.arr == True

        # Case value is a list of ints
        # noinspection PyTypeChecker
        b = self.cls(arr=[1, 3, 2, 4, -1])
        assert b.shape == (5,)
        assert b.dtype == np.int64
        assert np.all(b == [1, 3, 2, 4, -1])

        # Case tuple of floats
        # noinspection PyTypeChecker
        c = self.cls(arr=(2.3, 2.2, 4.5))
        assert c.shape == (3,)
        assert c.dtype == np.float64
        assert np.all(c == [2.3, 2.2, 4.5])

        # Case of BaseArray type
        # noinspection PyTypeChecker
        d = self.cls(arr=c)
        assert d.shape == (3,)
        assert d.dtype == np.float64
        assert np.all(d == [2.3, 2.2, 4.5])

    def test_coerce_to_numpy_invalid(self) -> None:
        # Case using a set is an object type
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            self.cls(arr={1, 2, 3})

        # Fails on unsupported objects
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            self.cls(arr=object())

        # Dict doesn't work
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            self.cls(arr={'abc': 124})

        # Not supporting complex arrays at this point
        with pytest.raises(ValidationError):
            self.cls(arr=np.ones(3, dtype=np.complex128))

        # Fail on a list of strings
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            self.cls(arr=['a', '23'])
            
    #
    # def test_freeze(self) -> None:
    #     warnings.warn("This method is deprecated", DeprecationWarning)
    #     assert True

    def test_array_interface(self) -> None:
        # The array interface should be identical to the input if it is not a copy
        value = self.cls(arr=(a := self.rand_32()))
        assert a.__array_interface__ == value.__array_interface__
        assert a.__array_function__ == value.arr.__array_function__

        # Case of a copy
        b = self.rand_32()
        value_copy = self.cls(arr=b.copy())
        assert b.__array_interface__ != value_copy.__array_interface__

    def test_transpose(self) -> None:
        # Test the transpose function matches that of numpy
        value = self.cls(arr=(a := self.rand_32()))

        assert np.all(a.T == value.T)
        assert value.T.shape == (3, 10)
        assert isinstance(a.T, np.ndarray)
        assert not isinstance(a.T, self.cls)

    def test_shape(self) -> None:
        # Test that the shape returns the same as the original numpy object
        value = self.cls(arr=(a := self.rand_32()))
        assert value.shape == a.shape

    def test_dtype(self) -> None:
        # Test that the dtype matches
        value = self.cls(arr=(a := self.rand_32()))
        assert value.dtype == a.dtype

    def test_ndim(self) -> None:
        # Test the NDIM match
        value = self.cls(arr=(a := self.rand_32()))
        assert value.ndim == a.ndim == value.arr.ndim

    def test_base(self) -> None:
        # Test the base property to show if a view
        a = self.rand_32()
        a_view = a.view()
        assert a_view.base is not None

        a = self.cls(arr=a_view)
        assert a.base is not None
        assert a.base.shape == a_view.shape
        assert a.base.dtype == a_view.dtype

    def test_size(self) -> None:
        a = BaseArray(arr=self.rand_32())
        assert a.size == np.prod(np.array(a.shape))

    def test_min(self) -> None:
        a = self.rand_32()
        # noinspection PyTypeChecker
        value = self.cls(arr=a)
        assert value.min() == a.min()
        assert np.min(value) == a.min()

    def test_max(self) -> None:
        a = self.rand_32()
        # noinspection PyTypeChecker
        value = self.cls(arr=a)
        assert value.max() == a.max()
        assert np.max(value) == a.max()

    def test_len(self) -> None:
        a = BaseArray(arr=self.rand_32())
        assert len(a) == self.base_shape[0]

    def test_copy_shallow(self) -> None:
        a = BaseArray(arr=self.rand_32())
        arr_2 = a.copy(deep=False)
        assert arr_2 is not a         # The base object differs
        assert arr_2.arr is a.arr     # But the array object is just a reference
        assert np.all(arr_2 == a)     # Copy should have exactly equal values (no precision loss)

    def test_copy_deep(self) -> None:
        # Basic case
        a = BaseArray(arr=self.rand_32())
        arr_2 = a.copy(deep=True)
        assert arr_2 is not a                     # The base object differs
        assert id(arr_2.arr) is not id(a.arr)     # The array object is also new
        assert np.all(arr_2 == a)

        # Test on passing positional arguments
        rand = np.random.rand(100,4)
        arr_3 = a.copy(rand, deep=True)
        assert np.all(arr_3 == rand)
        assert arr_3 is not rand
        assert id(arr_3.arr) is not id(rand)

        # Test on passing of update argument
        rand2 = np.random.rand(100,4)
        arr_4 = a.copy(update={'arr': rand2}, deep=True)
        assert np.all(arr_4 == rand2)
        assert arr_4 is not rand2
        assert id(arr_4.arr) is not id(rand2)

    def test_copy_update(self) -> None:
        # Test the positional array update object
        # numpy array
        a = BaseArray(arr=self.rand_32())
        b = self.rand_32()
        array_3 = a.copy(b)
        assert np.all(array_3.arr == b)
        assert array_3.arr is b     # Positional array bypasses deepcopy
        assert array_3.arr is not a.arr
        assert array_3 is not a

        # As BaseArray as a positional
        array_4 = a.copy(self.cls(arr=b))
        assert np.all(array_4.arr == b)
        assert array_4.arr is b
        assert array_4.arr is not a.arr
        assert array_4 is not a

        # As a list
        array_5 = a.copy(b.tolist())
        assert np.all(array_5.arr == b.tolist())
        assert array_5.arr is not a.arr
        assert array_5 is not a

        # And as an update kwarg
        array_6 = a.copy(update={'arr': b})
        assert np.all(array_6.arr == b)
        assert array_6.arr is b
        assert array_6.arr is not a.arr
        assert array_6 is not a

        # Test that update bypasses the deepcopy
        array_7 = a.copy(update={'arr': b}, deep=True)
        assert np.all(array_7.arr == b)
        assert array_7.arr is b     # Update array bypasses deepcopy
        assert array_7.arr is not a.arr
        assert array_7 is not a

        # Test when a BaseArray is passed as the update parameter
        array_8 = a.copy(update={'arr': self.cls(arr=b)})
        assert np.all(array_6.arr == b)
        assert array_8.arr is b
        assert array_8.arr is not a.arr
        assert array_8 is not a

    def test_numpy_functions(self) -> None:
        """
        Test a number of numpy functions
        """
        a = self.cls(arr=np.ones(self.base_shape))
        b = np.add(a, 1)
        assert np.all(b == 2)

        c = np.mean(a, axis=0)
        assert np.all(c == 1.0)
        assert c.shape == (3,)
        assert isinstance(b, np.ndarray)
        assert np.allclose(a, 1)

    def test_getitem(self) -> None:
        data = [[1, 2, 3,], [4, 5, 6], [-1.2, -5.6, -9.4], [2, -2, 2]]
        array_np = np.array(data)
        # noinspection PyTypeChecker
        array_base = self.cls(arr=data)

        # Case 1 - Single indexing
        result = array_base[3]
        assert np.all(array_np[3] == result)
        assert array_np[3] is not result
        assert isinstance(result, BaseArray)

        # Case 2 - Multi-sampling, no duplicates
        result = array_base[[0, 3]]
        assert np.all(array_np[[0, 3]] == result)
        assert array_np[[0, 3]] is not result
        assert isinstance(result, BaseArray)

        # Case 3 - Multi-sampling, duplicates
        result = array_base[[0, 1, 1, 3, 3]]
        assert np.all(array_np[[0, 1, 1, 3, 3]] == result)
        assert len(result) == 5
        assert result.shape == (5,3)
        assert id(array_np[[0, 1, 1, 3, 3]]) is not id(result)
        assert isinstance(result, BaseArray)

        # Case 4 - Slice
        result = array_base[::2]
        assert np.all(array_np[::2] == result)
        assert np.all(result == array_base[[0, 2]])
        assert isinstance(result, BaseArray)

        # Case 4 - Multi-dimension indexing
        # noinspection PyTypeChecker
        result = array_base[[0, 1, 1, 3], [2, 1, 2, 1]]
        assert np.all(array_np[[0, 1, 1, 3], [2, 1, 2, 1]] == result)
        assert np.all(result == [3, 5, 6, -2])
        assert isinstance(result, BaseArray)

        # Case 5 - Unsupported type (string)
        with pytest.raises(IndexError):
            # noinspection PyTypeChecker
            print(array_base["1"])

    def test_setitem(self) -> None:
        data = np.zeros((10, 3))
        array = self.cls(arr=data)

        # Case 1 - Single indexing
        a = array.copy()
        # Row indexing
        a[3] = 24
        assert np.all(a[3] == 24)
        assert np.all(a[2] == 0)
        # Element indexing
        a[1, 1] = 22
        assert a[1, 1] == 22
        assert a[1, 0] == 0
        assert isinstance(a, BaseArray)

        # Case 2 - Multi-indexing, constant
        a = array.copy()
        a[[1, 3, 4]] = 3
        for i in (1, 3):
            assert np.all(a[i] == 3)
        assert isinstance(a, BaseArray)

        # Case 3 - Multi-indexing, array
        a = array.copy()
        # Rows
        a[[1, 3, 4]] = [[0,1,2], [1, 2, 3], [2, 3, 4]]
        assert np.all(a[[1, 3, 4]] == [[0,1,2], [1, 2, 3], [2, 3, 4]])
        assert isinstance(a, BaseArray)

        # Case 4 - Slice
        a = array.copy()
        a[::4] = [[0,1,2], [1, 2, 3], [2, 3, 4]]
        assert np.all(a[::4] == [[0,1,2], [1, 2, 3], [2, 3, 4]])
        assert isinstance(a, BaseArray)

        # Case 4 - Multi-dimension indexing
        a = array.copy()
        # noinspection PyTypeChecker
        a[[0, 3, 5, 7], [1, 0, 1, 2]] = [2, 4, 6, 8]
        # noinspection PyTypeChecker
        assert np.all(a[[0, 3, 5, 7], [1, 0, 1, 2]] == [2, 4, 6, 8])

        # Case 5 - Unsupported type (string)
        with pytest.raises(IndexError):
            a['1'] = 3

    def test_logical_mixins(self) -> None:
        # Test that the behaviour results in the same logic as that of numpy
        random_1 = self.rand_32()
        random_2 = self.rand_32()
        a = self.cls(arr=random_1)
        b = self.cls(arr=random_2)

        assert np.all((random_1 < random_2) == (a < b))
        assert np.all((random_1 <= random_2) == (a <= b))
        assert np.all((random_1 > random_2) == (a > b))
        assert np.all((random_1 >= random_2) == (a >= b))
        assert np.all((random_1 == random_2) == (a == b))
        assert np.all((random_1 != random_2) == (a != b))

        # Ensure logical checks produce a boolean type
        c: npt.NDArray[np.bool_] = a > 3
        assert c.dtype == np.bool_
        c = a < 3
        assert c.dtype == np.bool_
        c = a <= 3
        assert c.dtype == np.bool_
        c = a >= 3
        assert c.dtype == np.bool_
        c = a == 3
        assert c.dtype == np.bool_
        c = a != 3
        assert c.dtype == np.bool_
        c = 3 > a
        assert c.dtype == np.bool_
        c = 3 < a
        assert c.dtype == np.bool_
        c = 3 <= a
        assert c.dtype == np.bool_
        c = 3 >= a
        assert c.dtype == np.bool_
        c = a == 3
        assert c.dtype == np.bool_
        c = a != 3
        assert c.dtype == np.bool_

    def test_numeric_mixins(self) -> None:
        a = self.cls(arr=np.random.rand(100, 100))
        # Base doesn't support math operators by default
        with pytest.raises(TypeError):
            a += 1
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            a - 1.0
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            a + 2.0
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            a / 0.5
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            a * 3

    def test_reshape(self) -> None:
        a = self.cls(arr=np.ones((100, 3)))
        b = np.zeros_like(a).copy()

        # Ensure a new object but all shapes and values match
        assert a is not b
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert b.ndim == 2
        assert a.shape == (100, 3)
        assert np.all(b == 0)
        assert a.size == 300

        # Compare against a view
        b = np.reshape(a, (3, 100))     # a view
        assert a.shape == (100, 3)            # The original shape has not changed
        assert b.shape == (3, 100)            # The new shape is different
        assert b.base is not a.arr            # The view base is a copy
        assert isinstance(a, BaseArray)        # Base is not changed
        assert isinstance(b, np.ndarray)      # Reshape returns a numpy array object
        assert b.shape == a.T.shape           # Transpose should match

        # Test another shape more complex
        b = np.reshape(a.arr, (5, 60))
        assert a.shape != (5, 60)
        assert b.shape == (5, 60)

        # Check that all values are equal
        assert np.sum(b) == 300 == np.sum(a)
        assert b.base is a.arr                  # This time the array is the base

    def test_view(self) -> None:
        a = BaseArray(arr=self.rand_32())
        view = a.view()
        assert view is not a
        assert view is not a.arr
        assert np.all(view == a)
        assert id(view.base) == id(a.arr)


class TestNumpyMixins(TestBaseArray):
    cls = NumericMixins

    def test_numeric_mixins(self) -> None:
        a = self.rand_64()
        f = self.cls(arr=a + 3)
        g = self.cls(arr=np.full_like(a, 2))
        b = self.cls(arr=a.copy())

        # Left operators
        c = b + 1
        assert np.allclose(c, a + 1)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = b - 1
        assert np.allclose(c, a - 1)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = b * 2
        assert np.allclose(c, a * 2)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = b / 2
        assert np.allclose(c, a / 2)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = f // 1
        assert np.allclose(c, 3)
        assert isinstance(c, NumericMixins)

        c = f % 1
        assert np.allclose(c, a)
        assert isinstance(c, NumericMixins)

        c = g ** 3
        assert np.allclose(c, 8)
        assert isinstance(c, NumericMixins)

        c = 1 + b
        assert np.allclose(c, a + 1)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = 1 - b
        assert np.allclose(c, 1 - a)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = 2 * b
        assert np.allclose(c, 2 * a)
        assert np.all(b != c)
        assert isinstance(c, NumericMixins)

        c = 2 / b
        assert np.allclose(c, 2 / a)
        assert np.all(c > a)
        assert isinstance(c, NumericMixins)

        c = 3.5 // g
        assert np.allclose(c, 1)
        assert isinstance(c, NumericMixins)

        c = 4 % f
        assert np.allclose(c, 1 - a)
        assert isinstance(c, NumericMixins)

        c = 3 ** g
        assert np.allclose(c, 9)
        assert isinstance(c, NumericMixins)

        # In place operators
        b += 1
        assert np.allclose(b, a + 1)
        assert isinstance(c, NumericMixins)

        b -= 1
        assert np.allclose(b, a)
        assert isinstance(c, NumericMixins)

        b *= 3
        assert np.allclose(b, a * 3)
        assert isinstance(c, NumericMixins)

        b /= 3
        assert np.allclose(b, a)
        assert isinstance(c, NumericMixins)

        b += 1
        b %= 1
        assert np.allclose(b, a)
        assert isinstance(c, NumericMixins)

        b //= 1
        assert np.allclose(b, np.zeros_like(a))
        assert isinstance(c, NumericMixins)

        b.arr = np.full_like(b.arr, 3)
        b **= 3
        assert np.allclose(b, 27)
        assert isinstance(c, NumericMixins)

        b.arr = np.full_like(b.arr, 8)
        c = divmod(b, 6)
        assert np.all(c[0] == 1)
        assert np.all(c[1] == 2)

        c = divmod(12, b)
        assert np.all(c[0] == 1)
        assert np.all(c[1] == 4)

        assert np.all(-b == (b * -1))

        temp = self.cls(arr=a.copy()) - 0.5
        assert np.all(abs(temp) >= 0)
        assert np.all(abs(temp) <= 0.5)

    def test_mat_mul_mixins(self):
        a_ = np.random.rand(5,3)
        b_ = np.random.rand(3,2)
        c_expected = a_ @ b_
        c_expected_inv = b_.T @ a_.T

        a_ = self.cls(arr=a_)
        b_ = self.cls(arr=b_)
        c_ = a_ @ b_
        assert c_.shape == (5,2)
        assert np.all(c_ == c_expected)
        assert isinstance(c_, self.cls)

        a_ = self.cls(arr=a_.T)
        b_ = self.cls(arr=b_.T)
        c_inv = a_.__rmatmul__(b_)
        assert c_inv.shape == (2, 5)
        assert np.all(c_inv == c_expected_inv)
        assert isinstance(c_inv, self.cls)

        a_ = np.random.rand(5,3)
        b_ = np.random.rand(3,3)
        c_expected = a_ @ b_
        a_ = self.cls(arr=a_)
        b_ = self.cls(arr=b_)
        a_ @= b_

        assert a_.shape == (5,3)
        assert np.all(a_ == c_expected)
        assert isinstance(a_, self.cls)

        b_ = np.random.rand(3,2)
        with pytest.raises(ValueError):
            a_ @= b_



class TestFixedLength(TestNumpyMixins):
    cls = FixedLengthArray

    def test_iter_method(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        # Check that each item returns the next row
        for i, vals in enumerate(array_fl):
            assert np.all(array_fl[i] == vals)

        # Check conversion to list
        as_list = list(array_fl)
        assert len(as_list) == len(array_fl)
        assert id(array_fl) != id(as_list)

    def test_create_mask_from_slice(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        # General case
        index = slice(0, 10, 3)
        mask = array_fl.create_mask(index)
        assert np.sum(mask) == 4
        assert np.all(mask[[0, 3, 6, 9]])
        assert mask.dtype == np.bool_

        # Reverse order
        index = slice(None, None, -3)
        mask = array_fl.create_mask(index)
        assert np.sum(mask) == 4
        assert np.all(mask[[9, 6, 3, 0]])

        # Reverse order + Negative limit
        index = slice(None, -1, 3)
        mask = array_fl.create_mask(index)
        assert np.sum(mask) == 3
        assert np.all(mask[[0, 3, 6]])

        # Reverse order + Negative limit
        index = slice(0, 4, None)
        mask = array_fl.create_mask(index)
        assert np.sum(mask) == 4
        assert np.all(mask[[0, 1, 2, 3]])

        # Slice for all points
        index = slice(None, None, None)
        mask = array_fl.create_mask(index)
        assert np.sum(mask) == 10
        assert np.all(mask[:])

    def test_create_mask_from_integer(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        val = 3
        mask = array_fl.create_mask(val)
        assert mask[val] == True
        assert np.sum(mask) == 1
        assert mask.dtype == np.bool_

        val = -2
        mask = array_fl.create_mask(val)
        assert mask[val] == True
        assert mask[len(array_fl) + val] == True
        assert np.sum(mask) == 1
        assert mask.dtype == np.bool_

    def test_create_mask_from_boolean(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        vals = np.random.randint(0, 2, array_fl.shape[0], dtype=bool)
        mask = array_fl.create_mask(vals)
        assert id(mask) == id(vals)
        assert np.all(mask == vals)
        assert mask.dtype == np.bool_

        for i in (8, 50, 100):
            with pytest.raises(ValueError):
                array_fl.create_mask(np.random.randint(0, 2, i, dtype=bool))
    
    def test_create_mask_from_sequence(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        mask = array_fl.create_mask((0, 1, 2))
        assert np.all(mask[[0, 1, 2]])
        assert np.sum(mask) == 3
        assert mask.dtype == np.bool_

        mask = array_fl.create_mask([-3, -4, -5])
        assert np.all(mask[[-3, -4, -5]])
        assert np.all(mask[[7, 6, 5]])
        assert np.sum(mask) == 3
        assert mask.dtype == np.bool_

    def test_create_mask_from_np_integers(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        # Test no duplicates
        vals = np.array([1, 3, 5])
        mask = array_fl.create_mask(vals)
        assert np.all(mask[[1, 3, 5]])
        assert np.sum(mask) == 3
        assert mask.dtype == np.bool_

        # Test duplicates
        vals = np.array([1, 1, 3, 3, 5])
        mask = array_fl.create_mask(vals)
        assert np.all(mask[[1, 3, 5]])
        assert np.sum(mask) == 3
        assert mask.dtype == np.bool_

        # Test out of range
        with pytest.raises(IndexError):
            array_fl.create_mask(np.array([0, 4, 15]))

    @pytest.mark.parametrize('value', ('1 2 3', True, np.random.rand(10), {1: '1', 2: '2'}))
    def test_invalid_create_mask_values(self, value: Any) -> None:
        array_fl = self.cls(arr=self.rand_32())

        with pytest.raises(ValueError) as e:
            array_fl.create_mask(value)

        assert type(e.value) in (DtypeError, ValueError, NoMatchError)

    def test_sample(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        indices = [1, 3, 4, 5]
        sample = array_fl.sample(indices)
        assert id(sample) != id(array_fl)
        assert id(sample.arr) != id(array_fl.arr)
        assert np.all(sample.arr == array_fl.arr[indices])
        assert len(sample) == 4
        assert len(sample) != len(array_fl)

        # Oversampling shouldn't work
        indices = [1,1,1,1, 3,3,3,3, 4,4,4,4, 5]
        sample = array_fl.sample(indices)
        assert len(sample) != len(indices)
        assert id(sample) != id(array_fl)
        assert id(sample.arr) != id(array_fl.arr)
        assert np.all(sample.arr == array_fl.arr[np.unique(indices)])
        assert len(sample) == 4
        assert len(sample) != len(array_fl)

    def test_reduce(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        base_vals = array_fl.copy()
        assert id(base_vals) != id(array_fl)
        assert id(base_vals.arr) != id(array_fl.arr)

        indices = [1, 3, 4, 5]
        base_vals.reduce(indices)
        assert len(base_vals) != len(array_fl)
        assert len(base_vals) == len(indices)
        assert np.all(base_vals == array_fl[indices])

    def test_extract(self) -> None:
        array_fl = self.cls(arr=self.rand_32())
        indices = [1, 3, 4, 5]
        base_vals = array_fl.copy()
        extracted = base_vals.extract(indices)

        # Check that the arrays are all unique
        assert id(extracted) != id(base_vals)
        assert id(extracted.arr) != id(base_vals.arr)
        assert id(array_fl.arr) != id(base_vals.arr)

        mask = array_fl.create_mask(indices)
        #Check the extracted values match the indices
        assert len(extracted) == len(indices)
        assert np.all(extracted == array_fl[mask])
        assert np.all(base_vals == array_fl[~mask])


class TestBaseVector(TestFixedLength):
    cls = BaseVector
    base_shape = (10,)

    def test_initialisation(self) -> None:
        data = np.random.rand(10)
        self.check_initialisation_options(data)

        vec = self.cls(arr=data)

        assert vec.size == 10
        assert len(vec) == 10
        assert np.sum(vec) < 10
        assert vec.shape == (10,)

    @pytest.mark.parametrize('value', (np.ones((10, 1)),
                                       np.ones((1, 10, 1)),
                                       np.ones((10, 1, 1)),
                                       BaseArray(arr=np.ones((1,10)))))
    def test_squeeze_coercion(self, value: npt.ArrayLike) -> None:
        vec = self.cls(arr=value)
        assert vec.size == 10
        assert len(vec) == 10
        assert np.sum(vec) == 10
        assert vec.shape == (10,)

    @pytest.mark.parametrize('value', (np.random.rand(10, 3), BaseArray(arr=np.random.rand(10, 2))))
    def test_vector_validation(self, value: npt.ArrayLike) -> None:
        with pytest.raises(ValidationError):
            self.cls(arr=value)

    def test_getitem(self) -> None:
        pass

    def test_setitem(self) -> None:
        pass

    def test_reshape(self) -> None:
        pass

    def test_numpy_functions(self) -> None:
        pass

    def test_mat_mul_mixins(self) -> None:
        pass

    def test_transpose(self) -> None:
        a = self.cls(arr=self.rand_32())
        assert np.all(a == a.T)


class TestHomogeneous(TestFixedLength):
    cls = HomogeneousArray

    def test_homogeneous(self) -> None:
        array = self.cls(arr=self.rand_32())
        h = array.H
        assert len(array) == len(h)
        assert isinstance(h, np.ndarray)
        assert array.shape[1]+1 == h.shape[1]
        assert np.all(h[:, -1] == 1)
        assert h.dtype == array.dtype
        assert np.all(array == h[:, :-1])


class TestArrayNx2(TestHomogeneous):
    cls = ArrayNx2
    base_shape = (10, 2)

    def test_transpose(self) -> None:
        a = self.rand_32().T    # (2, 10)

        b = self.cls(arr=a)
        assert np.all(b.T == a)
        assert b.shape[0] == a.shape[1]
        assert b.shape[1] == a.shape[0]

    def test_getitem(self) -> None:
        pass

    def test_setitem(self) -> None:
        pass

    def test_reshape(self) -> None:
        pass

    def test_numpy_functions(self) -> None:
        pass

    def test_coerce_to_numpy_valid(self) -> None:
        pass

    def test_mat_mul_mixins(self) -> None:
        pass


class TestArrayNx3(TestArrayNx2):
    cls = ArrayNx3
    base_shape = (10, 3)
