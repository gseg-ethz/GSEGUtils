import pytest
from abc import ABC
from typing import Callable

import numpy as np

from GSEGUtils.constants import HALF_PI, PI, TWO_PI
from GSEGUtils.validators import (
    validate_azimuth_angles,
    validate_inclination_angles,
    validate_spherical_angles,
    validate_horizontal_angles,
    validate_zenith_angles,
    validate_radius,
    coerce_wrapped_azimuth_angles,
    coerce_wrapped_horizontal_angles,
    validate_transposed_2d_array,
    convert_slice_to_integer_range,
    validate_in_range,
    normalize_min_max,
    linear_map_dtype,
    normalize_self,
    _normalize_base,
    normalize_uint8,
    normalize_uint16,
    normalize_int8,
    normalize_int16,
    normalize_int32,
    normalize_int64
)

COORDINATE_3D_PROPERTIES = ("x", "y", "z", "r", "v", "hz", "rho", "theta", "phi", "xyz", "spher")


class BaseAngleTestClass(ABC):
    main_test: Callable | None = None

    @pytest.mark.parametrize(
        "values, func",
        [
            ("str", main_test),
            ({"angle": 74}, main_test),
            ({1.3, 0.2, -1.3}, main_test),
        ],
    )
    def test_invalid_types(self, values: np.ndarray, func: Callable):
        with pytest.raises(TypeError):
            func(values)


class TestValidation(BaseAngleTestClass):
    class TestHzAngle(BaseAngleTestClass):
        main_test: Callable = validate_horizontal_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, -np.pi]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 2 * np.pi, 1.7, -np.pi - 3]), main_test),
                (float(72), main_test),
                (int(-44), main_test),
                (np.array([0, 1.3, np.pi, np.pi * 1.5, np.pi*2]), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestAzimuthAngle(BaseAngleTestClass):
        main_test: Callable = validate_azimuth_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, 2 * np.pi]), main_test),
                (np.linspace(0, TWO_PI, 1000, endpoint=True), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 3 * np.pi, 1.7, -np.pi - 3]), main_test),
                (np.linspace(-HALF_PI, HALF_PI, 1000, endpoint=True), main_test),
                (float(72), main_test),
                (int(-44), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestZenithAngles(BaseAngleTestClass):
        main_test: Callable = validate_zenith_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, np.pi]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 3 * np.pi, 1.7, -np.pi - 3]), main_test),
                ("not an array", main_test),
                (float(72), main_test),
                (np.array([-1.1, 0.5, 1.3]), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestInclinationAngles(BaseAngleTestClass):
        main_test: Callable = validate_inclination_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 1.3, -np.pi / 2, np.pi / 2]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, -np.pi]), main_test),
                (np.array([0, np.pi, 1.7, np.pi]), main_test),
                (float(72), main_test),
                (int(-44), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestRadiusDistance(BaseAngleTestClass):
        main_test: Callable = validate_radius

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 1.3, 200, 300000.21323]), main_test),
                (np.random.rand(100, 100), main_test),
                (np.zeros((100, 100)), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([-1.3, -2, np.inf, -np.pi]), main_test),
                (np.random.rand(100, 100) * -1, main_test),
                ([1.3, 2000, 23445.123], main_test),
                ((1.3, 2000, 23445.123), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestSphericalCoordinates(BaseAngleTestClass):
        main_test: Callable = validate_spherical_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (
                    np.array([[np.random.rand(10) * 100, np.random.rand(10) * TWO_PI - PI, np.random.rand(10) * PI]]),
                    main_test,
                )
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (
                    np.array(
                        [
                            np.random.rand(10) * 100 - 50,
                            np.random.rand(10) * TWO_PI - PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * TWO_PI - PI,
                            np.random.rand(10) * TWO_PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * TWO_PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * -TWO_PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    "Not an array", main_test,
                )
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


def test_coerce_wrapped_azimuths():
    original = np.linspace(0, TWO_PI, 1000000, endpoint=False)
    offset = 0.3455 * PI
    array = original + offset
    coerced_array = coerce_wrapped_azimuth_angles(array)
    assert validate_azimuth_angles(coerced_array) is not None
    coerced_array -= offset
    coerced_original_coordinates = coerce_wrapped_azimuth_angles(coerced_array)
    assert validate_azimuth_angles(coerced_original_coordinates) is not None
    assert np.allclose(coerced_original_coordinates, original)


def test_coerce_wrapped_horizontal_angles():
    original = np.linspace(PI, -PI, 1000000, endpoint=False)
    offset = 0.3455 * PI
    array = original + offset
    coerced_array = coerce_wrapped_horizontal_angles(array)
    assert validate_horizontal_angles(coerced_array) is not None
    coerced_array -= offset
    coerced_original_coordinates = coerce_wrapped_horizontal_angles(coerced_array)
    assert validate_horizontal_angles(coerced_original_coordinates) is not None
    assert np.allclose(coerced_original_coordinates, original)


@pytest.mark.parametrize("array", (np.random.rand(3, 10), np.random.rand(10, 3)))
def test_validate_Nx3_transposed(array):
    original = array.copy()
    if array.shape != (10, 3):
        original = original.T
    array = validate_transposed_2d_array(array, n=3)
    assert array.shape == (10, 3)
    assert np.allclose(array, original)


@pytest.mark.parametrize("array", (np.random.rand(2, 10), np.random.rand(10, 2)))
def test_validate_Nx2_transposed(array):
    original = array.copy()
    if array.shape != (10, 2):
        original = original.T
    array = validate_transposed_2d_array(array, n=2)
    assert array.shape == (10, 2)
    assert np.allclose(array, original)


def test_invalid_transposed_2d():
    a = np.random.rand(100, 100, 100)
    b = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        validate_transposed_2d_array(a, n=3)



@pytest.mark.parametrize(("slice_obj", "expected"), (
                         (slice(None, None, None), [i for i in range(10)]),
                         (slice(0, None, None), [i for i in range(10)]),
                         (slice(0, 5, None), [i for i in range(5)]),
                         (slice(3, 8, None), [i for i in range(3, 8)]),
                         (slice(3, 8, 2), [3, 5, 7]),
                         (slice(3, 9, 2), [3, 5, 7]),
                         (slice(3, 9, -1), []),
                         (slice(9, 3, -1), [9, 8, 7, 6, 5, 4]),
                         (slice(None, None, 3), [0, 3, 6, 9]),
                         (slice(-1, -3, -1), [9, 8]),
                         (slice(-1, -3, None), []),
                         (slice(-2, -10, -2), [8, 6, 4, 2]),
                         (slice(-2, None, -2), [8, 6, 4, 2, 0]),
                         (slice(None, 4, None), [0, 1, 2, 3]),
))
def test_slice_to_integer_range(slice_obj, expected):
    result = convert_slice_to_integer_range(slice_obj, 10)
    assert np.all(result == expected)

@pytest.mark.parametrize(("value", "v_min", "v_max"), (
    (np.full(2, 2), 1, 4),
    (np.arange(100), -1, 100),
    (np.linspace(-np.pi, np.pi, 100, endpoint=True), -np.pi-0.0001, np.pi+0.0001),
    ([-2, 7, 1004, 200.43], -34, 10000),
    (1, 0, 2)
))
def test_validate_in_range(value, v_min, v_max):
    assert validate_in_range(value, v_min, v_max) is None


def test_validate_in_range_invalid():
    # Both bounds broken
    with pytest.raises(ValueError):
        validate_in_range(np.arange(100), 20, 60)

    # Lower bound broken
    with pytest.raises(ValueError):
        validate_in_range(np.array([-100]), 10, 100)

    # Upper bound broken
    with pytest.raises(ValueError):
        validate_in_range(np.array([100123]), 10, 100)


@pytest.mark.parametrize(("func", "dtype"), (
                         (normalize_uint8, np.uint8),
                         (normalize_uint16, np.uint16),
                         (normalize_int8, np.int8),
                         (normalize_int16, np.int16),
                         (normalize_int32, np.int32),
                         (normalize_int64, np.int64)
))
def test_normalize_to_dedicated_int_dtype_funcs(func, dtype: np.dtype):
    values = np.random.rand(100).astype(np.float64)
    lower = np.iinfo(dtype).min
    upper = np.iinfo(dtype).max

    width = upper - lower

    values = np.ceil(values * width + lower).astype(dtype)
    computed = func(values)

    assert np.allclose(computed, values)


def test_normalize_min_max_basic_types():
    # Test case 1: Float to uint8
    float_array = np.array([0.0, 0.5, 1.0])
    result = normalize_min_max(float_array, 0, 255, np.uint8)
    assert result.dtype == np.uint8
    assert np.allclose(result, np.array([0, 128, 255]), atol=1)

    # Test case 2: Int to float
    int_array = np.array([0, 50, 100], dtype=np.int32)
    result = normalize_min_max(int_array, 0.0, 1.0, np.float32)
    assert result.dtype == np.float32
    assert np.allclose(result, np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # Test case 3: Boolean to int
    bool_array = np.array([False, True, False])
    result = normalize_min_max(bool_array, 0, 100, np.int32)
    assert result.dtype == np.int32
    assert np.allclose(result, np.array([0, 100, 0], dtype=np.int32))


def test_normalize_min_max_with_custom_bounds():
    # Test with custom v_min and v_max
    array = np.array([100, 200, 300, 400, 500])
    result = normalize_min_max(array, 0, 1, np.float32, v_min=200, v_max=400)
    assert np.allclose(result, np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32))


# def test_normalize_min_max_edge_cases():
#     # Test case 1: Single value array
#     single_value = np.array([42])
#     result = normalize_min_max(single_value, 0, 1, np.float32)
#     assert result.dtype == np.float32
#     assert np.array_equal(result, np.array([0], dtype=np.float32))
#
#     # Test case 2: Array with all same values
#     same_values = np.full(5, 10)
#     result = normalize_min_max(same_values, 0, 1, np.float32)
#     assert np.allclose(result, np.full(5, 0, dtype=np.float32))
#
#     # Test case 3: Empty array
#     empty_array = np.array([], dtype=np.float64)
#     result = normalize_min_max(empty_array, 0, 1, np.float32)
#     assert result.size == 0
#     assert result.dtype == np.float32


def test_normalize_min_max_dtype_conversions():
    # Test various dtype conversions
    dtypes_to_test = [
        (np.uint8, 0, 255),
        (np.uint16, 0, 65535),
        (np.int8, -128, 127),
        (np.int16, -32768, 32767),
        (np.int32, -2147483648, 2147483647),
        (np.float32, -1.0, 1.0),
        (np.float64, -1.0, 1.0)
    ]

    input_array = np.linspace(-100, 100, 1000)

    for dtype, lower, upper in dtypes_to_test:
        result = normalize_min_max(input_array, lower, upper, dtype)
        assert result.dtype == dtype
        assert result.min() >= lower
        assert result.max() <= upper


def test_normalize_min_max_invalid_inputs():
    # Test case 1: Invalid input type
    with pytest.raises(TypeError):
        normalize_min_max("not an array", 0, 1, np.float32)

    # Test case 2: Invalid target dtype
    with pytest.raises(TypeError):
        normalize_min_max(np.array([1, 2, 3]), 0, 1, "not a dtype")

    # Test case 3: v_max <= v_min
    with pytest.raises(ValueError):
        normalize_min_max(np.array([1, 2, 3]), 0, 1, np.float32, v_min=5, v_max=5)

    with pytest.raises(TypeError):
        array = np.array([[ 0.+0.j,  0.+0.j], [ 0.+0.j,  0.+0.j]])
        normalize_min_max(array, 0, 1, np.float32)


def test_normalize_min_max_multidimensional():
    # Test with 2D array
    array_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result = normalize_min_max(array_2d, 0, 255, np.uint8)
    assert result.shape == array_2d.shape
    assert result.dtype == np.uint8

    # Test with 3D array
    array_3d = np.random.rand(3, 4, 5)
    result = normalize_min_max(array_3d, -1, 1, np.float32)
    assert result.shape == array_3d.shape
    assert result.dtype == np.float32
    assert np.all((result >= -1) & (result <= 1))


def test_normalize_min_max_precision():
    # Test precision handling
    float_array = np.array([0.123456789, 0.987654321])

    # Test float32 precision
    result32 = normalize_min_max(float_array, 0, 1, np.float32)
    assert result32.dtype == np.float32

    # Test float64 precision
    result64 = normalize_min_max(float_array, 0, 1, np.float64)
    assert result64.dtype == np.float64

    # float64 should preserve more decimal places than float32
    assert np.abs(result64[0] - result32[0]) < 1e-6


def test_linear_map_dtype_same_type():
    # Test when input and target dtype are the same
    array = np.array([1, 2, 3], dtype=np.int32)
    result = linear_map_dtype(array, np.int32)
    assert np.array_equal(result, array)
    assert result.dtype == np.int32


@pytest.mark.parametrize(("src_type", "target_type"), (
        (np.int8, np.int16),
        (np.uint8, np.uint16),
        (np.uint16, np.uint32),
        (np.int8, np.uint8)
))
def test_linear_map_dtype_integer_conversions(src_type, target_type):
    # Create array with full range of source type
    info = np.iinfo(src_type)
    array = np.array([info.min, (info.max+info.min) / 2, info.max], dtype=src_type)

    result = linear_map_dtype(array, target_type)

    # Check dtype
    assert result.dtype == target_type

    # Check that relative ordering is preserved
    assert np.all(np.diff(result.astype(np.float64)) > 0)

    # Check bounds
    target_info = np.iinfo(target_type)
    assert result.min() >= target_info.min
    assert result.max() <= target_info.max


def test_linear_map_dtype_float_conversions():
    # Test float to float conversions
    src_type = np.float64
    target_type = np.float32

    array = np.array([0.0, 0.5, 1.0], dtype=src_type)
    result = linear_map_dtype(array, target_type)

    assert result.dtype == target_type
    assert np.allclose(result, array)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_linear_map_dtype_integer_to_float():
    # Test integer to float conversion
    int_types = [np.int8, np.int16, np.int32, np.uint8, np.uint16]
    float_types = [np.float32]

    for int_type in int_types:
        for float_type in float_types:
            info = np.iinfo(int_type)
            array = np.array([info.min, (info.max + info.min) / 2, info.max], dtype=int_type)

            result = linear_map_dtype(array, float_type)

            assert result.dtype == float_type
            assert np.all((result >= 0.0) & (result <= 1.0))
            assert np.allclose(result[1], 0.5, atol = 1/(info.max-info.min))  # Middle value should be mapped to 0.5


def test_linear_map_dtype_float_to_integer():
    # Test float to integer conversion
    array = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    # Test conversion to various integer types
    int_types = [np.int8, np.int16, np.uint8, np.uint16]

    for int_type in int_types:
        result = linear_map_dtype(array, int_type)
        info = np.iinfo(int_type)

        assert result.dtype == int_type
        assert result.min() >= info.min
        assert result.max() <= info.max

        # Check that relative ordering is preserved
        assert np.all(np.diff(result.astype(np.float64)) > 0)


def test_linear_map_dtype_edge_cases():
    # Test empty array
    empty_array = np.array([], dtype=np.float32)
    result = linear_map_dtype(empty_array, np.int32)
    assert result.size == 0
    assert result.dtype == np.int32

    # Test single value array
    single_value = np.array([42], dtype=np.int32)
    result = linear_map_dtype(single_value, np.float32)
    assert result.dtype == np.float32
    assert result.size == 1

    # Test array with all same values
    same_values = np.full(5, 10, dtype=np.int16)
    result = linear_map_dtype(same_values, np.uint8)
    assert result.dtype == np.uint8
    assert np.all(result == result[0])


def test_linear_map_dtype_invalid_inputs():
    # Test invalid input dtype
    with pytest.raises(TypeError):
        array = np.array(['a', 'b', 'c'])
        linear_map_dtype(array, np.float32)

    # Test invalid target dtype
    with pytest.raises(TypeError):
        array = np.array([1, 2, 3], dtype=np.int32)
        linear_map_dtype(array, np.complex64)

    # Test invalid input type
    with pytest.raises(ValueError):
        linear_map_dtype([1, 2, 3], np.float64)


def test_linear_map_dtype_multidimensional():
    # Test with 2D array
    array_2d = np.array([[0, 127, 255], [64, 192, 128]], dtype=np.uint8)
    result = linear_map_dtype(array_2d, np.float32)

    assert result.shape == array_2d.shape
    assert result.dtype == np.float32
    assert np.all((result >= 0.0) & (result <= 1.0))

    # Test with 3D array
    array_3d = np.random.randint(0, 255, size=(3, 4, 5), dtype=np.uint8)
    result = linear_map_dtype(array_3d, np.int16)

    assert result.shape == array_3d.shape
    assert result.dtype == np.int16


def test_normalize_self_integer_types():
    # Test with various integer types and their typical value ranges
    test_cases = [
        (np.int8, [2, 10, 30]),
        (np.int16, [-100, 0, 500]),
        (np.uint8, [50, 100, 200]),
        (np.uint16, [1000, 2000, 3000])
    ]

    for dtype, values in test_cases:
        # Create array with values well within the dtype limits
        array = np.array(values, dtype=dtype)
        info = np.iinfo(dtype)
        result = normalize_self(array)

        # Check dtype preservation
        assert result.dtype == dtype

        # Check that values are scaled to full dtype range
        expected = np.interp(array, [array.min(), array.max()], [info.min, info.max])
        np.testing.assert_array_equal(result, expected.astype(dtype))


def test_normalize_self_float_types():
    float_types = [np.float32, np.float64]
    test_arrays = [
        [-10.5, 0.0, 5.7],  # Mixed positive/negative
        [1.5, 2.0, 4.8],  # All positive
        [-5.0, -3.0, -1.0],  # All negative
        [100.5, 200.8, 300.1]  # Large values
    ]

    for dtype in float_types:
        for values in test_arrays:
            array = np.array(values, dtype=dtype)
            result = normalize_self(array)

            # Check dtype preservation
            assert result.dtype == dtype

            # Check normalization to [0, 1]
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

            # Check if relative ordering and spacing is preserved
            min_val, max_val = array.min(), array.max()
            expected = (array - min_val) / (max_val - min_val)
            np.testing.assert_allclose(result, expected, rtol=1e-6 if dtype == np.float32 else 1e-15)


def test_normalize_self_multidimensional():
    # Test 2D integer array
    array_2d = np.array([
        [10, 20, 30],
        [40, 50, 60]
    ], dtype=np.uint8)
    result_2d = normalize_self(array_2d)
    assert result_2d.shape == array_2d.shape
    assert result_2d.dtype == np.uint8

    # Test 3D float array
    array_3d = np.array([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]]
    ], dtype=np.float32)
    result_3d = normalize_self(array_3d)
    assert result_3d.shape == array_3d.shape
    assert result_3d.dtype == np.float32
    assert np.all((result_3d >= 0.0) & (result_3d <= 1.0))


def test_normalize_self_invalid_inputs():
    invalid_arrays = [
        (np.array([1 + 2j, 3 + 4j], dtype=np.complex64), TypeError),
        (np.array(['a', 'b', 'c']), TypeError)
    ]

    for array, expected_error in invalid_arrays:
        with pytest.raises(expected_error):
            normalize_self(array)


def test_normalize_base_same_dtype():
    """Test when input array already has target dtype"""
    # Test with float32
    array = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    result = _normalize_base(array, np.float32)
    assert np.array_equal(result, array)
    assert result.dtype == np.float32

    # Test with int32
    array = np.array([10, 20, 30], dtype=np.int32)
    result = _normalize_base(array, np.int32)
    assert np.array_equal(result, array)
    assert result.dtype == np.int32


def test_normalize_base_to_float():
    """Test normalization to floating point types"""
    test_cases = [
        (np.array([0, 128, 255], dtype=np.uint8), np.float32),
        (np.array([-32768, 0, 32767], dtype=np.int16), np.float64)
    ]

    for array, target_dtype in test_cases:
        result = _normalize_base(array, target_dtype)
        assert result.dtype == target_dtype
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 1.0)


def test_normalize_base_to_int():
    """Test normalization to integer types"""
    # Test with values already in [0,1] range
    float_array = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    for dtype in [np.uint8, np.int16]:
        result = _normalize_base(float_array, dtype)
        info = np.iinfo(dtype)
        assert result.dtype == dtype
        assert np.isclose(result[0], info.min)
        assert np.isclose(result[-1], info.max)

    # Test with values outside [0,1] range
    float_array = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    for dtype in [np.uint8, np.int16]:
        result = _normalize_base(float_array, dtype)
        info = np.iinfo(dtype)
        assert result.dtype == dtype
        assert np.all((result >= info.min) & (result <= info.max))


def test_normalize_base_multidimensional():
    """Test with multidimensional arrays"""
    # 2D array to float
    array_2d = np.array([[0, 128, 255], [128, 255, 0]], dtype=np.uint8)
    result = _normalize_base(array_2d, np.float32)
    assert result.shape == array_2d.shape
    assert result.dtype == np.float32
    assert np.all((result >= 0.0) & (result <= 1.0))

    # 3D array to int
    array_3d = np.random.uniform(0, 1, size=(2, 3, 4)).astype(np.float32)
    result = _normalize_base(array_3d, np.uint8)
    assert result.shape == array_3d.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))


def test_normalize_base_invalid_inputs():
    """Test invalid input handling"""
    invalid_arrays = [
        (np.array([1 + 2j, 3 + 4j], dtype=np.complex64), np.float32),
        (np.array(['a', 'b', 'c']), np.float32),
    ]

    for array, target_dtype in invalid_arrays:
        with pytest.raises((TypeError, ValueError)):
            _normalize_base(array, target_dtype)


def test_normalize_base_special_values():
    """Test handling of special floating point values"""
    array = np.array([np.inf, -np.inf, np.nan, 1.0, 0.0], dtype=np.float64)

    # To float
    result_float = _normalize_base(array, np.float32)
    assert result_float.dtype == np.float32
    assert np.all(np.isfinite(result_float[~np.isnan(result_float)]))

    # To int
    result_int = _normalize_base(array, np.uint8)
    assert result_int.dtype == np.uint8
    assert np.all((result_int >= 0) & (result_int <= 255))

