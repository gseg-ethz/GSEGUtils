"""Contract tests for GSEGUtils validators / coercers.

Tests assert the documented contract without referencing any pchandler symbol.
This is the structural firewall that prevents re-coupling per D-15.
"""

import numpy as np
import pytest

from GSEGUtils.base_arrays import BaseArray
from GSEGUtils.validators import validate_in_range


class TestValidateInRangeContract:
    """Contract tests for ``validate_in_range`` — exercises all four branches."""

    def test_in_range_does_not_raise(self) -> None:
        """Values fully inside ``[target_min, target_max]`` return ``None``."""
        assert validate_in_range(np.asarray([0.0, 0.5, 1.0]), 0.0, 1.0) is None

    def test_only_min_out_raises_lower_limit_message(self) -> None:
        """Only the minimum is below target_min — raises the lower-limit branch."""
        with pytest.raises(ValueError, match="exceeds lower limit"):
            validate_in_range(np.asarray([-1.0, 0.5, 1.0]), 0.0, 1.0)

    def test_only_max_out_raises_upper_limit_message(self) -> None:
        """Only the maximum is above target_max — raises the upper-limit branch."""
        with pytest.raises(ValueError, match="exceeds upper limit"):
            validate_in_range(np.asarray([0.0, 0.5, 2.0]), 0.0, 1.0)

    def test_both_out_raises_combined_message(self) -> None:
        """Both bounds violated — raises the combined-message branch (intentional, distinct from single-side)."""
        with pytest.raises(ValueError, match="exceeds bounds"):
            validate_in_range(np.asarray([-1.0, 0.5, 2.0]), 0.0, 1.0)


class TestCoerceArrayContract:
    """Contract tests for ``BaseArray._coerce_array`` — exercises all four input paths.

    The ``_coerce_array`` classmethod is reached via ``BaseArray`` construction
    (its ``@field_validator``). Tests use ``BaseArray`` directly — the base class
    is abstract via ``abc.ABC`` but Pydantic's BaseModel construction does not
    enforce abstractness, so a direct ``BaseArray(arr=...)`` call exercises the
    coercer without needing a concrete subclass.
    """

    def _wrap(self, value: object) -> BaseArray:
        """Return a ``BaseArray`` instance built from ``value`` (exercises the coercer)."""
        return BaseArray(arr=value)  # type: ignore[arg-type]

    def test_basearray_passthrough(self) -> None:
        """Wrapping a ``BaseArray`` unwraps via ``.arr`` and re-coerces (idempotent)."""
        inner = self._wrap(np.asarray([1.0, 2.0, 3.0]))
        outer = self._wrap(inner)
        np.testing.assert_array_equal(outer.arr, np.asarray([1.0, 2.0, 3.0]))

    def test_ndarray_passthrough(self) -> None:
        """Wrapping a plain ndarray stores the same data (with atleast_1d guarantee)."""
        wrapped = self._wrap(np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(wrapped.arr, np.asarray([1.0, 2.0, 3.0]))
        assert wrapped.arr.ndim >= 1

    def test_scalar_promotes_to_1d(self) -> None:
        """A scalar is promoted to a 1-D ndarray (atleast_1d guarantee)."""
        wrapped = self._wrap(5.0)
        assert wrapped.arr.ndim >= 1
        assert wrapped.arr.shape == (1,)

    def test_list_of_list_becomes_ndarray(self) -> None:
        """A nested list path through ``np.asarray`` + ``atleast_1d`` yields a 2-D ndarray."""
        wrapped = self._wrap([[1.0, 2.0], [3.0, 4.0]])
        assert wrapped.arr.ndim >= 2
        np.testing.assert_array_equal(wrapped.arr, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
