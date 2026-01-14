from functools import cached_property

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from GSEGUtils.base_arrays import BaseArray


class TestPydantic:
    def test_initialisation_of_multiple_instances(self):
        a = np.random.rand(10, 3)
        b = a + 3
        c = b * 2
        d = c / 4.2

        a = BaseArray(arr=a)
        b = BaseArray(arr=b)
        c = BaseArray(arr=c)
        d = BaseArray(arr=d)

        assert a is not b
        assert b is not c
        assert c is not d

        assert a.arr is not b.arr
        assert b.arr is not c.arr
        assert c.arr is not d.arr

        assert not np.allclose(a, b)
        assert not np.allclose(b, c)
        assert not np.allclose(c, d)

        assert a.model_config == b.model_config
        assert b.model_config == c.model_config
        assert c.model_config == d.model_config

    def test_exclude_on_dump(self):
        class A(BaseArray):
            num: int = Field(default=1)
            name: str | None = Field(default=None, exclude=True)

        a = A(arr=np.random.rand(10, 3), name="test")
        assert "test" == a.name
        a.name = "abcdef"

        assert "name" in a.__dict__.keys()

        dumped = a.model_dump()
        assert "name" not in dumped.keys()

        pydantic_copy = a.model_copy()
        base_array_copy = a.copy()

        assert "abcdef" == pydantic_copy.name

        # Model_copy by default would return the excluded values
        assert hasattr(pydantic_copy, "name")
        assert "abcdef" == pydantic_copy.name
        assert hasattr(base_array_copy, "name")
        assert getattr(base_array_copy, "name") is None

    def test_exclude_on_model_copy(self):
        class A(BaseModel):
            num: int = Field(default=1)
            name: str = Field(default="test", exclude=True)

        a = A()

        assert "name" in a.__dict__.keys()
        assert "test" in a.__dict__.values()

    def test_attributes_in_dict(self):
        class A(BaseModel):
            num: int = Field(default=1)
            name: str = Field(default="test", exclude=True)
            arr: list[int] = Field(default_factory=lambda: [1, 2, 3])

        a = A()

        assert len(a.__dict__.keys()) == 3
        for name in ("num", "name", "arr"):
            assert name in a.__dict__.keys()

        assert "method_not_in_dict" not in a.__dict__.keys()

    def test_model_config_overwrite(self):
        class A(BaseModel):
            model_config = ConfigDict()
            num: int = 1

        class B(A):
            model_config = ConfigDict(strict=True, frozen=True)

        a = A(num="1")
        assert a.num == 1
        a.num = 2
        assert a.num == 2

        # Strict now acting
        with pytest.raises(ValidationError):
            B(num="2")

        b = B(num=2)

        with pytest.raises(ValidationError):
            b.num = 1

    def test_cached_properties_excluded(self):
        class A(BaseModel):
            name: str = "abc"
            num: int = 1

            @cached_property
            def my_prop(self) -> int:
                return 14 + 14

        class B(BaseModel):
            name: str = "abc"
            num: int = 1

            @computed_field
            @cached_property
            def my_prop(self) -> int:
                return 14 + 14

        a = A()
        b = B()
        # Cached property not initially in dict
        assert "my_prop" not in a.__dict__
        assert "my_prop" not in b.__dict__
        assert hasattr(a, "my_prop")
        assert hasattr(b, "my_prop")

        c = a.my_prop
        d = b.my_prop
        assert c == d == 28
        assert "my_prop" in a.__dict__
        assert "my_prop" in b.__dict__

        a_copy = a.model_copy(deep=True)
        a_dumped = a.model_dump()

        b_copy = b.model_copy(deep=True)
        b_dumped = b.model_dump()

        # Show that the cached properties are dumped not passed onto the copy
        assert (
            "my_prop" in a_copy.__dict__.keys()
        )  # my_prop is still in the dict when using model_copy
        assert "my_prop" not in a_dumped.keys()
        assert hasattr(a, "my_prop")
        assert hasattr(a_copy, "my_prop")

        # Test that computed field dumps the property and has it as an attribute
        assert "my_prop" in b_copy.__dict__.keys()
        assert "my_prop" in b_dumped.keys()
        assert hasattr(b, "my_prop")
        assert hasattr(b_copy, "my_prop")
