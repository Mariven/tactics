"""
Test configuration and fixtures.
"""

# tests/conftest.py
import pytest
from typing import TypeVar, Callable, Any

T = TypeVar('T')
TestFn = Callable[[Any], bool]

# Your fixture utilities
def fixture(obj: T) -> Callable[[], T]:
    @pytest.fixture
    def _fixture() -> T:
        return obj
    return _fixture

# Common test objects that will be available to all tests
test_objects = {
    "dict_a1b2": {"a": 1, "b": 2},
    "list_123": [1, 2, 3],
    "empty_list": [],
    "str_hello": "hello",
    "int_100": 100,
}

# Register fixtures globally
globals().update({
    name: fixture(obj)
    for name, obj in test_objects.items()
})
