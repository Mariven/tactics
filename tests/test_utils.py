"""
Utilities for testing properties.
"""

# tests/test_utils.py
from typing import Callable, Any
import pytest

TestFn = Callable[[Any], bool]

def assert_prop(prop: TestFn, name: str = "") -> Callable[[Any], None]:
    """Create a test function that asserts a property holds"""
    def test(x: Any) -> None:
        assert prop(x), f"Property {name or prop.__name__} failed for input {x}"
    return test

# Property combinators
def and_then(prop1: TestFn, prop2: TestFn) -> TestFn:
    return lambda x: prop1(x) and prop2(x)

def or_else(prop1: TestFn, prop2: TestFn) -> TestFn:
    return lambda x: prop1(x) or prop2(x)

def not_(prop: TestFn) -> TestFn:
    return lambda x: not prop(x)
