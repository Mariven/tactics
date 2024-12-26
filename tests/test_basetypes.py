"""
Tests for the basetypes module.
"""

import pytest
from typing import TypeVar, Callable, Any
from functools import partial
from .test_utils import assert_prop, and_then, not_
from src.basetypes import is_container, is_concrete, is_typelike, join, schema_object, chain


T = TypeVar('T')
TestFn = Callable[[Any], bool]

def fixture(obj: T) -> Callable[[], T]:
    """Create a pytest fixture that always returns the same object"""
    @pytest.fixture
    def _fixture() -> T:
        return obj
    return _fixture

def assert_prop(prop: TestFn, name: str = "") -> Callable[[T], None]:
    """Create a test function that asserts a property holds for a given input"""
    def test(x: T) -> None:
        assert prop(x), f"Property {name or prop.__name__} failed for input {x}"
    return test

# Properties we want to test
props = {
    "is_container": is_container,
    "is_concrete": is_concrete,
    "is_typelike": is_typelike,
}

# Common test objects
test_objects = {
    "dict_a1b2": {"a": 1, "b": 2},
    "list_123": [1, 2, 3],
    "empty_list": [],
    "str_hello": "hello",
    "int_100": 100,
}

# Expected results for each property and test object
expected = {
    ("is_container", "dict_a1b2"): True,
    ("is_container", "list_123"): True,
    ("is_container", "empty_list"): True,
    ("is_container", "str_hello"): False,
    ("is_container", "int_100"): False,

    ("is_concrete", "dict_a1b2"): True,
    ("is_concrete", "list_123"): True,
    ("is_concrete", "empty_list"): False,
    ("is_concrete", "str_hello"): True,
    ("is_concrete", "int_100"): True,

    ("is_typelike", "dict_a1b2"): False,
    ("is_typelike", "list_123"): False,
    ("is_typelike", "empty_list"): False,
    ("is_typelike", "str_hello"): False,
    ("is_typelike", "int_100"): False,
}

# Basic property tests
def test_container_properties(dict_a1b2, list_123, str_hello, int_100) -> None:
    """Test container properties for various inputs"""
    assert is_container(dict_a1b2)
    assert is_container(list_123)
    assert not is_container(str_hello)
    assert not is_container(int_100)


# Automatically generate fixtures for all test objects
fixtures = {name: fixture(obj) for name, obj in test_objects.items()}

# Automatically generate test functions
def make_test(prop_name: str, obj_name: str) -> Callable[[], None]:
    """Create a test function for a given property and test object"""
    prop = props[prop_name]
    expected_result = expected.get((prop_name, obj_name))
    if expected_result is None:
        return None

    def test(request: pytest.FixtureRequest) -> None:
        obj = request.getfixturevalue(obj_name)
        assert prop(obj) == expected_result, \
            f"{prop_name}({obj}) should be {expected_result}"

    test.__name__ = f"test_{prop_name}_{obj_name}"
    return test

# Generate all test combinations
for prop_name in props:
    for obj_name in test_objects:
        test_fn = make_test(prop_name, obj_name)
        if test_fn:
            globals()[f"test_{prop_name}_{obj_name}"] = test_fn


# Property-based tests
container_but_not_type = and_then(is_container, not_(is_typelike))

@pytest.mark.parametrize("obj", [
    [1, 2, 3],
    {"a": 1, "b": 2},
    (1, 2, 3),
])
def test_container_but_not_type(obj) -> None:
    assert container_but_not_type(obj)

# Algebraic property tests
def test_join_properties(dict_a1b2, list_123, str_hello) -> None:
    # Commutativity
    assert join(dict_a1b2, list_123) == join(list_123, dict_a1b2)

    # Associativity
    assert join(join(dict_a1b2, list_123), str_hello) == \
           join(dict_a1b2, join(list_123, str_hello))

    # Schema compatibility
    assert schema_object(join(dict_a1b2, list_123)) == \
           join(schema_object(dict_a1b2), schema_object(list_123))

# We can also test properties that should hold across multiple inputs
def test_join_commutativity(dict_a1b2, list_123) -> None:
    """Test that join is commutative for different input types"""
    assert join(dict_a1b2, list_123) == join(list_123, dict_a1b2)

# Test algebraic properties
def test_join_associativity(dict_a1b2, list_123, str_hello) -> None:
    """Test that join is associative"""
    assert join(join(dict_a1b2, list_123), str_hello) == \
           join(dict_a1b2, join(list_123, str_hello))

# Test composition of operations
def test_schema_join_compatibility(dict_a1b2, list_123) -> None:
    """Test that schema_object and join work together as expected"""
    assert schema_object(join(dict_a1b2, list_123)) == \
           join(schema_object(dict_a1b2), schema_object(list_123))

# Property-based tests for more complex properties
def is_idempotent(x: Any) -> bool:
    """Test if join is idempotent for a given input"""
    return join(x, x) == x

# We can create higher-order test generators
def test_property_for_all(*objects: Any) -> Callable[[TestFn], None]:
    """Create a test that checks if a property holds for all given objects"""
    def test_all(prop: TestFn) -> None:
        for obj in objects:
            assert prop(obj), f"Property failed for {obj}"
    return test_all

def test_chain() -> None:
    from src.basetypes import chain

    @chain
    def example_fn(x: int, y: str, factor: float = 1.0) -> str:
        return f"{int(x * factor)}-{y.upper()}"

    # no arguments provided => partially applied
    step1 = example_fn()
    # pass integer => matched to x
    step2 = step1(5)
    # pass string => matched to y
    step3 = step2("hello")
    # pass factor => final call
    result = step2("hello", factor=2.0)
    assert result == "10-HELLO"

def test_fail_on_error() -> None:
    from src.basetypes import fail_on_error

    def risky(x: int) -> int:
        return 10 // x

    safe_risky = fail_on_error(risky, fail_value=-1)

    assert safe_risky(2) == 5
    # dividing by zero => exception => returns fail_value
    assert safe_risky(0) == -1
