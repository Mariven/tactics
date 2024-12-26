"""
Symbolic computation class that enables lazy evaluation of expressions.
"""
from __future__ import annotations

from supertypes import *

import pytest
from hypothesis import given, strategies as st


JUMP_OVER_EXCEPTIONS = None  # (ValueError, TypeError)
def record_call(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs) -> Any:
        print(f"Calling {fn.__name__} with args={args}, kwargs={kwargs} ...")
        try:
            value = fn(*args, **kwargs)
        except Exception as e:
            if JUMP_OVER_EXCEPTIONS is None or not isinstance(e, JUMP_OVER_EXCEPTIONS):
                raise
            print(f"\tJumped over {e.__class__.__name__} {e}")
            return None
        print(f"\tReturned {value} from {fn.__name__}")
        return value
    return wrapper


# Test cases using Hypothesis
@record_call
@given(x=st.integers(), y=st.integers())
def test_basic_arithmetic(x: int, y: int) -> None:
    print("test_basic_arithmetic", x, y)
    _x = Symbol('x')
    _y = Symbol('y')
    expr = _x + _y * 2
    assert expr(x=x, y=y) == x + y * 2

@record_call
@given(x=st.integers(), y=st.integers())
def test_comparison(x: int, y: int) -> None:
    print("test_comparison", x, y)
    _x = Symbol('x')
    _y = Symbol('y')
    expr = _x > _y
    assert expr(x=x, y=y) == (x > y)

@record_call
@given(x=st.integers())
def test_unary_operations(x: int) -> None:
    print("test_unary_operations", x)
    _x = Symbol('x')
    expr = -_x
    assert expr(x=x) == -x

@record_call
@given(data=st.lists(st.tuples(st.integers(), st.integers())))
def test_getitem(data: list[tuple[int, int]]) -> None:
    print("test_getitem", data)
    this = Symbol('this')
    first_element = this[0]
    mapped = map(first_element, data)
    assert list(mapped) == [item[0] for item in data]

@record_call
@given(x=st.integers(), y=st.integers())
def test_getattr(x: int, y: int) -> None:
    print("test_getattr", x, y)

    class Point:
        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    _p = Symbol('p')
    expr = _p.x + _p.y
    p = Point(x, y)
    assert expr(p=p) == x + y

@record_call
@given(x=st.integers(), y=st.integers())
def test_partial_evaluation(x: int, y: int) -> None:
    print("test_partial_evaluation", x, y)
    _x = Symbol('x')
    _y = Symbol('y')
    expr = _x + _y
    partial_expr = expr(x=x)
    assert partial_expr(y=y) == x + y

@record_call
@given(x=st.integers(), y=st.integers())
def test_persistent_values(x: int, y: int) -> None:
    print("test_persistent_values", x, y)
    _x = Symbol('x')
    _y = Symbol('y')
    expr = _x + _y
    expr._update_persistent({'x': x})
    assert expr(y=y) == x + y

@record_call
def test_error_handling() -> None:
    print("test_error_handling")
    _x = Symbol('x')
    with pytest.raises(ValueError):
        _x(y=1)  # Missing value for x

test_basic_arithmetic()
test_comparison()
test_unary_operations()
test_getitem()
test_getattr()
test_partial_evaluation()
test_persistent_values()
test_error_handling()

# Example usage (can be moved to documentation or a separate test file)
_x = Symbol('x')
_y = Symbol('y')
_z = Symbol('z')
_w = Symbol('w')

# Define expressions
exp = (_x**2 + _y**2)**0.5
print('exp(x=3, y=4) =', exp(x=3, y=4))  # Output: 5.0

exp2 = _z / (1 + _w)
exp3 = -exp**(-exp2)
result = exp3(x=3, y=4, z=5, w=0.5)
print('exp3(x=3, y=4, z=5, w=0.5) =', result)  # apparently, -0.004678

# Using positional arguments
print('(_x + 1)(4) =', (_x + 1)(4))  # Output: 5

# Mapping over a list
this = Symbol('this')
first_element = this[0]
data = [(1, 2), (3, 4), (5, 6)]
mapped = map(first_element, data)
print('First elements:', list(mapped))  # Output: [1, 3, 5]
