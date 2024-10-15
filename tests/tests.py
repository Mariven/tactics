"""Tests for src modules"""

from .utilities import *

import inspect
import functools
import types
import re
from typing import Any, Dict, List, Set, Tuple, Union, get_args, get_origin, TypeVar, Callable, Optional, get_type_hints
import operator

F = TypeVar('F', bound=Callable[..., Any])
# defer_kwargs

@defer_kwargs
def east_factory(strict: bool = True) -> Callable[[F], F]:
	"""
	A decorator factory that creates an 'east' decorator.
	Args:
		strict (bool, optional): A flag to set strictness. Defaults to True.
	Returns:
		Callable: A decorator function.
	"""
	def east_decorator(func: F) -> F:
		@functools.wraps(func)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			print(f"east_factory(strict={strict})")
			return func(*args, **kwargs)
		return wrapper
	return east_decorator

@defer_kwargs
def west_factory(strict: bool = True, ignore: Any = None) -> Callable[[F], F]:
	"""
	A decorator factory that creates a 'west' decorator.
	Args:
		strict (bool, optional): A flag to set strictness. Defaults to True.
		ignore (Any, optional): A value to ignore. Defaults to None.
	Returns:
		Callable: A decorator function.
	"""
	def west_decorator(func: F) -> F:
		@functools.wraps(func)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			print(f"west_factory(strict={strict}, ignore={ignore})")
			return func(*args, **kwargs)
		return wrapper
	return west_decorator

@east_factory()
@west_factory()
def f(x: int, _custom_param: Any = 0) -> int:
	"""
	A sample function decorated with east_factory and west_factory.
	Args:
		x (int): An integer input.
		_custom_param (Any, optional): A custom parameter. Defaults to 0.
	Returns:
		int: The input multiplied by 2.
	"""
	print(f"f({x}, {_custom_param})")
	return x * 2

# Test cases
f(-1)
	# west_factory(strict=True, ignore=None)
	# east_factory(strict=True)
	# f(-1, 0)
f(3, _east_factory_strict=False, _ignore="block")
	# west_factory(strict=True, ignore=block)
	# east_factory(strict=False)
	# f(3, 0)
f(2, _west_factory_strict=False, _east_factory_strict=None, _ignore="stock")
	# west_factory(strict=False, ignore=stock)
	# east_factory(strict=None)
	# f(2, 0)
f(1, _custom_param="base", _ignore="left")
	# west_factory(strict=True, ignore=left)
	# east_factory(strict=True)
	# f(1, base)

# Caching

@jsonl_cache('data/caches/tests.jsonl', ttl=100)
def expensive_function(x: int, y: int, func: Callable[[int], int]) -> int:
    """This is an expensive function that takes another function as an argument."""
    # Simulate an expensive operation
    time_sleep(5)
    return func(x + y)

@jsonl_cache('data/caches/tests.jsonl', ttl=5, allow_initialize=True)
def square(n: int) -> int:
    """Square the input."""
    return n * n

expensive_function(5, 10, id)



@router
def spline(x: float, coords: List[Tuple[int, int]], extra: str) -> Any:
	print(f"x: {x}, coords: {coords}, extra: {extra}")

# Test cases
spline(9.4, [(0, 0)], 'a')  # Should work normally
spline([(0, 0)], 9.4, 'b')  # Should route arguments correctly
spline('c', x=9.4, coords=[(0, 0)])  # Should work with explicit naming
spline('d', 9.4, coords=[(0, 0)])  # Should work with partial explicit naming
spline('e', [(0, 0)], x=9.4)  # Should work with mixed routing and explicit naming
