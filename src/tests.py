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
