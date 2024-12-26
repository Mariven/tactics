"""
Provides a set of interlacing wrappers for Python's functions and container types

Defines types:
    Type variables
        T, X, Y  # these are generic type variables
        T1, Ts   # e.g. T -> T1 for specifications, T, Ts -> T1 for modifications
        KT, VT   # for dictionaries
    Type aliases
        Func                Callable[..., Any]
        End[T]              Callable[[T], T]
        Hom[X, Y]           Callable[[X], Y]
        Decorator           Callable[[F], F]
        Object              dict[str, Any]
        Options             Object | None
        Pipe                Callable[[str, Options], str]

    map, filter             (f: Callable, L: list) -> list
    id                      (x: Any) -> Any
    comp                    (*args: Any) -> Fun
    reduce                  (oper: Callable[[T, T], T], values: list[T], base: T) -> T
    const                   (c: T) -> Callable[..., T]
    sub                     (pattern: str, oldstr: str, newstr: str) -> str
    get_attr                (obj: Any, attr: str) -> Any
    ...read, readlines, nonempty, cmd, get_attr
"""
from __future__ import annotations

import inspect
import re
import types
import typing

from pydantic import BaseModel

from typing import TypeVar, TypeAlias, Any, Generic, Literal, overload
from typing import get_args, get_origin, get_type_hints

from collections.abc import Hashable, Iterable, Generator, Callable, Iterator, Mapping, Container, Sequence, Coroutine

from functools import wraps, reduce

T, T1, X, Y = TypeVar('T'), TypeVar('T1'), TypeVar('X'), TypeVar('Y')
KT, VT = TypeVar('KT', bound=Hashable), TypeVar('VT')
End = Callable[[T], T]
Func = Callable[..., Any]
Typelike: TypeAlias = Any  # types._GenericAlias?
Decorator = End[Func]

Object = dict[str, Any]
Arguments: TypeAlias = dict[str, Any]

Options = Object | None
Pipe = Callable[[str, Options], str]

# Define a new __repr__ method responsible for displaying e.g. {"a": list[int], "b": UserData | None} instead of {"a": src.supertypes.List[int], "b": ModelMetaclass | None}
def custom_generic_alias_repr(obj: Any) -> str:
    """
    Custom representation for generic aliases.
    :param obj: The object to represent.
    :returns: A string representation of the object.
    """
    # Extract the base name of the type (e.g., 'List' from 'src.supertypes.List')
    if isinstance(obj, BaseModel):
        return obj.__class__.__name__
    base_name = getattr(obj, "_name", None) or getattr(obj, "__qualname__", None) or getattr(getattr(obj, "__origin__", None), "__name__", None) or str(obj)
    if args := getattr(obj, "__args__", None):
        new_args: list[Any] = []
        for arg in args:
            new_arg = repr(arg)
            if isinstance(arg, (type, typing._SpecialForm, typing._UnionGenericAlias)):
                new_arg = custom_generic_alias_repr(arg)
            if new_arg != "NoneType" or base_name != "Optional":
                new_args.append(new_arg)
        return f"{base_name}[{', '.join(new_args)}]"
    return base_name

# Backup the original __repr__ method
original_generic_alias_repr: Callable[[Any], str] = typing._GenericAlias.__repr__

# Temporarily override the __repr__ method
typing._GenericAlias.__repr__ = custom_generic_alias_repr


def is_type(obj: Any, t: Any) -> bool:
    """
    Checks if an object 'obj' complies with the given type 't'.
    Handles generic types (list, tuple, etc.), unions, optionals, and callables. For container types,
    each element is recursively verified against the parameterized types. For callables, basic
    annotation checks are performed.
    :param obj: The object to test.
    :param t: The type or type-like specification (including generics).
    :return: True if 'obj' is consistent with 't', False otherwise.
    """
    if t is Any:
        return True
    if isinstance(t, tuple):
        return any(is_type(obj, t_) for t_ in t)

    origin = get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        return any(is_type(obj, arg) for arg in get_args(t))
    if t is None or t is type(None):
        return obj is None

    if origin is list:
        if not isinstance(obj, list): return False
        if not get_args(t): return True
        return all(is_type(item, get_args(t)[0]) for item in obj)

    if origin is tuple:
        if not isinstance(obj, tuple): return False
        if not get_args(t): return True
        if len(get_args(t)) != len(obj): return False
        return all(is_type(obj[i], t_) for i, t_ in enumerate(get_args(t)))

    if origin is set:
        if not isinstance(obj, set): return False
        if not get_args(t): return True
        return all(is_type(item, get_args(t)[0]) for item in obj)

    if origin is dict:
        if not isinstance(obj, Mapping): return False
        if not get_args(t): return True
        if len(get_args(t)) != 2: return False
        key_type, value_type = get_args(t)
        return all(is_type(k, key_type) and is_type(v, value_type) for k, v in obj.items())

    if origin is Callable:
        if not callable(obj): return False
        if not get_args(t): return True

        return True  # TODO: check annotations

        args_type, return_type = get_args(t)

        # Get and check the callable's annotations
        try:
            annotations = obj.__annotations__
        except AttributeError:
            return args_type is ... and return_type is Any
        obj_return_type = annotations.get('return', Any)
        if not issubclass(obj_return_type, return_type):
            return False
        if args_type is ...:
            return True

        # Get and check parameter types
        sig = inspect.signature(obj)
        params = list(sig.parameters.values())
        if len(params) != len(args_type):
            return False
        for param, expected_type in zip(params, args_type):
            param_type = annotations.get(param.name, Any)
            if not is_type(param_type, expected_type):
                return False

        return True

    # For non-parameterized types, use regular isinstance
    try:
        return isinstance(obj, t)
    except TypeError:
        # Handle the case where t is a parameterized generic
        return isinstance(obj, get_origin(t) or t)


def chain(fn: Func) -> Func:
    """
    A decorator for automatic argument routing and partial application based on type hints.
    Allows arguments to be passed in any order, filling specific parameters by examining 'obj' or 'type(obj)'.
    If some arguments remain unassigned, returns a partially applied function. This effectively provides
    a form of flexible or dynamic currying.
    :param fn: The fully type-hinted function to decorate.
    :return: A function that can be invoked with arguments in any order and partial sets of arguments.
    """
    sig = inspect.signature(fn)
    parameters = list(sig.parameters.values())
    hints = typing.get_type_hints(fn)

    def make_wrapper(bound_arguments: Arguments) -> Func:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            local_bound_arguments = bound_arguments.copy()
            # Handle keyword arguments first
            for k, v in kwargs.items():
                local_bound_arguments[k] = v
            # Keep track of assigned params
            assigned_params = set(local_bound_arguments.keys())
            # Remaining parameters to be assigned
            remaining_params = [p for p in parameters if p.name not in assigned_params]
            args_list = list(args)
            while args_list:
                arg = args_list.pop(0)
                # If this is the last argument and there's one remaining parameter, fill it
                if not args_list and len(remaining_params) == 1:
                    local_bound_arguments[remaining_params[0].name] = arg
                    return fn(**local_bound_arguments)
                # Try to match arg to parameters not of type Any, first
                matched = False
                for p in remaining_params:
                    if p.name in assigned_params:
                        continue
                    param_type = hints.get(p.name, Any)
                    if param_type is Any:
                        continue  # skip Any for now
                    if is_type(arg, param_type):
                        local_bound_arguments[p.name] = arg
                        assigned_params.add(p.name)
                        remaining_params.remove(p)
                        matched = True
                        break
                if matched:
                    continue
                # Try parameters of type Any
                for p in remaining_params:
                    if p.name in assigned_params:
                        continue
                    param_type = hints.get(p.name, Any)
                    if param_type is Any:
                        local_bound_arguments[p.name] = arg
                        assigned_params.add(p.name)
                        remaining_params.remove(p)
                        matched = True
                        break
                if not matched:
                    msg = f"Cannot assign argument {arg} to any parameter"
                    raise TypeError(msg)
            # Check if all required parameters are assigned
            unassigned_params = [p for p in parameters if p.name not in assigned_params]
            missing_params = [p for p in unassigned_params if p.default == inspect.Parameter.empty and
                    p.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}]
            if not missing_params:  # All parameters are assigned; call the function
                return fn(**local_bound_arguments)
            return make_wrapper(local_bound_arguments)
        return wrapper
    return make_wrapper({})


def render_with(**kwargs) -> Decorator:
    """
    Decorator factory that processes the source code of a function, uncommenting lines that contain specific keys.
    All comments containing the keys in 'kwargs' will have their '#' prefix removed, and the key in the comment
    is replaced with kwargs[key] verbatim. Useful for toggling debug statements or injecting code aspects on-the-fly.
    For instance, if you want to inject a function that prints the current time, you can do:
    @render_with(var="time.time()")
    def my_function():
        (do something)
        # print(var)
        (do something else)
    This will turn the comment into the line `print(time.time())` where it stands.
    :param kwargs: Mappings from placeholder keys to the actual code strings to be substituted.
    :return: A decorator that transforms the function's source code before creation.
    """
    def decorator(fn: Func) -> Func:
        source = inspect.getsource(fn)
        fname = getattr(fn, '__name__', 'anonymous')
        new_name = f'_render_{fname}'
        lines = source.splitlines()
        while lines[0][0] == '@':
            lines = lines[1:]
        lines[0] = re.sub(r'^def\s+\w+\(', f'def {new_name}(', lines[0])
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                continue
            if any(k in line for k in kwargs):
                line = re.sub(r'^(\s*)#\s*', r'\1', line)
                for k, v in kwargs.items():
                    if not isinstance(v, str):
                        v = getattr(v, '__name__', str(v))
                    if k in line:
                        line = line.replace(k, v)
                lines[i] = line
        source = '\n'.join(lines)
        local_vars = {}
        exec(source, globals(), local_vars)
        return local_vars[new_name]
    return decorator

def fail_on_error(fn: Func, fail_value: Any = None) -> Func:
    """
    Wraps a function so that any exception is caught, returning 'fail_value' instead.
    This is a safe guard for operations that might throw due to type mismatch or external conditions,
    but we don't want them to propagate an exception. Instead, the function quietly returns 'fail_value'.
    :param fn: The function to wrap.
    :param fail_value: The value to return if an exception occurs.
    :return: The wrapped function that returns 'fail_value' upon any exception in 'fn'.
    """
    @wraps(fn)
    def safe_fn(*args, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as _:
            return fail_value
    return safe_fn

issubclass = fail_on_error(issubclass)

def is_typelike(obj: Any) -> bool:
    """
    Heuristic check if 'obj' is Python-type-like (including generics, union types, etc., unlike isinstance(obj, type)).
    For example, 'int | str', 'list[int]', 'Optional[SomeClass]', or plain 'int' are considered typelike.
    This function helps to differentiate actual types from instances. It does not return true on non-trivial schema objects.
    :param obj: The object to examine.
    :return: True if 'obj' appears to be a typing construct, otherwise False.
    """
    if isinstance(obj, (type, types.UnionType, typing.TypeVar, typing.ParamSpec, types.GenericAlias)):
        return True
    if hasattr(obj, "__origin__") and typing.get_origin(obj) is not None:
        return True
    if obj is Any or obj is ...:  # noqa: SIM103
        return True
    return False

class Preemptive:
    """
    Pre-emptive objects are objects that, when passed to a pre-emptable function,
    pre-empt the function's normal behavior via a special method.
    If x is an instance of a Preemptive class X and f is pre-emptable, then f(a, x, b, c=1) is pre-empted by x.
    This amounts to allowing X to modify the behavior of f and the arguments (typically of class X) that are given to f.
    """
    @classmethod
    def __sieve__(cls, arguments: Arguments) -> tuple[Arguments, Arguments]:
        # a subclass X(Preemptive) can use X.__sieve__(arguments)
        ingroup, outgroup = {}, {}
        for name, value in arguments.items():
            if isinstance(value, cls):
                ingroup[name] = value
            else:
                outgroup[name] = value
        return ingroup, outgroup


def preemptable(fn: Func) -> Func:
    """
    Marks a function as pre-emptable.
    If a function is pre-emptable, any Preemptive objects it receives can override (pre-empt) part or
    all of the function call by hooking into a structured approach that rebinds arguments. The typical usage
    relies on classes that inherit from Preemptive and define __preempt__.
    :param fn: The function to decorate as pre-emptable.
    :return: A function that can handle Preemptive objects among its arguments.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        arguments = inspect.signature(fn).bind(*args, **kwargs).arguments
        preemptors_by_class = {}  # keep track of pre-emptors by class;
        # if multiple pre-emptors of the same class are found, we'll try to
        # handle them all at once with their class's __preempt_multiple__ method
        # if it exists; if not, we'll handle them one at a time
        # (this helps us concatenate symbols)
        preemptors = []
        # this requires changing the order of pre-emption evaluation:
        # - iterate through arguments in given order
        # - if a pre-emptor is found, we'll go through all pre-emptors of that class first
        # - so if {a_i}: A, {b_i}: B, {c_i}: C, {d_i}: D, and arguments.values() = [a_1, c_1, c_2, d_1, b_1, a_2, b_2, c_3],
        # - we'll handle:
        # -   {a_1, a_2} via A.__preempt_multiple__ else {a_i}.__preempt__,
        # -   {c_1, c_2, c_3} via C.__preempt_multiple__ else {c_i}.__preempt__,
        # -   {d_1} via d_1.__preempt__,
        # -   {b_1, b_2} via B.__preempt_multiple__ else {b_i}.__preempt__
        for value in arguments.values():
            if isinstance(value, Preemptive) and hasattr(value, '__preempt__'):
                preemptors_by_class.setdefault(value.__class__, []).append(value)
                preemptors.append(value)

        if not preemptors:
            return fn(**arguments)

        def gn(**args) -> Any:
            return fn(**args)
        while preemptors:
            p = preemptors.pop(0)
            if not (pp := preemptors_by_class.get(p.__class__)):  # then it was handled when we popped the first member of its class
                continue
            gn, arguments = p.__class__.__preempt__(gn, arguments)
            del preemptors_by_class[p.__class__]
        return gn(**arguments)
    return wrapper


def is_container(x: Any) -> bool:
    """
    Check if x is considered a 'container' for iteration, ignoring strings.
    Specifically, checks for list, tuple, dict, set, or classes implementing __iter__ and __len__,
    yet not treating strings as containers here.
    :param x: The object to examine.
    :return: True if x is container-like (non-str), otherwise False.
    """
    if isinstance(x, str) or x is str:
        # Strings are not considered containers for our purposes
        return False
    if isinstance(x, Container) or issubclass(x, Container) or (hasattr(x, "__iter__") and hasattr(x, "__len__")):
        # Container classes: list, dict, set, tuple, str
        return True
    if any(isinstance(t, type) and is_container(t) for t in typing.get_args(x)):
        return is_typelike(x)
    return False

def is_concrete(x: Any) -> bool:
    """
    Determine if x is 'concrete' (non-type-like) by verifying it is neither None nor a type specification.
    Containers count as concrete only if they contain at least one wholly concrete element. For example,
    a list with no elements is not considered concrete. If x is a typed container, it's considered abstract.
    :param x: The object to check.
    :return: True if x is a concrete value, False otherwise.
    """
    if x is None:
        return False
    if is_container(x):
        if is_typelike(x):
            return False
        if len(x) == 0:
            # empty containers can be both, but we'll consider them abstract
            return False
        if isinstance(x, dict):
            return any(is_concrete(v) for v in x.values())
        return all(is_concrete(i) for i in x)
    return not is_typelike(x)


def schema_object(x: Any) -> Any:
    """
    Build a schema-like representation of 'x', called its 'schema object',  which might be an instance or a typed annotation.
    For containers, recurses into their elements to produce a parallel structure containing type
    or partial type information (or the object type if not purely abstract).
    :param x: The object or type to transform.
    :return: A schema-like object describing the structure or type of x.
    """
    if is_typelike(x):
        return x
    if is_container(x):
        if len(x) == 0:
            return type(x)
        if isinstance(x, dict):
            return {k: schema_object(v) for k, v in x.items()}
        if isinstance(x, tuple):
            # tuples are heterogeneous: each element may be a different type
            return tuple(map(schema_object, x))
        if isinstance(x, set):
            return {reduce(join, map(schema_object, x))}
        # lists are homogeneous: each element must be the same type. so we join all observed types into a single type
        return [reduce(join, map(schema_object, x))]
    return type(x)

def join(X: Any, Y: Any) -> Any:
    """
    Combine two schema/object representations into a union or merged container type.
    If either X or Y is concrete (contains actual values), attempts to unify them by converting into
    a corresponding schema. For container structures, merges them recursively.
    :param X: The first object or type representation.
    :param Y: The second object or type representation.
    :return: A unified representation capturing aspects of both X and Y.
    """
    if X is None or Y is None:
        return Y if X is None else X
    crx, cry = is_concrete(X), is_concrete(Y)
    cnx, cny = is_container(X), is_container(Y)
    if crx or cry:
        # deal with concrete types
        return join(schema_object(X) if crx else X, schema_object(Y) if cry else Y)
    if not cnx or not cny:
        # deal with non-containers
        return (type_annotation(X) if cnx else X) | (type_annotation(Y) if cny else Y)
    if len(X) == 0 or len(Y) == 0:
        # deal with empty containers
        return (type(X) | Y) if len(X) == 0 else (X | type(Y))
    # strategies for joining containers of the same type
    if isinstance(X, dict) and isinstance(Y, dict):
        keys = set(X) | set(Y)
        return {k: join(X.get(k), Y.get(k)) for k in keys}
    if isinstance(X, list) and isinstance(Y, list):
        return [join(X[0], Y[0])]
    if isinstance(X, tuple) and isinstance(Y, tuple):
        return tuple(join(x, y) for x, y in zip(X, Y))
    if isinstance(X, set) and isinstance(Y, set):
        return {join(x, y) for x, y in zip(X, Y)}
    # X and Y are non-empty containers of different types, so the best we can do is a union of the type annotations
    return type_annotation(X) | type_annotation(Y)

def type_annotation(x: Any) -> Typelike:
    """
    Construct a type annotation for 'x' by analyzing its structure.
    For built-in containers, attempts to produce a generic type annotation (e.g. list[T], dict[K, V]),
    recursing over elements to unify their types. If x is already a recognized type, returns it.
    :param x: The object to build a type annotation for.
    :return: A type-like representation summarizing x's structure.
    """
    if is_typelike(x):
        return x
    if is_container(x):
        if len(x) == 0:
            return type(x)
        if isinstance(x, dict):
            return dict[reduce(join, map(type_annotation, x.keys())), reduce(join, map(type_annotation, x.values()))]
        if isinstance(x, list):
            return list[reduce(join, map(type_annotation, x))]
        if isinstance(x, tuple):
            return tuple(map(type_annotation, x))
        if isinstance(x, set):
            return set(map(type_annotation, x))
        return [type_annotation(i) for i in x]
    return type(x)


def map(f: Callable[[X], Y], items: list[X]) -> list[Y]:
    """
    Applies a function 'f' to each element in 'items', returning a new list of results.
    :param f: A callable taking X and returning Y.
    :param items: The list of X items to process.
    :return: A new list of type Y with the function applied to each element.
    """
    return [f(i) for i in items]

def filter(f: Callable[[T], bool], items: list[T]) -> list[T]:
    """
    Filters elements in 'items' using the boolean predicate 'f'.
    Each item i is kept if and only if f(i) is True.
    :param f: A callable returning True for items to keep, False otherwise.
    :param items: The list of items to filter.
    :return: A new list of items for which f(item) was True.
    """
    return [i for i in items if f(i)]

def id(x: T) -> T:
    """
    The identity function.
    :param x: The input value.
    :returns: The input value.
    """
    return x

def comp(*args) -> Func:
    """
    Composes a series of functions.
    :param args: The functions to compose.
    :returns: A new function that is the composition of the input functions.
    """
    def _(x) -> Callable:
        n_min_comp = 2
        if len(args) == n_min_comp:
            return args[0](args[1](x))
        return comp(*args[1:])(args[0](x))
    return _

# @super_func
# def reduce(opn: Callable[[T, T], T], values: list[T], base: T) -> T:
#     """
#     Reduces a list of values using a binary operator.
#     :param opn: The binary operator to use for reduction.
#     :param values: The list of values to reduce.
#     :param base: The base value to use for the reduction.
#     """
#     if len(values) == 0:
#         return base
#     if len(values) == 1:
#         return values[0]
#     return opn(values[0], reduce(opn, values[1:], base))

def const(c: T) -> Callable[..., T]:
    """
    const(c) is a constant function that returns c regardless of its input.
    :param c: The value to return.
    :returns: The constant function.
    """
    def _(*args, **kwargs) -> T:
        return c
    return _

def sub(pattern: str, repl: str, string: str) -> str:
    """
    Returns a function that replaces occurrences of pattern in the given string with repl.
    :param pattern: The pattern to search for.
    :param repl: The replacement string.
    :returns: The substituting function.
    """
    return re.sub(pattern, repl, string)

def _read(fp: str) -> str:
    """
    Reads the contents of a file.
    :param fp: The file path.
    :returns: The contents of the file as a string.
    """
    with open(fp) as f:
        return f.read()

def _readlines(fp: str) -> list:
    """
    Reads the lines of a file.
    :param fp: The file path.
    :returns: A list of lines in the file.
    """
    with open(fp) as f:
        return list(f.readlines())

read, readlines = _read, _readlines

def nonempty(f: Func, items: list) -> list:
    """
    Applies a function to each item in a list and filters out the empty results.
    :param f: The function to apply to each item.
    :param items: The list of items.
    :returns: A new list containing the non-empty results.
    """
    return [f(y) for y in items if f(y)]

def cmd(input: str) -> Any:
    import subprocess
    results = subprocess.run(input, capture_output=True, check=False)
    return results.stdout.decode('utf-8')
