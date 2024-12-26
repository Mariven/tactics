"""
Provides a set of interlacing wrappers for Python's functions and container types
Contains:
    class Fun
        check               (obj: Any) -> bool
        form                (sig: str, call: str, fn: Callable | None = None, return_former: bool = False, wrap: bool = False) -> Callable

    class Dict(dict)
        check               (obj: Any) -> bool
        filter              (self, predicate: Callable) -> dict
        map, mapKeys        (self, f: Callable) -> dict
        pop, get            (self, key: Any, default: Any = None) -> Any
        keys, values, items (self) -> list

    class List(list)
        check               (obj: Any) -> bool
        map                 (self: list[X], f: Callable[[X], Y]) -> list[Y]
        filter              (self: list[T], f: Callable[[T], bool]) -> list[T]
        asyncMap            (func: Callable[[X], Y], arr: list[X], workers: int = 32) -> list[Y]

    class Tuple(list)
        check               (obj: Any) -> bool
        map, filter         (self, f: Callable) -> Tuple

    map, filter             (f: Callable, L: list) -> list
    id                      (x: Any) -> Any
    comp                    (*args) -> Fun
    reduce                  (opn: Callable, values: Any, base: Any) -> Any
    const                   (c: Any, x: Any) -> Any
    sub                     (pattern: str, oldstr: str, newstr: str) -> str
    get_attr                (obj: Any, attr: str) -> Any
    ...read, readlines, nonempty, match, split, strip, select, get, cmd, prnt
"""
from __future__ import annotations

from src.basetypes import *

import builtins
import functools
import inspect
import itertools
import json
import logging
import operator
from pydantic_core import core_schema
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed


class SymbolBase(type):
    """Metaclass for Symbol that automatically creates operator methods."""
    _indexer = itertools.count()

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        def create_method(method_name: str) -> Callable[[Symbol, Mask[X]], Symbol]:
            def method(self: Symbol, other: Mask[X]) -> Symbol:
                op_name = method_name.strip('_')
                func = lambda obj: getattr(obj, method_name)(other)
                return func
            return method

        print_infix_with = lambda symbol: lambda x, y: f"({x} {symbol} {y})"
        print_prefix_with = lambda symbol: lambda x: f"{symbol}({x})"
        print_postfix_with = lambda symbol: lambda x: f"({x}){symbol}"
        for method in [
                'add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'pow',
                'lshift', 'rshift', 'and', 'xor', 'or',
                'neg', 'pos', 'abs', 'invert',
                'lt', 'le', 'eq', 'ne', 'gt', 'ge'
                ]:
            for prefix in ['__', '__r']:
                method_name = f'{prefix}{method}__'
                attrs[method_name] = create_method(method_name)
        obj: type = super().__new__(cls, name, bases, attrs)
        return obj

Mask: TypeAlias = 'Symbol[T]' | T
# An x_: Symbol[T] is thought of as a 'generalized element' of T
#    , or a morphism with codomain T, in the same sense that x: 1 -> T is an element of T.
# t_: Mask[T] represents something to be treated as an element of T.

class Symbol(Generic[T], Preemptive, metaclass=SymbolBase):
    """
    A symbolic computation class that enables lazy evaluation of expressions. Symbols can be combined and manipulated to form
        abstract computations that are only executed when specific values are provided.
    E.g., let `_a, ..., _z = Symbol('a') ... Symbol('z') ` Then, you might write:
        - `poly = _a**2 + _3*_a + 1` for the function a=>a^2+3a+1, and call it with poly(3) to get 16
        - `f = _x.ducks * _y.geese + _x.geese` for a function you can call as `f(x = YourBirds, y = MyBirds)`
        - With `this = Symbol('this')`, `this[0]` is equivalent to `operator.itemgetter(0)`
            and can be used like `this[0]([1,2,3])` to get 1.
    Really, we're just exploiting magic methods in order to enable a clean syntax for defining expressions.
    All built-in binary functions and comparison operators are supported, and most unary operators are, too.

    Inheriting Preemptive allows symbols to act like free variables in pre-emptable functions:
        - if f(a, b, c) = a + b + c is pre-emptable and _x = Symbol('x'), then
            f(1, _x**2, 3) is a function sending x to 1 + x^2 + 3, so f(1, _x**2, 3)(5) = 1 + 5**2 + 3 = 29
    """
    @overload
    def __init__(self, symbol: str) -> None: ...

    @overload
    def __init__(self, binds: str | list[Symbol], defn: Callable[[Symbol[T], dict], T]) -> None: ...

    def __init__(self,
                 symbol: str | None = None,
                 binds: str | list[Symbol] = [],
                 defn: Callable[[Symbol[T], dict], T] = lambda s, kw: kw.get(s._symbol),
                 printer: Callable[[str | tuple[str, ...]], str] = lambda *s: "[{}]".format(", ".join(s)),
                 config: Options = {}
        ) -> None:
        """
        :param symbol: The string used to indicate the symbol. If none, the symbol is 'anonymous', but will be given a standardized name.
        :param binds: A list of Symbol objects that this one depends on.
        :param defn: Method of evaluating the expression on some parameters.
        :param printer: Method of forming a string representation of the symbol from a string representation of its binds.
        :param config: Configuration options for the Symbol object.
        """
        if not isinstance(binds, (str, list)):
            raise TypeError("binds must be either a string or a list of Symbol objects")

        self._binds: list[Symbol] = binds
        self._anonymous: bool = symbol is None
        self._defn: Callable[[Arguments], T]
        if self._anonymous:
            self._symbol: str = f'a_{next(Symbol._indexer)}'
            if len(self._binds) == 0:  # literally just called Symbol()
                raise ValueError("Can't have an anonymous Symbol with no binds")
        else:
            self._symbol: str = symbol
        self._printer: Callable[[str | tuple[str, ...]], str] = printer

        def _defn(kwargs: Arguments) -> T:
            try:
                return defn(self, kwargs)
            except Exception as e:
                msg = f"Error evaluating {self._symbol}: {e!s}"
                raise ValueError(msg) from e
        self._defn = _defn
        self._cfg = config

        self._subs: list[Symbol] = self._binds if self._anonymous or len(self._binds) > 1 else []
        self._roots: set[str] = {self._symbol} if not self._anonymous else set.union(*[sub._roots for sub in self._subs], set())
        self._arity: int = len(self._roots)
        self._router: dict[str, list[Symbol]] = {root: [] for root in self._roots}
        self._route_back: dict[str, list[Symbol]] = {root: [] for root in self._roots}
        self._order: list[str]
        if len(self._roots) == 1:
            self._order = list(self._roots)
        else:
            self._order = [self._symbol] if not self._anonymous else list(reduce(
                lambda a, b: dict.fromkeys(a, None) | dict.fromkeys(b, None),
                self._roots,
                {}
            ))

        if self._anonymous:
            self._order = sorted(self._order, key=lambda x: [x in sub._roots for sub in self._subs].index(True))
        self._cache: dict[str, Any] = {}
        self._persistent: dict[str, Any] = {}

        # Add support for lazy evaluation chains
        self._lazy_chain: list[Callable] = []
        self._transformers: dict[str, Callable] = {}

        # Establish bidirectional routing
        for sub in self._subs:
            for root in sub._roots:
                self._router[root].append(sub)
                sub._route_back[root].append(self)

        # Internal variables:
        #   _arity: number of roots
        #   _anonymous: whether the Symbol is anonymous
        #   _binds: list of Symbols that this one depends on
        #   _cache: dict mapping each root to the cached value of the Symbol
        #   _cfg: configuration options for the Symbol object
        #   _defn: method of evaluating the expression on some parameters
        #   _order: list of all Symbols that depend on this one
        #   _persistent: dict mapping each root to the persistent value of the Symbol
        #   _roots: set of all Symbols that this one depends on
        #   _route_back: dict mapping each root to a list of Symbols that it depends on
        #   _router: dict mapping each root to a list of Symbols that depend on it
        #   _subs: list of Symbols that this one depends on

    def _route(self, kwargs: Arguments) -> list[tuple[Symbol, Arguments]]:
        """
        Route kwargs to the appropriate Symbol objects.
        :param kwargs: The keyword arguments to route.
        :returns: A list of tuples, where each tuple contains a Symbol and a dictionary of keyword arguments that should be passed to it.
        """
        if self._subs:
            return [(sub, {root: val for root, val in kwargs.items() if root in sub._roots}) for sub in self._subs]
        return [(self, kwargs)]

    def _config(self, key: str, new_val: Any | None = None, default: Any | None = None) -> Any:
        """
        Get or set a configuration option for the Symbol object.
        :param key: The key to get or set.
        :param new_val: The new value to set (optional).
        :param default: The default value to return if the key is not found (optional).
        :returns: The value of the configuration option.
        """
        if new_val is not None:
            self._cfg[key] = new_val
        return self._cfg.get(key, default)

    @classmethod
    def __preempt__(cls, fn: Callable, arguments: dict) -> tuple[Callable, dict]:
        class_args, remaining_args = cls.__sieve__(arguments)

        def fn_out(**args) -> Callable[[Symbol, dict], Any]:
            def new_defn(_self: Symbol, symkwargs: dict) -> Any:
                return fn( **({k: v(**symkwargs) for k, v in class_args.items()} | args) )
            return new_defn

        def make_symbol(**args) -> Symbol:
            return Symbol(binds = list(class_args.values()), defn = fn_out(**args))
        return make_symbol, remaining_args

    def __call__(self, *args, **kwargs) -> T:
        """
        Evaluate the Symbol object with the given arguments.
        :param args: Positional arguments matching the order of variables.
        :param kwargs: Keyword arguments matching variable names.
        :return: Result of the evaluation.
        """
        try:
            if len(self._roots) == 0:
                return self._defn({})
            if len(args) + len(kwargs) > self._arity:
                if not self._cfg('extra_kwargs', default=True):
                    msg = f"Too many arguments: expected {self._arity}, got {len(args) + len(kwargs)}"
                    raise ValueError(msg)

            if len(args) == self._arity == 1:
                args, kwargs = [], {next(iter(self._roots)): args[0]}
            if args:
                for var, arg in zip(self._order, args):
                    kwargs[var] = arg
            missing_vars = self._roots - kwargs.keys()
            if missing_vars == self._roots:
                # We want to accept extraneous variables so closures can be freely passed in,
                #   but if not a single variable is a root, something is wrong.
                msg = f"No values provided for {self._symbol} by call {args, kwargs}"
                raise ValueError(msg)
            if missing_vars:
                if not self._config('partial_eval', default=True):
                    msg = f"Missing values for variables: {missing_vars}"
                    raise ValueError(msg)
                kwargs = {k: kwargs[k] for k in self._order if k in kwargs}

            # Merge with persistent values
            full_kwargs = {**self._persistent, **kwargs}

            # Evaluate and update cache
            self._cache.update(full_kwargs)

            # If ALL roots were filled, we evaluate the function directly
            if not self._roots - full_kwargs.keys():
                result = self._defn(full_kwargs)
                # Apply lazy evaluation chain
                if self._lazy_chain:
                    for func in self._lazy_chain:
                        result = func(result)
                self._cache[self._symbol] = result
                return result

            # Otherwise, we create a new symbol with the partial evaluation

            # Common pattern: to create a new Symbol from an existing one,
            #   the new definition should have _self at the start, for the new symbol to use, but
            #   it should use self._defn, not _self._defn (which would cause a circular reference).
            def new_defn(_self, kwargs: Arguments) -> T:
                return self._defn(full_kwargs | kwargs)

            sym = Symbol(binds=[self], defn=new_defn)
            sym._roots = {x for x in sym._roots if x not in full_kwargs}
            sym._arity = len(sym._roots)
            return sym

        except Exception as e:
            msg = f"Error evaluating {self._symbol}: {e!s}"
            if self._config('debug', False):
                print("Traceback:")
                import traceback
                traceback.print_exc()
            raise ValueError(msg) from e

    def maybe(self) -> Symbol[T | None]:
        """
        Convert to optional type that handles exceptions gracefully.
        :returns: A new Symbol object that returns None if evaluation fails.
        """
        def safe_eval(_self, kwargs: Arguments) -> T | None:
            try:
                return self._defn(kwargs)
            except Exception:
                return None
        return Symbol(binds=[self], defn=safe_eval)

    def default(self, value: T) -> Symbol[T]:
        """
        Provide a default value if evaluation fails.
        :param value: The default value to return if evaluation fails.
        :returns: A new Symbol object that returns the default value if evaluation fails.
        """
        def default_eval(_self, kwargs: Arguments) -> T:
            try:
                return self._defn(kwargs)
            except Exception:
                return value
        return Symbol(binds=[self], defn=default_eval)

    def guard(self, predicate: Callable[[T], bool]) -> Symbol[T]:
        """
        Add a validation predicate that must pass.
        :param predicate: The predicate function to validate the result.
        :returns: A new Symbol object that applies the predicate to the result.
        """
        def guarded_eval(_self, kwargs: Arguments) -> T:
            result = self._defn(kwargs)
            if not predicate(result):
                msg = f"Guard failed for {self._symbol}"
                raise ValueError(msg)
            return result
        return Symbol(binds=[self], defn=guarded_eval)

    def tap(self, func: Callable[[T], Any]) -> Symbol[T]:
        """
        Add a side effect without modifying the value.
        :param func: The function to apply to the result.
        :returns: A new Symbol object that applies the function to the result.
        """
        def tapped_eval(_self, kwargs: Arguments) -> T:
            result = self._defn(kwargs)
            func(result)
            return result
        return Symbol(binds=[self], defn=tapped_eval)

    def _update_persistent(self, updates: dict[str, Any]) -> None:
        """
        Update persistent values and propagate changes upwards.
        :param updates: A dictionary of updates to apply.
        """
        self._persistent.update(updates)
        for key, val in updates.items():
            for parent in self._route_back.get(key, []):
                parent._update_persistent({key: val})

    def _copy(self, original: Symbol[T]) -> Symbol[T]:
        """
        Create a shallow copy of the Symbol object.
        :param original: The Symbol object to copy.
        :returns: A new Symbol object that is a shallow copy of the original.
        """
        self._roots = original._roots.copy()
        self._router = original._router.copy()
        self._order = original._order.copy()
        return self

    def _total_copy(self, original: Symbol[T]) -> Symbol[T]:
        """
        Create a deep copy of the Symbol object.
        :param original: The Symbol object to copy.
        :returns: A new Symbol object that is a deep copy of the original.
        """
        self._anonymous = original._anonymous
        self._arity = original._arity
        self._binds = original._binds.copy()
        self._cache = original._cache.copy()
        self._cfg = original._cfg.copy()
        self._defn = original._defn
        self._order = original._order.copy()
        self._persistent = original._persistent.copy()
        self._roots = original._roots.copy()
        self._router = original._router.copy()
        self._route_back = original._route_back.copy()
        self._subs = original._subs.copy()
        self._symbol = original._symbol
        return self

    def __getitem__(self, key: Any) -> Symbol[Any]:
        def wrapped_getitem(_self, kwargs: Arguments) -> Any:
            return self._defn(kwargs)[key]
        new_printer = lambda s: f"{s}[{key}]"
        return Symbol(binds=[self], defn=wrapped_getitem, printer=new_printer)

    def __getattr__(self, name: str) -> Symbol[Any] | Any:
        if name.startswith('_'):
            # marks internal attributes
            try:
                return object.__getattribute__(self, name)
            except AttributeError as e:
                msg = f"Symbol has no attribute {name}"
                raise AttributeError(msg) from e

        def wrapped_getattr(_self, kwargs: Arguments) -> Any:
            return getattr(self._defn(kwargs), name)
        return Symbol(binds=[self], defn=wrapped_getattr, printer=lambda s: f"{s}.{name}")

    def __str__(self) -> str:
        """Return a string representation of the Symbol."""
        return f"Symbol({self._symbol})"

    def __repr__(self) -> str:
        """Return a detailed string representation of the Symbol."""
        return self._printer(*[sub.__repr__() for sub in self._binds]) if self._binds else self._symbol

    @staticmethod
    def _binary_operator(method_name: str, op_func: Callable[[T, T1], X]) -> Callable[[Symbol[T], Mask[T1]], Symbol[X]]:
        """
        Creates a binary operator method for Symbol.
        :param method_name: The name of the method.
        :param op_func: The operator function to apply.
        :returns: A function that applies the operator to Symbol objects.
        """

        def method(self: Symbol[T], other: Mask[T1]) -> Symbol[X]:
            if isinstance(other, Symbol):
                # Create a new anonymous node
                def new_defn(_self: Symbol, kwargs: Arguments) -> X:
                    try:
                        return op_func(*[sub(**kwargs) for sub, kwargs in _self._route(kwargs)])
                    except Exception as e:
                        msg = f"Error in {method_name}: {e}"
                        raise ValueError(msg) from e
                new_printer = lambda s, t: f"({self!r} {method_name.strip('_')} {other!r})"
                return Symbol(binds=[self, other], defn=new_defn, printer=new_printer)

            def new_defn(_self, kwargs: Arguments) -> X:
                try:
                    return op_func(self._defn(kwargs), other)
                except Exception as e:
                    msg = f"Error in {method_name} with constant: {e}"
                    raise ValueError(msg) from e
            new_printer = lambda s: f"({self!r} {method_name.strip('_')} {other})"
            return Symbol(binds=[self], defn=new_defn, printer=new_printer)
        method.__name__ = method_name
        return method

    @staticmethod
    def _r_binary_operator(method_name: str, op_func: Callable[[T1, T], X]) -> Callable[[Symbol[T], Mask[T1]], Symbol[X]]:
        """
        Create a reverse binary operator method for Symbol.

        :param method_name: The name of the method.
        :param op_func: The operator function to apply.
        :return: A function that applies the operator to Symbol objects.
        """
        def method(self: Symbol[T], other: Mask[T1]) -> Symbol[X]:
            def new_defn(_self, kwargs: Arguments) -> X:
                try:
                    return op_func(other, self._defn(kwargs))
                except Exception as e:
                    msg = f"Error in reverse binary operation {method_name}: {e!s}"
                    raise ValueError(msg) from e
            new_printer = lambda s: f"({other} {method_name.strip('_')} {self!r})"
            return Symbol(binds=[self], defn=new_defn, printer=new_printer)
        method.__name__ = method_name
        return method

    @staticmethod
    def _unary_operator(method_name: str, op_func: Callable[[T], X]) -> Callable[[Symbol[T]], Symbol[X]]:
        """
        Create a unary operator method for Symbol.

        :param method_name: The name of the method.
        :param op_func: The operator function to apply.
        :return: A function that applies the operator to a Symbol object.
        """
        def method(self: Symbol[T]) -> Symbol[X]:
            def new_defn(_self, kwargs: Arguments) -> X:
                try:
                    return op_func(self._defn(kwargs))
                except Exception as e:
                    msg = f"Error in unary operation {method_name}: {e!s}"
                    raise ValueError(msg) from e
            new_printer = lambda _: f"{method_name.strip('_')}({self!r})"
            return Symbol(None, binds=[self], defn=new_defn, printer=new_printer)
        method.__name__ = method_name
        return method

    @staticmethod
    def _comparison_operator(method_name: str, op_func: Callable[[T, T1], bool]) -> Callable[[Symbol[T], Mask[T1]], Symbol[bool]]:
        """
        Create a comparison operator method for Symbol.

        :param method_name: The name of the method.
        :param op_func: The operator function to apply.
        :return: A function that applies the operator to Symbol objects.
        """
        def method(self: Symbol[T], other: Mask[T1]) -> Symbol[bool]:
            if isinstance(other, Symbol):
                def new_defn(_self: Symbol[bool], kwargs: Arguments) -> bool:
                    try:
                        return op_func(*[sub(**kwargs) for sub, kwargs in _self._route(kwargs)])
                    except Exception as e:
                        msg = f"Error in comparison operation {method_name}: {e!s}"
                        raise ValueError(msg) from e
                new_symbol = Symbol(None, binds=[self, other], defn=new_defn)
                new_symbol._roots = self._roots.union(other._roots)
                new_symbol._router = {k: self._router.get(k, []) + other._router.get(k, [])
                                    for k in new_symbol._roots}
                new_symbol._order = self._order.copy()
                new_symbol._order.extend(v for v in other._order if v not in new_symbol._order)
            else:
                def new_defn(_, kwargs: Arguments) -> bool:
                    try:
                        return op_func(self._defn(kwargs), other)
                    except Exception as e:
                        msg = f"Error in comparison operation {method_name} with constant: {e!s}"
                        raise ValueError(msg) from e
                new_symbol = Symbol(None, binds=[self], defn=new_defn)
            return new_symbol
        method.__name__ = method_name
        return method

    @staticmethod
    def _r_comparison_operator(method_name: str, op_func: Callable[[T1, T], bool]) -> Callable[[Symbol[T], Mask[T1]], Symbol[bool]]:
        """
        Create a reverse comparison operator method for Symbol.

        :param method_name: The name of the method.
        :param op_func: The operator function to apply.
        :return: A function that applies the operator to Symbol objects.
        """
        def method(self: Symbol[T], other: Mask[T1]) -> Symbol[bool]:
            return Symbol._comparison_operator(method_name, op_func)(other, self)
        method.__name__ = method_name
        return method

binary_op_names = ['add', 'and_', 'floordiv', 'lshift', 'matmul', 'mod', 'mul', 'or_', 'pow', 'rshift', 'sub', 'truediv', 'xor']

ops = {
    'binary': {f"__{k}__".replace('___', '__'): getattr(operator, k) for k in binary_op_names},
    'r_binary': {f"__r{k}__".replace('___', '__'): getattr(operator, k) for k in binary_op_names},
    'unary': {f"__{k}__": getattr(operator, k) for k in ['abs', 'invert', 'neg', 'pos']
         } | {f"__{k.__name__}__": k for k in [float, int, iter, len, list, next, set, tuple]},
    'comparison': {f"__{k}__": getattr(operator, k) for k in ['lt', 'le', 'eq', 'ne', 'gt', 'ge']},
}
# Assign operator methods to Symbol
for op_type, op_dict in ops.items():
    for method_name, op_func in op_dict.items():
        setattr(Symbol, method_name, getattr(Symbol, f"_{op_type}_operator")(method_name, op_func))


this = Symbol("this")

def arm(fn_builder: Callable[..., Callable[..., T]], *fn_args, **fn_kwargs) -> Callable[..., T]:
    """
    Arms a function builder with a set of arguments to apply to built functions.
    Can be used with symbols to create dependent functions.
    Idea: map(els, this.method) returns a list of element methods, but map(els, arm(this.method)) actually computes the method at each element.
    Ex: List(strings).filter(this.strip) doesn't work, since it filters on (truthy) strip *methods*, but List(strings).filter(arm(this.strip)) actually filters on the method applied to each element, returning the desired list of non-whitespace strings.
    :param fn_builder: The function builder to arm.
    :param fn_args: The arguments to arm the function builder with.
    :param fn_kwargs: The keyword arguments to arm the function builder with.
    :returns: A new function that, when called, will produce the original function builder with the given arguments.
    """
    def fn(*args, **kwargs) -> T:
        return fn_builder(*args, **kwargs)(*fn_args, **fn_kwargs)
    return fn


class Fun:
    """
    Function wrapper that allows for FP-like manipulations and partial evaluation (via Meta objects)
        Lots of galaxy-brained notation for function manipulation:
            f * g is the composition f(g(x))
            f @ L is the map of f over L, i.e., [f(x) for x in L]
            f / g returns f(x) if that works and isn't None, else (g(x) if g callable else g)
            f ** n is the function that returns f(f(...(f(x))...)) for n iterations
            chaining: 3 >> f >> g is equivalent to g(f(3))
    """
    # function wrapper for functional programming-style syntactic sugar
    def __init__(self,
                 func: str | Func,
                 form: tuple[str, str] | None = None,
                 args: list[Any] | None = None,
                 **kwargs: Any
        ) -> None:
        """
        Initialize a Fun object.

        :param func: A function or a string representation of a function to be wrapped.
        :param form: An optional tuple containing signature and call order for function transformation.
        :param args: An optional list of arguments to be appended to the function call.
        :param kwargs: Additional keyword arguments to be used in the function call.
        """
        if args is None:
            args = []
        if isinstance(func, str):
            tokens = []
            for token in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', func):
                if token not in tokens and token not in dir(builtins) and all(y + '.' + token not in func for y in dir(builtins)):
                    tokens.append(token)
            func: Func = eval('lambda ' + ', '.join(tokens) + ': ' + func)
        i = 0
        self.func: Func = func
        recursion_limit = 10
        while isinstance(func, Fun):
            # the function being wrapped should not be a wrapper itself, but maybe there'll be a point to re-wrapping
            self.func = self.func.func
            i += 1
            if i > recursion_limit:
                raise Exception("Error instantiating superfunction: can't find original function")
        self.type = type
        self.kwargs = kwargs
        self.args = args
        if form is not None:
            self.func = Fun.form(form[0], form[1], func)
        self.arguments = list(inspect.signature(self.func).parameters.keys())

    def __name__(self) -> str:
        return str(self.func.__name__)

    def __doc__(self) -> str:
        return str(self.func.__doc__)

    def __call__(self, *args, **kwargs) -> Any:
        """
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :returns: The result of the function call
        """
        while (len(args) == 1 and isinstance(args[0], (tuple, Tuple))):
            args = args[0]
        args = list(args) + self.args
        kwargs |= self.kwargs

        if (k := len(args) + len(kwargs)) < (n := len(self.arguments)):
            selfargs = list(self.arguments)
            filled_by_args = {selfargs[i]: v for i, v in enumerate(args)}
            a = [x for x in selfargs if x not in filled_by_args and x not in kwargs]  # unprovided arguments

            def newfun(*args2, **kwargs2) -> Any:
                if len(args2) > len(a):
                    msg = f"Too many arguments provided: {args2}"
                    raise ValueError(msg)
                filled_by_args2 = {a[i]: args2[i] for i in range(len(args2))}
                return self.func(**(filled_by_args | filled_by_args2 | kwargs | kwargs2))
            f = Fun(newfun)
            f.arguments = a
            return f

        rv = self.func(*args, **kwargs)
        if callable(rv):
            return Fun(rv)
        return rv

    def __setitem__(self, name: str, value: Any) -> None:
        self.kwargs[name] = value

    def __class_getitem__(cls, name: Any) -> Any:
        # Replicate Callable.__getitem__, with a few extra cases
        # since product type domains are supported by Callable (Callable[[T, U], V] is a T x U -> V)
        #   we'll use lists to represent product types and tuples to represent function types
        #   so Fun[T, [U, V]] is a T -> U x V = Callable[T, tuple[U, V]]
        #      Fun[T, (U, V)] is a T -> (U -> V) = Callable[T, Callable[U, V]]
        #      Fun[[T, U], V] is a T x U -> V = Callable[[T, U], V]
        #      Fun[(T, U), V] is a (T -> U) -> V = Callable[Callable[T, U], V]
        # this doesn't handle nested product and function types too well yet
        #   e.g. Fun[ [T, [T, (U, T)], (U, V)], [T, (U, V)] ] should be a T x (T x (U -> T)) x (U -> V) -> (T x (U -> V))
        if is_typelike(name):
            return Callable[..., name]  # Fun[T] is a function with output type T

        def is_type_tree(T: Typelike | list | tuple) -> bool:
            if isinstance(T, (list, tuple)):
                return all(is_type_tree(x) for x in T)
            return is_typelike(T)

        if isinstance(name, tuple) and all(is_type_tree(x) for x in name):
            if len(name) == 0:
                return Callable[..., Any]  # not clear why you would do this, but it's allowed
            if len(name) == 1:
                return Callable[..., name[0]]
            if len(name) > 2:
                return Callable[[name[:-1]], name[-1]]
            # length is exactly 2
            dom, cod = name
            if isinstance(dom, tuple):
                dom = Fun[dom]
            if isinstance(cod, tuple):
                cod = Fun[cod]
            if isinstance(cod, list):
                cod = tuple[tuple(cod)]
            if isinstance(dom, list):
                return Callable[[*dom], cod]
            return Callable[[dom], cod]
        return None

    def __getitem__(self, name: Any) -> Any:
        # if a is a string, f[a] returns value of local arg a
        if isinstance(name, str):
            return self.kwargs.get(name, None)
        # if a is a list, f[a] maps f over a
        if isinstance(name, (list, List, tuple, Tuple)):
            return List([self(x) for x in name])
        return None

    def __delitem__(self, name) -> None:
        if name in self.kwargs:
            self.kwargs.__delitem__(name)

    @classmethod
    def __preempt__(cls, fn: Callable, arguments: dict) -> tuple[Callable, dict]:
        class_args, remaining_args = cls.__sieve__(arguments)
        return lambda **args: fn( **({k: v.func for k, v in class_args.items()} | args) ), remaining_args

    @staticmethod
    def check(obj: Any) -> bool:
        """
        Checks if the given object is a function or subclass of Fun.
        :param obj: The object to check.
        :returns: True if the object is a function, False otherwise.
        """
        return isinstance(obj, (Callable, Fun))

    def __mul__(self, other) -> Fun:
        """
        Composition of functions: (f * g)(x) = f(g(x))
        Mnemonic: closest symbol to circ
        :param other: The function to compose with the current function.
        :returns: A new Fun object representing the composition.
        """
        def composed(*args, **kwargs) -> Any:
            return self(other(*args, **kwargs))
        ret = Fun(composed)
        ret.arguments = other.arguments
        return ret

    def __rmul__(self, other) -> Fun:
        """
        Composition of functions: (f * g)(x) = f(g(x))
        :param other: The function to compose with the current function.
        :returns: A new Fun object representing the composition.
        """
        def composed(*args, **kwargs) -> Any:
            return other(self(*args, **kwargs))
        ret = Fun(composed)
        ret.arguments = self.arguments
        return ret

    def __add__(self, other) -> Fun:
        """
        'Horizontal' composition of functions: (f + g)(x) = (f(x), g(x))
        Compatible with chaining: 3 >> f >> g + h is equivalent to (f(g(3)), h(g(3)))
        However, only useful for functions that return a single value
        Mnemonic: f 'and' g
        :param other: The function to compose with the current function.
        :returns: A new Fun object representing the horizontal composition.
        """
        def added(*a) -> Any:
            val = self(a[0])
            combiner = lambda v, w: (*v, w) if isinstance(v, tuple) else (v, w)
            # so (f + g + h)(x) = (f(x), g(x), h(x)) rather than ((f(x), g(x)), h(x))
            # has to be asymmetric because addition is evaluated left-to-right
            if len(a) != 1:
                return combiner(val, other(a[1]))
            return combiner(val, other(a[0]))
        ret = Fun(added)
        ret.arguments = self.arguments
        return ret

    def __matmul__(self, other: Iterable) -> Iterable:
        """
        Map a function over a list: f @ L = [f(x) for x in L]
        Mnemonic: f 'at' each element of L
        :param other: The list to map the function over.
        :returns: A new list with the function applied to each element.
        """
        def composed(iterable: Iterable) -> Iterable:
            return [self.func(x) for x in iterable]
        return Fun(composed)(other)

    def __getattr__(self, arg) -> Any:
        """
        Acts as normal, with a cutout for Haskell-style composition
        :param arg: The attribute to retrieve.
        :returns: The attribute value.
        """
        if isinstance(arg, Fun):
            return self * arg
        return object.__getattribute__(self, arg)

    def __truediv__(self, other) -> Callable | None:
        """
        Function: f / g returns f(x) if that works and isn't None, else (g(x) if g callable else g)
        Mnemonic: f 'else' g
        If you want to return a callable g verbatim, use f / const(g)
        :param other: The fallback function or value.
        :returns: A new Fun object representing the conditional application.
        """
        def newfun(*args, **kwargs) -> Any:
            try:
                res = self.func(*args, **kwargs)
                assert res is not None
                return res
            except Exception:
                try:
                    return (other.func(*args, **kwargs) if isinstance(other, Fun) else other(*args, **kwargs))
                except Exception:
                    return None
        f = Fun(newfun)
        f.arguments = self.arguments
        return f

    def __pow__(self, n: int) -> Fun:
        """
        Function: repeated application; f ** n = f(f(...(f(x))...)) for n iterations
        Mnemonic: (standard mathematical notation, but with Python's power notation instead of superscript notation f^n)
        :param n: The number of times to apply the function.
        :returns: A new Fun object representing the repeated application.
        """
        if n == 0:
            return Fun(lambda x: x)
        if n < 0:
            raise ValueError("Raising a function to a negative power is not supported")
        n_recurse_at = 3
        if n < n_recurse_at:
            return [Fun(lambda x: x), self, self * self][n]
        return (self**2)**(n // 2) if n % 2 == 0 else self * (self**(n - 1))

    def __rrshift__(self, x) -> Any:
        """
        Function: 3 >> f >> g is equivalent to g(f(3))
            f >> g alone is equivalent to g compose f (sometimes written f;g)
        Mnemonic: feeding inputs into function pipelines
        :param x: The value to apply the function to.
        :returns: The result of applying the function to the value.
        """
        return self * x if isinstance(x, Fun) else self(x)

    @staticmethod
    def form(sig_order: str, call_order: str, fn: Callable | None = None, return_former: bool = False, wrap: bool = False) -> Fun:
        """
        Utility for variadic combinator constructor with generalized notation.
        :param sig_order: The order of the arguments in the function signature.
        :param call_order: The order of the arguments in the function call.
        :param fn: The function to wrap.
        :param return_former: Whether to return the higher-order forming function rather than the function it wraps.
        :param wrap: Whether to wrap the function in a Fun object.
        :returns: The wrapped function.
        """
        """
        - spacing denotes separation of composition, i.e. currying
            - ex. sig = 0 1 creates the function g(a)(b) := call[f]
            - while sig = 01 creates g($0, $1) := call[f]
            - here call[f] is the evaluation tree for the input f specified by call
        - parentheses denote sub-evaluations
            - ex. call = 01(2) creates sig[g] := f(a, b(c))
            - while call = 3 3(3(0)1)2 2(1) creates sig[g] := f(d)(d(d(a), b), c)(c(b))
        - you can't use parentheses in sig, because that's incoherent
        - so form('01 2', '201', f) returns the function g: a, b -> c -> f(c, a, b)
        - also accepts alt notation w/ wsigs instead of nums, ',' instead of juxtaposition, and ';' instead of ' ', e.g.
            - form("func, L; callback", "callback(func, L)", display)
            - is the same as form('01 2', '2(01)', display)
            - just don't try to *combine* sequence-space notation with comma-semicolon notation
        - ~all classic combinators can be expressed as forms of the identity operator `id`:
            - apply = form('0 1', '0(1)', id): (a -> b) -> a -> b
            - const = form('0 1', '0', id): a -> b -> a
            - compose = form('0 1 2', '0(1(2))', id) (b -> c) -> (a -> b) -> a -> c
            - fix = form('0', '0(0)', id): (a -> a) -> a
            - on = form('0 1 2 3', '0(1(2))(1(3))', id): (b -> b -> c) -> (a -> b) -> a -> a -> c
            - chain = form('0 1 2', '0(1(2))(2)', id): (a -> b -> c) -> (b -> a) -> b -> c
            - ap = form('0 1 2', '0(2)(1(2))', id): (a -> b -> c) -> (a -> b) -> a -> c
            - applyTo = form('0 1', '1(0)', id): a -> (a -> b) -> a
            - converge = form('0 1 2 3', '0(1(3))(2(3))', id): (b -> c -> d) -> (a -> b) -> (a -> c) -> a -> d
            - join = form('0 1', '0(1)(1)', id): (a -> a -> b) -> a -> b
        """
        rx_comma = r'[,;]'
        rx_space = r' +'
        Gr = lambda *rxs: r'(' + (r'|'.join(rxs) if len(rxs) > 1 else rxs[0]) + r')'
        rx_digit = r'[0-9]'
        rx_tchar = r'[a-zA-Z0-9_]'
        rx_non_tchar = r'[^a-zA-Z0-9_]'
        while isinstance(fn, Fun):
            fn = fn.func
        sig, call = [re.sub(Gr(rx_comma) + rx_space, r'\1', y) for y in [sig_order, call_order]]
        # remove extraneous spaces from word-notation inputs
        word_notation = bool(re.findall(rx_comma, sig))
        tokens = re.findall(rx_tchar + '+' * word_notation, sig)
        # get tokens
        initial_params = sorted(set(tokens), key=tokens.index)

        for idx, tok in enumerate(initial_params):
            # convert word-notation to number-notation
            sig = sig.replace(tok, str(idx))
            call = call.replace(tok, str(idx))

        for delim_old, delim_new in [(',', ''), (';', ' ')]:
            sig = re.sub(delim_old, delim_new, sig)
            call = re.sub(delim_old, delim_new, call)

        indices = [int(i) for i in re.findall(rx_digit, call)]
        # find all indices used in call...
        call += " "
        for x in range(max(indices) + 1):
            # and append all unused indices to its end (so 03 becomes 03 12)
            if x not in indices:
                call += str(x)
        call = "_ORIG(" + call.strip().replace(" ", ") (") + ")"
        # _ORIG(03) (12)
        sig = "lambda " + ": lambda ".join(sig.split(" ")) + ": "
        # "012 3" becomes "lambda 012: lambda 3: "
        for _ in range(len(call) + len(sig)):
            call, sig = [re.sub(Gr(rx_digit, r'\)') + Gr(rx_digit), r"\1, \2", x) for x in [call, sig]]
            # "lambda 0, 1, 2: lambda 3: " and "_ORIG(0, 3) (1, 2)"
        sig += call.replace(" ", "")
        # now the new nested lambda is fully formed, but with numerical indices
        params_list = initial_params if fn is None else list(inspect.signature(fn).parameters.keys())
        for i, j in enumerate(params_list):
            if j[0] in '0123456789_':
                j = 'x' + j
            sig = sig.replace(str(i), j)
            # replace numerical indices with valid tokens
        for _, j in enumerate(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            if ' ' + j in sig:
                # using a function with fewer params than indices in sig might leave 'lambda 1' etc. in sig
                sig = re.sub(Gr(rx_non_tchar) + Gr(rx_digit), r'\1x\2', sig)
        former = eval("lambda _ORIG: " + sig)
        # former = lambda _ORIG: lambda x0, x1, x2: lambda x3: _ORIG(x0, x3)(x1, x2)
        # so if g = former(f), then g(a, b, c)(d) = f(a, d)(b, c)
        if fn is None or return_former:
            return former
        if wrap:
            return Fun(former(fn))
        return former(fn)

class Dict(dict[KT, VT], Generic[KT, VT]):
    """
    Dictionary wrapper that improves functionality and integration with Fun
    Allows for attribute-like access to keys, e.g. D.x is equivalent to D['x'] (as in JS and Lua)
    Operations:
        D[X: Callable] = D.filter(X)
        D[X: Tuple] = D[X[0]][X[1:]]
        D[X: list] = [D[x] for x in X]
        D[X: other] = D.get(X, None)
        D + E = D | E = {k: E.get(k, D.get(k)) for k in set(E.keys()+D.keys())}
        D & E = {k: E.get(k, D.get(k)) for k in D}
        D - E = {k: D.get(k) for k in D if k not in E}
        D ^ E = (D - E) + (E - D)
        D * f = {f(k): v for k, v in D.items()}
        f * D = {k: f(v) for k, v in D.items()}
    """
    def __init__(self, items: dict[KT, VT] = {}, **kwargs: VT) -> None:
        if not isinstance(items, dict):
            items = dict(items)
        for k, v in itertools.chain(items.items(), kwargs.items()):
            if isinstance(v, (list, tuple, dict)) and not isinstance(v, (List, Tuple, Dict)):
                # recursively convert nested containers, for chained attributes and such
                items[k] = ({dict: Dict, tuple: Tuple, list: List}[type(v)])(v)
        super(Dict, self).__init__(items)
        self.__dict__ = items

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        if hasattr(source_type, '__args__'):
            key_type, value_type = source_type.__args__
            return core_schema.dict_schema(keys_schema=handler(key_type), values_schema=handler(value_type))
        return core_schema.dict_schema()

    def __class_getitem__(cls, item) -> Any:
        # Replicate typing annotations for Dict
        return dict[item]

    def __getitem__(self, key: Any) -> Any:
        # ordinary element retrieval: like dict.__getitem__, but fails gracefully
        if isinstance(key, list):
            return [self[k] for k in key]
        if isinstance(key, tuple):
            if len(key) == 1:
                return self[key[0]]
            return None if (key[0] not in self or self[key[0]] is None) else self[key[0]][key[1:]]
        if isinstance(key, dict):
            new_obj = Dict()
            for k, v in key.items():
                keyHashable, keyContainer = isinstance(v, Hashable), isinstance(v, (Dict, Tuple))
                _, selfContainer = isinstance(self[k], Hashable), isinstance(self[k], (Dict, Tuple))
                if keyHashable and k in self:
                    new_obj[v] = self[k]
                elif k in self:
                    new_obj[k] = self[k][v] if (selfContainer or keyContainer) else self[k]
            return new_obj
        if isinstance(key, str):
            return super(Dict, self).get(key, None)
        if isinstance(key, int):
            return super(Dict, self)[key]
        return super(Dict, self).get(key, None)

    @staticmethod
    def check(obj: Any) -> bool:
        """
        Checks if the given object is a Dict or subclass of Dict.
        :param obj: The object to check.
        :returns: True if the object is a Dict, False otherwise.
        """
        return (isinstance(obj, (Dict, dict)))

    def filter(self, predicate: Callable[..., bool] = lambda x: bool(x)) -> dict[KT, VT]:  # noqa: PLW0108
        # only keeps k such that P(k, self[k])
        """
        Filters the dictionary based on a predicate function. If the predicate is unary, it filters based on keys. If binary, it filters based on both keys and values.
        :param predicate: The predicate function to filter the dictionary.
        :returns: A new dictionary containing the filtered items.
        """
        if len(self) == 0:
            return Dict()
        # figure out arity of P
        if (predicate.__code__.co_argcount) == 1:
            return Dict({k: v for k, v in self.items() if predicate(k)})
        return Dict({k: v for k, v in self.items() if predicate(k, v)})

    def valuefilter(self, predicate: Callable[..., bool] = lambda x: bool(x)) -> dict[KT, VT]:  # noqa: PLW0108
        # only keeps k such that P(self[k])
        return self.filter(lambda k, v: predicate(v))

    def map(self, *fn) -> dict:  # applies f to each value in self
        """
        Applies a function to each value in the dictionary.
        :param fn: The function to apply to the values. If multiple functions are provided, they are applied in sequence.
        :returns: A new dictionary with the values transformed.
        """
        mapped = self
        for f in fn:
            mapped = Dict({k: f(v) for k, v in mapped.items()})
        return mapped

    def mapKeys(self, *fn) -> dict:  # applies f to each key in self
        """
        Applies a function to each key in the dictionary.
        :param fn: The function to apply to the keys. If multiple functions are provided, they are applied in sequence.
        :returns: A new dictionary with the keys transformed.
        """
        mapped = self
        for f in fn:
            mapped = Dict({f(k): v for k, v in mapped.items()})
        return mapped

    def strip(self, obj: Any = None) -> dict:  # removes all keys that map to None
        """
        Removes all keys that map to a given object from the dictionary.
        :param obj: The object to remove as a dictionary value. By default, None.
        :returns: A new dictionary with the keys removed.
        """
        return Dict({k: v for k, v in self.items() if v != obj})

    def pop(self, key: Any, default: Any = None) -> Any:
        """
        Attempts to remove and return the value associated with a given key from the dictionary. If the key is not found, returns a default value.
        :param key: The key to remove and return the value for.
        :param default: The default value to return if the key is not found. By default, None.
        :returns: The value associated with the key, or the default value if the key is not found.
        """
        if key in self:
            value = self[key]
            del self[key]
            return value
        return default

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Attempts to retrieve the value associated with a given key from the dictionary. If the key is not found, returns a default value.
        :param key: The key to retrieve the value for.
        :param default: The default value to return if the key is not found. By default, None.
        :returns: The value associated with the key, or the default value if the key is not found.
        """
        if isinstance(key, str):
            return super(Dict, self).get(key, default)
        if isinstance(key, (list, tuple, dict)):
            return self[key]
        return default

    def has(self, key: Any) -> bool:
        """
        Checks if a given key is present in the dictionary.
        :param key: The key to check for.
        :returns: True if the key is present, False otherwise.
        """
        return key in self

    def keysort(self, key: Callable[[KT], Any] = lambda x: x, reverse: bool = False) -> dict:
        """
        Sorts the dictionary by keys using the given key function.
        :param key: The key function to use for sorting.
        :param reverse: Whether to sort in reverse order.
        :returns: A new dictionary with the items sorted by keys.
        """
        return Dict(sorted(self.items(), key=lambda x: key(x[0]), reverse=reverse))

    def valsort(self, key: Callable[[VT], Any] = lambda x: x, reverse: bool = False) -> dict:
        """
        Sorts the dictionary by values using the given key function.
        :param key: The key function to use for sorting.
        :param reverse: Whether to sort in reverse order.
        :returns: A new dictionary with the items sorted by values.
        """
        return Dict(sorted(self.items(), key=lambda x: key(x[1]), reverse=reverse))

    def json(self) -> str:
        return json.dumps(self)

    # general idea being: f * D applies f to keys, D * f applies f to values
    def __mul__(self, f) -> dict:
        return self.map(f)

    def __rmul__(self, f) -> dict:
        return self.mapKeys(f)

    # D - E removes all keys in E from D, whether E is a list or a dict; if E is neither, D - E is just a safe delete
    def __sub__(self, minor) -> dict:
        if isinstance(minor, (list, List, tuple, Tuple, dict, Dict)):
            return Dict({k: v for k, v in self.items() if k not in minor})
        return Dict({k: v for k, v in self.items() if k != minor})

    def __rsub__(self, major) -> dict:
        if not isinstance(major, (dict, Dict)):
            raise TypeError("Can't subtract a dictionary from a non-dictionary")
        return Dict({k: v for k, v in major.items() if k not in self})

    def __delitem__(self, key) -> None:
        # like ordinary but safe
        if key in self:
            super(Dict, self).__delitem__(key)

    def __add__(self, other) -> dict:
        # D + E = D | E = {k: E.get(k, D.get(k)) for k in set(E.keys()+D.keys())}
        return Dict(self.__dict__ | other)

    def __radd__(self, other) -> dict:
        return Dict(other | self.__dict__)

    # D | E is the usual unrestricted write operation
    def __or__(self, other) -> dict:
        return Dict(self.__dict__ | other)

    def __ror__(self, other) -> dict:
        return Dict(other | self.__dict__)

    # D & E is a restricted write operation: imprints elements of E onto elements of D but doesn't add elements not in D
    def __and__(self, other) -> dict:
        return Dict({k: (other[k] if k in other else self[k]) for k in self})

    def __rand__(self, other) -> dict:
        return Dict({k: (self[k] if k in self else other[k]) for k in other})

    # D ^ E is the usual exclusive or, consisting of all keys in one dictionary but not the other (making it symmetric)
    def __xor__(self, other) -> dict:
        return Dict({k: self[k] for k in self if k not in other} | {k: other[k] for k in other if k not in self})

    def __rxor__(self, other) -> dict:
        return Dict({k: self[k] for k in self if k not in other} | {k: other[k] for k in other if k not in self})

    def keys(self) -> list:
        return list(super(Dict, self).keys())

    def values(self) -> list:
        return list(super(Dict, self).values())

    def items(self) -> list:
        return list(super(Dict, self).items())

class List(list[T], Generic[T]):
    """
    Wrapper for lists
    """
    def __init__(self, iterable: Any = None) -> None:
        if iterable is None:
            iterable = []
        if not isinstance(iterable, list):
            iterable = list(iterable)
        for i, v in enumerate(iterable):
            if isinstance(v, (list, tuple, dict)) and not isinstance(v, (List, Tuple, Dict)):
                iterable[i] = ({dict: Dict, tuple: Tuple, list: List}[type(v)])(v)
        super(List, self).__init__(iterable)

    @property
    def head(self) -> T:
        if len(self) == 0:
            raise ValueError('head of empty list')
        return self[0]

    @property
    def tail(self) -> List:
        return List(self[1:])

    def __getattr__(self, attr: str) -> Any:
        try:
            return getattr(list, attr)
        except AttributeError:
            try:
                return self[attr]
            except AttributeError:
                return None

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        if hasattr(source_type, '__args__'):
            item_type = source_type.__args__[0]
            return core_schema.list_schema(items_schema=handler(item_type))
        return core_schema.list_schema()

    def __class_getitem__(cls, item) -> Any:
        # Replicate typing annotations for List
        return list[item]

    def __getitem__(self, idx) -> Any:
        # ordinary element retrieval: like list.__getitem__, but with support for
        #    - None: L[None] = L
        #    - Tuple: L[T] treats T as a series of nested lookups (don't need to write L[(i, j, k)], since L[i, j, k] is equivalent)
        #        - List([0, [1, [2, [3]]]])[1, 1, 1] = 3
        #    - Callable: L[f] treats f as a predicate to filter L by
        #        - List([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).filter(this % 2 == 0) = [0, 2, 4, 6, 8]
        #    - list: L[L2] treats L2 as a list of indices to look up ('horizontal' indexing, as opposed to 'vertical' indexing with tuples)
        #        - List([0, 1, 2, 3, 4, ':)', 5, 6, 7, 8, 9])[[1, 3, 5, 7]] = [1, 3, ':)', 6]
        #    - str: L[s] treats L as a list of dictionaries to query
        #        - List([{'a': 0, 'b': 1}, {'a': 2, 'b': 3}, {'a': 4, 'b': 5}])['a'] = [0, 2, 4]
        if idx is None:  # L[None] = L
            return self
        if isinstance(idx, tuple):  # L[T] treats T as a series of nested lookups
            if len(idx) == 0:
                return self
            if len(idx) == 1:
                return self[idx[0]]
            return self[idx[0]][Tuple(idx[1:])]
        if isinstance(idx, Callable):  # L[f] treats f as a predicate to filter L by
            return List([obj for obj in self if idx(obj)])
        if isinstance(idx, list):  # L[L2] treats L2 as a list of indices to look up
            return List([self[i] for i in idx])
        if isinstance(idx, str):  # L[s] treats L as a list of dictionaries to query
            if not isinstance(self[0], dict):
                raise ValueError(f"Cannot index a list of non-dict objects with a string: {self}")
            return List([obj[idx] for obj in self])
        return super(List, self).__getitem__(idx)

    def __getattribute__(self, attr: str) -> Any:
        # attempts super(List, self).__getattribute__(attr) to get default behavior,
        #   but if that fails, acts like __getitem__[attr] (for attribute-like access across a list of dictionaries)
        try:
            return super(List, self).__getattribute__(attr)
        except AttributeError:
            return self[attr]

    @staticmethod
    def check(obj: Any) -> bool:
        """
        Checks if the given object is a list or subclass of List.
        :param obj: The object to check.
        :returns: True if the object is a list, False otherwise.
        """
        return isinstance(obj, (List, list))

    def map(self, *fn) -> List:
        """
        Applies a function to each element of the list.
        :param fn: The function to apply. If multiple functions are provided, they are applied in sequence.
        :returns: A new list with the function applied to each element.
        """
        mapped = self
        for f in fn:
            mapped = list(map(f, mapped))
        return List(mapped)

    def forEach(self, *fn) -> None:
        """
        Applies a function to each element of the list in place.
        :param fn: The function to apply to each element. If multiple functions are provided, they are applied in sequence.
        """
        for i, x in enumerate(self):
            for f in fn:
                x = f(x)
            self[i] = x

    def filter(self, *fn) -> List:
        """
        Filters the elements of the list based on the given predicate function. Returns a new List object containing the filtered elements.
        :param fn: The predicate function to filter the elements. If multiple functions are provided, they are applied in sequence.
        :returns: A new List object containing the filtered elements.
        """
        filtered = self
        for f in fn:
            filtered = List(list(filter(f, filtered)))
        return filtered

    def strip(self, obj: Any = None) -> List:  # removes obj from either end of the list
        """
        Removes consecutive instances of obj from either end of the list.
        :param obj: The object to remove. Defaults to None.
        :returns: The modified list.
        """
        L = self.copy()
        while len(L) > 0 and L[0] == obj:
            L = L[1:]
        while len(L) > 0 and L[-1] == obj:
            L = L[:-1]
        return L

    @staticmethod
    def async_map(func: Fun, arr: list, workers: int = 32) -> List:
        """
        Applies a function to each element of a list asynchronously using multiple workers.
        :param func: The function to apply to each element.
        :param arr: The list of elements.
        :param workers: The number of worker threads to use. Defaults to 32.
        :returns: A new list with the function applied to each element.
        """
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return List(executor.map(func, arr))

    def json(self, save_as: str | None = None) -> str:
        text = json.dumps(self)
        if save_as is not None:
            with open(save_as, 'w') as f:
                f.write(text)
        return text

    def jsonl(self, save_as: str | None = None) -> str:
        text = '\n'.join(json.dumps(item) for item in self)
        if save_as is not None:
            with open(save_as, 'w') as f:
                f.write(text)
        return text


class Tuple(list[T], Generic[T]):
    """
    Tuple wrapper
    """
    def __init__(self, items=None) -> None:
        if items is None:
            items = []
        if not isinstance(items, list):
            items = list(items)
        for i, v in enumerate(items):
            if isinstance(v, (list, tuple, dict)) and not isinstance(v, (List, Tuple, Dict)):
                items[i] = ({list: List, tuple: Tuple, dict: Dict}[type(v)])(v)
        super(Tuple, self).__init__(items)

    def __getattr__(self, attr: Any) -> Any:
        if attr == 'head':
            if len(self) == 0:
                raise ValueError('head of empty tuple')
            return self[0]
        if attr == 'tail':
            return Tuple(self[1:])
        return getattr(list, attr)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        if hasattr(source_type, '__args__'):
            item_schemas = [handler(arg) for arg in source_type.__args__]
            return core_schema.tuple_schema(items_schema=item_schemas)
        return core_schema.tuple_schema()

    def __class_getitem__(cls, item) -> Any:
        return tuple[item]

    def __getitem__(self, idx) -> Any:
        if idx is None:
            return self
        if isinstance(idx, (tuple, Tuple)):
            if len(idx) == 0:
                return self
            if len(idx) == 1:
                return self[idx[0]]
            return self[idx[0]][Tuple(idx[1:])]
        return super(Tuple, self).__getitem__(idx)

    @staticmethod
    def check(obj: Any) -> bool:
        """
        Checks if the given object is a tuple or subclass of Tuple.
        :param obj: The object to check.
        :returns: True if the object is a tuple, False otherwise.
        """
        return isinstance(obj, (tuple, Tuple))

    def map(self, *fn) -> Tuple:
        """
        Applies a function to each element of the tuple.
        :param fn: The function to apply. If multiple functions are provided, they are applied in sequence.
        :returns: A new tuple with the function applied to each element.
        """
        mapped = self
        for f in fn:
            mapped = list(map(f, mapped))
        return Tuple(mapped)

    def filter(self, *fn) -> Tuple:
        """
        Filters the elements of the tuple based on the given predicate function. Returns a new Tuple object containing the filtered elements.
        :param fn: The predicate function to filter the elements. If multiple functions are provided, they are applied in sequence.
        :returns: A new Tuple object containing the filtered elements.
        """
        filtered = self
        for f in fn:
            filtered = list(filter(f, filtered))
        return Tuple(filtered)

def super_func(func: Func) -> Func:
    """
    Decorator that converts a function into a Fun object.
    :param func: The function to convert.
    :returns: A Fun object that wraps the function.
    """
    if isinstance(func, Fun):
        func = func.func

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        output = func(*args, **kwargs)
        for pair in [(list, List), (tuple, Tuple), (dict, Dict)]:
            if isinstance(output, pair[0]) and not isinstance(output, pair[1]):
                return pair[1](output)
        return output
    return Fun(wrapper)

def non_super_func(func: Callable[[X], Y]) -> Callable[[X], Y]:  # for clarity
    return func

@super_func
def map(f: Callable[[X], Y], items: list[X]) -> list[Y]:
    """
    Applies a function to each item in a given list.
    :param f: The function to apply to each item.
    :param items: The list of items to apply the function to.
    :returns: A new list with the function applied to each item.
    """
    return [f(i) for i in items]

@super_func
def filter(f: Callable[[T], bool], items: list[T]) -> list[T]:
    """
    Filters a list of items based on the predicate f.
    :param f: The predicate function to filter the items.
    :param items: The list of items to filter.
    :returns: A new list containing the items that satisfy the predicate.
    """
    return [i for i in items if f(i)]

@non_super_func
def id(x: T) -> T:
    """
    The identity function.
    :param x: The input value.
    :returns: The input value.
    """
    return x

@super_func
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

@super_func
def const(c: T, _: Any) -> T:
    """
    A function that discards its second input to return the first.
    Curried, const(c) is a constant function that returns c regardless of its input.
    :param c: The value to return.
    :param _: The value to discard.
    :returns: The value c.
    """
    return c

# @super_func
# def sub(pattern, repl) -> Callable[str, str]:
#     """
#     Returns a function that replaces occurrences of pattern in the given string with repl.
#     :param pattern: The pattern to search for.
#     :param repl: The replacement string.
#     :returns: The substituting function.
#     """
#     return lambda string: re.sub(pattern, repl, string)

# sub = Fun.form('01 2', '012', re.sub)

@super_func
def sub(pattern: str, repl: str, string: str) -> str:
    """
    Returns a function that replaces occurrences of pattern in the given string with repl.
    :param pattern: The pattern to search for.
    :param repl: The replacement string.
    :returns: The substituting function.
    """
    return re.sub(pattern, repl, string)

@super_func
def _read(fp: str) -> str:
    """
    Reads the contents of a file.
    :param fp: The file path.
    :returns: The contents of the file as a string.
    """
    with open(fp) as f:
        return f.read()

@super_func
def _readlines(fp: str) -> list:
    """
    Reads the lines of a file.
    :param fp: The file path.
    :returns: A list of lines in the file.
    """
    with open(fp) as f:
        return List(f.readlines())

read, readlines = _read, _readlines

@super_func
def nonempty(f: Fun, items: list) -> list:
    """
    Applies a function to each item in a list and filters out the empty results.
    :param f: The function to apply to each item.
    :param items: The list of items.
    :returns: A new list containing the non-empty results.
    """
    return [f(y) for y in items if f(y)]

@super_func
def cmd(input: str) -> Any:
    results = subprocess.run(input, capture_output=True, check=False)
    return results.stdout.decode('utf-8')

select = Fun(lambda *args: Fun(lambda L: (lambda a, b, c: L[a:b:c])(*(slice(*args).indices(len(L))))))  # select(a)(L) = L[:a]; select(b,c)(L) = L[b:c]
match = Fun(re.match) / Dict({'groups': lambda: (3,)})
split = Fun(str.split, ('01', '10'))
strip = Fun(str.strip, args=[None])
get = Fun('a[b]')

prnt = Fun(lambda x: print(x, end = ''))


def get_attr(obj: Any, attr: str) -> Any:
    """
    Recursively retrieves nested attributes, dictionary keys, or list/tuple indices from an object.
    Supports optional attributes (ending with '?') and list comprehensions (e.g., '.[attr]').
    :param obj: The object to retrieve the attribute from.
    :param attr: The attribute string, which may include dots for nesting, '?' for optional attributes, and '[expr]' for list comprehensions over iterables.
    :returns: The retrieved attribute value(s).
    """
    def _get_attr(obj: Any, attr_parts: list) -> Any:
        if not attr_parts:
            return obj
        key, opt = attr_parts[0], attr_parts[0].endswith('?')
        if opt:
            key = key[:-1]

        # Handle list comprehension syntax "[expr]"
        if key.startswith('[') and key.endswith(']'):
            # Apply the expression inside brackets to each item in obj
            sub_attr = key[1:-1]  # Remove the '[' and ']'
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                result = []
                for item in obj:
                    try:
                        value = _get_attr(item, [sub_attr])
                        result.append(value)
                    except Exception:
                        if not opt:
                            raise
                        result.append(None)
                # Recurse into the rest of the attr_parts
                return _get_attr(result, attr_parts[1:])
            msg = f"Cannot iterate over object of type {type(obj)} for list comprehension"
            raise TypeError(msg)
        try:
            if isinstance(obj, dict):
                obj = obj.get(key)
            elif isinstance(obj, (list, tuple)):
                obj = obj[int(key)]
            else:
                try:
                    obj = getattr(obj, key)
                except AttributeError:
                    return None
        except (AttributeError, KeyError, IndexError, ValueError, TypeError):
            if opt:
                return None
            raise
        # Recurse into the rest of the attr_parts
        return _get_attr(obj, attr_parts[1:])

    # Parse the attribute string into parts
    def parse_attribute_string(attr: str) -> list:
        parts, i, n = [], 0, len(attr)
        while i < n:
            if attr[i] == '.':
                i += 1  # Skip the dot
            if i < n and attr[i] == '[':
                # Collect everything inside the brackets
                start, depth = i, 1
                i += 1
                while i < n and depth > 0:
                    depth += {'[': 1, ']': -1}.get(attr[i], 0)
                    i += 1
                if depth > 0:
                    raise ValueError("Mismatched '[' in attribute string")
                parts.append(attr[start:i])  # Include the ']' character
            else:
                # Collect regular attribute name
                start = i
                while i < n and attr[i] not in {'.', '['}:
                    i += 1
                parts.append(attr[start:i])
        return parts

    attr_parts = parse_attribute_string(attr)
    return _get_attr(obj, attr_parts)
