"""
Provides a set of interlacing wrappers for Python's functions and container types
Contains:
    class Null
    class Void
    class Fail
    is_list, is_dict, is_tuple, is_set, is_container, is_type
        (x: Any) -> bool
    class Multi(list)
        is_container
            (obj: Any) -> bool
        walk
            (obj: Any, path: Optional[list] = None) -> Any
        build_skeleton
            (obj: Any) -> Any
        set_nested
            (obj: Any, path: list, value: Any) -> Any
    class MetaMeta(type)
    class Meta(metaclass=MetaMeta)
        is_meta
            (cls, obj) -> bool
    class Fun
        check
            (cls: Any) -> bool
        form
            (ord: str, call: str, fn: Optional[Callable] = None, return_former: bool = False, wrap: bool = False) -> Callable
    class Dict(dict)
        check
            (cls) -> bool
        filter
            (self, predicate: Callable) -> Dict
        map, mapKeys
            (self, f) -> Dict
        pop, get
            (self, key: Any, default: Any = None) -> Any
        get
            (self, key: Any, default: Any = None) -> Any
        keys, values, items
            (self) -> list
    class List(list)
        check
            (cls) -> bool
        map
            (self: List[X], f: Callable[[X], Y]) -> List[Y]
        filter
            (self: List[T], f: Callable[[T], bool]) -> List[T]
        asyncMap
            (func: Callable[[X], Y], arr: List[X], workers: int = 32) -> List[Y]
    class Tuple(list)
        check
            (cls: Any) -> bool
        map, filter
            (self, f) -> Tuple
    map, filter
        (f, L) -> List
    id
        (x: Any) -> Any
    comp
        (*args) -> Fun
    reduce
        (opn: Callable, values: Any, base: Any) -> Any
    const
        (c, x) -> Any
    sub
        (pattern, oldstr, newstr) -> str
    Fun read, readlines, nonempty, match, split, strip, select, get, cmd, prnt
"""
from __future__ import annotations

import builtins
import inspect
import logging
import re
from collections.abc import Hashable, Iterable, Generator, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import singledispatch, singledispatchmethod, update_wrapper
from functools import wraps as functools_wraps
from inspect import signature as inspect_signature
from itertools import chain as itertools_chain, product as itertools_product
from operator import itemgetter as operator_itemgetter
from pydantic_core import core_schema
from subprocess import run as subprocess_run
from types import MemberDescriptorType
from typing import List as typing_List, Dict as typing_Dict, Tuple as typing_Tuple, Set as typing_Set
from typing import Any, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

# Should look at https://boltons.readthedocs.io/en/latest/ and https://github.com/jab/bidict

T, X, Y = TypeVar('T'), TypeVar('X'), TypeVar('Y')
End = Callable[[T], T]
Hom = Callable[[X], Y]
F = TypeVar('F', bound=Callable[..., Any])
Decorator = End[F]

class Null:
    """
    A null object. Methods are to treat it as though it wasn't there.
    Semantically, 
        - {"a": None} is a dictionary with a value None, while {"a": Null} is empty.
        - add(3, None) should raise an error b/c you can't add 3 and None, while add(3, Null) should raise an error b/c add takes two arguments.
    """
    pass

class Void:
    """
    A void object.
    """
    pass

class Fail:
    """
    A failure object.
    """
    pass

def is_list(x: Any) -> bool:
    return isinstance(x, (list, List, typing_List))

def is_dict(x: Any) -> bool:
    return isinstance(x, (dict, Dict, typing_Dict))

def is_tuple(x: Any) -> bool:
    return isinstance(x, (tuple, Tuple, typing_Tuple))

def is_set(x: Any) -> bool:
    return isinstance(x, (set, typing_Set))

def is_container(x: Any) -> bool:
    return is_list(x) or is_dict(x) or is_tuple(x) or is_set(x)

def is_type(x: Any) -> bool:
    return isinstance(x, type)

class Multi(list):
    """
    A class that represents multi-valued parameters.
    Wrapped functions distribute over Multi-params: F(<2,3,4>)=<F(2),F(3),F(4)>.
    """
    def __init__(self, *args) -> None:
        super(Multi, self).__init__(args)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Multi instance.
        """
        return f"<{', '.join([repr(w) for w in self])}>"

    def __str__(self) -> str:
        """
        Returns a string representation of the Multi instance.
        """
        return f"<{', '.join([str(w) for w in self])}>"

    @staticmethod
    def is_container(obj: Any) -> bool:
        """
        Checks if the given object is a container type.
        :param obj: The object to check.
        :returns: True if the object is a container type, False otherwise.
        """
        return isinstance(obj, (Multi, typing_List, typing_Dict, typing_Tuple, list, dict, tuple, set))

    @staticmethod
    def walk(obj: Any, path: List = []) -> Any:
        """
        Recursively traverses the given object, yielding the path and value of each element.        
        :param obj: The object to traverse.
        :param path: The current path (optional).
        :yields: A tuple containing the path and value of each element.
        """
        if isinstance(obj, Multi):
            for i, item in enumerate(obj):
                yield from Multi.walk(item, [*path, (i, obj)])
        elif Multi.is_container(obj):
            if isinstance(obj, (typing_Dict, dict)):
                for k, v in obj.items():
                    yield from Multi.walk(v, [*path, (k, obj)])
            elif isinstance(obj, (typing_List, list, typing_Tuple, tuple, set)):
                for i, item in enumerate(obj):
                    yield from Multi.walk(item, [*path, (i, obj)])
        else:
            yield path, obj

    @staticmethod
    def build_skeleton(obj: Any) -> Any:
        """
        Recursively builds a skeleton of the given object, replacing non-container values with None.
        :param obj: The object to build a skeleton for.
        :returns: The skeleton of the given object.
        """
        return build_skeleton(obj)

def register_all(types: Tuple) -> Decorator:
    def wrapper(func):
        dispatcher = singledispatch(func)
        for t in types:
            dispatcher.register(t)(func)
        update_wrapper(dispatcher, func)
        return dispatcher
    return wrapper


@singledispatch
def build_skeleton(obj: Any) -> Any:
    """Base case for non-container types."""
    return None

@register_all((Multi, dict, list, tuple, set))
def build_skeleton(obj: Any) -> Any:
    if isinstance(obj, Multi):
        return Multi(*[build_skeleton(item) for item in obj])
    elif isinstance(obj, (dict, typing_Dict)):
        return type(obj)({k: build_skeleton(v) for k, v in obj.items()})
    elif isinstance(obj, (list, typing_List, tuple, typing_Tuple)):
        return type(obj)(build_skeleton(item) for item in obj)
    elif isinstance(obj, (set, typing_Set)):
        return set()  # Always return a regular set for the skeleton
    return None # Should never reach here with registered types


Multi.build_skeleton = staticmethod(build_skeleton)



@singledispatch
def set_nested(obj: Any, path: list, value: Any) -> Any:
    raise TypeError("Object is not a supported container type")

@register_all((dict, list, Multi))
def _(obj: Any, path: list, value: Any) -> Any:
    for key, _ in reversed(path):
        obj[key] = value
        value = obj
    return value

@register_all((tuple,))
def _(obj: Any, path: list, value: Any) -> Any:
    obj = list(obj)
    for key, _ in reversed(path):
        obj[key] = value
        value = obj
    return obj


Multi.set_nested = staticmethod(set_nested)


class MetaMeta(type):
    '''
    Necessary for making the compression trick that Meta uses work
    '''
    def __new__(cls, name: str, bases: typing_Tuple[type, ...], attrs: typing_Dict[str, Any]) -> type:
        def create_method(method_name: str) -> Callable:
            def method(self, other: str) -> Callable:
                op_name = method_name.strip('_')
                if method_name.startswith('__r'):
                    repr_string = f"λe.{op_name}({other!r}, e)"
                else:
                    repr_string = f"λe.{op_name}(e, {other!r})"

                func = lambda obj: getattr(obj, method_name)(other)
                func.__repr__, func.__qualname__, func.__name__ = (lambda: repr_string), repr_string, repr_string
                func.__str__ = lambda: repr_string
                return func
            return method

        for method in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow',
                        'lshift', 'rshift', 'and', 'xor', 'or', 'lt', 'le', 'eq',
                        'ne', 'gt', 'ge']:
            for prefix in ['__', '__r']:
                method_name = f'{prefix}{method}__'
                attrs[method_name] = create_method(method_name)
        return super().__new__(cls, name, bases, attrs)


class Meta(metaclass=MetaMeta):
    '''
    Treated as blanks by wrapped functions -- useful for partial evaluation
    '''
    def __init__(self) -> None:
        self.is_meta = True

    def __repr__(self) -> str:
        return "<Meta object>"

    def __getitem__(self, key: Any) -> Any:
        return operator_itemgetter(key)

    def __contains__(self, key: Any) -> Any:
        return lambda obj: key in obj

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Meta)

    def __subclasscheck__(self, subclass: Any) -> bool:
        return issubclass(subclass, Meta)

    def __getattribute__(self, name: str) -> Any:
        if name == '_is_meta':
            return True
        if name.startswith('__') and name.endswith('__'):
            if name in {'__getattribute__', '__instancecheck__', '__subclasscheck__'}:
                return getattr(self, name)
            if name in {'__mro__', '__class__', '__init__'}:
                return getattr(type(self), name)

            def method(*args, **kwargs) -> Callable:
                repr_string = f"λe.{name[2:-2]}(e, {', '.join(map(repr, args))})"
                func = lambda obj: getattr(obj, name)(*args, **kwargs)
                func.__repr__, func.__name__, func.__qualname__ = (lambda: repr_string), repr_string, repr_string
                return func
            return method

        def attr_func(obj) -> Any:
            return getattr(obj, name)
        attr_func.__repr__, attr_func.__qualname__, attr_func.__name__ = (lambda: f"λe.{name}(e)"), f"λe.{name}(e)", f"λe.{name}(e)"
        return attr_func

    @classmethod
    def is_meta(cls, obj) -> bool:
        """Check if an object is a Meta instance without using isinstance."""
        return hasattr(obj, 'is_meta') and obj.is_meta


this = Meta()

class Fun:
    '''
    Function wrapper that allows for FP-like manipulations and partial evaluation (via Meta objects)
        Lots of galaxy-brained notation for function manipulation:
            f * g is the composition f(g(x))
            f @ L is the map of f over L, i.e., [f(x) for x in L]
            f / g returns f(x) if that works and isn't None, else (g(x) if g callable else g)
            f ** n is the function that returns f(f(...(f(x))...)) for n iterations
            chaining: 3 >> f >> g is equivalent to g(f(3))
    '''
    # function wrapper for functional programming-style syntactic sugar
    def __init__(self, func, form = None, args = None, **kwargs) -> None:
        # args is to be appended to the arguments of a call
        if args is None:
            args = []
        if isinstance(func, str):
            tokens = []
            for token in re.findall('[a-zA-Z_][a-zA-Z0-9_]*', func):
                if token not in tokens and token not in dir(builtins) and all(y + '.' + token not in func for y in dir(builtins)):
                    tokens.append(token)
            func = eval('lambda ' + ', '.join(tokens) + ': ' + func)
        i = 0
        self.func = func
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
        self.parameters = inspect_signature(self.func).parameters
        self.arguments = list(self.parameters.keys())

    def __name__(self) -> str:
        return self.func.__name__

    def __doc__(self) -> str:
        return self.func.__doc__

    def __call__(self, *args, **kwargs) -> Any:
        while (len(args) == 1 and isinstance(args[0], (tuple, Tuple))):
            args = args[0]
        args = list(args) + self.args
        kwargs |= self.kwargs

        # Check for Multi objects and handle them asynchronously
        multi_args = []
        for arg in itertools_chain(args, kwargs.values()):
            multi_args.extend(path for path, value in Multi.walk(arg) if isinstance(value, Multi))

        if multi_args:
            skeleton_args = Multi.build_skeleton(args)
            skeleton_kwargs = Multi.build_skeleton(kwargs)

            multi_values = [Multi.set_nested(args if isinstance(path[-1][1], (list, tuple, List, Tuple)) else kwargs, path[:-1], value) for path, value in multi_args]

            combinations = list(itertools_product(*multi_values))

            with ThreadPoolExecutor() as executor:
                futures = []
                for combo in combinations:
                    current_args = Multi.set_nested(skeleton_args, [], args)
                    current_kwargs = Multi.set_nested(skeleton_kwargs, [], kwargs)
                    for (path, _), value in zip(multi_args, combo):
                        if isinstance(path[-1][1], (list, tuple, List, Tuple)):
                            Multi.set_nested(current_args, path, value)
                        else:
                            Multi.set_nested(current_kwargs, path, value)
                    futures.append(executor.submit(self.func, *current_args, **current_kwargs))

                results = [future.result() for future in futures]

            return Multi(*results)

        # partial evaluation code
        if (k := len(args) + len(kwargs)) < (n := len(self.arguments)):
            selfargs = list(self.arguments)
            a = selfargs[len(args):]
            first_unprovided = min(selfargs.index(y) for y in a if y not in kwargs)
            kw_in_first = {k: v for (k, v) in kwargs.items() if k in selfargs[:first_unprovided]}
            kw_in_second = {k: v for (k, v) in kwargs.items() if k not in selfargs[:first_unprovided]}
            call = ''.join(map(str, range(n)))
            ord = call.replace(str(k), ' ' + str(k))
            return Fun(Fun.form(ord, call, self.func)(*args, **kw_in_first), **kw_in_second)
        if any(Meta.is_meta(x) for x in args):
            indices = [i for i, x in enumerate(args) if Meta.is_meta(x)]
            args = [x for x in args if not Meta.is_meta(x)]

            def helper(args=args, indices=indices, *args2) -> Any:
                args = args[:]
                for num, idx in enumerate(sorted(indices)):
                    args.insert(idx, args2[num])
                return args
            return Fun(lambda *args2: self.func(helper(args, indices, *args2)))
        rv = self.func(*args, **kwargs)
        if callable(rv):
            return Fun(rv)
        return rv

    def __setitem__(self, name: str, value: Any) -> None:
        self.kwargs[name] = value

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
    def check(cls: Any) -> bool:
        return isinstance(cls, (Fun, Callable))

    def __mul__(self, other) -> 'Fun':
        # (f*g)(x) = f(g(x))
        def _(*args, **kwargs) -> Any:
            return self(other(*args, **kwargs))
        return Fun(_)

    def __rmul__(self, other) -> 'Fun':
        # (g*f)(x) = g(f(x))
        def _(*args, **kwargs) -> Any:
            return other(self(*args, **kwargs))
        return Fun(_)

    def __add__(self, other) -> 'Fun':
        def _(*a) -> Any:
            if len(a) != 1:
                return (self(a[0]), other(a[1]))
            # if type(a[0]) == tuple and len(a[0]) != 1:
            # return (self(a[0][0]), other(a[0][1]))
            return (self(a[0]), other(a[0]))
        return Fun(_)

    def __matmul__(self, other) -> 'Fun':
        # f @ L = map(f)(L) = [f(x) for x in L]
        def _(iterable) -> Any:
            return [self.func(x) for x in iterable]
        return Fun(_)

    def __getattr__(self, arg) -> Any:
        if isinstance(arg, Fun):
            return self * arg
        return object.__getattribute__(self, arg)

    def __truediv__(self, other) -> Optional['Fun']:
        # (f/g)(x) = f(x) if that works, else g(x), else None
        def _(*args, **kwargs) -> Any:
            try:
                return self.func(*args, **kwargs)
            except Exception:
                try:
                    return (other.func(*args, **kwargs) if isinstance(other, Fun) else other(*args, **kwargs))
                except Exception:
                    return None
        return _

    def __pow__(self, n: int) -> 'Fun':
        # repeats a function n times, with n=0 yielding the identity
        n_recurse_at = 3
        if n < n_recurse_at:
            return [Fun(lambda x: x), self, self * self][n]
        return (self**2)**(n // 2) if n % 2 == 0 else self * (self**(n - 1))

    def __rrshift__(self, x) -> Any:  # stupid chaining
        return self * x if isinstance(x, Fun) else self(x)

    @staticmethod
    def form(ord: str, call: str, fn: Optional[Callable] = None, return_former: bool = False, wrap: bool = False) -> Callable:
        """
        Utility for variadic combinator constructor with generalized notation.
        :param ord: The order of the arguments in the function signature.
        :param call: The order of the arguments in the function call.
        :param fn: The function to wrap.
        :param return_former: Whether to return the higher-order forming function rather than the function it wraps.
        :param wrap: Whether to wrap the function in a Fun object.
        :returns: The wrapped function.
        """
        """
        - spacing denotes separation of composition, i.e. currying
            - ex. ord = 0 1 creates the function g(a)(b) := call[f]
            - while ord = 01 creates g($0, $1) := call[f]
            - here call[f] is the evaluation tree for the input f specified by call
        - parentheses denote sub-evaluations
            - ex. call = 01(2) creates ord[g] := f(a, b(c))
            - while call = 3 3(3(0)1)2 2(1) creates ord[g] := f(d)(d(d(a), b), c)(c(b))
        - you can't use parentheses in ord, because that's incoherent
        - so form('01 2', '201', f) returns the function g: a, b -> c -> f(c, a, b)
        - also accepts alternate notation with words instead of numbers,
        - commas instead of juxtaposition, and semicolons instead of spaces, e.g.
            - form("func, L; callback", "callback(func, L)", display)
            - is the same as form('01 2', '2(01)', display)
            - just don't try to *combine* sequence-space notation with comma-semicolon notation
        - classic combinators can all be written in terms of id:
        - writing `0 1 -> 01` for form('0 1', '0(1)', id), we have
            - apply = 0 1 -> 0(1)
                - (a -> b) -> a -> b
            - const = 0 1 -> 0
                - K: a -> b -> a
            - compose = 0 1 2 -> 0(1(2))
                - B: (b -> c) -> (a -> b) -> a -> c
            - fix = 0 -> 0(0)
                - Y: (a -> a) -> a
            - on = 0 1 2 3 -> 0(1(2))(1(3))
                - P: (b -> b -> c) -> (a -> b) -> a -> a -> c
            - chain = 0 1 2 -> 0(1(2))(2)
                - S_: (a -> b -> c) -> (b -> a) -> b -> c
            - ap = 0 1 2 -> 0(2)(1(2))
                - S: (a -> b -> c) -> (a -> b) -> a -> c
            - applyTo = 0 1 -> 1(0)
                - T: a -> (a -> b) -> a
            - converge = 0 1 2 3 -> 0(1(3))(2(3))
                - S2: (b -> c -> d) -> (a -> b) -> (a -> c) -> a -> d
            - join = 0 1 -> 0(1)(1)
                - W: (a -> a -> b) -> a -> b
        """
        while isinstance(fn, Fun):
            fn = fn.func
        # '3,0;1(2)'
        ord, call = [re.sub('([,;]) +', r'\1', y) for y in [ord, call]]
        tokens = re.findall('[a-zA-Z0-9_]' + '+' * bool(re.findall('[,;]', ord)), ord)
        initial_params = []
        for t in tokens:
            if t not in initial_params:
                initial_params.append(t)
        for i, j in enumerate(initial_params):
            ord = ord.replace(j, str(i))
            call = call.replace(j, str(i))
        ord, call = [re.sub(',', '', re.sub(';', ' ', y)) for y in [ord, call]]

        indices = [int(i) for i in re.findall("[0-9]", call)]
        call += " " + "".join([str(x) for x in range(max(indices) + 1) if x not in indices])
        call = "ORIGINAL(" + call.replace(" ", ") (") + ")"
        ord = "lambda " + ": lambda ".join(ord.split(" ")) + ": "
        for _ in range(len(call) + len(ord)):
            call, ord = [re.sub(r"([0-9]|\))([0-9])", "\\1, \\2", x) for x in [call, ord]]
        ord += call.replace(" ", "")
        params_list = initial_params if fn is None else list(inspect_signature(fn).parameters.keys())
        for i, j in enumerate(params_list):
            if j[0] in '0123456789_':
                j = 'x' + j
            ord = ord.replace(str(i), j)
        for _, j in enumerate('0123456789'):
            if ' ' + j in ord:  # using a function with fewer params than indices in ord might leave 'lambda 1' etc. in ord
                ord = re.sub(r'([^a-zA-Z0-9])(\d)', r'\1x\2', ord)
        # print("lambda Fun: "+ord)
        former = eval("lambda ORIGINAL: " + ord)
        return former if (fn is None or return_former) else (Fun(former(fn)) if wrap else former(fn))

# e.g. if Sub = Curry_01_2(re.sub)
# then Sub('a','b')('yam') = 'ybm'
# so you can chain the monadic Sub(s, t) functions together as
# in_string >> Sub(s1, t1) >> Sub(s2, t2) >> ... >> Sub(sn, tn)
# for n in range(2, 4):
# 	s = ''.join(str(i) for i in range(n))
# 	col = lambda L: ''.join(str(i) for i in L)
# 	for m in range(n-1):
# 		for p in itertools.permutations(range(n), n):
# 			invp = tuple(p.index(i) for i in range(n))
# 			#print('Curry_'+col(p[:m+1])+'_'+col(p[m+1:])+' = Fun.form("'+s[:m+1]+' '+s[m+1:]+'", "'+col(invp[:m+1]) + col(invp[m+1:])+'", id)')
# 			vars()['Curry_'+col(p[:m+1])+'_'+col(p[m+1:])] = Fun.form(s[:m+1]+' '+s[m+1:], col(invp[:m+1]) + col(invp[m+1:]))

class Dict(dict):
    '''
    Wrapper for dictionaries that allows for more functionality and integration with F
        Allows for attribute-like access to keys, e.g. D.x is equivalent to D['x'] (as in JS and Lua)
        The operator * treats the dictionary as a function from keys to values, so (D * f)[k] = D[f(k)] and (f * D)[k] = f(D[k])
    D + E = D | E = {k: E.get(k, D.get(k)) for k in set(E.keys()+D.keys())}
    '''
    def __init__(self, items=None) -> None:
        if items is None:
            items = {}
        for k, v in items.items():
            if isinstance(v, (list, tuple, dict)) and not isinstance(v, (List, Tuple, Dict)):
                items[k] = ({dict: Dict, tuple: Tuple, list: List}[type(v)])(v)
        super(Dict, self).__init__(items)
        self.__dict__ = items

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        if hasattr(source_type, '__args__'):
            key_type, value_type = source_type.__args__
            return core_schema.dict_schema(keys_schema=handler(key_type), values_schema=handler(value_type))
        return core_schema.dict_schema()

    @classmethod
    def __getitem__(cls, item: Any = None) -> Any:
        if item is None:
            return typing_Dict[cls]
        return typing_Dict[cls, item]

    @singledispatchmethod
    def __getitem__(self, key: Any) -> Any:
        return super(Dict, self).get(key, None)

    @__getitem__.register(list)
    def _(self, key: typing_List) -> typing_List:
        return List([self[k] for k in key])

    @__getitem__.register(tuple)
    def _(self, key: typing_Tuple) -> Any:
        if len(key) == 1:
            return self[key[0]]
        return None if (key[0] not in self or self[key[0]] is None) else self[key[0]][key[1:]]

    @__getitem__.register(dict)
    def _(self, key: dict) -> Dict:
        new_obj = Dict()
        for k, v in key.items():
            keyHashable, keyContainer = isinstance(v, Hashable), isinstance(v, (Dict, Tuple))
            selfHashable, selfContainer = isinstance(self[k], Hashable), isinstance(self[k], (Dict, Tuple))
            if keyHashable and k in self:
                new_obj[v] = self[k]
            elif k in self:
                new_obj[k] = self[k][v] if (selfContainer or keyContainer) else self[k]
        return new_obj

    def __getattr__(self, attr: str) -> Any:
        return getattr(dict, attr)

    @classmethod
    def check(cls) -> bool:
        return (isinstance(cls, (Dict, dict)))

    def filter(self, predicate: Callable[..., bool]) -> 'Dict':  # only keeps k such that P(k, self[k])
        if len(self) == 0:
            return Dict()
        # figure out arity of P
        if (predicate.__code__.co_argcount) == 1:
            return Dict({k: v for k, v in self.items() if predicate(k)})
        return Dict({k: v for k, v in self.items() if predicate(k, v)})

    def map(self, f) -> 'Dict':  # applies f to each value in self
        return Dict({k: f(v) for k, v in self.items()})

    def mapKeys(self, f) -> 'Dict':  # applies f to each key in self
        return Dict({f(k): v for k, v in self.items()})

    def pop(self, key: Any, default: Any = None) -> Any:
        if key in self:
            value = self[key]
            del self[key]
            return value
        return default

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str):
            return super(Dict, self).get(key, default)
        if isinstance(key, (list, tuple, dict)):
            return self[key]
        return default

    def has(self, key: Any) -> bool:
        return key in self

    def __mul__(self, f) -> 'Dict':
        return self.map(f)

    def __rmul__(self, f) -> 'Dict':
        return self.mapKeys(f)

    # general idea being: f * D applies f to keys, D * f applies f to values
    def __sub__(self, minor) -> 'Dict':
        # D - E removes all keys in E from D
        if isinstance(minor, (list, List, dict, Dict)):
            return Dict({k: v for k, v in self.items() if k not in minor})
        return Dict({k: v for k, v in self.items() if k != minor})

    def __rsub__(self, major) -> 'Dict':
        if not isinstance(major, (dict, Dict)):
            raise ValueError("Can't subtract a dictionary from a non-dictionary")
        return Dict({k: v for k, v in major.items() if k not in self})

    def __delitem__(self, key) -> None:
        # like ordinary but safe
        if key in self:
            super(Dict, self).__delitem__(key)

    def __add__(self, other) -> 'Dict':
        # D + E = D | E = {k: E.get(k, D.get(k)) for k in set(E.keys()+D.keys())}
        return Dict(self.__dict__ | other)
    def __radd__(self, other) -> 'Dict':
        return Dict(other | self.__dict__)
    def __or__(self, other) -> 'Dict':
        return Dict(self.__dict__ | other)
    def __ror__(self, other) -> 'Dict':
        return Dict(other | self.__dict__)
    # D & E imprints elements of E onto elements of D; doesn't add elements not in D
    def __and__(self, other) -> 'Dict':
        return Dict({k: other[k] for k in self if k in other})
    def __rand__(self, other) -> 'Dict':
        return Dict({k: self[k] for k in other if k in self})


    def keys(self) -> list:
        return list(super(Dict, self).keys())

    def values(self) -> list:
        return list(super(Dict, self).values())

    def items(self) -> list:
        return list(super(Dict, self).items())

class List(list):
    """
    Wrapper for lists
    """
    def __init__(self, w=None) -> None:
        if w is None:
            w = []
        for i, v in enumerate(w):
            if isinstance(v, (list, tuple, dict)) and not isinstance(v, (List, Tuple, Dict)):
                w[i] = ({dict: Dict, tuple: Tuple, list: List}[type(v)])(v)
        super(List, self).__init__(w)

    def __getattr__(self, attr: str) -> Any:
        if attr == 'head':
            if len(self) == 0:
                raise ValueError('head of empty list')
            return self[0]
        if attr == 'tail':
            return List(self[1:])
        return getattr(list, attr)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]) -> core_schema.CoreSchema:
        if hasattr(source_type, '__args__'):
            item_type = source_type.__args__[0]
            return core_schema.list_schema(items_schema=handler(item_type))
        return core_schema.list_schema()

    @classmethod
    def __getitem__(cls, item = None) -> Any:
        if item is None:
            return typing_List[cls]
        return typing_List[cls, item]

    @singledispatchmethod
    def __getitem__(self, idx) -> Any:
        return super(List, self).__getitem__(idx)

    @__getitem__.register(type(None))
    def _(self, idx) -> Any:
        return self

    @__getitem__.register(tuple)
    def _(self, idx) -> Any:
        if len(idx) == 0:
            return self
        if len(idx) == 1:
            return self[idx[0]]
        return self[idx[0]][Tuple(idx[1:])]

    @__getitem__.register(Callable)
    def _(self, idx) -> Any:
        return List([obj for obj in self if idx(obj)])

    @__getitem__.register(list)
    def _(self, idx) -> Any:
        return List([obj[idx] for obj in self])

    @classmethod
    def check(cls) -> bool:
        return isinstance(cls, (List, list))

    def map(self, f) -> 'List':
        return List(list(map(f, self)))

    def filter(self, f) -> 'List':
        return List(list(filter(f, self)))

    @staticmethod
    def asyncMap(func: Callable, arr: list, workers: int = 32) -> 'List':
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return List(executor.map(func, arr))

class Tuple(list):
    """
    Tuple wrapper
    """
    def __init__(self, items=None) -> None:
        if items is None:
            items = []
        if not isinstance(items, list):
            items = list(items)
        for i, v in enumerate(items):
            if isinstance(v, (list, tuple, dict))  and not isinstance(v, (List, Tuple, Dict)):
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

    @classmethod
    def __getitem__(cls, item = None) -> Any:
        if item is None:
            return typing_Tuple[cls]
        return typing_Tuple[cls, item]

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

    @classmethod
    def check(cls: Any) -> bool:
        return isinstance(cls, (tuple, Tuple))

    def map(self, f) -> 'Tuple':
        return Tuple(list(map(f, self)))

    def filter(self, f) -> 'Tuple':
        return Tuple(list(filter(f, self)))

def super_func(func: Callable) -> Callable:
    if isinstance(func, Fun):
        func = func.func
    @functools_wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        for pair in [(list, List), (tuple, Tuple), (dict, Dict)]:
            if isinstance(output, pair[0]) and not isinstance(output, pair[1]):
                return pair[1](output)
        return output
    return Fun(wrapper)

def non_super_func(func: Callable[[X], Y]) -> Callable[[X], Y]: # for clarity
    return func

@super_func
def map(f: Callable[[X], Y], L: List[X]) -> List[Y]:
    """Applies f to each element in L"""
    return [f(i) for i in L]
@super_func
def filter(f: Callable[[T], bool], L: List[T]) -> List[T]:
    """Filters elements in L based on f"""
    return [i for i in L if f(i)]

@non_super_func
def id(x: T) -> T:
    """Identity function"""
    return x

@super_func
def comp(*args) -> Fun:
    """Function composition"""
    def _(x) -> Fun:
        n_min_comp = 2
        if len(args) == n_min_comp:
            return args[0](args[1](x))
        return comp(*args[1:])(args[0](x))
    return _

@super_func
def reduce(opn: Callable[[T, T], T], values: List[T], base: T) -> T:
    """Reduces a list of values using a binary operator"""
    if len(values) == 0:
        return base
    if len(values) == 1:
        return values[0]
    return opn(values[0], reduce(opn, values[1:], base))

@super_func
def const(c: T, x: Any) -> T:
    """Constant function that always returns c"""
    return c

@super_func
def sub(pattern, oldstr, newstr) -> str:
    """Replaces occurrences of pattern in oldstr with newstr"""
    return re.sub(pattern, newstr, oldstr)

# not going to remember which way it goes, so may as well make it symmetrical
@super_func
def join(x, y) -> Any:
    if isinstance(x, str) and isinstance(y, (list, List, tuple, Tuple)):
        return x.join(y)
    return y.join(x)

@super_func
def _read(fp: str) -> str:
    with open(fp) as f:
        return f.read()

@super_func
def _readlines(fp: str) -> List:
    with open(fp) as f:
        return List(f.readlines())

read, readlines = _read, _readlines

@super_func
def nonempty(f: Callable, L: List) -> List:
    return [f(y) for y in L if f(y)]

select = Fun(lambda *args: Fun(lambda L: (lambda a, b, c: L[a:b:c])(*(slice(*args).indices(len(L))))))  # select(a)(L) = L[:a]; select(b,c)(L) = L[b:c]
match = Fun(re.match) / Dict({'groups': lambda: (3,)})
split = Fun(str.split, ('01', '10'))
strip = Fun(str.strip, args=[None])
get = Fun('a[b]')
cmd = Fun(lambda y: subprocess_run(y, capture_output=True, check=False).stdout.decode('utf-8'))
prnt = Fun(lambda x: print(x, end = ''))