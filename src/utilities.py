r"""
Miscellaneous utilities for simplifying and analyzing this package's other modules.

Contains:
    defer_kwargs               (factory: Callable[..., Decorator]) -> Callable[..., Decorator]
    log_and_callback           (func: Callable, view: Callable) -> Callable
    show_call                  (func: Callable, view: Callable) -> Callable
    freeze_args                (*f_args, **f_kwargs) -> Decorator
    distribute                 (exclude: list[str], output: str, tabulate_all: bool, after: Callable, threads: int, timeouts: dict[str, float | None] | None) -> Callable
    merge_kwargs_into          (key: str, except_for: list[str]) -> Decorator
    jsonl_cache                (path: str, ttl: int, allow_initialize: bool = False, allow_clean: bool = True) -> Decorator
    preformatting_factory      (formatters: dict[str, Callable]) -> Decorator
    router                     (func: F) -> F
    gen_stream                 (generator) -> str
    print_stream               (generator) -> None
    get_parameter_values       (func: F, args: list[Any], kwargs: Object) -> Object
    profile                    (obj: Any, max_depth: int, prefix: str, visited: list[Any] | None, depth: int) -> list[Any]
    fuzzy_in                   (s1: str, s2: str) -> bool
    regularize                 (obj: Any) -> Any
    query                      (objects: list[dict], key: str, value: Any, on_failure: dict) -> dict
    query_all                  (objects: list[dict], key: str, value: Any) -> list[dict]
    make_lines                 (text: str, row_len: int, separators: list[str], newlines: list[str]) -> list[tuple[int, str]]
    gen_pseudoword             (length: int, state: int) -> str
"""
from __future__ import annotations

from src.basetypes import *
from src.supertypes import *

import datetime
import json
import os
import random
import requests
import sqlite3
import time
import types

# meta-decorator
def defer_kwargs(
    decorator_factory: Callable[..., Decorator]
) -> Callable[..., Decorator]:
    """
    A meta-decorator that allows deferring keyword arguments for multiple chained decorator factories.

    Any kwargs that are prefixed with underscores and match the parameter names of the underlying factories
    will automatically be routed to the appropriate decorator. This is helpful for combining multiple
    decorators that need to receive distinct keyword arguments in a single function call.
    For example, if you have a decorator factory deco(arg=1) and function f(x), then
        defer_kwargs(deco)(f)(x, _deco_arg=2) will call deco(arg=2)(f)(x).

    :param decorator_factory: A function that returns a decorator (a "factory for decorators").
    :return: A callable that, when given arguments, returns a decorator which can handle deferred kwargs.
    """
    @functools.wraps(decorator_factory)
    def deferring_factory(
        *factory_args: Any,
        **factory_kwargs: Any
    ) -> Decorator:
        def deferred_decorator(func: Func) -> Func:
            in_wrapped: bool = hasattr(func, '_add_opts_wrapped')
            all_factories = (func._all_factories if in_wrapped else []) + [decorator_factory]
            original_func = func._original_func if in_wrapped else func

            @functools.wraps(original_func)
            def resolving_func(*func_args: Any, **func_kwargs: Any) -> Any:
                sig = inspect.signature(original_func)
                base_params: set[str] = set(sig.parameters.keys())

                base_kwargs: Object = {k: v for k, v in func_kwargs.items() if k in base_params}
                potential_dec_kwargs: Object = {k: v for k, v in func_kwargs.items() if k not in base_params}

                factory_specific_kwargs = {dec: {} for dec in all_factories}
                for factory in all_factories:
                    factory_sig = inspect.signature(factory)
                    factory_params: set[str] = set(factory_sig.parameters.keys())
                    factory_params -= {'args', 'kwargs'}

                    prefix = f"_{factory.__name__}_"
                    prefixed_kwargs = {k[len(prefix):]: v for k, v in potential_dec_kwargs.items() if k.startswith(prefix)}
                    factory_specific_kwargs[factory].update(prefixed_kwargs)

                for k, v in potential_dec_kwargs.items():
                    if not k.startswith('_'):
                        continue
                    print(k, factory.__name__)
                    if not k.startswith(tuple(f"_{factory.__name__}_" for factory in all_factories)):
                        unprefixed_k = k[1:]
                        matching_factories = [factory for factory in all_factories if unprefixed_k in (set(inspect.signature(factory).parameters.keys()) - {'args', 'kwargs'})]
                        if len(matching_factories) > 1:
                            msg = f"Ambiguous kwarg {unprefixed_k} belongs to {', '.join(factory.__name__ for factory in matching_factories)}"
                            raise ValueError(msg)
                        if len(matching_factories) == 1:
                            factory_specific_kwargs[matching_factories[0]][unprefixed_k] = v

                decorated_func = original_func
                for factory in reversed(all_factories):
                    updated_factory_kwargs = factory_kwargs.copy()
                    updated_factory_kwargs.update(
                        factory_specific_kwargs[factory])
                    decorated_func = factory(
                        *factory_args,
                        **updated_factory_kwargs
                    )(decorated_func)

                for factory, kwargs in factory_specific_kwargs.items():
                    for k in kwargs:
                        func_kwargs.pop(f"_{factory.__name__}_{k}", None)
                        func_kwargs.pop(f"_{k}", None)

                return decorated_func(
                    *func_args,
                    **(base_kwargs | func_kwargs)
                )

            resolving_func._add_opts_wrapped = True
            resolving_func._original_func = original_func
            resolving_func._all_factories = all_factories
            return resolving_func

        return deferred_decorator
    return deferring_factory

# decorator factory
def log_and_callback(func: Callable, view: Callable = logging.info) -> Callable:
    """
    A decorator to provide logging and optional callback behavior for async methods on a class instance.

    Each time the decorated function is called on 'self', its name and the class name will be logged via
    the provided 'view' callable, then the original function is awaited and its result is returned. If the
    instance has a 'callback' method, that callback will be invoked with signature callback(self, method_name, result).

    :param func: The async function to decorate.
    :param view: The logging function, defaults to logging.info.
    :return: An async function wrapper that logs and optionally invokes a callback.
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        view(f"Calling {func.__name__} on {self.__class__.__name__}")
        result = await func(self, *args, **kwargs)
        if hasattr(self, 'callback') and callable(self.callback):
            await self.callback(self, func.__name__, result)
        return result
    return wrapper

# decorator factory
def show_call(func: Callable, view: Callable = print) -> Callable:
    """
    A decorator to print function calls and return values for debugging.

    The decorated function, when called, prints its arguments and the resulting return value using the
    provided 'view' function (print by default). This is synchronous usage: it does not handle async calls.

    :param func: The function to decorate.
    :param view: The function used for printing (default: builtin print).
    :return: A function wrapper that prints arguments and return results.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        args_str = ", ".join(map(str, args))
        kwargs_str = ", ".join((str(a) + "=" + str(b) for a, b in kwargs.items()))
        nonempty_str = filter(lambda x: x.strip(), [args_str, kwargs_str])
        view(f'{func.__name__}({", ".join(nonempty_str)})')
        out = func(*args, **kwargs)
        view(f'\t = {out}')
        return out
    return wrapper

# decorator factory
def freeze_args(*f_args, **f_kwargs) -> Decorator:
    """
    Fix some named arguments of a function, or prepend some unnamed arguments to the function's call signature.
    This effectively creates a new function with those arguments bound or prepended, leaving the rest
    of the arguments unmodified.

    :param f_args: Positional arguments to be prepended to the function call.
    :param f_kwargs: Keyword arguments to be bound to the decorated function.
    :return: A decorator that, when applied to a function, returns a new function with the specified
             arguments fixed and/or prepended.
    """
    def decorator(fn: Func) -> Func:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            return fn(*(f_args + args), **(f_kwargs | kwargs))
        return wrapper
    return decorator

# decorator factory
@defer_kwargs
def distribute(
        exclude: str | list[str] = [],
        outputs: dict[str, str] = {"value": "value", "order": "order"},
        tabulate_all: bool = False,
        after: Func = lambda **x: x,
        threads: int = 1,
        timeouts: dict[str, float | None] | None = None
    ) -> Decorator:
    """
    A decorator factory to enable distributive (parallel or serial) execution over list-typed parameters.

    There is a special format for outputs: it is a list with one dict for each call, all with keys "value" and "order",
        along with values for all varying parameters used in the call. Thus, you often want to use after to transform
        the output of the function into the format you want, e.g. `after=lambda **x: x["value"]`.

    If a decorated function is called with any parameter(s) that are lists (except for those in 'exclude'), it will
    run the function for each combination of items in those lists, returning a list of dictionaries of results.
    Each result dictionary will have the result under outputs["value"] and a zero-based sequence index under
    outputs["order"]. Optionally, tabulate_all controls whether the resulting dictionaries include non-list
    parameters as well.

    TODO: set safety measures in place, since it might be easy to accidentally trigger 5400 different completions at once
    TODO: add a progress bar and a way to cancel the operation without losing already-computed results
    TODO: add a (contextualizable, combinable) rate limiting system
    :param exclude: Parameter name(s) to exclude from distribution, so that those are not iterated over if they are lists.
    :param outputs: Dict specifying where to place the resulting function's return value and the iteration order in the
                    final dictionary, in keys "value" and "order".
    :param tabulate_all: If True, each result contains a copy of all named arguments. If False, only the distributed
                         arguments differ in each result dict.
    :param after: A callable that post-processes each result dictionary (e.g., for filtering or transformation).
    :param threads: Number of threads for parallel execution; 1 means serial.
    :param timeouts: Optional dictionary specifying 'local' and/or 'global' timeouts in seconds.
    :return: A decorator that modifies a function to evaluate distributively over list-typed parameters.
    """
    if isinstance(exclude, str):
        exclude = [exclude]

    timeouts = timeouts or {}
    local_timeout = timeouts.get('local')
    global_timeout = timeouts.get('global')

    def distribute_decorator(func: Func) -> Func:
        """
        Modifies a function to evaluate distributively across lists.
        :param func: The function to be modified.
        :returns: The modified function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_args = dict(zip(arg_names, args))
            all_args.update(kwargs)

            if any((isinstance(v, list) and k not in exclude) for k, v in all_args.items()):
                # list_names = [key for key in all_args if isinstance(all_args[key], list) and key not in exclude]
                list_args = {k: v if (isinstance(v, list) and k not in exclude) else [v] for k, v in all_args.items()}
                combinations = list(itertools.product(*list_args.values()))
                optimal_threads = min([threads, len(combinations)])
                results = []

                if optimal_threads > 1:
                    with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                        # Submit all tasks
                        future_to_combo = {
                            executor.submit(func, **dict(zip(list_args.keys(), combo))): (combo, idx)
                            for idx, combo in enumerate(combinations)
                        }

                        try:
                            # Use as_completed with a timeout
                            for future in as_completed(future_to_combo.keys(), timeout=global_timeout):
                                combo, idx = future_to_combo[future]
                                try:
                                    combo_dict = dict(zip(list_args.keys(), combo))
                                    if not tabulate_all:
                                        combo_dict = {k: v for k, v in combo_dict.items()
                                                    if (isinstance(all_args.get(k), list) and k not in exclude)}
                                    # Use local timeout for individual tasks
                                    result = future.result(timeout=local_timeout)
                                    combo_dict[outputs["value"]] = result
                                    combo_dict[outputs["order"]] = idx
                                    results.append(after(**combo_dict))
                                except TimeoutError:
                                    print(f"Task {idx} ({str(combo_dict)[:100]}) timed out after {local_timeout} seconds")
                                    future.cancel()
                                except Exception as e:
                                    print(f"Task {idx} ({str(combo_dict)[:100]}) generated an exception: {e}")

                        except TimeoutError:
                            print(f"Global timeout after {global_timeout} seconds")
                            # Cancel all remaining futures
                            for fut in future_to_combo:
                                fut.cancel()

                    return results

                # Serial execution (when threads <= 1)
                for idx, combo in enumerate(combinations):
                    combo_dict = dict(zip(list_args.keys(), combo))
                    result = func(**combo_dict)
                    if not tabulate_all:
                        combo_dict = {k: v for k, v in combo_dict.items()
                                    if (isinstance(all_args.get(k), list) and k not in exclude)}
                    combo_dict[outputs["value"]] = result
                    combo_dict[outputs["order"]] = idx
                    results.append(after(**combo_dict))
                return results

            return func(*args, **kwargs)
        return wrapper
    return distribute_decorator

# decorator factory
def merge_kwargs_into(key: str, except_for: list[str] = []) -> Decorator:
    """
    Produces a decorator merging extra kwargs into a dictionary-type argument of a function.

    For a function with a dictionary argument 'key', calling that function with additional kwargs merges
    them into that dictionary. The 'except_for' list can specify keys to exclude from merging. This is
    handy when a function expects a dictionary argument but you want to supply some portion of it as kwargs.

    :param key: Name of the dictionary argument into which extra kwargs are merged.
    :param except_for: Names (or patterns) to exclude from merging into that dictionary.
    :return: A decorator that merges the relevant subset of kwargs into the function's dictionary argument.
    """
    def excepted(k: str) -> bool:
        return any(re.match(excuse, k) for excuse in except_for)

    def decorator(f: Func) -> Func:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            to_merge = {k: v for k, v in kwargs.items() if not (excepted(k) or k == key)}
            kwargs = {k: v for k, v in kwargs.items() if (excepted(k) or k == key)}
            kwargs[key] = {**kwargs.get(key, {}), **to_merge}
            return f(*args, **kwargs)
        return wrapper
    return decorator


# decorator factory
def jsonl_cache(path: str, ttl: int, allow_initialize: bool = False, allow_clean: bool = True) -> Decorator:
    """
    Decorator factory for caching function results in a JSON Lines file.

    Each unique function call (based on hash of function name and arguments) is stored along with an expiration
    time. If a cached result is found (and not expired), it is returned immediately. Otherwise, the function
    is called and its result is appended to the JSONL file. Expired entries can be optionally cleaned on read.

    :param path: The path to the JSONL file used for caching.
    :param ttl: Time-to-live in seconds for each cached entry.
    :param allow_initialize: If True, creates the cache file if it doesn't exist. Otherwise, raises an error if missing.
    :param allow_clean: If True, removes expired entries from the cache file on each read.
    :return: A decorator that adds JSONL-based caching to a function, keyed on function name and arguments.
    """
    def serialize(obj: Any, short_functions: bool = False, dump: bool = False) -> Any:
        """
        A helper function for serializing input objects, including functions.
        :param obj: The object to be serialized.
        :returns: The serialized object.
        """
        if isinstance(obj, Callable):
            if short_functions:
                return f"{obj.__name__}#{hash_fn(obj.__name__ + obj.__doc__)}"
            return {"name": obj.__name__, "docstring": obj.__doc__}
        try:
            dumps = json.dumps(obj)
            return dumps if dump else obj
        except (TypeError, OverflowError):
            return repr(obj)

    def hash_fn(x: Any) -> str:
        """
        A helper function for hashing input objects.
        :param x: The object to be hashed.
        :returns: The hashed object.
        """
        if not isinstance(x, str):
            x = serialize(x, dump=True)
        short = hash(x) % 2**32
        return hex(short)[2:]

    def cache_decorator(f: Func) -> Func:
        """
        Allows a function to cache its outputs in a JSONL file.
        :param f: The function whose outputs are to be cached.
        :returns: The transformed function.
        """
        @functools.wraps(f)
        def caching_func(*args, **kwargs) -> Any:
            """
            Wraps the function to cache its outputs.
            :param args: The positional arguments of the function.
            :param kwargs: The keyword arguments of the function.
            :returns: The output of the function.
            """
            serialized_args = json.dumps([serialize(arg) for arg in args])
            serialized_kwargs = json.dumps({k: serialize(v) for k, v in kwargs.items()})
            # Create a unique identifier for this function call
            call_hash = hash_fn((
                f.__name__,
                f.__doc__,
                serialized_args,
                serialized_kwargs
            ))
            # Check if the cache file exists
            if os.path.exists(path):
                do_clean = False
                do_return = (False, None)
                with open(path) as cache_file:
                    for line in cache_file:
                        entry = json.loads(line)
                        if entry['id'] == call_hash:
                            if entry['expires'] > int(time.time()):
                                do_return = (True, entry['output'])
                            elif entry['expires'] <= int(time.time()) and allow_clean:
                                do_clean = True
                if do_clean:
                    with open(path) as cache_file:
                        entries = [json.loads(line) for line in cache_file]
                    with open(path, 'w') as cache_file:
                        for entry in entries:
                            if entry['expires'] > int(time.time()):
                                json.dump(entry, cache_file)
                                cache_file.write('\n')
                if do_return[0]:
                    return do_return[1]
            elif not allow_initialize:
                msg = f"Cache file at path '{path}' does not exist"
                raise FileNotFoundError(msg)
            else:
                try:
                    with open(path, 'w') as cache_file:
                        cache_file.write('')
                except FileNotFoundError as e:
                    msg = f"Could not create cache file at path '{path}'"
                    raise FileNotFoundError(msg) from e
            # If not found in cache or expired, call the function
            result = f(*args, **kwargs)
            # Create a new cache entry
            current_time = int(time.time())
            new_entry = {
                "id": call_hash,
                "function": serialize(f, True),
                "input": {
                    "args": [serialize(arg, True) for arg in args],
                    "kwargs": {k: serialize(v, True) for k, v in kwargs.items()}
                },
                "output": result,
                "created": current_time,
                "expires": current_time + ttl,
            }
            # Append the new entry to the cache file
            try:
                with open(path, 'a') as cache_file:
                    json.dump(new_entry, cache_file)
                    cache_file.write('\n')
            except FileNotFoundError as e:
                msg = f"Could not write to cache file at path '{path}'"
                raise FileNotFoundError(msg) from e
            except (TypeError, OverflowError) as e:
                raise ValueError("Cache entry is not JSON serializable") from e
            return result
        return caching_func
    return cache_decorator

# decorator factory
def preformatting_factory(formatters: dict[str, Callable]) -> Decorator:
    """
    Creates a decorator that applies specific formatter functions to named parameters before passing them on.

    Often used to standardize or sanitize inputs (e.g., normalizing URLs) so that the decorated function doesn't
    receive multiple inconsistent representations of the same data. Each parameter in 'formatters' is replaced
    by formatters[param_name](value) if present.

    :param formatters: A dict mapping argument names to callables. Each corresponding arg is replaced with the
                       callable's return value.
    :return: A decorator that applies pre-processing to the specified function arguments.
    """
    # this decorator factory is used to ensure that the url is in the correct format before being passed to the function
    # so that the cache doesn't have to deal with multiple versions of the same url
    def preformatting_decorator(func: Func) -> Func:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            input_values = get_parameter_values(func, args, kwargs)
            f_kwargs = input_values.copy()
            for key, value in input_values.items():
                if key in formatters:
                    f_kwargs[key] = formatters[key](value)
            return func(**f_kwargs)
        return wrapper
    return preformatting_decorator

# decorator

def router(func: Func) -> Func:
    """
    A decorator that attempts to route positional arguments to the parameters based on type hints.

    This decorator matches *args to appropriate parameters by comparing their dynamic types against the
    annotated parameter types of the original function. Any leftover or ambiguous arguments remain as-is
    if they do not obviously match a typed parameter.

    :param func: The function to decorate.
    :return: The decorated function that tries to route positional arguments by analyzing function signature.
    """
    comparison_matrix = {
        'Any': { 'list': 60, 'tuple': 65, 'dict': 55, 'set': 50 },
        'list': { 'tuple': 5 , 'list': -1 },
        'tuple': { 'list': 5 , 'tuple': -1 },
        'dict': { 'set': 5 , 'dict': -1 },
        'set': { 'dict': 5 , 'tuple': 3 , 'list': 3 , 'set': -1 },
        'float': { 'int': 3, 'float': -1 },
        'int': { 'float': 3, 'int': -1 }
    }

    def match_args_to_params(args: tuple, param_names: list[str], param_types: list[Any]) -> Object:
        """Match arguments to parameters based on types."""
        matched_args = {}
        args = list(args)
        for name, param_type in zip(param_names, param_types):
            this_best = sorted(([e[0], compare_types(param_type, standardize_type_literal(e[1]))] for e in enumerate(args)), key=lambda x: x[1])
            # matched_args[name] = this_best
            i, _ = this_best[0]
            matched_args[name] = args.pop(i)
        return matched_args

    def compare_types(type1: Any, type2: Any, epsilon: float = 0.1) -> float:
        """Compare two types based on a partial order defined by weights and a comparison matrix."""
        if isinstance(type1, type):
            type1 = standardize_type(type1)
        if isinstance(type2, type):
            type2 = standardize_type(type2)
        if isinstance(type1, str) and isinstance(type2, str):
            if type1 == type2:
                return -1
            if type1 in comparison_matrix and type2 in comparison_matrix[type1]:
                return comparison_matrix[type1][type2]
            if type2 in comparison_matrix and type1 in comparison_matrix[type2]:
                return -comparison_matrix[type2][type1]

        if isinstance(type1, tuple) and isinstance(type2, tuple):
            # Compare container types
            container1, *args1 = type1
            container2, *args2 = type2
            if container1 == container2 and container1[0] in {'list', 'tuple', 'dict', 'set'}:
                return compare_types(args1[0], args2[0], epsilon * 0.01) + (compare_types(args1[1], args2[1], epsilon * 0.01) if container1 == 'dict' else 0)
            # Different container types, use comparison matrix
            if container1 in comparison_matrix and container2 in comparison_matrix[container1]:
                return comparison_matrix[container1][container2]
            if container2 in comparison_matrix and container1 in comparison_matrix[container2]:
                return -comparison_matrix[container2][container1]

        if isinstance(type1, list) and isinstance(type2, list):
            return compare_types(type1[0], type2[0], epsilon * 0.01)

        return 0.0  # If all else fails, assume they are equal

    def standardize_type(typ: Any) -> list[Any]:
        """Convert a type annotation into a standardized format."""
        origin = get_origin(typ)
        args = get_args(typ)
        if args:
            if origin is typing._UnionGenericAlias or origin is types.UnionType:
                return functools.reduce(operator.iadd, [standardize_type(arg) for arg in args], [])
            if origin in {list, List}:
                return [('list', standardize_type(args[0]))]
            if origin in {dict, Dict}:
                return [('dict', standardize_type(args[0]), standardize_type(args[1]))]
            if origin in {tuple, Tuple}:
                return [('tuple', *functools.reduce(operator.iadd, [standardize_type(arg) for arg in args], []))]
            if origin in {set, typing.Set}:
                return [('set', standardize_type(args[0]))]
        label = re.sub(r'<class [\'"](\w+)[\'"]>', r'\1', str(typ)) if isinstance(typ, type) else str(typ)
        if label.split('.')[-1].lower() in {'list', 'dict', 'tuple', 'set'}:
            return [tuple([label.split('.')[-1].lower()] + ([['Any'], ['Any']] if 'ict' in label else [['Any']]))]
        return [label]

    def standardize_type_literal(obj: Any) -> list[Any]:
        # should convert e.g. [(0, 0)] into standardize_type(list[tuple[int, int]]) = [('list', [('tuple, (['int'], ['int']))])]
        if isinstance(obj, type):
            return standardize_type(obj)
        if not isinstance(obj, (list, dict, tuple, set, List, typing.Dict, typing.Tuple, typing.Set)):
            return [re.sub(r'<class [\'"](\w+)[\'"]>', r'\1', str(type(obj)))]
        if isinstance(obj, (list, typing.List)):
            return [('list', standardize_type_literal(obj[0]))]
        if isinstance(obj, (dict, typing.Dict)):
            return [('dict', standardize_type_literal(obj[0]), standardize_type_literal(obj[1]))]
        if isinstance(obj, (tuple, typing.Tuple)):
            return [('tuple', *functools.reduce(operator.iadd, [standardize_type_literal(arg) for arg in obj], []))]
        if isinstance(obj, (set, typing.Set)):
            return [('set', standardize_type_literal(obj[0]))]
        return None

    @functools.wraps(func)
    def wrapper(*w_args: Any, **w_kwargs: Any) -> Any:
        # print()
        # Separate explicitly named arguments
        explicit_args = {}
        remaining_kwargs = {}
        for name, value in w_kwargs.items():
            if name in func_params:
                explicit_args[name] = value
            else:
                remaining_kwargs[name] = value

        # Get the remaining parameters that need to be filled
        remaining_param_names = [name for name in func_params if name not in explicit_args]
        remaining_param_types = [standardize_type(type_hints[name]) for name in remaining_param_names]
        # Match the remaining arguments to the parameters
        matched_args = match_args_to_params(w_args, remaining_param_names, remaining_param_types)
        # print(remaining_param_names, remaining_param_types, matched_args, remaining_kwargs, explicit_args)

        # Call the original function with the routed arguments
        return func(**(matched_args | remaining_kwargs | explicit_args))

    sig = inspect.signature(func)
    func_params = sig.parameters
    type_hints = get_type_hints(func)
    return wrapper

def gen_stream(generator) -> Generator[str, None, None]:
    r"""
    Wraps a data-yielding generator as a server-sent event (SSE) stream.

    For each chunk produced by 'generator', yields a string in SSE JSON format: "data: {\"text\": <chunk>}\n\n".
    Once the generator is exhausted, yields one final "data: [DONE]\n\n" marker.

    :param generator: An iterator or generator yielding data chunks (strings).
    :yield: SSE-formatted strings for each chunk.
    """
    for chunk in generator:
        if chunk:
            # Format the chunk as a server-sent event
            yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"

def print_stream(generator, view: Callable = print) -> None:
    """
    Prints each chunk from a data-yielding generator, optionally appending them to a single output.

    Intended for streaming text or SSE-like sequences. For each chunk, 'view' is called. The default is the built-in print.

    :param generator: A generator or iterator yielding data chunks.
    :param view: A callable used to output each chunk, default is print().
    :return: None
    """
    for chunk in generator:
        if chunk:
            view(chunk, end='')

def get_parameter_values(func: Func, args: list[Any], kwargs: Object) -> Object:
    """
    Retrieve bound parameter values from a function call, combining positional and keyword arguments.

    This function uses the function's signature to figure out which parameter gets which argument (including
    defaults). It returns a dictionary mapping parameter names to their final values.

    :param func: The function whose call parameters come from args and kwargs.
    :param args: The positional arguments as called.
    :param kwargs: The keyword arguments as called.
    :return: A dictionary of resolved parameter names mapping to the argument values in the call.
    """
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments.items())

def profile(
    obj: Any,
    max_depth: int = 5,
    prefix: str = "",
    visited: list | None = None,
    depth: int = 0
) -> list:
    """
    Recursively examines the attributes of 'obj', returning a list of descriptive strings.

    Skips special methods (names like __x__) and re-inspects objects up to a certain depth to avoid recursion loops.
    If an attribute is a container (list, dict, etc.), it may be partially expanded. This is intended to provide a
    broad overview of an object's structure, useful for debugging.

    :param obj: The root object to profile.
    :param max_depth: Maximum recursion depth.
    :param prefix: A string prefix used for indentation or naming in the output lines.
    :param visited: A list of object IDs visited so far (to prevent repeated recursion).
    :param depth: The current depth of the recursion.
    :return: A list of strings describing each attribute encountered.
    """
    from types import MemberDescriptorType
    if visited is None or depth == 0:
        visited = []
    MAX_LINE_LENGTH = 40
    if depth > max_depth or id(obj) in visited:
        return []
    visited.append(id(obj))

    prefix = "\t" * depth
    type_name = lambda attr, full="", literal=False: prefix + (full + ": " if full else "") + (attr if literal else type(attr).__name__)
    sample = lambda gen: next(iter(gen))
    strings = []
    # Get all attributes
    attrs = dir(obj)
    for attr_name in attrs:
        full_name = f"{prefix}.{attr_name}" if prefix and depth == 0 else attr_name
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue  # Skip special methods
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            strings.append(type_name("[inaccessible]", full_name, literal=True))
            continue

        if isinstance(attr, (int, float, bool)):
            strings.append(type_name(str(attr), full_name, literal=True))
        elif isinstance(attr, str):
            if len(attr) < MAX_LINE_LENGTH:
                strings.append(type_name(f'"{attr}"', full_name, literal=True))
            else:
                strings.append(type_name(f'"{attr[:MAX_LINE_LENGTH]}[{len(attr) - 10} more...]"', full_name, literal=True))
        elif (islist := isinstance(attr, list)) or isinstance(attr, (tuple, set)):
            if attr:
                element_type = type_name(sample(attr)).strip()
                strings.append(f"{type_name(attr, full_name)}[{element_type}]")
            else:
                strings.append(type_name(("empty " + type(attr).__name__ if not islist else "[]"), full_name, literal=True))
        elif isinstance(attr, dict):
            if attr:
                key_type = type_name(sample(attr.keys())).strip()
                value_type = type_name(sample(attr.values())).strip()
                strings.append(type_name(f"dict[{key_type}, {value_type}]", full_name, literal=True))
            else:
                strings.append(type_name("{}", full_name, literal=True))
        elif inspect.isroutine(attr):
            # This will catch functions, methods, built-in functions, and other callable objects
            if inspect.isfunction(attr):
                label = "function"
            elif inspect.ismethod(attr):
                label = "method"
            else:
                label = "routine"
            try:
                sig = inspect.signature(attr)
                params = []
                for p in sig.parameters.values():
                    if isinstance(p.annotation, str):
                        p_type = p.annotation
                    else:
                        try:
                            p_type = p.annotation.__name__
                        except Exception:
                            p_type = ""
                    p_type = (": " + p_type) if (p_type and "empty" not in p_type) else ""
                    p_def = " = " + repr(p.default) if p.default != inspect.Parameter.empty else ""
                    params.append(f"{p.name}{p_type}{p_def}")
                if isinstance(sig.return_annotation, str):
                    out_type = sig.return_annotation
                else:
                    try:
                        out_type = sig.return_annotation.__name__
                    except Exception:
                        out_type = ""
                return_anno = (" -> " + out_type) if (out_type and "empty" not in out_type) else ""
                # for p in sig.parameters.values():
                # 	p_type = (": " + p.annotation.__name__) if p.annotation != inspect.Parameter.empty else ""
                # 	p_def = " = " + repr(p.default) if p.default != inspect.Parameter.empty else ""
                # 	params.append(f"{p.name}{p_type}{p_def}")
                # return_anno = (" -> " + sig.return_annotation.__name__) if sig.return_annotation != inspect.Signature.empty else ""
                strings.append(type_name(f"{label}({', '.join(params)}){return_anno}", full_name, literal=True))
            except (ValueError, TypeError, UnicodeEncodeError):
                # If we can"t get a signature, fall back to the docstring
                doc = attr.__doc__
                if doc:
                    first_line = doc.split("\n")[0]
                    strings.append(type_name(f"{label}: {first_line}", full_name, literal=True))
                else:
                    strings.append(type_name(f"{label}", full_name, literal=True))
        elif isinstance(attr, MemberDescriptorType):
            strings.append(type_name(f"member_descriptor: {attr.__doc__ or ''}", full_name, literal=True))
        elif not ((ismod := inspect.ismodule(attr)) or inspect.isclass(attr)):
            strings.append(type_name(attr, full_name))
        else:
            strings.append(type_name("Module" if ismod else "Class", full_name, literal=True))
            if depth in {0, 1, 2} or not ismod:
                strings += profile(attr, max_depth, full_name, visited, depth + 1)
    return strings

def fuzzy_in(s1: str, s2: str) -> bool:
    """
    Check if s1 is a subsequence (not necessarily contiguous) of s2.

    This function attempts to match each character of s1 in order within s2. If s1 appears within s2 in
    the correct sequence (though not necessarily contiguously), returns True; otherwise False.

    :param s1: The potential subsequence string.
    :param s2: The string to search within.
    :return: True if s1 is a non-contiguous subsequence of s2, else False.
    """
    # !jonge algorithm
    if not s1 or s1 in s2:
        return True
    if len(s1) > len(s2):
        return False
    try:
        idx = s2.index(s1[0])
        return fuzzy_in(s1[1:], s2[idx + 1:])
    except Exception:
        return False

@super_func
def regularize(obj: Any) -> Any:
    """
    Convert various container or model objects to a more standard dictionary/list/primitive structure.

    If 'obj' is a list-like or dict-like structure (or a pydantic model, or something with a `dict()` method),
    it recursively converts them into standard Python lists or dictionaries. Primitives remain unchanged.

    :param obj: Any object or nested container.
    :return: A regularized version of 'obj' where container elements become built-in dicts, lists, or primitives.
    """
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, (list, List, tuple, Tuple)):
        return [regularize(x) for x in obj]
    if isinstance(obj, (dict, Dict)):
        return {k: regularize(v) for k, v in obj.items()}
    regularization_methods = [lambda x: x.__dict__, lambda x: x.dict(), dict]
    r = None
    for method in regularization_methods:
        try:
            r = method(obj)
            if not isinstance(r, dict):
                raise TypeError("Output of regularize not a dict")
        except Exception:
            pass  # print(type(obj), str(obj)[:20], e)
    return regularize(r)


@super_func
def select_keys(obj, keys) -> Any:
    """
    Recursively select or rename keys in a nested dictionary-like object.

    - If 'keys' is a string, returns the value of that key in 'obj' if it exists, else None.
    - If 'keys' is a list, returns a Dict containing only those key-value pairs whose keys are in the list.
    - If 'keys' is a dict, allows renaming (string to string) or further nested selects:
        {outer_key: new_key, outer_key2: {inner_key: new_name, ...}}

    :param obj: A dictionary-like object to select from.
    :param keys: The key(s) to select. May be string, list, or recursively structured dict.
    :return: A new object or value following the selection/renaming pattern.
    """
    new_obj = Dict({})
    if isinstance(keys, str):
        new_obj = obj.get(keys, None)
    elif type(keys) in {list, List}:
        for i in keys:
            if i in obj:
                new_obj[i] = obj[i]
    elif type(keys) in {dict, Dict}:
        for (k, v) in keys.items():
            if k in obj:
                if type(obj[k]) not in {dict, Dict}:
                    if isinstance(v, str):
                        new_obj[v] = obj[k]
                    else:
                        new_obj[k] = obj[k]
                else:
                    new_obj[k] = select_keys(obj[k], v)
    return new_obj

def query(objects: list[dict], key: str, value: Any, on_failure: dict = {}) -> dict:
    """
    Retrieve the first dictionary from 'objects' whose [key] matches 'value'. If not found, return on_failure.

    :param objects: List of dictionaries to search.
    :param key: The key in each dictionary to check.
    :param value: The matching criterion for that key.
    :param on_failure: A fallback dictionary if no match is found.
    :return: The first matching dictionary, or 'on_failure' if none match.
    """
    for obj in objects:
        if key in obj and obj[key] == value:
            return obj
    return on_failure

def query_all(objects: list[dict], key: str, value: Any) -> list[dict]:
    """
    Return all dictionaries from 'objects' whose [key] matches 'value'.

    :param objects: List of dictionaries to search.
    :param key: The key in each dictionary to check.
    :param value: The matching criterion for that key.
    :return: A list of all dictionaries where dict[key] == value.
    """
    valid = List([])
    for obj in objects:
        if key in obj and obj[key] == value:
            valid.append(obj)
    return valid

def make_lines(
    text: str,
    row_len: int = 80,
    separators: list[str] = [" "],
    newlines: list[str] = ["\n"]
) -> list[tuple[int, str]]:
    """
    Break a chunk of text into lines, respecting a maximum row length and optional newline characters.

    The function slices text into lines of up to 'row_len' characters, trying to cut at a suitable separator.
    If a newline character is encountered, it forces a line break.

    :param text: The input text to be split.
    :param row_len: Maximum characters allowed on a single line.
    :param separators: Characters considered as valid breakpoints within a line.
    :param newlines: Characters that force an immediate newline.
    :return: A list of lines in the form (line_number, line_content).
    """

    lines = []
    current_line = ""
    line_number = 1

    i = 0
    while i < len(text):
        char = text[i]

        if char in newlines:
            # Start a new line
            lines.append((line_number, current_line))
            line_number += 1
            current_line = ""
            i += 1  # Move to the next character after the newline

        elif len(current_line) + 1 > row_len:
            # Line is full, try to split at a separator

            # Find the last separator within the allowed line length
            last_separator_index = -1
            for j in range(0, len(current_line) - 1, -1):
                if current_line[j] in separators:
                    last_separator_index = j
                    break
            if last_separator_index != -1:
                # Split at the last separator
                lines.append((line_number, current_line[: last_separator_index]))
                line_number += 1
                current_line = current_line[last_separator_index:]
            else:
                while i < len(text) and text[i] not in separators + newlines:
                    current_line += text[i]
                    i += 1
                lines.append((line_number, current_line))
                line_number += 1
                current_line = text[i] if i < len(text) and text[i] not in newlines else ""
            i += 1
        else:
            # Add the character to the current line
            current_line += char
            i += 1

    # Append the last line
    if current_line:
        lines.append((line_number, current_line))

    return lines

# make_lines("""When I was young, I'd listen to the radio, waiting for my favorite songs.""", 8)

@super_func
def gen_pseudoword(length: int, state: int = 0) -> str:
    """
    Generates a pseudoword of a given length, potentially for placeholders or random short IDs.

    Uses a naive approach with vowels and consonants, occasionally appending combined patterns.
    The 'state' parameter influences random choice among vowels/consonants, for the recursive call.

    :param length: The number of characters to include in the pseudoword.
    :param state: A numeric state that can randomly shift the balance of consonant/vowel patterns. Leave it at 0.
    :return: A randomly generated pseudoword string of length 'length'.
    """
    vowel_p = 0.3
    p, is_vowel, vowels, consonants = length, random.random() < vowel_p, 'aeio', 'bcdfgklmnprstv'
    options_dict = {i[0]: random.choice(i[1:]) for i in ['ai', 'eia', 'oui', 'blr', 'chlr', 'dr', 'ffl', 'kh', 'll', 'ndg', 'ph', 'rh', 'sh', 'th', 'whr']}
    choice = random.choice([vowels + consonants, vowels, consonants][state])
    return '' if p < 1 else (choice + is_vowel * options_dict.get(choice, '') + gen_pseudoword(p - 1, 1 + (choice in vowels)))[:p + state]
