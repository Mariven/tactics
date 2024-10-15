r"""
Miscellaneous utilities for simplifying and analyzing this package's other modules.

With:
    T, X, Y generic
    End       := Callable[[T], T]
    F         := Callable[..., Any]
    Object    := Dict[str, Any]
    Decorator := End[F]
    Pipe      := Callable[[str, Optional[Object]], str]
Contains:
    defer_kwargs
        (factory: Callable[..., Decorator]) -> Callable[..., Decorator]
    log_and_callback
        (func: Callable, view: Callable = logging.info) -> Callable
    show_call
        (func: Callable, view: Callable = print) -> Callable
    freeze_args
        (*f_args, **f_kwargs) -> Decorator
    distribute
        (exclude: List[str], output: str, tabulate_all: bool, after: Callable, threads: int) -> Callable
    jsonl_cache
        (path: str, ttl: int, allow_initialize: bool = False, allow_clean: bool = True) -> Decorator
    preformatting_factory
        (formatters: Dict[str, Callable]) -> Decorator
    router
        (func: F) -> F
    gen_stream
        (generator) -> str
    print_stream
        (generator) -> None
    get_parameter_values
        (func: F, args: List[Any], kwargs: Object) -> Object
    profile
        (obj: Any, max_depth: int, prefix: str, visited: Optional[List], depth: int) -> List
    fuzzy_in
        (s1: str, s2: str) -> bool
    regularize
        (obj: Any) -> Any
    query
        (objects: List[Dict], key: str, value: Any, on_failure: Dict) -> Dict
    query_all
        (objects: List[Dict], key: str, value: Any) -> List[Dict]
    make_lines
        (text: str, row_len: int, separators: List[str], newlines: List[str]) -> List[Tuple[int, str]]
    gen_pseudoword
        (length: int, state: int) -> str
"""

from .supertypes import *

from json import load as json_load, loads as json_loads, dump as json_dump, dumps as json_dumps
from os import path as os_path
from time import time as time_time
import inspect, operator, sqlite3, types

Object = typing_Dict[str, Any]
Pipe = Callable[[str, Optional[Object]], str]

# meta-decorator
def defer_kwargs(factory: Callable[..., Decorator]) -> Callable[..., Decorator]:
    """
    A meta-decorator that allows kwargs of multiple chained decorator factories to be specified
    in a call of a decorated function by adding the prefix "_{factory_name}_" or just "_"
    if the kwarg is unique to that decorator.
    :param factory: The decorator factory to be enhanced.
    :returns: An enhanced version of the decorator factory that supports deferred kwargs.
    """
    @functools_wraps(factory)
    def deferring_factory(*fac_args: Any, **fac_kwargs: Any) -> Decorator:
        def wrapper(func: F) -> F:
            in_wrapped: bool = hasattr(func, '_add_opts_wrapped')
            all_factories = (func._all_factories if in_wrapped else []) + [factory]
            original_func = func._original_func if in_wrapped else func

            @functools_wraps(original_func)
            def resolving_func(*func_args: Any, **func_kwargs: Any) -> Any:
                sig = inspect.signature(original_func)
                base_params: Set[str] = set(sig.parameters.keys())

                base_kwargs: Object = {k: v for k, v in func_kwargs.items() if k in base_params}
                potential_dec_kwargs: Object = {k: v for k, v in func_kwargs.items() if k not in base_params}

                fac_specific_kwargs: Dict[Callable, Object] = {dec: {} for dec in all_factories}
                for fac in all_factories:
                    fac_sig = inspect.signature(fac)
                    fac_params: Set[str] = set(fac_sig.parameters.keys()) - {'args', 'kwargs'}

                    prefix = f"_{fac.__name__}_"
                    prefixed_kwargs = {k[len(prefix):]: v for k, v in potential_dec_kwargs.items() if k.startswith(prefix)}
                    fac_specific_kwargs[fac].update(prefixed_kwargs)

                for k, v in potential_dec_kwargs.items():
                    if k.startswith('_') and not k.startswith(tuple(f"_{fac.__name__}_" for fac in all_factories)):
                        unprefixed_k = k[1:]
                        matching_facs = [fac for fac in all_factories if unprefixed_k in (set(inspect.signature(fac).parameters.keys()) - {'args', 'kwargs'})]
                        if len(matching_facs) > 1:
                            msg = f"Ambiguous kwarg {unprefixed_k} belongs to {', '.join(fac.__name__ for fac in matching_facs)}"
                            raise ValueError(msg)
                        if len(matching_facs) == 1:
                            fac_specific_kwargs[matching_facs[0]][unprefixed_k] = v

                decorated_func = original_func
                for fac in reversed(all_factories):
                    updated_fac_kwargs = fac_kwargs.copy()
                    updated_fac_kwargs.update(fac_specific_kwargs[fac])
                    decorated_func = fac(*fac_args, **updated_fac_kwargs)(decorated_func)

                for fac, kwargs in fac_specific_kwargs.items():
                    for k in kwargs:
                        func_kwargs.pop(f"_{fac.__name__}_{k}", None)
                        func_kwargs.pop(f"_{k}", None)

                return decorated_func(*func_args, **(base_kwargs | func_kwargs))

            resolving_func._add_opts_wrapped = True
            resolving_func._original_func = original_func
            resolving_func._all_factories = all_factories
            return resolving_func

        return wrapper
    return deferring_factory

# decorator factory
def log_and_callback(func: Callable, view: Callable = logging.info) -> Callable:
    """
    Decorator to add logging and callback functionality to methods.
    :param func: The function to be decorated.
    :param view: The function to use for logging.
    :returns: The decorated function.
    """
    @functools_wraps(func)
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
    Decorator to print the arguments and return value of a function when it is called.
    :param func: The function to be decorated.
    :returns: The decorated function.
    """
    @functools_wraps(func)
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
    Fixes some named arguments of a function, or prepends some unnamed arguments.
    :param f_args: The unnamed arguments to be prepended.
    :param f_kwargs: The named arguments to be fixed.
    :return: A decorator that fixes the arguments of a function.
    """
    def decorator(fn: F) -> F:
        @functools_wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            return fn(*(f_args + args), **(f_kwargs | kwargs))
        return wrapper
    return decorator

# decorator factory
@defer_kwargs
def distribute(
        exclude: Union[str, List[str]] = [],
        outputs: Dict[str, str] = {"value": "value", "order": "order"},
        tabulate_all: bool = False,
        after: F = lambda **x: x,
        threads: int = 1
    ) -> Decorator:
    """
    Factory for producing decorators that modify functions to evaluate distributively across lists. E.g., if you have a function f(foo: int, bar: str), then prepending @distribute() allows you to call f(foo=[1,2,3],bar="a") and receive [{foo: 1, order: 0, value: f(foo=1, bar="a")}, {foo: 2, order: 1...}, ...].
    TODO: set safety measures in place, since it might be easy to accidentally trigger 5400 different completions at once
    TODO: add a progress bar and a way to cancel the operation without losing already-computed results
    TODO: add a (contextualizable, combinable) rate limiting system
    :param exclude: Prevents certain named parameters from being distributed over if lists, e.g. if they"re lists originally
    :param outputs: The keys that the utility outputs are assigned to in the resulting dicts.
    :param tabulate_all: Controls whether each dict contains all named arguments, or just the actively changing ones.
    :param after: A function called upon all named parameters along with value to transform or filter outputs.
    :param threads: Controls the number of threads to run simultaneously, default is 1 (i.e. serial), will activate parallelism when greater than 1.
    :return: A decorator that modifies a function to evaluate distributively across lists.
    """
    if isinstance(exclude, str):
        exclude = [exclude]

    def distribute_decorator(func: F) -> F:
        """
        Modifies a function to evaluate distributively across lists.
        :param func: The function to be modified.
        :return: The modified function.
        """
        @functools_wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_args = dict(zip(arg_names, args))
            all_args.update(kwargs)
            if any((isinstance(v, list) and k not in exclude) for k, v in all_args.items()):
                # list_names = [key for key in all_args if isinstance(all_args[key], list) and key not in exclude]
                list_args = {k: v if (isinstance(v, list) and k not in exclude) else [v] for k, v in all_args.items()}
                combinations = list(itertools_product(*list_args.values()))
                optimal_threads = min([threads, len(combinations)])
                results = []
                if optimal_threads > 1:
                    with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                        future_to_combo = {
                            executor.submit(func, **dict(zip(list_args.keys(), combo))): (combo, idx)
                            for idx, combo in enumerate(combinations)
                        }
                        for future in as_completed(future_to_combo):
                            combo, idx = future_to_combo[future]
                            try:
                                result = future.result()
                                combo_dict = dict(zip(list_args.keys(), combo))
                                if not tabulate_all:
                                    combo_dict = {k: v for k, v in combo_dict.items() if (isinstance(all_args.get(k), list) and k not in exclude)}
                                combo_dict[outputs["value"]] = result
                                combo_dict[outputs["order"]] = idx
                                results.append(after(**combo_dict))
                            except Exception as e:
                                print(f"Generated an exception: {e}")
                    return results
                for idx, combo in enumerate(combinations):
                    combo_dict = dict(zip(list_args.keys(), combo))
                    result = func(**combo_dict)
                    if not tabulate_all:
                        combo_dict = {k: v for k, v in combo_dict.items() if (isinstance(all_args.get(k), list) and k not in exclude)}
                    combo_dict[outputs["value"]] = result
                    combo_dict[outputs["order"]] = idx
                    results.append(after(**combo_dict))
                return results
            return func(*args, **kwargs)
        return wrapper
    return distribute_decorator

# decorator factory
def jsonl_cache(path: str, ttl: int, allow_initialize: bool = False, allow_clean: bool = True) -> Decorator:
    """
    Factory for producing decorators that cache the results of functions in a specified JSONL file.
    :param path: The path to the cache file.
    :param ttl: The time-to-live of the cache entries in seconds.
    :param allow_initialize: Whether to allow the cache file to be created if it does not exist.
    :param allow_clean: Whether to delete expired cache entries on the spot.
    :return: A decorator that caches the results of a function.
    """
    def serialize(obj: Any, short_functions: bool = False, dump: bool = False) -> Any:
        """
        A helper function for serializing input objects, including functions.
        :param obj: The object to be serialized.
        :return: The serialized object.
        """
        if isinstance(obj, Callable):
            if short_functions:
                return f"{obj.__name__}#{hash_fn(obj.__name__ + obj.__doc__)}"
            return {"name": obj.__name__, "docstring": obj.__doc__}
        try:
            dumps = json_dumps(obj)
            return dumps if dump else obj
        except (TypeError, OverflowError):
            return repr(obj)

    def hash_fn(x: Any) -> str:
        """
        A helper function for hashing input objects.
        :param x: The object to be hashed.
        :return: The hashed object.
        """
        if not isinstance(x, str):
            x = serialize(x, dump=True)
        short = hash(x) % 2**32
        return hex(short)[2:]

    def cache_decorator(f: F) -> F:
        """
        Allows a function to cache its outputs in a JSONL file.
        :param f: The function whose outputs are to be cached.
        :return: The transformed function.
        """
        @functools_wraps(f)
        def caching_func(*args, **kwargs) -> Any:
            """
            Wraps the function to cache its outputs.
            :param args: The positional arguments of the function.
            :param kwargs: The keyword arguments of the function.
            :return: The output of the function.
            """
            serialized_args = json_dumps([serialize(arg) for arg in args])
            serialized_kwargs = json_dumps({k: serialize(v) for k, v in kwargs.items()})
            # Create a unique identifier for this function call
            call_hash = hash_fn((
                f.__name__,
                f.__doc__,
                serialized_args,
                serialized_kwargs
            ))
            # Check if the cache file exists
            if os_path.exists(path):
                do_clean = False
                do_return = (False, None)
                with open(path) as cache_file:
                    for line in cache_file:
                        entry = json_loads(line)
                        if entry['id'] == call_hash:
                            if entry['expires'] > int(time_time()):
                                do_return = (True, entry['output'])
                            elif entry['expires'] <= int(time_time()) and allow_clean:
                                do_clean = True
                if do_clean:
                    with open(path) as cache_file:
                        entries = [json_loads(line) for line in cache_file]
                    with open(path, 'w') as cache_file:
                        for entry in entries:
                            if entry['expires'] > int(time_time()):
                                json_dump(entry, cache_file)
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
            current_time = int(time_time())
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
                    json_dump(new_entry, cache_file)
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
def preformatting_factory(formatters: Dict[str, F]) -> Decorator:
    """
    A decorator factory that allows you to specify formatters for any of the input parameters of a function before it is called.
    :param formatters: A dictionary mapping parameter names to formatter functions that should be applied to the input values.
    :return: A decorator that applies the specified formatters to the input values before calling the function.
    """
    # this decorator factory is used to ensure that the url is in the correct format before being passed to the function
    # so that the cache doesn't have to deal with multiple versions of the same url
    def preformatting_decorator(func: F) -> F:
        @functools_wraps(func)
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

def router(func: F) -> F:
	"""
	Decorator that routes arguments to parameters based on type hints.
	:param func: The function to decorate
	:return: The decorated function capable of reordering inputs to match parameters
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

	def match_args_to_params(args: Tuple[Any, ...], param_names: List[str], param_types: List[Any]) -> Object:
		"""Match arguments to parameters based on types."""
		matched_args = {}
		args = list(args)
		for name, param_type in zip(param_names, param_types):
			this_best = sorted(([e[0], compare_types(param_type, standardize_type_literal(e[1]))] for e in enumerate(args)), key=lambda x: x[1])
			# matched_args[name] = this_best
			i, diff = this_best[0]
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

	def standardize_type(typ: Any) -> List[Any]:
		"""Convert a type annotation into a standardized format."""
		origin = get_origin(typ)
		args = get_args(typ)
		if args:
			if origin is Union or origin is types.UnionType:
				return functools.reduce(operator.iadd, [standardize_type(arg) for arg in args], [])
			if origin in {list, List}:
				return [('list', standardize_type(args[0]))]
			if origin in {dict, Dict}:
				return [('dict', standardize_type(args[0]), standardize_type(args[1]))]
			if origin in {tuple, Tuple}:
				return [('tuple', *functools.reduce(operator.iadd, [standardize_type(arg) for arg in args], []))]
			if origin in {set, Set}:
				return [('set', standardize_type(args[0]))]
		label = re.sub(r'<class [\'"](\w+)[\'"]>', r'\1', str(typ)) if isinstance(typ, type) else str(typ)
		if label.split('.')[-1].lower() in {'list', 'dict', 'tuple', 'set'}:
			return [tuple([label.split('.')[-1].lower()] + ([['Any'], ['Any']] if 'ict' in label else [['Any']]))]
		return [label]

	def standardize_type_literal(obj: Any) -> List[Any]:
		# should convert e.g. [(0, 0)] into standardize_type(List[Tuple[int, int]]) = [('list', [('tuple, (['int'], ['int']))])]
		if isinstance(obj, type):
			return standardize_type(obj)
		if not isinstance(obj, (list, dict, tuple, set, List, Dict, Tuple, Set)):
			return [re.sub(r'<class [\'"](\w+)[\'"]>', r'\1', str(type(obj)))]
		if isinstance(obj, (list, List)):
			return [('list', standardize_type_literal(obj[0]))]
		if isinstance(obj, (dict, Dict)):
			return [('dict', standardize_type_literal(obj[0]), standardize_type_literal(obj[1]))]
		if isinstance(obj, (tuple, Tuple)):
			return [('tuple', *functools.reduce(operator.iadd, [standardize_type_literal(arg) for arg in obj], []))]
		if isinstance(obj, (set, Set)):
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

def gen_stream(generator) -> str:
    """
    Generates a stream of server-sent events from a generator.
    :param generator: A generator that yields chunks of data.
    :yields: Formatted server-sent event strings.
    """
    for chunk in generator:
        if chunk:
            # Format the chunk as a server-sent event
            yield f"data: {json_dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"

def print_stream(generator, view: Callable = print) -> None:
    """
    Prints a stream of server-sent events from a generator.
    :param generator: A generator that yields chunks of data.
    :param view: The function to use for printing the server-sent event strings.
    :yields: none.
    """
    for chunk in generator:
        if chunk:
            view(chunk, end='')

def get_parameter_values(func: F, args: List[Any], kwargs: Object) -> Object:
    """
    Get the values of the parameters of a function that were passed as arguments.
        E.g. for calling a function f(a, b=1, c=-1) as f(2, c=9), this function returns {'a':2, 'b':1, 'c':9}.
    :param func: The function to get the parameter values of.
    :param args: The positional arguments that were passed to the function.
    :param kwargs: The keyword arguments that were passed to the function.
    :return: A dictionary mapping all parameter names to their values.
    """
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments.items())

def profile(obj: Any, max_depth: int = 5, prefix: str = "", visited: Optional[List] = None, depth: int = 0) -> List:
    """
    like dir but recursive and much better in every way
    I should really put this somewhere where it"ll get more use
    max_module_depth is to prevent garden-variety modules that appear deep in from being expanded
    """
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
    """fuzzy_in"""
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
def regularize(obj) -> Any:
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
        except Exception as e:
            pass  # print(type(obj), str(obj)[:20], e)
    return regularize(r)


@super_func
def select_keys(obj, keys) -> Any:
    """
    renaming:  {link:foo.com}, {link:url} -> {url:foo.com}
    selection: {a:1,b:2,...,z:26}, d -> 4
    filtering: {a:1,b:2,...,z:26}, [a,c,v] -> {a:1,c:3,v:22}
    recursion: {ab:{om:{a:1,z:26},du:ff}}, {ab:{du:ee,om:[b,z]}} -> {ab:{ee:ff,om:{z:26}}}
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

def query(objects: List[Dict], key: str, value: Any, on_failure: Dict = {}) -> Dict:
    """
    Get the first object from a list of dictionaries that matches a key==value query.
    If no match is found, return on_failure.
    :param objects: List of dictionaries to search through.
    :param key: Key to search for.
    :param value: Value to search for.
    :param on_failure: Value to return if no match is found.
    :return: First object that matches the query, or on_failure if no match is found.
    """
    for obj in objects:
        if key in obj and obj[key] == value:
            return obj
    return on_failure

def query_all(objects: List[Dict], key: str, value: Any) -> List[Dict]:
    """
    Get all objects from a list of dictionaries that match a key==value query.
    :param objects: List of dictionaries to search through.
    :param key: Key to search for.
    :param value: Value to search for.
    :return: List of all objects that match the query.
    """
    valid = List([])
    for obj in objects:
        if key in obj and obj[key] == value:
            valid.append(obj)
    return valid

def make_lines(text: str, row_len: int = 80, separators: List[str] = [" "], newlines: List[str] = ["\n"]) -> List[Tuple[int, str]]:
    """
    Parses text into lines, respecting row length and separators.
    :param text: The text to be parsed.
    :param row_len: The maximum length of each line.
    :param separators: A list of characters considered separators.
    :param newlines: A list of characters that trigger a new line.
    :return: A list of lines, where each line is a list containing its 1-based index and the line content as a string.
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
    '''
    Generates a pseudoword of a given length, for nonce IDs and such
    '''
    vowel_p = 0.3
    p, is_vowel, vowels, consonants = length, random.random() < vowel_p, 'aeio', 'bcdfgklmnprstv'
    options_dict = {i[0]: random.choice(i[1:]) for i in 'ai eia oui blr chlr dr ffl kh ll ndg ph rh sh th whr'.split()}
    choice = random.choice([vowels + consonants, vowels, consonants][state])
    return '' if p < 1 else (choice + is_vowel * options_dict.get(choice, '') + gen_pseudoword(p - 1, 1 + (choice in vowels)))[:p + state]
