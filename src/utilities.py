"""
Miscellaneous utilities for simplifying and analyzing this package's other modules.

Contains:
    def gen_stream(generator) -> str
    def print_stream(generator) -> None
    def profile(obj: Any, max_depth: int, prefix: str, visited: Optional[List], depth: int) -> List
    def defer_kwargs(factory: Callable[..., Callable[[F], F]]) -> Callable[..., Callable[[F], F]]
    @defer_kwargs
    def distribute(exclude: Optional[List[str]], output: str, tabulate_all: bool, after: Callable, threads: int) -> Callable:
    def fuzzy_in(s1: str, s2: str) -> bool
    @Fun
    def regularize(obj) -> Any
    @Fun
    def gen_pseudoword(length: int, state: int) -> str
"""

from .supertypes import *

from json import load as json_load, loads as json_loads, dump as json_dump, dumps as json_dumps
from types import MemberDescriptorType as types_MemberDescriptorType
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed

def gen_stream(generator) -> str:
    """gen_stream"""
    for chunk in generator:
        if chunk:
            # Format the chunk as a server-sent event
            yield f"data: {json_dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"

def print_stream(generator) -> None:
    """print_stream"""
    for chunk in generator:
        if chunk:
            print(chunk, end='')

def log_and_callback(func: Callable) -> Callable:
    """Decorator to add logging and callback functionality to methods."""
    @functools_wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        logging.info(f"Calling {func.__name__} on {self.__class__.__name__}")
        result = await func(self, *args, **kwargs)
        if hasattr(self, 'callback') and callable(self.callback):
            await self.callback(self, func.__name__, result)
        return result
    return wrapper

def show_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        args_str = ", ".join(map(str, args))
        kwargs_str = ", ".join((str(a)+"="+str(b) for a, b in kwargs.items()))
        nonempty_str = filter(lambda x: x.strip(), [args_str, kwargs_str])
        print(f'{func.__name__}({", ".join(nonempty_str)})')
        out = func(*args, **kwargs)
        print(f'\t = {out}')
        return out
    return wrapper

def profile(obj: Any, max_depth: int = 5, prefix: str = "", visited: Optional[List] = None, depth: int = 0) -> List:
    """
    like dir but recursive and much better in every way
    I should really put this somewhere where it"ll get more use
    max_module_depth is to prevent garden-variety modules that appear deep in from being expanded
    """
    if visited is None:
        visited = []
    MAX_LINE_LENGTH = 40
    if depth == 0:
        visited = []
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
        elif isinstance(attr, types_MemberDescriptorType):
            strings.append(type_name(f"member_descriptor: {attr.__doc__ or ''}", full_name, literal=True))
        elif not ((ismod := inspect.ismodule(attr)) or inspect.isclass(attr)):
            strings.append(type_name(attr, full_name))
        else:
            strings.append(type_name("Module" if ismod else "Class", full_name, literal=True))
            if depth in [0, 1, 2] or not ismod:
                strings += profile(attr, max_depth, full_name, visited, depth + 1)
    return strings

F = TypeVar('F', bound=Callable[..., Any])
def defer_kwargs(factory: Callable[..., Callable[[F], F]]) -> Callable[..., Callable[[F], F]]:
    """
    A decorator that allows kwargs of multiple chained decorator factories to be specified
    in a call of a decorated function by adding the prefix "_{factory_name}_" or just "_"
    if the kwarg is unique to that decorator.
    Args:
        factory (Callable): The decorator factory to be enhanced.
    Returns:
        Callable: An enhanced version of the decorator factory that supports deferred kwargs.
    """
    @functools_wraps(factory)
    def deferring_factory(*fac_args: Any, **fac_kwargs: Any) -> Callable[[F], F]:
        def wrapper(func: F) -> F:
            in_wrapped: bool = hasattr(func, '_add_opts_wrapped')
            all_factories = (func._all_factories if in_wrapped else []) + [factory]
            original_func = func._original_func if in_wrapped else func

            @functools_wraps(original_func)
            def resolving_func(*func_args: Any, **func_kwargs: Any) -> Any:
                sig = inspect.signature(original_func)
                base_params: Set[str] = set(sig.parameters.keys())

                base_kwargs: Dict[str, Any] = {k: v for k, v in func_kwargs.items() if k in base_params}
                potential_dec_kwargs: Dict[str, Any] = {k: v for k, v in func_kwargs.items() if k not in base_params}

                fac_specific_kwargs: Dict[Callable, Dict[str, Any]] = {dec: {} for dec in all_factories}
                for fac in all_factories:
                    fac_sig = inspect.signature(fac)
                    fac_params: Set[str] = set(fac_sig.parameters.keys()) - {'args', 'kwargs'}

                    prefix = f"_{fac.__name__}_"
                    prefixed_kwargs = {k[len(prefix):]: v for k, v in potential_dec_kwargs.items() if k.startswith(prefix)}
                    fac_specific_kwargs[fac].update(prefixed_kwargs)

                for k, v in potential_dec_kwargs.items():
                    if k.startswith('_') and not k.startswith(tuple(f"_{fac.__name__}_" for fac in all_factories)):
                        unprefixed_k = k[1:]
                        matching_facs = [fac for fac in all_factories if unprefixed_k in (set(inspect.signature(fac).parameters.keys()) - {'args','kwargs'})]
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

# def distribute(...)
@defer_kwargs
def distribute(exclude: Optional[List[str]] = None, output: str = "value", tabulate_all: bool = False, after: Callable = lambda **x: x, threads: int = 1) -> Callable:
    """
    factory for producing decorators that modify functions to evaluate distributively across lists
    e.g., if you have a function f(foo: int, bar: str), then prepending @distribute() allows you to call f(foo=[1,2,3],bar="a") and receive [{"foo": 1, value: f(foo=1, bar="a")}, {"foo": 2, ...}, ...]
    `exclude` prevents certain named parameters from behaving like this, e.g. if they"re already lists
    `output` controls the key that evaluation outputs are assigned to
    `tabulate_all` controls whether each dict contains all named arguments, or just the actively changing ones
    `after` is a function called upon all named parameters along with value to transform or filter outputs
    `threads` controls the number of threads to run simultaneously, default is 1 (i.e. serial), will activate parallelism when greater than 1. produces a near-linear speedup for API calls; openAI and fireworks allow these API keys up to 3000 and 600 requests per minute, so there"s no problem setting threads to say 50 (except for price)
    should set safety measures in place, since it might be easy to accidentally trigger 5400 different completions at once
    """
    if exclude is None:
        exclude = []
    if isinstance(exclude, str):
        exclude = [exclude]

    def distribute_decorator(func) -> Callable:
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
                            executor.submit(func, **dict(zip(list_args.keys(), combo))): combo
                            for combo in combinations
                        }
                        for future in as_completed(future_to_combo):
                            combo = future_to_combo[future]
                            try:
                                result = future.result()
                                combo_dict = dict(zip(list_args.keys(), combo))
                                if not tabulate_all:
                                    combo_dict = {k: v for k, v in combo_dict.items() if (isinstance(all_args.get(k), list) and k not in exclude)}
                                combo_dict[output] = result
                                results.append(after(**combo_dict))
                            except Exception as e:
                                print(f"Generated an exception: {e}")
                    return results
                for combo in combinations:
                    combo_dict = dict(zip(list_args.keys(), combo))
                    result = func(**combo_dict)
                    if not tabulate_all:
                        combo_dict = {k: v for k, v in combo_dict.items() if (isinstance(all_args.get(k), list) and k not in exclude)}
                    combo_dict[output] = result
                    results.append(after(**combo_dict))
                return results
            return func(*args, **kwargs)
        return wrapper
    return distribute_decorator

def fuzzy_in(s1: str, s2: str) -> bool:
    """fuzzy_in"""
    # !jonge algorithm
    if s1 == "" or s1 in s2:
        return True
    if len(s1) > len(s2):
        return False
    try:
        idx = s2.index(s1[0])
        return fuzzy_in(s1[1:], s2[idx + 1:])
    except Exception:
        return False

@Fun
def regularize(obj) -> Any:
    if type(obj) in {str, bool, int, float, type(None)}:
        return obj
    if type(obj) in {list, List, tuple, Tuple}:
        new_type = {list: List, List: List, tuple: Tuple, Tuple: Tuple}[type(obj)]
        return new_type([regularize(x) for x in obj])
    if type(obj) in {dict, Dict}:
        return Dict({k: regularize(v) for k, v in obj.items()})
    try:
        return regularize(obj.dict())
    except Exception:
        pass
    try:
        return regularize(obj.__dict__)
    except Exception:
        pass
    return regularize(dict(obj))

@Fun
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
    elif type(keys) in [list, List]:
        for i in keys:
            if i in obj:
                new_obj[i] = obj[i]
    elif type(keys) in [dict, Dict]:
        for (k, v) in keys.items():
            if k in obj:
                if type(obj[k]) not in [dict, Dict]:
                    if isinstance(v, str):
                        new_obj[v] = obj[k]
                    else:
                        new_obj[k] = obj[k]
                else:
                    new_obj[k] = select_keys(obj[k], v)
    return new_obj

@Fun
def gen_pseudoword(length: int, state: int = 0) -> str:
    '''
    Generates a pseudoword of a given length, for nonce IDs and such
    '''
    vowel_p = 0.3
    p, is_vowel, vowels, consonants = length, random.random() < vowel_p, 'aeio', 'bcdfgklmnprstv'
    options_dict = {i[0]: random.choice(i[1:]) for i in 'ai eia oui blr chlr dr ffl kh ll ndg ph rh sh th whr'.split()}
    choice = random.choice([vowels + consonants, vowels, consonants][state])
    return '' if p < 1 else (choice + is_vowel * options_dict.get(choice, '') + gen_pseudoword(p - 1, 1 + (choice in vowels)))[:p + state]
