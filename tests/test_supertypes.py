"""
Test the supertypes module
"""
from functools import reduce
import pytest
from hypothesis import given, strategies as st
from src.supertypes import List, Dict, Fun, Symbol, Tuple, this, arm
from src.basetypes import preemptable
from typing import Any, Callable

@pytest.fixture
def sample_list() -> List:
    return List([1, 2, 3, 4, 5])

@pytest.fixture
def sample_dict() -> Dict:
    return Dict({'a': 1, 'b': 2, 'c': 3})

class TestList:
    def test_init(self, sample_list) -> None:
        assert sample_list == [1, 2, 3, 4, 5]
        # Test nested container conversion
        nested = List([{'a': [1, 2]}, [3, 4], (5, 6)])
        assert isinstance(nested[0], Dict)
        assert isinstance(nested[1], List)
        assert isinstance(nested[2], Tuple)

    def test_map(self, sample_list) -> None:
        result = sample_list.map(lambda x: x * 2)
        assert result == [2, 4, 6, 8, 10]
        # Test chained mapping
        result = sample_list.map(lambda x: x * 2, lambda x: x + 1)
        assert result == [3, 5, 7, 9, 11]

    def test_filter(self, sample_list) -> None:
        result = sample_list.filter(lambda x: x % 2 == 0)
        assert result == [2, 4]
        # Test chained filtering
        result = sample_list.filter(lambda x: x > 2, lambda x: x < 5)
        assert result == [3, 4]

    def test_getitem(self, sample_list) -> None:
        assert sample_list[1] == 2
        assert sample_list[1:] == [2, 3, 4, 5]
        # Test advanced indexing
        assert sample_list[[0, 2, 4]] == [1, 3, 5]
        assert sample_list[lambda x: x > 3] == [4, 5]

    def test_getattr(self, sample_list) -> None:
        assert sample_list.head == 1
        assert sample_list.tail == List([2, 3, 4, 5])
        # Test attribute access on list of dicts
        lst = List([{'a': 1}, {'a': 2}])
        assert lst.a == [1, 2]

    def test_check(self) -> None:
        assert List.check(List([1, 2, 3])) is True
        assert List.check([1, 2, 3]) is True
        assert List.check({'a': 1}) is False
        assert List.check(None) is False

    def test_for_each(self, sample_list) -> None:
        original = sample_list.copy()
        sample_list.forEach(lambda x: x * 3)
        assert sample_list == [v * 3 for v in original]
        # Test chained forEach
        sample_list.forEach(lambda x: x + 1, lambda x: x * 2)
        assert sample_list == [(v * 3 + 1) * 2 for v in original]

    def test_async_map(self, sample_list) -> None:
        f = lambda x: x + 10
        result = List.async_map(f, sample_list, workers=2)
        assert result == [x + 10 for x in sample_list]
        # Test with exception handling
        g = lambda x: 1 / x if x != 3 else None
        result = List.async_map(g, sample_list)
        assert None in result

    def test_strip(self) -> None:
        lst = List([None, None, 1, 2, None])
        result = lst.strip(None)
        assert result == [1, 2]
        # Test with other values
        lst = List([0, 0, 1, 2, 0, 3, 0, 0])
        result = lst.strip(0)
        assert result == [1, 2, 0, 3]

    def test_json_and_jsonl(self, sample_list) -> None:
        json_output = sample_list.json()
        assert isinstance(json_output, str)
        assert "[1,2,3,4,5]" in json_output.replace(" ", "")
        jsonl_output = sample_list.jsonl()
        assert isinstance(jsonl_output, str)
        assert "1\n2\n3\n4\n5" in jsonl_output.replace(" ", "")

class TestDict:
    def test_init(self, sample_dict) -> None:
        assert sample_dict == {'a': 1, 'b': 2, 'c': 3}
        # Test nested container conversion
        nested = Dict({'a': [1, 2], 'b': {'c': 3}, 'd': (4, 5)})
        assert isinstance(nested['a'], List)
        assert isinstance(nested['b'], Dict)
        assert isinstance(nested['d'], Tuple)

    def test_map(self, sample_dict) -> None:
        result = sample_dict.map(lambda x: x * 2)
        assert result == {'a': 2, 'b': 4, 'c': 6}
        # Test chained mapping
        result = sample_dict.map(lambda x: x * 2, lambda x: x + 1)
        assert result == {'a': 3, 'b': 5, 'c': 7}

    def test_filter(self, sample_dict) -> None:
        result = sample_dict.filter(lambda k, v: v % 2 == 0)
        assert result == {'b': 2}
        # Test value-only filter
        result = sample_dict.valuefilter(lambda v: v > 2)
        assert result == {'c': 3}

    def test_getitem(self, sample_dict) -> None:
        assert sample_dict['a'] == 1
        assert sample_dict.get('d', 4) == 4
        # Test advanced indexing
        assert sample_dict[['a', 'c']] == [1, 3]
        assert sample_dict[{'a': 'x', 'b': 'y'}] == {'x': 1, 'y': 2}

    def test_pop(self, sample_dict) -> None:
        value = sample_dict.pop('b')
        assert value == 2
        assert 'b' not in sample_dict
        # Test pop with default
        value = sample_dict.pop('z', 10)
        assert value == 10

    def test_has(self, sample_dict) -> None:
        assert sample_dict.has('a') is True
        assert sample_dict.has('d') is False
        # Test with non-string keys
        d = Dict({1: 'a', 2: 'b'})
        assert d.has(1) is True
        assert d.has(3) is False

    def test_keys_values_items(self, sample_dict) -> None:
        assert sample_dict.keys() == ['a', 'b', 'c']
        assert sample_dict.values() == [1, 2, 3]
        assert sample_dict.items() == [('a', 1), ('b', 2), ('c', 3)]
        # Test after modification
        sample_dict['d'] = 4
        assert 'd' in sample_dict

    def test_arithmetic_operations(self, sample_dict) -> None:
        other_dict = Dict({'b': 3, 'd': 4})
        assert (sample_dict + other_dict) == {'a': 1, 'b': 3, 'c': 3, 'd': 4}
        assert (sample_dict - 'b') == {'a': 1, 'c': 3}
        assert (sample_dict & other_dict) == {'a': 1, 'b': 3, 'c': 3}
        # Test XOR operation
        other_dict['e'] = 1
        assert (sample_dict ^ other_dict) == {'a': 1, 'c': 3, 'd': 4, 'e': 1}

    def test_map_keys(self, sample_dict) -> None:
        result = sample_dict.mapKeys(lambda k: k.upper())
        assert result == {'A': 1, 'B': 2, 'C': 3}
        # Test chained key mapping
        result = sample_dict.mapKeys(lambda k: k.upper(), lambda k: f"key_{k}")
        assert result == {'key_A': 1, 'key_B': 2, 'key_C': 3}

    def test_strip(self) -> None:
        d = Dict({'a': 1, 'b': None, 'c': 3, 'd': None})
        stripped = d.strip(None)
        assert stripped == {'a': 1, 'c': 3}
        # Test with other values
        d = Dict({'a': 0, 'b': 1, 'c': 0})
        stripped = d.strip(0)
        assert stripped == {'b': 1}

    def test_sorting(self) -> None:
        d = Dict({'x': 5, 'a': 2, 'z': 10})
        key_sorted = d.keysort()
        assert list(key_sorted.keys()) == ['a', 'x', 'z']
        val_sorted = d.valsort()
        assert list(val_sorted.values()) == [2, 5, 10]
        # Test with custom key functions
        key_sorted = d.keysort(key=ord)
        val_sorted = d.valsort(key=lambda x: -x, reverse=True)
        assert list(val_sorted.values()) == [2, 5, 10]

    def test_json(self) -> None:
        d = Dict({'a': 1, 'b': 2})
        output = d.json()
        assert isinstance(output, str)
        assert '"a": 1' in output
        assert '"b": 2' in output
        # Test nested structures
        d = Dict({'a': [1, 2], 'b': {'c': 3}})
        output = d.json()
        assert '"a": [1, 2]' in output
        assert '"b": {"c": 3}' in output

    def test_multiplication(self) -> None:
        # D * f => apply f to values,  f * D => apply f to keys
        d = Dict({'k1': 1, 'k2': 2})
        multiply_values = d * (lambda v: v * 3)
        assert multiply_values == {'k1': 3, 'k2': 6}, "Should multiply each value by 3"

        uppercase_keys = (lambda k: k.upper()) * d
        assert uppercase_keys == {'K1': 1, 'K2': 2}, "Should uppercase each key"
        # Test chained operations
        result = (lambda k: f"key_{k}") * (d * (lambda v: v + 1))
        assert result == {'key_k1': 2, 'key_k2': 3}

class TestFun:
    def test_init(self) -> None:
        f = Fun(lambda x: x * 2)
        assert f(3) == 6
        # Test with string function
        g = Fun("x + y")
        assert g(x=1, y=2) == 3

    def test_composition(self) -> None:
        f = Fun(lambda x: x * 2)
        g = Fun(lambda x: x + 1)
        h = f * g
        assert h(3) == 8  # (3 + 1) * 2
        # Test multiple composition
        i = f * g * f
        assert i(3) == 14  # ((3 * 2) + 1) * 2

    def test_map(self) -> None:
        f = Fun(lambda x: x * 2)
        result = f @ [1, 2, 3]
        assert result == [2, 4, 6]
        # Test with generator
        result = f @ range(3)
        assert result == [0, 2, 4]

    def test_parameter_storage(self) -> None:
        f = Fun(lambda x, y: x + y)
        f['y'] = 3
        assert f(2) == 5
        # Test parameter deletion
        del f['y']
        assert f(2, 3) == 5

    def test_partial_composition(self) -> None:
        f = Fun(lambda x, y, z: x + y + z)
        g = f(1)
        assert g(2, 3) == 6
        h = f(1, 2)
        assert h(3) == 6
        # Test with keyword arguments
        i = f(z=3)
        assert i(1, 2) == 6

    def test_form_basic(self) -> None:
        add = Fun.form('0 1', '01', lambda x, y: x + y)
        assert add(1)(2) == 3
        # Test with return_former=True
        former = Fun.form('0 1', '01', return_former=True)
        assert former(lambda x, y: x * y)(2)(3) == 6

    def test_form_complex(self) -> None:
        compose = Fun.form('0 1 2', '012', lambda f, g, x: f(g(x)))
        f = lambda x: x * 2
        g = lambda x: x + 1
        assert compose(f)(g)(3) == 8
        # Test with nested calls
        h = Fun.form('0 1 2', '0(1(2))22', lambda x, y, z: x + y + z)
        assert h(lambda x: x * -1)(lambda x: x + 0.5)(3) == 2.5  # -(3 + 0.5) + 3 + 3

    def test_form_alternate_notation(self) -> None:
        on = Fun.form("func, L; callback", "func, L, callback", lambda func, L, callback: callback(func(L[0]), func(L[1])))
        assert on(lambda x: x * 2, [3, 4])(lambda a, b: a + b) == 14
        # Test with different separators
        post_fold = Fun.form("op, base; vals, folder", "folder(op, vals, base)", lambda out: out**2)
        assert post_fold(lambda x, y: x + y, 0)([1, 2, 3], reduce) == 36

    def test_power_operator(self) -> None:
        f = Fun(lambda x: x + 1)
        g = f ** 3
        assert g(0) == 3
        # Test with negative power
        with pytest.raises(ValueError):
            _ = f ** -1

    def test_division_operator(self) -> None:
        f = Fun(lambda x: 1 / x if x != 0 else None)
        g = lambda x: 999
        h = f / g
        assert h(2) == 0.5
        assert h(0) == 999
        # Test with another Fun object
        i = f / Fun(lambda x: -1)
        assert i(0) == -1

    def test_getattr(self) -> None:
        class TestClass:
            def method(self, x) -> int:
                return x * 2
        f = Fun(TestClass().method)
        assert f(3) == 6
        # Test attribute error
        with pytest.raises(AttributeError):
            _ = f.nonexistent

    def test_rshift_operator(self) -> None:
        f = Fun(lambda x: x + 1)
        g = Fun(lambda x: x * 2)
        result = 1 >> f >> g
        assert result == 4
        # Test with multiple shifts
        result = 1 >> f >> g >> f
        assert result == 5

    def test_add_operator(self) -> None:
        f = Fun(lambda x: x + 1)
        g = Fun(lambda x: x * 2)
        h = f + g
        assert h(3) == (4, 6)
        # Test with more than two functions
        i = Fun(lambda x: x ** 2)
        j = f + g + i
        assert j(3) == (4, 6, 9)

    def test_matmul_operator(self) -> None:
        f = Fun(lambda x: x * 2)
        result = f @ [1, 2, 3]
        assert result == [2, 4, 6]
        # Test with generator
        result = f @ range(3)
        assert result == [0, 2, 4]

    def test_check_method(self) -> None:
        assert Fun.check(Fun(lambda x: x))
        assert Fun.check(lambda x: x)
        assert not Fun.check(5)
        # Test with method objects

        class TestClass:
            def method(self) -> None: pass
        assert Fun.check(TestClass.method)

class TestSymbol:
    """Tests for the Symbol class, including pre-emption and advanced usage."""
    def test_basic_operations(self) -> None:
        x = Symbol('x')
        y = Symbol('y')
        # Test arithmetic operations
        expr = x + 2 * y
        assert expr(x=1, y=2) == 5
        # Test comparison operations
        expr = x > y
        assert expr(x=3, y=2) is True
        # Test unary operations
        expr = -x
        assert expr(x=3) == -3

    def test_anonymous_symbols(self) -> None:
        # Test creation of anonymous symbols
        with pytest.raises(ValueError):
            sym = Symbol()
        sym = Symbol('x') + Symbol('y')
        assert sym._anonymous
        # Test operations with anonymous symbols
        expr = sym + 1
        assert expr(6, -1) == 6

    def test_symbol_composition(self) -> None:
        x = Symbol('x')
        y = Symbol('y')
        # Test nested operations
        expr = (x + y) * (x - y)
        assert expr(x=3, y=2) == 5
        # Test with functions
        expr = abs(x - y)
        assert expr(x=2, y=3) == 1

    def test_attribute_access(self) -> None:
        class Point:
            def __init__(self, x, y) -> None:
                self.x = x
                self.y = y
        p = Symbol('p')
        expr = p.x + p.y
        assert expr(p=Point(2, 3)) == 5
        # Test nested attributes
        expr = p.x.real
        assert expr(p=Point(2 + 0j, 3)) == 2

    def test_indexing(self) -> None:
        arr = Symbol('arr')
        # Test basic indexing
        expr = arr[0] + arr[-1]
        assert expr(arr=[1, 2, 3]) == 4
        # Test slicing
        expr = arr[1:]
        assert expr(arr=[1, 2, 3]) == [2, 3]

    def test_error_handling(self) -> None:
        x = Symbol('x')
        # Test missing variable
        with pytest.raises(ValueError):
            x(y=1)
        # Test type error
        expr = x + 'string'
        with pytest.raises(ValueError):
            expr(x=1)

    def test_maybe_and_default(self) -> None:
        x = Symbol('x')
        # Test maybe
        expr = (1 / x).maybe()
        assert expr(x=2) == 0.5
        assert expr(x=0) is None
        # Test default
        expr = (1 / x).default(999)
        assert expr(x=2) == 0.5
        assert expr(x=0) == 999

    def test_guard_and_tap(self) -> None:
        x = Symbol('x')
        # Test guard
        expr = x.guard(lambda v: v > 0)
        assert expr(x=1) == 1
        with pytest.raises(ValueError):
            expr(x=-1)
        # Test tap
        side_effect = []
        expr = x.tap(side_effect.append)
        assert expr(x=1) == 1
        assert side_effect == [1]

    def test_persistent_values(self) -> None:
        x = Symbol('x')
        y = Symbol('y')
        expr = x + y
        # Test updating persistent values
        expr._update_persistent({'x': 1})
        assert expr(y=2) == 3
        # Test overriding persistent values
        assert expr(x=3, y=2) == 5

    def test_preemption(self) -> None:
        @preemptable
        def operation(a, b, c) -> int:
            return a + b + c
        x = Symbol('x')
        # Test basic preemption
        expr = operation(1, x + 1, 3)
        assert expr(x=2) == 6
        # Test multiple symbols
        y = Symbol('y')
        expr = operation(x + 1, y * 2, 3)
        assert expr(x=2, y=3) == 9

    def test_arm(self) -> None:
        # Test arm
        strings = ['a ', ' b', ' c ']
        expr = List(strings).map(arm(this.strip))
        assert expr == ['a', 'b', 'c']

    @given(x=st.integers(), y=st.integers())
    def test_property_based(self, x: int, y: int) -> None:
        # Property-based testing with hypothesis
        a = Symbol('a')
        b = Symbol('b')
        # Commutativity of addition
        expr1 = a + b
        expr2 = b + a
        assert expr1(a=x, b=y) == expr2(a=x, b=y)
        # Associativity of multiplication
        expr1 = (a * b) * 2
        expr2 = a * (b * 2)
        assert expr1(a=x, b=y) == expr2(a=x, b=y)

    def test_pow_operator(self) -> None:
        f = Fun(lambda x: x + 1)
        g = f ** 3
        assert g(1) == 4  # (1 + 1) + 1 + 1
        assert g(2) == 5  # (2 + 1) + 1 + 1
        h = Fun(lambda x: x * 2)
        i = h ** 2
        assert i(2) == 8  # (2 * 2) * 2
        assert i(3) == 12  # (3 * 2) * 2
        assert g(0) == 3  # (0 + 1) + 1 + 1
        j = Fun(lambda x: x - 1)
        k = j ** 4
        assert k(10) == 6  # (((10 - 1) - 1) - 1) - 1
        assert k(5) == 1  # (((5 - 1) - 1) - 1) - 1
        m = Fun(lambda x: x / 2)
        n = m ** 3
        assert n(16) == 2  # ((16 / 2) / 2) / 2
        assert n(8) == 1  # ((8 / 2) / 2) / 2

    def test_chaining_arguments(self) -> None:
        # test partial application with positional and keyword combos
        f = Fun(lambda x, y, z: x + y + z)
        step1 = f(2)  # partially apply x=2
        step2 = step1(y=5)  # partially apply y=5
        assert step2(z=3) == 10
        step3 = step1(z=10)
        assert step3(y=2) == 14

    def test_exception_safe_div(self) -> None:
        # test f / g, ensuring no crash if f fails
        f = Fun(lambda x: 1 / x if x != 0 else None)
        fallback = Fun(lambda x: 999)
        combined = f / fallback
        assert combined(2) == 0.5
        assert combined(0) == 999

    def test_function_reassignment(self) -> None:
        # confirm that re-wrapping an internal function won't break
        f = Fun(lambda x: x + 1)
        f.func = lambda x: x * 10
        assert f(2) == 20

    def test_basic_evaluation(self) -> None:
        x = Symbol("x")
        y = Symbol("y")
        expr = x + 2 * y
        assert expr(x=1, y=2) == 5, "Symbolic expression (x + 2*y) with x=1,y=2 => 5"

    def test_partial_evaluation(self) -> None:
        x = Symbol("x")
        y = Symbol("y")
        expr = x * y + 10
        part = expr(x=2)
        # part is now a Symbol awaiting y
        assert part(y=4) == 18

    def test_subscript_and_getattr(self) -> None:
        # symbol-based itemgetter
        item = Symbol("arr")[0]
        assert item(arr=[10, 20]) == 10
        # symbol-based getattr

        class A:
            v = 100
        attr = Symbol("some_obj").v
        assert attr(some_obj=A()) == 100

    def test_maybe(self) -> None:
        # maybe returns None if evaluation fails
        x = Symbol("x")
        expr = (1 / x).maybe()
        assert expr(x=2) == 0.5
        assert expr(x=0) is None, "Division by zero => returns None with maybe"

    def test_default(self) -> None:
        x = Symbol("x")
        expr = (1 / x).default(999)
        assert expr(x=2) == 0.5
        assert expr(x=0) == 999, "Division by zero => returns default 999"

    def test_guard(self) -> None:
        x = Symbol("x")
        positive = x.guard(lambda val: val > 0)
        assert positive(x=10) == 10
        with pytest.raises(ValueError):
            positive(x=-1)

    def test_config_options(self) -> None:
        x = Symbol('x', config={'extra_kwargs': False})
        # Test extra kwargs handling
        with pytest.raises(TypeError):
            x(x=1, y=2)
        # Test partial evaluation config
        x = Symbol('x', config={'partial_eval': False})
        expr = x + 1
        with pytest.raises(ValueError):
            expr()

    def test_guard(self) -> None:
        x = Symbol("x")
        positive = x.guard(lambda val: val > 0)
        assert positive(x=10) == 10
        with pytest.raises(ValueError):
            positive(x=-1)

    def test_config_options(self) -> None:
        x = Symbol('x', config={'extra_kwargs': False})
        # Test extra kwargs handling
        with pytest.raises(ValueError):
            x(x=1, y=2)
        # Test partial evaluation config
        x = Symbol('x', config={'partial_eval': False})
        expr = x + 1
        with pytest.raises(ValueError):
            expr()

    def test_copy_operations(self) -> None:
        x = Symbol('x')
        expr = x + 1
        # Test shallow copy
        copied = Symbol('y')._copy(expr)
        assert copied._roots == expr._roots
        # Test deep copy
        copied = Symbol('z')._total_copy(expr)
        assert copied._symbol == expr._symbol
        assert copied._roots == expr._roots

    def test_lazy_evaluation(self) -> None:
        x = Symbol('x')
        # Test lazy chain
        expr = x + 1
        expr._lazy_chain.append(lambda v: v * 2)
        assert expr(x=2) == 6
        # Test multiple transformations
        expr._lazy_chain.append(lambda v: v + 1)
        assert expr(x=2) == 7

    def test_routing(self) -> None:
        x = Symbol('x')
        y = Symbol('y')
        expr = x + y
        # Test routing with partial application
        partial = expr(x=1)
        assert partial._roots == {'y'}
        # Test routing with full application
        result = expr(x=1, y=2)
        assert result == 3

    def test_printer_customization(self) -> None:
        x = Symbol('x')
        y = Symbol('y')
        # Test custom printer
        expr = Symbol(binds=[x, y],
                     defn=lambda s, kw: kw['x'] + kw['y'],
                     printer=lambda *s: f"sum({', '.join(s)})")
        assert repr(expr) == "sum(x, y)"

    def test_complex_expressions(self) -> None:
        # Test complex mathematical expressions
        x = Symbol('x')
        y = Symbol('y')
        z = Symbol('z')
        expr = (x**2 + y**2)**0.5 * z
        assert expr(x=3, y=4, z=2) == 10.0
        # Test nested function calls
        expr = (x - y)**(z - y)
        assert expr(x=5, y=3, z=1) == (5 - 3)**(1 - 3)

    def test_symbol_reuse(self) -> None:
        x = Symbol('x')
        # Test reusing symbol in multiple expressions
        expr1 = x + 1
        expr2 = x * 2
        assert expr1(x=3) == 4
        assert expr2(x=3) == 6
        # Test symbol in combined expression
        expr3 = expr1 * expr2
        assert expr3(x=3) == 24

    def test_type_handling(self) -> None:
        x = Symbol('x')
        # Test with different types
        expr = x + 1
        assert expr(x=1.5) == 2.5
        with pytest.raises(ValueError):
            expr(x="hello")
        # Test with collections
        expr = x[0] + x[-1]
        assert expr(x=[1, 2, 3]) == 4
        assert expr(x="hello") == "ho"

    def test_error_propagation(self) -> None:
        x = Symbol('x')
        expr = 1 / x
        # Test error in evaluation
        with pytest.raises(ValueError):
            expr(x=0)
        # Test error in guard
        expr = x.guard(lambda v: v > 0)
        with pytest.raises(ValueError):
            expr(x=-1)
        # Test error with maybe
        expr = (1 / x).maybe()
        assert expr(x=0) is None

    def test_tap(self, capsys) -> None:
        x = Symbol("x")
        tapped = x.tap(lambda val: print(f"Tapped: {val}"))
        result = tapped(10)
        captured = capsys.readouterr()
        assert "Tapped: 10" in captured.out
        assert result == 10

    def test_preemption(self) -> None:
        """Symbol inherits Preemptive, so test that a preemptable function can be overridden by a Symbol."""
        @preemptable
        def some_operation(a, b) -> int:
            return a + b

        x = Symbol("b")  # Means "I want to supply b as a symbolic parameter"
        # Now calling some_operation(5, x+10)
        fused = some_operation(5, x + 10)
        # fused is a new Symbol that depends on x
        assert fused(b=2) == 17  # (5 + (2+10))

    def test_operator_combinations(self) -> None:
        x = Symbol("x")
        y = Symbol("y")
        expr = (x - 3) * (y + 2)
        assert expr(x=5, y=4) == (2 * 6), "(5 - 3) * (4 + 2) => 2*6=12"

    def test_call_with_extras(self) -> None:
        # test ignoring extra kwargs if extra_kwargs == True
        x = Symbol("x")
        expr = x * 3
        assert expr(x=2, dummy="hello") == 6

    def test_call_too_many_arguments_disallowed(self) -> None:
        # set extra_kwargs to False => raise TypeError if extraneous provided
        x = Symbol("x", config={"extra_kwargs": False})
        expr = x * 2
        assert expr._arity == 1
        assert expr._cfg['extra_kwargs'] is False
        with pytest.raises(ValueError):
            _ = expr(x=2, y=99)
