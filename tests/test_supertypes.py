import pytest
from supertypes import List, Dict, Fun, Multi

@pytest.fixture
def sample_list():
    return List([1, 2, 3, 4, 5])

@pytest.fixture
def sample_dict():
    return Dict({'a': 1, 'b': 2, 'c': 3})

class TestList:
    def test_init(self, sample_list):
        assert sample_list == [1, 2, 3, 4, 5]

    def test_map(self, sample_list):
        result = sample_list.map(lambda x: x * 2)
        assert result == [2, 4, 6, 8, 10]

    def test_filter(self, sample_list):
        result = sample_list.filter(lambda x: x % 2 == 0)
        assert result == [2, 4]

    def test_getitem(self, sample_list):
        assert sample_list[1] == 2
        assert sample_list[1:] == [2, 3, 4, 5]

    def test_getattr(self, sample_list):
        assert sample_list.head == 1
        assert sample_list.tail == [2, 3, 4, 5]

    def test_check(self):
        assert List.check(List([1, 2, 3])) == True
        assert List.check([1, 2, 3]) == True
        assert List.check({'a': 1}) == False

    def test_asyncMap(self):
        def square(x):
            return x ** 2
        result = List.asyncMap(square, [1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

class TestDict:
    def test_init(self, sample_dict):
        assert sample_dict == {'a': 1, 'b': 2, 'c': 3}

    def test_map(self, sample_dict):
        result = sample_dict.map(lambda x: x * 2)
        assert result == {'a': 2, 'b': 4, 'c': 6}

    def test_filter(self, sample_dict):
        result = sample_dict.filter(lambda k, v: v % 2 == 0)
        assert result == {'b': 2}

    def test_getitem(self, sample_dict):
        assert sample_dict['a'] == 1
        assert sample_dict.get('d', 4) == 4

    def test_pop(self, sample_dict):
        value = sample_dict.pop('b')
        assert value == 2
        assert 'b' not in sample_dict

    def test_has(self, sample_dict):
        assert sample_dict.has('a') == True
        assert sample_dict.has('d') == False

    def test_keys_values_items(self, sample_dict):
        assert sample_dict.keys() == ['a', 'b', 'c']
        assert sample_dict.values() == [1, 2, 3]
        assert sample_dict.items() == [('a', 1), ('b', 2), ('c', 3)]

    def test_arithmetic_operations(self, sample_dict):
        other_dict = Dict({'b': 3, 'd': 4})
        assert (sample_dict + other_dict) == {'a': 1, 'b': 3, 'c': 3, 'd': 4}
        assert (sample_dict - 'b') == {'a': 1, 'c': 3}
        assert (sample_dict & other_dict) == {'b': 3}

class TestFun:
    def test_init(self):
        f = Fun(lambda x: x * 2)
        assert f(3) == 6

    def test_composition(self):
        f = Fun(lambda x: x * 2)
        g = Fun(lambda x: x + 1)
        h = f * g
        assert h(3) == 8  # (3 + 1) * 2

    def test_map(self):
        f = Fun(lambda x: x * 2)
        result = f @ [1, 2, 3]
        assert result == [2, 4, 6]

    def test_parameter_storage(self):
        f = Fun(lambda x, y: x + y, args=[1])
        assert f(2) == 3
        f['y'] = 3
        assert f(2) == 5

    def test_partial_composition(self):
        f = Fun(lambda x, y, z: x + y + z)
        g = f(1)
        assert g(2, 3) == 6
        h = f(1, 2)
        assert h(3) == 6

    def test_interaction_with_multi(self):
        f = Fun(lambda x: x * 2)
        m = Multi(1, 2, 3)
        result = f(m)
        assert result == Multi(2, 4, 6)

    def test_form_basic(self):
        add = Fun.form('01', '01', lambda x, y: x + y)
        assert add(1)(2) == 3

    def test_form_complex(self):
        compose = Fun.form('0 1 2', '0(1(2))', lambda f, g, x: f(g(x)))
        f = lambda x: x * 2
        g = lambda x: x + 1
        assert compose(f)(g)(3) == 8

    def test_form_alternate_notation(self):
        on = Fun.form("func, L; callback", "callback(func, L)", lambda func, L, callback: callback(func(L[0]), func(L[1])))
        assert on(lambda x: x * 2)([3, 4])(lambda a, b: a + b) == 14

    def test_power_operator(self):
        f = Fun(lambda x: x + 1)
        g = f ** 3
        assert g(0) == 3

    def test_division_operator(self):
        f = Fun(lambda x: x if x > 0 else None)
        g = lambda x: 0
        h = f / g
        assert h(5) == 5
        assert h(-5) == 0

    def test_getattr(self):
        class TestClass:
            def method(self, x):
                return x * 2
        f = Fun(TestClass().method)
        assert f(3) == 6

    def test_rshift_operator(self):
        f = Fun(lambda x: x + 1)
        g = Fun(lambda x: x * 2)
        result = 1 >> f >> g
        assert result == 4

    def test_add_operator(self):
        f = Fun(lambda x: x + 1)
        g = Fun(lambda x: x * 2)
        h = f + g
        assert h(3) == (4, 6)

    def test_matmul_operator(self):
        f = Fun(lambda x: x * 2)
        result = f @ [1, 2, 3]
        assert result == [2, 4, 6]

    def test_truediv_operator(self):
        f = Fun(lambda x: x if x > 0 else None)
        g = 0
        h = f / g
        assert h(5) == 5
        assert h(-5) == 0

    def test_check_method(self):
        assert Fun.check(Fun(lambda x: x))
        assert Fun.check(lambda x: x)
        assert not Fun.check(5)

    def test_pow_operator(self):
        f = Fun(lambda x: x + 1)
        g = f ** 3
        assert g(0) == 3  # (0 + 1) + 1 + 1
        assert g(1) == 4  # (1 + 1) + 1 + 1
        assert g(2) == 5  # (2 + 1) + 1 + 1

        h = Fun(lambda x: x * 2)
        i = h ** 2
        assert i(2) == 8  # (2 * 2) * 2
        assert i(3) == 12  # (3 * 2) * 2

        j = Fun(lambda x: x - 1)
        k = j ** 4
        assert k(10) == 6  # (((10 - 1) - 1) - 1) - 1
        assert k(5) == 1  # (((5 - 1) - 1) - 1) - 1

        l = Fun(lambda x: x / 2)
        m = l ** 3
        assert m(8) == 1  # ((8 / 2) / 2) / 2
        assert m(16) == 2  # ((16 / 2) / 2) / 2

class TestMulti:
    def test_init(self):
        m = Multi(1, 2, 3)
        assert list(m) == [1, 2, 3]

    def test_walk(self):
        m = Multi(1, [2, 3], {'a': 4})
        paths = list(Multi.walk(m))
        assert len(paths) == 4
        assert ([0], 1) in paths
        assert ([1, 0], 2) in paths
        assert ([1, 1], 3) in paths
        assert ([2, 'a'], 4) in paths

    def test_build_skeleton(self):
        m = Multi(1, [2, 3], {'a': 4})
        skeleton = Multi.build_skeleton(m)
        assert isinstance(skeleton, Multi)
        assert isinstance(skeleton[1], List)
        assert isinstance(skeleton[2], Dict)