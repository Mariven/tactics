"""
Tests for the table module.
"""

import pytest
import tempfile
import contextlib
import os
import json
from src.table import (
    column,
    find_duplicates,
    is_typelike,
    is_container,
    is_concrete,
    join,
    schema_object,
    is_free,
    is_categorical,
    get_categories,
    get_keys,
    index_categorical,
    index_free,
    index,
    deindex,
    compute_schema_sketch,
    parse_json_with_metadata,
    MetadataParsingError,
)
from typing import Any, Callable, Generator

test_data_short = [{'val': x, 'name': y, 'cat': z} for x, y, z in [
    (4, 'a', 'A'), (2, 'b', 'B'), (1, 'c', 'A'), (4, 'd', 'B'), (5, 'e', 'A'), (3, 'f', 'B'), (1, 'g', 'A')
]]
test_data = {
    'people': [
        {'name': 'Jenny', 'height': 63, 'birthday': {'m': 8, 'd': 21}, 'blood': 'O'},
        {'name': 'Alma', 'height': 67, 'birthday': {'m': 5, 'd': 14}, 'blood': 'O'},
        {'name': 'Nathan', 'height': 71, 'birthday': {'m': 9, 'd': 9}, 'blood': 'B'},
        {'name': 'Eva', 'height': 64, 'birthday': {'m': 3, 'd': 4}, 'blood': 'A'},
        {'name': 'Peter', 'height': 70, 'birthday': {'m': 12, 'd': 30}, 'blood': 'AB'},
        {'name': 'Sara', 'height': 66, 'birthday': {'m': 7, 'd': 17}, 'blood': 'A'},
        {'name': 'Tom', 'height': 68, 'birthday': {'m': 1, 'd': 1}, 'blood': 'B'},
        {'name': 'Lily', 'height': 65, 'birthday': {'m': 10, 'd': 31}, 'blood': 'O'},
        {'name': 'David', 'height': 69, 'birthday': {'m': 2, 'd': 28}, 'blood': 'AB'},
        {'name': 'Alice', 'height': 62, 'birthday': {'m': 6, 'd': 23}, 'blood': 'A'},
        {'name': 'Bob', 'height': 72, 'birthday': {'m': 11, 'd': 11}, 'blood': 'B'},
        {'name': 'Sophia', 'height': 61, 'birthday': {'m': 4, 'd': 5}, 'blood': 'AB'},
        {'name': 'Jack', 'height': 67, 'birthday': {'m': 7, 'd': 22}, 'blood': 'O'},
        {'name': 'John', 'height': 70, 'birthday': {'m': 1, 'd': 23}, 'blood': 'A'},
        {'name': 'Mary', 'height': 68, 'birthday': {'m': 3, 'd': 15}, 'blood': 'B'},
        {'name': 'Emma', 'height': 66, 'birthday': {'m': 8, 'd': 12}, 'blood': 'AB'},
        {'name': 'Henry', 'height': 71, 'birthday': {'m': 10, 'd': 28}, 'blood': 'A'},
        {'name': 'Anna', 'height': 64, 'birthday': {'m': 12, 'd': 21}, 'blood': 'B'},
        {'name': 'Charlie', 'height': 63, 'birthday': {'m': 1, 'd': 15}, 'blood': 'O'},
        {'name': 'Olivia', 'height': 62, 'birthday': {'m': 2, 'd': 6}, 'blood': 'AB'},
        {'name': 'George', 'height': 72, 'birthday': {'m': 3, 'd': 8}, 'blood': 'A'},
        {'name': 'Sophie', 'height': 67, 'birthday': {'m': 4, 'd': 17}, 'blood': 'B'},
        {'name': 'Carmen', 'height': 61, 'birthday': {'m': 4, 'd': 7}, 'blood': 'O'},
        {'name': 'Dan', 'height': 69, 'birthday': {'m': 6, 'd': 30}, 'blood': 'B'},
        {'name': 'Elena', 'height': 63, 'birthday': {'m': 2, 'd': 5}, 'blood': 'A'},
        {'name': 'Fred', 'height': 70, 'birthday': {'m': 9, 'd': 2}, 'blood': 'O'},
        {'name': 'Gail', 'height': 64, 'birthday': {'m': 5, 'd': 13}, 'blood': 'AB'},
        {'name': 'Hank', 'height': 71, 'birthday': {'m': 1, 'd': 29}, 'blood': 'B'},
        {'name': 'Irene', 'height': 65, 'birthday': {'m': 10, 'd': 27}, 'blood': 'A'},
        {'name': 'Kate', 'height': 66, 'birthday': {'m': 6, 'd': 12}, 'blood': 'AB'},
        {'name': 'Larry', 'height': 72, 'birthday': {'m': 10, 'd': 20}, 'blood': 'B'},
        {'name': 'Nancy', 'height': 67, 'birthday': {'m': 1, 'd': 18}, 'blood': 'A'},
    ]
}['people']

def test_is_typelike() -> None:
    assert is_typelike(int)
    assert is_typelike(str)
    assert is_typelike(int | str)
    assert is_typelike(list[int])
    assert is_typelike(dict[str, float])
    assert is_typelike(Any)
    assert is_typelike(tuple[int, ...])
    assert not is_typelike(4)
    assert not is_typelike('b')
    assert not is_typelike((False, True))
    assert is_typelike(Any | None)
    assert is_typelike(Callable[..., Any])

def test_is_container() -> None:
    assert is_container([1, 2, 3])
    assert not is_container("hello")
    assert not is_container(100)
    assert is_container({"a": 1, "b": 2})
    assert is_container((1, 2, 3))
    assert is_container(range(10))
    assert not is_container(int)
    assert not is_container(str)
    assert is_container(set())
    assert is_container(type('X', (list,), {}))
    assert not is_container(type('X', (int,), {}))

def test_is_concrete() -> None:
    assert is_concrete([1, 2, 3])
    assert is_concrete("hello")
    assert is_concrete(100)
    assert is_concrete({"a": 1, "b": 2})
    assert is_concrete((1, 2, 3))
    assert is_concrete(range(10))
    assert not is_concrete(int)
    assert not is_concrete(set())
    assert not is_concrete(type('X', (list,), {}))
    assert not is_concrete(type('X', (int,), {}))
    assert not is_concrete({'a': str, 'b': bool})
    assert is_concrete({'a': 'hello', 'b': bool})
    assert not is_concrete({'a': str, 'b': {'c': str, 'd': {'e': str}}})
    assert not is_concrete({'a': str, 'b': {'c': str, 'd': {'e': str, 'f': None}}})

def test_schema_object() -> None:
    assert schema_object("hello") is str
    assert schema_object(100) is int
    assert schema_object([1, 2, 3]) == [int]
    assert schema_object(int | str) == int | str
    assert schema_object({"a": 1, "b": 2}) == {"a": int, "b": int}
    assert schema_object((1, 2, 3.1, (True,))) == (int, int, float, (bool,))
    assert schema_object({'a': str, 'b': bool}) == {'a': str, 'b': bool}

def test_join() -> None:
    assert join(str, "hello") is str
    assert join(str, int) == str | int
    assert join(str, [1, 2, 3]) == str | list[int]
    assert join([1, 2, 3], "hello") == list[int] | str
    assert join([1, 2, 3], [4, 'b', 6]) == [int | str]
    assert join((1, 2, 3), (4, 'b', 6)) == (int, int | str, int)
    assert join([bool], [[[{True: False}]]]) == [bool | list[list[dict[bool, bool]]]]
    assert join({'a': str}, {'a': ['b', 'c', 'd']}) == {'a': str | list[str]}
    assert join({'a': str}, {'b': int}) == {'a': str, 'b': int}
    assert join({'a': str}, {'a': int, 'b': float}) == {'a': str | int, 'b': float}

def test_column() -> None:
    assert column(test_data_short, 'val') == [4, 2, 1, 4, 5, 3, 1]
    assert column(test_data_short, 'name') == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert column(test_data, 'height') == [63, 67, 71, 64, 70, 66, 68, 65, 69, 62, 72, 61, 67, 70, 68, 66, 71, 64, 63, 62, 72, 67, 61, 69, 63, 70, 64, 71, 65, 66, 72, 67]
    assert column(test_data, 'birthday.m') == [8, 5, 9, 3, 12, 7, 1, 10, 2, 6, 11, 4, 7, 1, 3, 8, 10, 12, 1, 2, 3, 4, 4, 6, 2, 9, 5, 1, 10, 6, 10, 1]
    assert column(test_data, ['birthday', 'm']) == [8, 5, 9, 3, 12, 7, 1, 10, 2, 6, 11, 4, 7, 1, 3, 8, 10, 12, 1, 2, 3, 4, 4, 6, 2, 9, 5, 1, 10, 6, 10, 1]

def test_find_duplicates() -> None:
    assert find_duplicates(test_data_short, 'val') == [4, 1]
    assert find_duplicates(test_data_short, 'name') == []
    assert find_duplicates(test_data, 'name') == []
    assert sorted(find_duplicates(test_data, 'height')) == sorted([64, 61, 65, 67, 71, 70, 66, 68, 62, 72, 63, 69])

def test_is_free() -> None:
    assert not is_free(test_data_short, 'val')
    assert is_free(test_data_short, 'name')
    assert not is_free(test_data_short, 'cat')
    assert is_free(test_data, 'name')
    assert not is_free(test_data, 'height')
    assert not is_free(test_data, 'birthday.m')
    assert not is_free(test_data, 'birthday.d')
    assert not is_free(test_data, 'blood')

def test_is_categorical() -> None:
    assert not is_categorical(test_data_short, 'val')
    assert not is_categorical(test_data_short, 'name')
    assert is_categorical(test_data_short, 'cat')
    assert not is_categorical(test_data, 'name')
    assert not is_categorical(test_data, 'height')
    assert not is_categorical(test_data, 'birthday.m')
    assert not is_categorical(test_data, 'birthday.d')
    assert is_categorical(test_data, 'blood')

def test_get_categories() -> None:
    assert sorted(get_categories(test_data_short, 'name')) == sorted(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    assert sorted(get_categories(test_data_short, 'cat')) == sorted(['A', 'B'])
    assert sorted(get_categories(test_data, 'blood')) == sorted(['A', 'B', 'AB', 'O'])

def test_get_keys() -> None:
    assert get_keys(test_data_short) == [['val'], ['name'], ['cat']]
    assert get_keys(test_data) == [['name'], ['height'], ['birthday', 'm'], ['birthday', 'd'], ['blood']]

def test_index_categorical() -> None:
    assert index_categorical(test_data_short, 'cat') == {
        'A': [{'val': 4, 'name': 'a'}, {'val': 1, 'name': 'c'}, {'val': 5, 'name': 'e'}, {'val': 1, 'name': 'g'}],
        'B': [{'val': 2, 'name': 'b'}, {'val': 4, 'name': 'd'}, {'val': 3, 'name': 'f'}]
    }

def test_index_free() -> None:
    assert index_free(test_data_short, 'name') == {
        'a': {'val': 4, 'cat': 'A'},
        'b': {'val': 2, 'cat': 'B'},
        'c': {'val': 1, 'cat': 'A'},
        'd': {'val': 4, 'cat': 'B'},
        'e': {'val': 5, 'cat': 'A'},
        'f': {'val': 3, 'cat': 'B'},
        'g': {'val': 1, 'cat': 'A'}
    }

_sort_by = lambda key: lambda data: sorted(data, key=lambda x: x[key])
_flip = lambda key: lambda data: deindex(index(data, key), key)
_sort_key = "name"
_free_key = "name"
_cat_key = "cat"

assert _sort_by(_sort_key)(_flip(_cat_key)(test_data_short)) == _sort_by(_sort_key)(test_data_short)
assert _sort_by(_sort_key)(_flip(_free_key)(test_data_short)) == _sort_by(_sort_key)(test_data_short)

def test_index() -> None:
    assert index(test_data_short, 'cat') == index_categorical(test_data_short, 'cat')
    assert index(test_data_short, 'name') == index_free(test_data_short, 'name')

def test_deindex() -> None:
    assert deindex(index_categorical(test_data_short, 'cat'), 'cat') == [
        {'val': 4, 'name': 'a', 'cat': 'A'},
        {'val': 1, 'name': 'c', 'cat': 'A'},
        {'val': 5, 'name': 'e', 'cat': 'A'},
        {'val': 1, 'name': 'g', 'cat': 'A'},
        {'val': 2, 'name': 'b', 'cat': 'B'},
        {'val': 4, 'name': 'd', 'cat': 'B'},
        {'val': 3, 'name': 'f', 'cat': 'B'}
    ]
    assert deindex(index_free(test_data_short, 'name'), 'name') == [
        {'val': 4, 'cat': 'A', 'name': 'a'},
        {'val': 2, 'cat': 'B', 'name': 'b'},
        {'val': 1, 'cat': 'A', 'name': 'c'},
        {'val': 4, 'cat': 'B', 'name': 'd'},
        {'val': 5, 'cat': 'A', 'name': 'e'},
        {'val': 3, 'cat': 'B', 'name': 'f'},
        {'val': 1, 'cat': 'A', 'name': 'g'}
    ]

def test_compute_schema_sketch() -> None:
    assert compute_schema_sketch(int) == "int"
    assert compute_schema_sketch(str) == "str"
    assert compute_schema_sketch(list[int]) == "list[int]"
    assert compute_schema_sketch(dict[str, int]) == "dict[str, int]"
    assert compute_schema_sketch({'a': 1, 'b': 2}) == "<str>: int"
    assert compute_schema_sketch({'a': [1, 2], 'b': [3, 4]}) == "<str>: [int]"
    assert compute_schema_sketch({'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}) == "<str>: <str>: int"
    assert compute_schema_sketch({1: 'a', 2: 'b'}) == "{int}: str"
    assert compute_schema_sketch({(1, 2): 'a', (3, 4): 'b'}) == "{(int, int)}: str"
    assert compute_schema_sketch([1, 2, 3]) == "[int]"
    assert compute_schema_sketch([[1, 2], [3, 4]]) == "[[int]]"
    assert compute_schema_sketch([{'a': 1}, {'b': 2}]) == "[<str>: int]"
    assert compute_schema_sketch([{'a': [1, 2]}, {'b': [3, 4]}]) == "[<str>: [int]]"
    assert compute_schema_sketch([{'a': {'x': 1}}, {'b': {'y': 2}}]) == "[<str>: <str>: int]"
    assert compute_schema_sketch([1, 'a', 2, 'b']) == "[int, str]"
    assert compute_schema_sketch([{'a': 1}, {'a': 'x'}, {'b': 2}]) == "[(<str>: int), (<str>: str)]"
    assert compute_schema_sketch([{'a': 1, 'b': 2}, {'a': 'x'}, {'c': 3.1}]) == "[(<str>: int), (<str>: str), (<str>: float)]"
    assert compute_schema_sketch([{'a': 1}, {'a': 2}, {'a': 3}]) == "[<str>: int]"
    assert compute_schema_sketch([{'a': 1}, {'b': 2}, {'c': 3}]) == "[<str>: int]"
    assert compute_schema_sketch([{'a': 1, 'b': 2}, {'c': 3, 'd': 4}]) == "[<str>: int]"
    assert compute_schema_sketch([{'a': {'x': 1}}, {'a': {'x': 2}}, {'a': {'x': 3}}]) == "[<str>: <str>: int]"
    assert compute_schema_sketch([{'a': {'x': 1}}, {'a': {'y': 2}}, {'a': {'z': 3}}]) == "[<str>: <str>: int]"
    assert compute_schema_sketch([{'a': {'x': 1, 'y': 2}}, {'a': {'x': 3, 'z': 4}}, {'a': {'y': 5, 'z': 6}}]) == "[<str>: <str>: int]"
    assert compute_schema_sketch([1, 2, 3, 4, 5]) == "[int]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b']) == "[int, str]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6]) == "[int, str]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1}]) == "[int, str, (<str>: int)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}]) == "[int, str, (<str>: int)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}]) == "[int, str, (<str>: int), (<str>: str)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}]) == "[int, str, (<str>: int), (<str>: str), (<str>: float)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2]]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int]]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2], ['a', 'b']]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int], [str]]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2], ['a', 'b'], [1, 'a']]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int], [str], [int, str]]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2], ['a', 'b'], [1, 'a'], {'y': {'p': 3}}]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int], [str], [int, str], (<str>: <str>: int)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2], ['a', 'b'], [1, 'a'], {'y': {'p': 3}}, {'y': {'p': 'q'}}]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int], [str], [int, str], (<str>: <str>: int), (<str>: <str>: str)]"
    assert compute_schema_sketch([1, 2, 3, 'a', 'b', 4, 5, 6, {'x': 1, 'y': 2}, {'x': 'p', 'y': 'q'}, {'x': 3.1}, [1, 2], ['a', 'b'], [1, 'a'], {'y': {'p': 3}}, {'y': {'p': 'q'}}, {'y': {'r': 4.1}}]) == "[int, str, (<str>: int), (<str>: str), (<str>: float), [int], [str], [int, str], (<str>: <str>: int), (<str>: <str>: str), (<str>: <str>: float)]"


def create_temp_json() -> Generator[Callable[[dict], str], None, None]:
    """Fixture to create and clean up temporary JSON files"""
    temp_files = []

    def _create_file(data: dict) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_files.append(f.name)
            return f.name

    yield _create_file

    # Cleanup
    for file in temp_files:
        with contextlib.suppress(OSError):
            os.remove(file)

create_temp_json_fix = pytest.fixture(create_temp_json)

def test_basic_alias_replacement(create_temp_json_fix) -> None:
    """Test basic alias replacement functionality"""
    data = {
        "metadata": {
            "primary_table": "users",
            "aliases": [
                {"users": {
                    "id": ["user_id", "name"],
                    "email": ["contact", "mail"]
                }}
            ]
        },
        "users": [
            {"user_id": "alice", "contact": "alice@test.com"},
            {"name": "bob", "mail": "bob@test.com"}
        ]
    }

    file_path = create_temp_json_fix(data)
    result = parse_json_with_metadata(file_path)

    expected = [
        {"id": "alice", "email": "alice@test.com"},
        {"id": "bob", "email": "bob@test.com"}
    ]
    assert result["users"] == expected

def test_defaults_with_substitution(create_temp_json_fix) -> None:
    """Test defaults with dynamic substitution"""
    data = {
        "metadata": {
            "primary_table": "items",
            "defaults": [
                {"items": {
                    "id": "item_${indices[-1]}",
                    "label": "${name}",
                    "type": "default_type"
                }}
            ]
        },
        "items": [
            {"name": "First"},
            {"name": "Second", "type": "custom"}
        ]
    }

    file_path = create_temp_json_fix(data)
    result = parse_json_with_metadata(file_path)

    expected = [
        {"name": "First", "id": "item_0", "label": "First", "type": "default_type"},
        {"name": "Second", "id": "item_1", "label": "Second", "type": "custom"}
    ]
    assert result["items"] == expected

def test_nested_structures(create_temp_json_fix) -> None:
    """Test handling of nested structures"""
    data = {
        "metadata": {
            "primary_table": "departments",
            "aliases": [
                {"departments": {
                    "id": ["dept_id"],
                    "employees": {"id": ["emp_id"]}
                }}
            ],
            "defaults": [
                {"departments": {
                    "location": "HQ",
                    "employees": {"status": "active"}
                }}
            ]
        },
        "departments": [
            {
                "dept_id": "eng",
                "employees": [
                    {"emp_id": "e1", "name": "Alice"},
                    {"emp_id": "e2", "name": "Bob", "status": "contractor"}
                ]
            }
        ]
    }

    file_path = create_temp_json_fix(data)
    result = parse_json_with_metadata(file_path)

    expected = [
        {
            "id": "eng",
            "location": "HQ",
            "employees": [
                {"id": "e1", "name": "Alice", "status": "active"},
                {"id": "e2", "name": "Bob", "status": "contractor"}
            ]
        }
    ]
    assert result["departments"] == expected

def test_error_handling(create_temp_json_fix) -> None:
    """Test error handling for invalid inputs"""

    # Test missing primary table
    data = {
        "metadata": {},
        "users": []
    }
    with pytest.raises(MetadataParsingError, match="metadata.primary_table is required"):
        parse_json_with_metadata(create_temp_json_fix(data))

    # Test invalid metadata type
    data = {
        "metadata": "invalid",
        "users": []
    }
    with pytest.raises(MetadataParsingError, match="metadata must be an object"):
        parse_json_with_metadata(create_temp_json_fix(data))

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        parse_json_with_metadata("nonexistent.json")

    # Test non-strict mode
    data = {
        "metadata": "invalid",
        "users": []
    }
    result = parse_json_with_metadata(create_temp_json_fix(data), strict=False)
    assert result == data

def test_complex_example(create_temp_json_fix) -> None:
    """Test a complex example combining multiple features"""
    data = {
        "metadata": {
            "primary_table": "organizations",
            "aliases": [
                {"organizations": {
                    "id": ["org_id", "name"],
                    "projects": {
                        "id": ["proj_id"],
                        "members": {"role": ["position"]}
                    }
                }}
            ],
            "defaults": [
                {"organizations": {
                    "type": "company",
                    "projects": {
                        "status": "active",
                        "members": {"access_level": "${role}_access"}
                    }
                }}
            ]
        },
        "organizations": [
            {
                "org_id": "acme",
                "projects": [
                    {
                        "proj_id": "p1",
                        "title": "Project 1",
                        "members": [
                            {"user": "alice", "position": "admin"},
                            {"user": "bob", "position": "dev"}
                        ]
                    }
                ]
            }
        ]
    }

    file_path = create_temp_json_fix(data)
    result = parse_json_with_metadata(file_path)

    expected = [
        {
            "id": "acme",
            "type": "company",
            "projects": [
                {
                    "id": "p1",
                    "title": "Project 1",
                    "status": "active",
                    "members": [
                        {"user": "alice", "role": "admin", "access_level": "admin_access"},
                        {"user": "bob", "role": "dev", "access_level": "dev_access"}
                    ]
                }
            ]
        }
    ]
    assert result["organizations"] == expected

def test_edge_cases(create_temp_json_fix) -> None:
    """Test edge cases and corner conditions"""
    # Test empty table
    data = {
        "metadata": {
            "primary_table": "users",
            "aliases": [{"users": {"id": ["user_id"]}}]
        },
        "users": []
    }
    result = parse_json_with_metadata(create_temp_json_fix(data))
    assert result["users"] == []

    # Test single dict instead of list
    data = {
        "metadata": {
            "primary_table": "user",
            "aliases": [{"user": {"id": ["user_id"]}}]
        },
        "user": {"user_id": "alice"}
    }
    result = parse_json_with_metadata(create_temp_json_fix(data))
    assert result["user"] == {"id": "alice"}

    # Test invalid indices expression
    data = {
        "metadata": {
            "primary_table": "items",
            "defaults": [
                {"items": {"id": "item_${indices[1+2]}"}}  # Invalid expression
            ]
        },
        "items": [{}]
    }
    result = parse_json_with_metadata(create_temp_json_fix(data))
    assert result["items"][0]["id"] == ""  # Should fail safely

def test_invalid_alias_structures(create_temp_json_fix) -> None:
    """Test handling of invalid alias structures"""
    # Test invalid alias type
    data = {
        "metadata": {
            "primary_table": "users",
            "aliases": "invalid"
        },
        "users": [{"name": "alice"}]
    }
    result = parse_json_with_metadata(create_temp_json_fix(data), strict=False)
    assert result["users"] == [{"name": "alice"}]

    # Test invalid alias mapping
    data = {
        "metadata": {
            "primary_table": "users",
            "aliases": [
                {"users": "invalid"},
                {"users": {"id": "invalid"}}  # Should be list
            ]
        },
        "users": [{"name": "alice"}]
    }
    result = parse_json_with_metadata(create_temp_json_fix(data), strict=False)
    assert result["users"] == [{"name": "alice"}]

def test_invalid_defaults_structures(create_temp_json_fix) -> None:
    """Test handling of invalid defaults structures"""
    # Test invalid defaults type
    data = {
        "metadata": {
            "primary_table": "users",
            "defaults": "invalid"
        },
        "users": [{"name": "alice"}]
    }
    result = parse_json_with_metadata(create_temp_json_fix(data), strict=False)
    assert result["users"] == [{"name": "alice"}]

    # Test invalid defaults mapping
    data = {
        "metadata": {
            "primary_table": "users",
            "defaults": [
                {"users": "invalid"}
            ]
        },
        "users": [{"name": "alice"}]
    }
    result = parse_json_with_metadata(create_temp_json_fix(data), strict=False)
    assert result["users"] == [{"name": "alice"}]
