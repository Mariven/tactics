"""
Data structures and utilities for working with tables.
"""

from __future__ import annotations
import json
from src.supertypes import *

logger = logging.getLogger(__name__)

"""
Example data:
Row  + Table 1 (TStats)                        Row + Table 2 (TVersus)                +
     | name   | height | birthday | blood |        | order  | name   | matches        |
     |        |        | m  || d  |       |        |        |        | against || won |
     |--------|--------|----||----|-------|        |--------|--------|---------||-----|
   0 | Jenny  | 63     | 08 || 21 | O     |      0 | 2      | Nathan | Alma    || F   |
   1 | Alma   | 67     | 05 || 14 | O     |        | "      | "      | Jenny   || T   |
   2 | Nathan | 71     | 09 || 09 | B     |        | "      | "      | Jenny   || F   |
     +------------------------------------+        |--------|--------|---------||-----|
                                                 1 | 5      | Alma   | Nathan  || T   |
                                                   |--------|--------|---------||-----|
                                                 2 | 1      | Jenny  | Nathan  || F   |
                                                   | "      | "      | Nathan  || T   |
                                                   +----------------------------------+
= TStats                                       = TVersus
{ name: str,                                   { order: int,
  height: int,                                   name: str,
  birthday: {m: int, d: int},                    matches: [
  blood: Literal['O','A','B','AB'] }              { against: str, won: bool } ] }

Notation:
    A 'schema' `X` is a means of serializing a 'kind' `G = *X` of entity. It has 'keys' `k` representing 'fields' `f = *k` of the kind.
        - * is just pointer notation; we can say that G is the referent of X, and f is the referent of k.
    An 'object' `x` is the X-representation of a determinate instance `g = *x` of kind *X. Its 'values' are the property values of g.
        - So (G, g, f) corresponds by reference to (X, x, k).
    A 'primary' key is a key for which each object x has a uniquely identifying value.
    A 'categorical' key is a key whose values correspond to partitions of the kind. Primary keys are only trivially categorical.
        - This notion is fuzzy at the edges, dependent on how many categories there are abstractly and in practice.
        - e.g. pet species might be categorical among thirty people, with three categories ['cat', 'dog', 'parrot'],
            but thirty million people will have too many categories to be feasible (someone has a pet pangolin).
        - So some categories are 'unbounded' (as with species), while others are 'bounded' (there will only ever be four blood types).

    A table `TX` is a collection of objects sharing a common schema `X`.
      - A kind may have multiple schemas for different purposes, and a schema may have multiple tables for different contexts.
    A table is...
        array-indexed by a primary key when given as an array of objects with explicit primary values:
            Ex 1 (name-array):
                'people': [
                    {name: 'Jenny', height: 63, birthday: {m: 8, d: 21}, blood: 'O'},
                    {name: 'Alma', height: 67, birthday: {m: 5, d: 14}, blood: 'O'},
                    {name: 'Nathan', height: 71, birthday: {m: 9, d: 09}, blood: 'B'}
                ]
        object-indexed by a primary key when given as an object whose keys are primary values pointing to their unique objects:
            Ex 2 (name-object):
                'people': {
                    'Jenny': {height: 63, birthday: {m: 8, d: 21}, blood: 'O'},
                    'Alma': {height: 67, birthday: {m: 5, d: 14}, blood: 'O'},
                    'Nathan': {height: 71, birthday: {m: 9, d: 09}, blood: 'B'}
                }
        array- or object-indexed by a categorical key when data is grouped by that key's values in array or object form:
            Ex 3 (blood-object name-object):
                'people': {
                    'O': {
                        'Jenny': {height: 63, birthday: {m: 8, d: 21}},
                        'Alma': {height: 67, birthday: {m: 5, d: 14}}
                    },
                    'B': {
                        'Nathan': {height: 71, birthday: {m: 9, d: 09}}
                    }
                }
            Ex 4 (blood-object name-array):
                'people': {
                    'O': [
                        {name: 'Jenny', height: 63, birthday: {m: 8, d: 21}},
                        {name: 'Alma', height: 67, birthday: {m: 5, d: 14}}
                    ],
                    'B': [
                        {name: 'Nathan', height: 71, birthday: {m: 9, d: 09}}
                    ]
                }
            Ex 5 (blood-array name-object):
                'people': [
                    {'blood': 'O', 'people': {'Jenny': {...}, 'Alma': {...}}},
                    {'blood': 'B', 'people': {'Nathan': {...}}}
                ]
            Ex 6 (blood-array name-array):
                'people': [
                    {'blood': 'O', 'people': [{name: 'Jenny', ...}, {name: 'Alma', ...}]},
                    {'blood': 'B', 'people': [{name: 'Nathan', ...}]}
                ]
        A table can always be indexed by its primary key, and may be indexed at shallower levels by categorical keys as well.
    A key is 'free' if it indexes objects in a table, and 'reserved' otherwise (if it can be written as an explicit string in the schema).
    In the same way, an object is 'free' if its keys are free, and 'reserved' otherwise.
    
    A 'sketch' is a brief structural representation of a schema, using a simplified syntax.
    - `A: B` = dict[A, B]
    - `A, B` = A | B
    - `A?` = A | None
    - `[A]` = list[A]
    - Parentheses mark subobjects where necessary.
    - For `A` a key type, `<A>: B` denotes reserved keys, while `{A}: B` denotes free keys.
    Sketches can be converted to Python type annotations, with Literal types for reserved keys and basic types for free keys.
    So the above examples would be sketched as:
    1. <str>: [<str>: str, int, (<str>: int)]
    2. <str>: {str}: [<str>: int, (<str>: int), str]
    3. <str>: {str}: {str}: <str>: int, (<str>: int)
    4. <str>: {str}: [<str>: str, int, (<str>: int)]
    5. <str>: [<str>: str, ({str}: int, (<str>: int))]
    6. <str>: [<str>: str, [<str>: str, int, (<str>: int)]]

    The grammar for the sketch syntax is as follows:
        Expr         ::= ListExpr | DictExpr | UnionExpr
        ListExpr     ::= '[' Expr ']'
        DictExpr     ::= DictKey ': ' UnionExpr
        UnionExpr    ::= DictExpr | UnionMember (', ' UnionMember)*
        UnionMember  ::= UnionItem | UnionItem '?'
        UnionItem    ::= ListExpr | Type | '(' DictExpr ')'
        DictKey      ::= '{' Type '}' | '<' Type '>' | Type
        Type         ::= 'str' | 'int' | and so on ...

    A schema is given in 'object' form when given as an object conforming to the schema,
        but with non-container values and free/categorical keys replaced by their types.
        - Categorical indices may be 'tabulated', or replaced by tuples of their values
          e.g. `('O', 'A', 'B', 'AB'): {'name': str, 'birthday': {'m': int, 'd': int}}` instead of `str: ...`
        - Categorical and free indices, and implicit values, may be 'annotated', or paired with their identities
          e.g. `(str, 'name'): {'birthday': {'m': int, 'd': int}, 'blood': str}` instead of `str: ...`
    Examples:
    1. {'people': [{'name': str, 'height': int, 'birthday': {'m': int, 'd': int}, 'blood': str}]}
    2. {'people': {str: {'height': int, 'birthday': {'m': int, 'd': int}, 'blood': str}}}
    3. {'people': {str: {str: {'height': int, 'birthday': {'m': int, 'd': int}}}}}
    4. {'people': {str: [{'name': str, 'height': int, 'birthday': {'m': int, 'd': int}}]}}
    5. {'people': [{'blood': str, 'people': {str: {'height': int, 'birthday': {'m': int, 'd': int}}}}]}
    6. {'people': [{'blood': str, 'people': [{'name': str, 'height': int, 'birthday': {'m': int, 'd': int}}]}]}

    Normality conditions for schemas:
    (A) All fields have distinct keys.
        Example (pet!name and species by owner!name):
        -: {'name': str, 'pets': [{'name': str, 'species': str}]}
        Normalization strategies:
            1. rename colliding keys
               +: {'name': str, 'pets': [{'petname': str, 'species': str}]}
               this is the most general solution
            2. make one colliding key implicit
               +: {'name': str, 'pets': {str: [str]}}
               works because pet!name's only sibling field is categorical
            3. index by one of the colliding keys
               +: {'pets': {str: [{'name': str, 'species': str}]}}
               works because owner!name is a primary key
    (B0) All keys are reserved, categorical, implicit, or free. (No restraints).
         {(str, 'owner!name'): {(str, 'pet!name'): (str, 'species')}} (without annotations: {str: {str: str}} )
    (B1) All keys are reserved, categorical, or implicit, save possibly one free primary key.
         {(str, 'owner!name'): {(str, 'species'): [(str, 'pet!name')]}} (without annotations: {str: {str: [str]}} )
    (B2) All keys are reserved, categorical, or implicit.
         [{'name': str, (str, 'species'): [(str, 'pet!name')]}]
    (B3) All keys are reserved or categorical.
         [{'name': str, (str, 'species'): [{'petname': str}]}]
    (B4) All keys are reserved.
         [{'name': str, 'pets': [{'petname': str, 'species': str}]}]
    (C)
    (D) There are no subtables. (This generally requires massive redundancy).
         [{'name': str, 'petname': str, 'species': str}]

"""

"""
    # record.json → <str>: {str}: <str>: (<str>: int, str, bool), int, [str], (<str>: str?)
    # messages.json → {str}: <str>: [str], (<str>: str?)
    # descriptions.json → {str}: <str>: int, str, [str], [<str>: str, int]
    record.users["alice"]: {
        "data": {
            "followers_count": int,        | the number of followers alice has
            "friends_count": int,          | the number of users alice follows
            "name": str,                   | the user's name
            "screen_name": str,            | the user's screen name
            "description": str,            | the user's description
            "profile_image": str,          | the user's profile image
            ...
        },
        "depth": int,                      | alice's degrees of separation from the root user
        "follows_to": list[str],           | the users that alice follows
        "follows_from": list[str],         | the users that follow alice
        "cursors": {
            "follows_to": Optional[str],   | the cursor for alice's follows list, if it wasn't completely fetched
            "follows_from": Optional[str]  | the cursor for alice's followers list, if it wasn't completely fetched
        }
    }

    with open("visible/descriptions.json", "r") as file:
        descriptions = json.load(file)

    descriptions["alice"]: {
        "confidence": int,                | the confidence score for the description
        "description": str,               | the description of the user
        "interests": list[str],           | the user's interests
        "aptitudes": list[{               | the user's aptitudes
            "field": str,                 |     the field of expertise
            "level": int                  |     the user's proficiency in the field, on a scale of 1 to 5
        }]
    }

    with open("visible/messages.json", "r") as file:
        messages = json.load(file)

    messages["alice"]: {
        "tweets": list[str],             | the user's tweets
        "replies": list[str],            | the user's replies
        "cursors": {
            "tweets": str | None,        | the cursor for the user's tweets, if it wasn't completely fetched
            "replies": str | None        | the cursor for the user's replies, if it wasn't completely fetched
        }
    }

"""

Key: TypeAlias = int | str | tuple[int | str]
Table: TypeAlias = list[dict] | dict[str, list[dict]] | dict[str, dict]


def column(table: list[dict], key: Key) -> list[Any]:
    """
    Extracts a column of values from a table based on a specified key.

    The function supports nested keys for dictionaries and numerical indices for lists/tuples.
    It navigates through the structure of each row in the table to retrieve the value associated with the key.
    If a row does not contain the specified key or if the path to a nested key is incomplete, it appends None to the result.

    :param table: A list of dictionaries representing the table. Each dictionary should have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) to extract values for.
                Nested keys can be specified as a string with components separated by dots (e.g., 'a.b.c')
                or as a list of strings and/or integers (e.g., ['a', 'b', 0]).
    :return: A list of values corresponding to the specified key for each row in the table.
             If a row does not have the key, the corresponding value in the list will be None.
    """
    if isinstance(key, str):
        key = key.split(".") if "." in key else [key]
    if isinstance(key, int):
        return [row[key] for row in table]
    col = []
    for row in table:
        val = row
        for k in key:
            if isinstance(val, dict):
                val = val.get(k)
            elif isinstance(val, (list, tuple)) and isinstance(k, int):
                val = val[k] if 0 <= k < len(val) else None
            else:
                val = None
            if val is None:
                break
        col.append(val)
    return col

def find_duplicates(table: list[dict], key: Key) -> list[Any]:
    """
    Finds and returns duplicate values associated with a specified key in a table.

    This function is designed to identify duplicate values within a column of a table,
    where the column is identified by the provided key. The key may represent a simple
    field or a nested field within the dictionaries that make up the table rows.
    Duplicate values are those that appear more than once across all rows.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) for which
                duplicate values are to be identified. Nested keys can be specified as a string
                with components separated by dots (e.g., 'a.b.c') or as a list of strings
                and/or integers (e.g., ['a', 'b', 0]).
    :return: A list of duplicate values found for the specified key. If no duplicates
             are found, an empty list is returned.
    """
    col = column(table, key)
    v_counts = {}
    duplicates = []
    for v in col:
        sv = json.dumps(v)
        v_counts[sv] = v_counts.get(sv, 0) + 1
        if v_counts[sv] > 1 and v not in duplicates:
            duplicates.append(v)
    return duplicates

def is_free(table: list[dict], key: Key) -> bool:
    """
    Checks if a given key is "free" in a table, meaning all its values are unique.

    A key is considered "free" if no value associated with this key is repeated across
    the entire table. This implies that each value for this key is unique and can be
    used as a unique identifier for rows in the table.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) to check
                for uniqueness. Nested keys can be specified as a string with components
                separated by dots (e.g., 'a.b.c') or as a list of strings and/or integers
                (e.g., ['a', 'b', 0]).
    :return: True if the key is "free" (all values are unique), False otherwise.
    """
    return len(find_duplicates(table, key)) == 0

def is_categorical(table: list[dict], key: Key) -> bool:
    """
    Determines if a key in a table is categorical based on the distribution of its values.

    A key is considered categorical if the number of distinct values associated with it
    is relatively small compared to the total number of rows in the table. Specifically,
    a key is deemed categorical if the square of the number of distinct values is less than
    or equal to the number of rows. This heuristic suggests that the key divides the table
    into a manageable number of categories.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) to evaluate
                for categorical properties. Nested keys can be specified as a string with
                components separated by dots (e.g., 'a.b.c') or as a list of strings and/or
                integers (e.g., ['a', 'b', 0]).
    :return: True if the key is considered categorical based on the defined heuristic, False otherwise.
    """
    col = column(table, key)
    return len(set(col))**2 <= len(col)

def get_categories(table: list[dict], key: Key) -> list[Any]:
    """
    Retrieves the list of distinct categories (unique values) for a given key in a table.

    This function extracts all unique values associated with the specified key across
    all rows of the table. The key can represent a simple field or a nested field within
    the dictionaries that make up the table rows.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) for which
                unique values (categories) are to be retrieved. Nested keys can be specified
                as a string with components separated by dots (e.g., 'a.b.c') or as a list
                of strings and/or integers (e.g., ['a', 'b', 0]).
    :return: A list of unique values (categories) found for the specified key.
    """
    categories = []
    for v in column(table, key):
        if v not in categories:
            categories.append(v)
    return categories

def get_keys(table: Table) -> list[list[Key]]:
    """
    Retrieves all unique keys from a table, including nested keys represented as lists.

    This function traverses the structure of the input table, which can be either a list of
    dictionaries or a nested dictionary. It identifies all unique keys, including those nested
    within dictionaries or lists, and returns them as a list of tuples. Each tuple represents
    a path to a key in the table structure, with nested keys represented by sequences of
    strings or integers within the tuple.

    :param table: The table data, which can be either a list of dictionaries or a nested dictionary.
    :return: A list of tuples, where each tuple represents a unique key or a path to a nested key
            in the table. Nested keys are represented by sequences of strings or integers within
            the tuple.
    """
    keys = []
    if isinstance(table, dict):
        for k, v in table.items():
            if isinstance(v, (dict, list)):
                for nk in get_keys(v):
                    if not isinstance(nk, list):
                        nk = [nk]
                    if [k] + nk not in keys:
                        keys.append([k] + nk)
            elif [k] not in keys:
                keys.append([k])
    elif isinstance(table, list):
        for row in table:
            if isinstance(row, dict):
                for k, v in row.items():
                    if isinstance(v, (dict, list)):
                        for nk in get_keys(v):
                            if not isinstance(nk, list):
                                nk = [nk]
                            if [k] + nk not in keys:
                                keys.append([k] + nk)
                    elif [k] not in keys:
                        keys.append([k])
            elif isinstance(row, list):
                for item in row:
                    if isinstance(item, (dict, list)):
                        for nk in get_keys(item):
                            if not isinstance(nk, list):
                                nk = [nk]
                            if nk not in keys:
                                keys.append(nk)
    return keys

def index_categorical(table: list[dict], key: Key) -> dict[Any, list[dict]]:
    """
    Indexes a table by a categorical key, grouping rows into lists based on the key's unique values.

    This function transforms a table (represented as a list of dictionaries) into a dictionary
    where each key is a unique value from the specified categorical key, and each value is a list
    of dictionaries. Each dictionary in these lists represents a row from the original table that
    matches the corresponding category value. The original key used for categorization is removed
    from each row to avoid redundancy.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure and contain the
                  specified categorical key.
    :param key: The key (or nested key path as a string or list of strings/integers) to be used
                for categorizing the rows. This key should be categorical, meaning it has a
                relatively small number of distinct values compared to the total number of rows.
                Nested keys can be specified as a string with components separated by dots
                (e.g., 'a.b.c') or as a list of strings and/or integers (e.g., ['a', 'b', 0]).
    :return: A dictionary where keys are the unique values of the categorical key, and values are
             lists of dictionaries. Each dictionary in the list represents a row from the original
             table that matches the category, with the categorical key removed.
    """
    categories = {c: [] for c in get_categories(table, key)}
    for d in table:
        categories[d[key]].append(d.copy())
        del categories[d[key]][-1][key]
    return categories

def index_free(table: list[dict], key: Key) -> dict[Any, dict]:
    """
    Indexes a table by a free key, creating a dictionary where each key is a unique value of the
    specified key, and each value is the corresponding row from the table.

    This function is used when the specified key is known to have unique values across all rows
    of the table (i.e., it is a "free" key). It constructs a dictionary where each key is a unique
    value of the free key, and the corresponding value is the entire row (as a dictionary) from
    the original table that matches this unique value. The free key is removed from each row in
    the resulting dictionary to avoid redundancy.

    :param table: A list of dictionaries, where each dictionary represents a row in the table.
                  It is assumed that all dictionaries have a consistent structure and contain the
                  specified free key.
    :param key: The key (or nested key path as a string or list of strings/integers) to be used
                for indexing the rows. This key should be free, meaning each value associated with
                this key is unique across all rows. Nested keys can be specified as a string with
                components separated by dots (e.g., 'a.b.c') or as a list of strings and/or
                integers (e.g., ['a', 'b', 0]).
    :return: A dictionary where keys are the unique values of the free key, and values are
             dictionaries representing the corresponding rows from the original table, with the
             free key removed.
    """
    indices = dict.fromkeys(column(table, key))
    for d in table:
        indices[d[key]] = d.copy()
        del indices[d[key]][key]
    return indices

def index(data: list[dict], key: Key) -> dict[str, dict | list[dict]]:
    """
    Indexes a list of dictionaries (table) based on the specified key, determining whether to use
    categorical or free indexing.

    This function decides the appropriate indexing method based on whether the provided key is
    categorical or free. If the key is identified as categorical (having a limited number of
    distinct values relative to the table size), it performs categorical indexing. If the key is
    identified as free (having unique values across all rows), it performs free indexing.

    :param data: A list of dictionaries, where each dictionary represents a row in the table.
                 It is assumed that all dictionaries have a consistent structure.
    :param key: The key (or nested key path as a string or list of strings/integers) by which
                to index the table. The function determines whether this key is categorical or
                free and applies the corresponding indexing method. Nested keys can be specified
                as a string with components separated by dots (e.g., 'a.b.c') or as a list of
                strings and/or integers (e.g., ['a', 'b', 0]).
    :return: A dictionary resulting from either categorical or free indexing:
             - If categorical indexing is used, the keys are the unique values of the categorical
               key, and the values are lists of dictionaries representing the rows that match each
               category.
             - If free indexing is used, the keys are the unique values of the free key, and the
               values are dictionaries representing the corresponding rows.
    """
    return index_categorical(data, key) if is_categorical(data, key) else index_free(data, key)

def deindex(data: dict[str, dict | list[dict]], new_key: str) -> list[dict]:
    """
    Converts an indexed dictionary back into a list of dictionaries, effectively reversing the
    indexing operation.

    This function handles dictionaries that were previously indexed either by a free key or a
    categorical key. It reconstructs the original table format (a list of dictionaries) from
    the indexed structure. When the input dictionary was indexed by a free key, each value
    (which is a dictionary representing a row) is augmented with the free key's value under the
    new key name specified by `new_key`. When the input dictionary was indexed by a categorical
    key, each value (which is a list of dictionaries) is flattened into a single list, and each
    dictionary is augmented with the categorical key's value under the new key name.

    :param data: A dictionary that was previously created by either `index_free` or `index_categorical`.
                 - If indexed by a free key, the dictionary's keys are the unique values of the free
                   key, and the values are dictionaries representing individual rows.
                 - If indexed by a categorical key, the dictionary's keys are the unique values of
                   the categorical key, and the values are lists of dictionaries, each representing
                   a row that matches the category.
    :param new_key: The name of the key to be added to each row dictionary, holding the value of
                    the original indexing key (either free or categorical).
    :return: A list of dictionaries, representing the original table structure before indexing.
             Each dictionary in the list corresponds to a row in the original table, augmented
             with the `new_key` field that contains the value of the original indexing key.
    """
    ret = []
    if len(data) == 0:
        return ret
    sample = next(iter(data))
    if isinstance(data[sample], dict):  # free key
        for f, v in data.items():
            ret.append({**v, new_key: f})
    else:  # categorical key
        for f, lv in data.items():
            ret.extend({**v, new_key: f} for v in lv)
    return ret

def compute_schema_sketch(schema: Any) -> str:
    """
    Computes a sketch representation of a given schema or data structure.

    The sketch is a simplified string representation that captures the essential
    structure of the schema, using a custom syntax:
      - 'A: B' represents a dictionary with keys of type A and values of type B.
      - '<A>: B' represents a dictionary with reserved keys of type A.
      - '{A}: B' represents a dictionary with free keys of type A.
      - '[A]' represents a list of type A.
      - 'A, B' represents a union of types A and B.
      - 'A?' represents an optional type A (equivalent to A | None).
      - Parentheses are used to denote sub-objects when necessary.

    :param schema: The schema or data structure to represent. Can be a type, a
                   data structure composed of basic types, lists, dictionaries,
                   or any combination thereof.
    :return: A string representing the schema in sketch syntax.
    """
    if isinstance(schema, list):
        if len(schema) == 0:
            return "[]"
        sketches = []
        for item in schema:
            c = compute_schema_sketch(item)
            if isinstance(item, dict) and len(schema) > 1:
                c = f"({c})"
            if c not in sketches:
                sketches.append(c)
        res = f"[{', '.join(sketches)}]"
        if res.startswith("[(") and res.endswith(")]") and "," not in res:
            return f"[{res[2:-2]}]"
        return res
    if isinstance(schema, dict):
        if len(schema) == 0:
            return "{}"
        first_key = next(iter(schema.keys()))
        if all(isinstance(k, type(first_key)) for k in schema):
            key_sketch = compute_schema_sketch(first_key)
            value_sketch = compute_schema_sketch(reduce(join, schema.values()))
            if all(isinstance(k, str) and k.isidentifier() for k in schema):
                return f"<{key_sketch}>: {value_sketch}"
            return f"{{{key_sketch}}}: {value_sketch}"
        items_sketch = [
            [compute_schema_sketch(k), compute_schema_sketch(v)]
            for k, v in schema.items()
        ]
        if len(items_sketch) > 1:
            for i, (k, v) in enumerate(items_sketch):
                if isinstance(v, dict):
                    items_sketch[i] = [k, f"({v})"]
        items_sketch = ", ".join([f"{k}: {v}" for k, v in items_sketch])
        return f"{items_sketch}"
    if isinstance(schema, tuple):
        return "(" + ", ".join(compute_schema_sketch(item) for item in schema) + ")"
    if schema is type(None):
        return "?"
    if is_typelike(schema):
        if a := get_args(schema):
            if o := get_origin(schema):
                if o.__name__ == "UnionType":
                    return ", ".join(compute_schema_sketch(item) for item in a)
                if o.__name__ == "Optional":
                    return f"{compute_schema_sketch(a[0])}?"
                return f"{o.__name__}[{', '.join(compute_schema_sketch(item) for item in a)}]"
            return f"{schema.__name__}[{', '.join(compute_schema_sketch(item) for item in a)}]"
        return schema.__name__ if hasattr(schema, "__name__") else str(schema)
    return type(schema).__name__

class MetadataParsingError(Exception):
    """Custom exception for metadata parsing errors"""
    pass

def parse_json_with_metadata(data: str | dict, strict: bool = True) -> dict:
    """
    Parses a JSON file with metadata-driven processing rules.
    The function:
    1. Reads metadata to identify the primary table and processing rules
    2. Applies alias replacements from metadata["aliases"]
    3. Applies defaults from metadata["defaults"] with dynamic substitution
    :param data: Path to the JSON file, or the JSON data itself.
    :param strict: If True, raises exceptions for validation errors. If False, logs warnings.
    :return: Dict containing the processed data
    :raises:
        MetadataParsingError: For metadata-related errors when strict=True
        json.JSONDecodeError: For invalid JSON
        FileNotFoundError: When file doesn't exist
    """
    def _apply_aliases(data: list[dict] | dict, alias_obj: list[dict] | dict, last_key: str = "", silent: bool = True) -> Any:
        """
        Builds a mapping of aliases to canonical keys for each table.
        :param data: The data to process
        :param alias_obj: List of alias definitions from metadata
        :return: Dict mapping table names to {alias: canonical_key} mappings
        """
        warning = logger.warning if not silent else lambda msg: None
        if isinstance(data, list):
            return [_apply_aliases(item, alias_obj, last_key + f"[{i}]", silent) for i, item in enumerate(data)]

        if isinstance(alias_obj, list):
            for alias_subobj in alias_obj:
                data = _apply_aliases(data, alias_subobj, last_key, silent)
            return data

        if not isinstance(alias_obj, dict):
            warning(f"Invalid alias object: {alias_obj}")
            return data

        if not isinstance(data, dict):
            warning(f"Invalid data object: {data}")
            return data

        for k, v in alias_obj.items():
            if isinstance(v, dict):  # {a: {b: ...}}
                data[k] = _apply_aliases(data[k], v, last_key + f"[{k}]", silent)
                continue
            if isinstance(v, str):
                v = [v]
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                if isinstance(v[0], dict):  # {a: [{b: ...}, {c: ...}]}
                    if k in data:
                        data[k] = _apply_aliases(data[k], v[0], last_key + f"[{k}]", silent)
                    else:
                        warning(f"Key {k} not found in data at {last_key}")
                    continue
                if isinstance(v[0], str):  # {a: [b, c]}
                    syn_found = False
                    for syn in v:
                        if syn == k:
                            continue
                        if syn in data:
                            syn_found = True
                            if k in data:
                                if type(data[k]) is not type(data[syn]):
                                    warning(f"Both {k} and {syn} are present in data at {last_key}, but their types do not match")
                                    continue
                                warning(f"Both {k} and {syn} are present in data at {last_key}; joining them")
                                data[k] = join(data[k], data[syn])
                                del data[syn]
                                continue
                            data[k] = data[syn]
                            del data[syn]
                    if not syn_found:
                        if k not in data:
                            warning(f"Neither {k} nor any of its synonyms found in data at {last_key}")
                    continue
                warning(f"Invalid alias object list value: {v[0]} at {last_key}")
                continue
            warning(f"Invalid alias object value: {v} at {last_key}")

        return data

    def _apply_defaults(data: list[dict] | dict, defaults_obj: list[dict] | dict, indices: list[int] = [], last_key: str = "", silent: bool = True) -> Any:
        """
        Applies defaults to a data structure.
        :param data: The data to process
        :param defaults_obj: List of default values to apply
        :param indices_stack: Stack of current indices for substitution
        :return: Dict mapping table names to {alias: canonical_key} mappings
        """
        warning = logger.warning if not silent else lambda msg: None

        if isinstance(data, list):
            return [_apply_defaults(item, defaults_obj, indices + [i], last_key + f"[{i}]", silent) for i, item in enumerate(data)]

        if isinstance(defaults_obj, list):
            for defaults_subobj in defaults_obj:
                data = _apply_defaults(data, defaults_subobj, indices, last_key, silent)
            return data

        if not isinstance(defaults_obj, dict):
            warning(f"Invalid defaults object: {defaults_obj} at {last_key}")
            return data

        if not isinstance(data, dict):
            warning(f"Invalid data object: {data} at {last_key}")
            return data

        for k, v in defaults_obj.items():
            if isinstance(v, dict):  # {a: {b: ...}}
                if k not in data:
                    data[k] = {}
                data[k] = _apply_defaults(data[k], v, indices, last_key + f"[{k}]", silent)
            elif isinstance(v, list):
                if len(v) != 0 and isinstance(v[0], dict):  # {a: [{b: ...}, {c: ...}]}
                    for i, _ in enumerate(data[k]):
                        for vj in v:
                            data[k][i] = _apply_defaults(data[k][i], vj, indices + [i], last_key + f"[{k}][{i}]", silent)
                elif k not in data:
                    data[k] = v
            elif k not in data:
                data[k] = v
                if isinstance(v, str) and re.findall(r'\$\{([^\}]+)\}', v):
                    data[k] = _render_substitutions(v, data, indices, last_key + f"[{k}]")
        return data

    def _render_substitutions(val: Any, context: dict, indices: list[int], last_key: str = "") -> Any:
        """
        Renders substitutions in a value using context and indices.
        :param val: Value to process substitutions in
        :param context: Dictionary of context values for substitution
        :param indices: List of indices for array access
        :return: Processed value with substitutions applied
        """
        if not isinstance(val, str):
            return val

        patterns = re.finditer(r'\$\{([^\}]+)\}', val, re.IGNORECASE)
        if not patterns:
            return val
        for pattern in patterns:
            match = pattern.group(1).strip()
            try:
                replacement = str(eval(match, {"indices": indices} | context))
            except Exception as e:
                logger.warning(f"Error in substitution '{match}' at {last_key}: {e!s}")
                replacement = ""
            val = val.replace(pattern.group(0), replacement)
        return val

    try:
        if isinstance(data, str):
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(f)
    except json.JSONDecodeError as e:
        logger.exception(f"Invalid JSON in {data}")
        raise
    except FileNotFoundError:
        logger.exception(f"File not found: {data}")
        raise

    # Validate metadata structure
    if not isinstance(data, dict):
        msg = f"Root JSON must be an object, got {type(data)}"
        if strict:
            raise MetadataParsingError(msg)
        logger.warning(msg)
        return data

    if "metadata" not in data:
        return data
    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        msg = f"metadata must be an object, got {type(metadata)}"
        if strict:
            raise MetadataParsingError(msg)
        logger.warning(msg)
        return data

    # Process aliases and defaults
    try:
        if "aliases" in metadata:
            data = _apply_aliases(data, metadata["aliases"], silent = True)
        if "defaults" in metadata:
            data = _apply_defaults(data, metadata["defaults"], silent = True)
            if "aliases" in metadata:
                # missing keywords should only raise warnings if they're not filled in by defaults
                # but we can't fill in defaults before we check for aliases, since that allows for data conflicts
                # so we apply aliases while silencing warnings, apply defaults, then reapply aliases without silencing warnings
                data = _apply_aliases(data, metadata["aliases"], silent = False)

        return data

    except Exception as e:
        msg = f"Error processing metadata: {e!s}"
        if strict:
            raise MetadataParsingError(msg) from e
        logger.exception(msg)
        return data

class Table:
    def __init__(self, data) -> None:
        self.metadata = None
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)
        if "metadata" in data:
            self.metadata = data["metadata"]
            data = parse_json_with_metadata(data)
            if "primary_table" in data["metadata"]:
                if data["metadata"]["primary_table"] in data:
                    data = data[data["metadata"]["primary_table"]]
        self.data = data

    @property
    def schema_object(self) -> Any:
        return schema_object(self.data)

    @property
    def schema_sketch(self) -> Any:
        return compute_schema_sketch(self.data)

    def save(self, path, filetype) -> None:
        pass
