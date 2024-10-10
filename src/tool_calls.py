"""
Tools and utilities for handling tools
"""
from __future__ import annotations

from .utilities import *

import ast
import inspect
import os
import re
import requests
from exa_py.api import Exa
from openai import OpenAI
from PIL import Image, PngImagePlugin
from subprocess import Popen as subprocess_Popen, PIPE as subprocess_PIPE, run as subprocess_run

tool_ids = ["google-search", "google-cx-id", "exa", "jina"]

# data/secrets.json format:
# { data : [{ id : "openai", name : "OpenAI API Key", value : "sk-abcd..." }] }
with open('data/secrets.json', encoding='utf-8') as f:
	secrets_table = Dict(json_load(f))

keys_tools = {item.id : item.value for item in secrets_table.secrets if item.id in tool_ids}

clients_tools = {
    "exa": Exa(api_key=keys_tools['exa']),
}


"""
Class: Tool

Description:
The Tool class provides functionalities for saving, loading, and generating schemas for functions. It also includes methods for invoking the tool, generating docstrings automatically, and converting human-readable function descriptions into schemas.

Methods:
- __init__: Initializes the Tool object.
- save: Saves an image file with embedded schema and function source.
- load: Loads an image file to retrieve schema and function source.
- gen_schema: Generates a JSON schema for a function based on its signature and docstring.
- auto_schema: Automatically generates a schema for a function by inferring its purpose and parameters.
- schema_from_string: Converts a human-readable function description into a proper schema.
- __call__: Invokes the tool by calling the function it encapsulates.
- describe: Prints a description of the tool based on the generated schema.
"""
class Tool:
    """Wrapper for tools"""
    def __init__(self, tool_file_or_function: Optional[Any] = None, load_passive: bool = False) -> None:
        """
        Initialize a Tool object.
        :param tool_file_or_function: Either a file path (str) or a callable function.
        :param load_passive: If True, don't automatically load the tool file.
        """
        self.schema: Optional[dict] = None
        self.source: Optional[str] = None
        self.func: Optional[Callable] = None
        self.empty: bool = False
        self.tool_file: Optional[str] = None
        if isinstance(tool_file_or_function, str):
            self.tool_file = tool_file_or_function
            if not load_passive:
                if self.tool_file.endswith('.png'):
                    self.load_png()
                elif self.tool_file.endswith('.py'):
                    self.load_py()
                else:
                    raise Exception("Tool file must be a PNG or Python file")
        elif callable(tool_file_or_function):
            self.func = tool_file_or_function
            try:
                self.source = inspect.getsource(self.func)
            except Exception:
                print(f'Warning: could not get source for tool function {self.func.__name__}')
            try:
                if not self.schema:
                    self.schema = self.gen_schema(self.func)
            except Exception:
                print(f'Warning: could not get schema for tool function {self.func.__name__}')

    def save_png(self, image_file: str, tool_file: Optional[str] = None) -> str:
        """
        Save the tool function as a PNG file with embedded schema and source code.
        :param image_file: The path of the image file to save.
        :param tool_file: Optional file path to save the tool to. If None, uses self.tool_file.
        :returns: The path of the saved tool file.
        :raises Exception: If no tool file is specified or if the image file is not a PNG.
        """
        if tool_file is None:
            if self.tool_file is None:
                raise Exception("Tool file to save to not specified")
            tool_file = self.tool_file
        try:
            suffix = tool_file.split('/')[-1].split('.')[-1]
            if not re.fullmatch(r'[a-z]{,4}', suffix):
                raise ValueError('Invalid file suffix')
        except Exception as e:
            raise Exception('Tool filepath must have a regular ending (".png", ".tool", ...)') from e
        if not image_file.endswith('.png'):
            raise Exception("Image file must be a PNG")
        if self.schema is None or (self.func is None and self.source is None):
            if self.schema is None:
                missing = "schema"
                if self.func is None:
                    missing += " or function"
            else:
                missing = "function"
            msg = f'Function does not have a {missing} to save'
            raise Exception(msg)
        image = Image.open(image_file)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Schema", json_dumps(self.schema))
        if self.source is None:
            self.source = inspect.getsource(self.func)
        meta.add_text("Source", self.source)
        image.save(tool_file, "PNG", pnginfo=meta)
        return tool_file

    def save_py(self, tool_file: Optional[str] = None) -> str:
        """
        Save the tool function as a Python file.
        :param tool_file: Optional file path to save to. If None, uses self.tool_file.
        :returns: The path of the saved file.
        :raises Exception: If no tool file is specified or if the function is not found.
        """
        if tool_file is None:
            if self.tool_file is None:
                raise Exception("Tool file to save to not specified")
            tool_file = self.tool_file
        if not tool_file.endswith('.py'):
            raise Exception("Tool file must be a Python file")
        if self.source is None:
            if self.func is None:
                raise Exception("Function not found")
            self.source = inspect.getsource(self.func)
        with open(tool_file, 'w') as f:
            f.write(self.source)
        return tool_file

    def load_png(self, tool_file: Optional[str] = None) -> Tool:
        """
        Load a tool function from a PNG file.
        :param tool_file: Optional file path to load from. If None, uses self.tool_file.
        :returns: The Tool instance.
        :raises Exception: If no tool file is specified or if schema retrieval fails.
        """
        if tool_file is None:
            if self.tool_file is None:
                raise Exception("Tool file to load from not specified")
            tool_file = self.tool_file
        if not tool_file.endswith('.png'):
            raise Exception("Tool file must be a PNG")
        try:
            image = Image.open(tool_file)
            meta = image.info
        except Exception as e:
            raise Exception("Could not open image file") from e
        if "Source" in meta:
            self.source = meta["Source"]
            # Recreate the function object from the source using exec
            local_dict = {}
            exec(self.source, globals(), local_dict)
            # Assuming the function will always be named within the source
            func_ast = ast.parse(self.source)
            func_name = func_ast.body[0].name
            self.func = local_dict[func_name]
        if "Schema" in meta:
            self.schema = json_loads(meta["Schema"])
        if self.schema is None:
            try:
                self.schema = self.gen_schema(self.func)
            except Exception as e:
                raise Exception("Could not retrieve schema from function object") from e
        return self

    def load_py(self, tool_file: Optional[str] = None) -> Tool:
        """
        Load a tool function from a Python file.
        :param tool_file: Optional file path to load from. If None, uses self.tool_file.
        :returns: The Tool instance.
        :raises Exception: If no tool file is specified or if schema retrieval fails.
        """
        if tool_file is None:
            if self.tool_file is None:
                    raise Exception("Tool file to load from not specified")
            tool_file = self.tool_file
        if not tool_file.endswith('.py'):
            raise Exception("Tool file must be a Python file")
        with open(tool_file) as f:
            self.source = f.read()
        local_dict = {}
        exec(self.source, globals(), local_dict)
        func_ast = ast.parse(self.source)
        func_name = func_ast.body[0].name
        self.func = local_dict[func_name]
        if self.schema is None:
            try:
                self.schema = self.gen_schema(self.func)
            except Exception as e:
                raise Exception("Could not retrieve schema from function object") from e
        return self

    # gen_schema will work on any function f with head provided by auto_schema(f, return_docstring=True)
    def gen_schema(self, func: Optional[Callable] = None, hide: List[str] = None) -> Dict[str, Any]:
        """
        Generates a JSON schema for a given function based on its signature and docstring.
        :param func: The function for which to generate the schema.
        :param hide: List of parameter names to hide from the schema.
        :returns: A dictionary representing the JSON schema of the function.
        """
        if hide is None:
            hide = ["self"]
        if func is None:
            try:
                func = self.func
            except Exception as e:
                raise Exception("Could not retrieve function for gen_schema call") from e

        def get_type_schema(typ) -> Dict:
            origin = get_origin(typ)
            args = get_args(typ)

            if origin in {list, List, typing_List}:
                return {"type": "array", "items": get_type_schema(args[0])}
            if origin in {dict, Dict, typing_Dict}:
                return {
                    "type": "object",
                    "additionalProperties": get_type_schema(args[1]),
                }
            if origin is Union:
                return {"oneOf": [get_type_schema(arg) for arg in args]}
            return {"type": {
                int: "integer",
                str: "string",
                float: "number",
                bool: "boolean",
            }.get(typ, "string")}
        doc = func.__doc__
        if doc is None:
            raise Exception("Function docstring not found")
        description, *controls = (re.sub(r'^\s+', '', x).strip() for x in re.split(r'\n[\s\t]*?\:', doc))
        description = re.sub([*re.findall("^(\\s+)", description), "\n"][0], '\n', description).strip()
        param_descriptions = {}
        for c in controls:
            if c.startswith('schema:'):
                full_schema = eval(c.split('schema:')[-1].strip())
                return self.parse_schema(full_schema, func.__name__, description)
            if c.startswith('param'):
                param_name = c.strip().split(' ')[1].split(':')[0]
                param_desc = ':'.join(c.strip().split(':')[1:]).strip()
                param_descriptions[param_name] = param_desc.strip()

        param_schema = {"type": "object", "properties": {}, "required": []}
        for name, param in inspect.signature(func).parameters.items():
            if name not in hide:
                type_schema = get_type_schema(param.annotation)
                if param.default is param.empty:
                    param_schema["required"].append(name)
                param_schema["properties"][name] = {
                    **type_schema,
                    "description": param_descriptions.get(name, ""),
                }

        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": param_schema,
            },
        }
        if func == self.func:
            self.schema = schema
        return schema

    def auto_schema(self, func: Optional[Callable] = None, provider_model: str = "beta:coder", return_docstring: bool = False) -> Union[Dict[str, Any], str]:
        """
        Generates a tool schema for a given function based on its signature and inferred purpose.
        :param func: The function for which to generate the schema.
        :param providermodel: The provider and model to use for generating the schema. Defaults to "beta:coder" (resolves to "deepseek-beta:deepseek-coder").
        :param return_docstring: Whether to return the generated docstring instead of the schema. Defaults to False.
        :returns: A dictionary representing the tool schema of the function, or a string containing the docstring if return_docstring is True.
        """
        if func is None:
            try:
                func = self.func
            except Exception as e:
                raise Exception("Could not retrieve function for gen_schema call") from e
        prompt = f"Please rewrite the declaration of this Python function to include valid parameters for the input and output types, and write a docstring containing a brief description of what it does and a brief description of what each of its parameters does, in the form \n```\ndef f(a: type, b: type [...]) -> type\n\t\"\"\"\n\tfunction purpose\n\t:param a: role of parameter a\n\t:param b: role of parameter b\n\t[...]\n\t:returns: what the output is\"\"\"\n\tpass\n```\nAssume that the `typing` module and any modules used by the function have already been imported, and literally write \"pass\" instead of writing the rest of the code. Do not continue beyond the docstring. Here is the function:\n```python\n{inspect.getsource(func)}\n```\n\nHere's the revised function declaration with valid parameters and a docstring: ```python\n"
        func_str = text_completion(
            prompt=prompt,
            model=provider_model,
            max_tokens=1024,
            suffix='\n```', return_output=True, temperature=0.1)
        if return_docstring:
            return func_str
        try:
            # common failure modes
            func_str = re.sub(r"args ?: ?([lL]ist)([,\)])", r"args: \1 = []\2", func_str)
            func_str = re.sub(r"kwargs ?: ?([dD]ict)([,\)])", r"kwargs: \1 = {}\2", func_str)
            # Parse the function string to an AST
            func_ast = ast.parse(func_str).body[0]
            # Split the docstring into lines and extract description & params
            lines = [i.strip() for i in ast.get_docstring(func_ast).strip().split("\n")]
            param_pattern = re.compile(r":param (\w+): (.+)")
            desc_lines, param_descs, in_desc = [], {}, True
            for line in lines:
                if in_desc and line.startswith((":param", ":return")):
                    in_desc = False
                if in_desc:
                    desc_lines.append(line)
                elif match := param_pattern.match(line):
                    param_descs[match.group(1)] = match.group(2)
            params_schema = {"type": "object", "properties": {}, "required": []}
            for arg in func_ast.args.args:
                if arg.annotation:
                    arg_type = {
                        "int": "integer", "str": "string", "float": "number",
                        "bool": "boolean", "list": "array", "dict": "object",
                    }.get(ast.get_source_segment(func_str, arg.annotation), "string")
                else:
                    arg_type = "string"
                arg_desc = param_descs.get(arg.arg, "")
                params_schema["properties"][arg.arg] = {
                    "type": arg_type, "description": arg_desc}
            num_default = len(func_ast.args.args) - len(func_ast.args.defaults)
            params_schema["required"] = [arg.arg for arg in func_ast.args.args[:num_default]]
            schema = {
                "type": "function",
                "function": {
                    "name": func_ast.name,
                    "description": " ".join(desc_lines).strip(),
                    "parameters": params_schema},
            }
            return schema
        except Exception as e:
            print("Failed: " + str(e) + ". Output: \n\n" + func_str)

    @staticmethod
    def schema_from_string(s: str) -> Dict[str, Any]:
        '''
        schema_from_string(s: str)
        Takes a human-readable typed function description like this and turns it into a proper schema
        :param s: A human-readable typed function description
        :returns: A schema object
        '''
        s = s.replace('\t', '').strip()
        pattern = '^(?:def )?([A-Za-z_0-9]+)\\((.*?)(?:\\):?\\s*\n|\\) ->.*?\n)(?:["\' ]*\n)?(\\s*\\w.*?)\n(\\s*(?:(?::param )?.*\n?)*)(?:\n?:?returns: (.*?))?\n?(?:["\' ]*\n)?(?:pass\n?)?'
        if (L := re.match(pattern, s)):
            func_name, arg_types, func_desc, arg_descs, out_desc = L.groups()
        else:
            msg = f"Couldn't parse string {s} as schema"
            raise Exception(msg)
        # print(L.groups())
        required_args = [i.strip().split(':')[0] for i in arg_types.split(',') if '=' not in i]
        arg_types = {i: {'int': 'integer', 'list': 'array', 'dict': 'object', 'str': 'string', 'bool': 'boolean', 'callable': 'function'}[j.lower()] for (i, j) in re.findall('([A-Za-z_0-9]+): ?([A-Za-z]+)', arg_types)}
        arg_descs = dict([re.match('(?::param )?([A-Za-z_0-9]+): (.*)', i).groups() for i in arg_descs.strip().split('\n')])
        args = list(arg_types.keys())
        return {'type': 'function', 'function': {'name': func_name, 'description': func_desc, 'parameters': {'type': 'object', 'properties': {i: {'type': arg_types[i], 'description': arg_descs[i]} for i in args}, 'required': required_args}}}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the tool function.

        :param args: Positional arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        :returns: The result of calling the function.
        :raises Exception: If the tool is empty.
        """
        if self.empty:
            raise Exception("Can't call tool method as tool is empty!")
        return self.func(*args, **kwargs)

    def describe(self) -> Optional[str]:
        """
        Generate a description of the tool based on its schema.

        :returns: A string description of the tool, or None if no schema is available.
        """
        if not self.schema:
            return None
        params = "\n\t".join([v["type"] + " " + k + ": " + v["description"] for k, v in self.schema["function"]["parameters"]["properties"].items()])
        return f'Tool {self.schema["function"]["name"]}({", ".join(list(self.schema["function"]["parameters"]["properties"]))}): {self.schema["function"]["description"]}\n\t{params}'

    def parse_schema(self, K: Dict[str, Any], name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        # shortcut for compressing long tool schemas
        # example:
        # x = {'list edits': {'': "A list of edits",
        # 'dict edit': {'': '',
        # 'int id': "A line number to overwrite",
        # "list lines": {'': "New lines to insert here",
        # "str line": ''}}}}
        # Agent.tools.append(parse_schema(x, "line_editor", "Edit some lines in the present document"))
        def interp(s) -> (bool, str, str):
            s = s.strip()
            s2 = re.sub(r'(?:\w*)\?? ?[\W]*(\w+)[\W]*', r'\1', s)
            if len(s2) == 0:
                return False, "null", ""
            required = ("!" in s or "? " not in s)
            if '**' in s or ('{' in s and '}' in s) or 'dict ' in s:  # **docs, dict docs, {docs}
                t = "object"
            elif '*' in s or ('[' in s and ']' in s) or 'list ' in s:  # *vals, list vals, [vals]
                t = "array"
            elif '#' in s or '+' in s or 'int ' in s or 'float ' in s:  # +lbs, #lbs, int lbs
                t = "number"
            elif ('?' in s and (('? ' not in s) or s.count('?') > 1)) or 'bool ' in s:  # bool alive, alive?, ?alive
                t = "boolean"
            else:  # not a permitted type
                t = "function" if s.endswith('()') else "string"
            return required, t, s2

        D = {}
        for k, v in K.items():
            if not k:
                continue
            required, t, s = interp(k)
            D[s] = {"type": t}
            if t == "array":
                down = "items"
            elif t == "object":
                down = "properties"
            else:
                if isinstance(v, str):  # descriptions
                    D[s]["description"] = v
                elif isinstance(v, list):  # enums without descriptions
                    D[s]["enum"] = v
                elif isinstance(v, dict):  # multiple fields, e.g. {'str size': {'enum': ['S','M','L'], 'description': 'Shirt size'}} yields {'size': {'type': 'string', 'enum': ...}}
                    D[s] |= v
                continue
            D[s]["description"] = v['']
            if not D[s]["description"]:
                del D[s]["description"]
            D[s][down] = self.parse_schema(v)
            if t == "object":
                D[s]["required"] = [interp(i)[2] for i in v if interp(i)[0]]
        if name:
            return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": D, "required": [interp(i)[2] for i in K if interp(i)[0]]}}}
        return D

class Toolbox:
    """A collection of tools"""
    def __init__(self, tools: List[Union[Tool, Callable]] = None) -> None:
        """
        Initialize a Toolbox with a list of tools.

        :param tools: A list of Tool objects or callable functions to be converted to Tools.
        """
        if tools is None:
            tools = []
        self.tools: List[Tool] = []
        self.tool_dict: Dict[str, Tool] = {}
        for t in tools:
            if isinstance(t, Tool):
                self.tools.append(t)
                self.tool_dict[t.schema["function"]["name"]] = t
            elif callable(t):
                try:
                    T = Tool(t)
                    self.tools.append(T)
                    self.tool_dict[T.schema["function"]["name"]] = T
                except Exception:
                    print('Warning: could not instantiate tool from callable ' + t.__name__ + '  (' + str(t) + '), discarding')
            else:
                print('Warning: Could not recognize object ' + str(t) + ' of type ' + str(type(t)) + ' as a tool, discarding')

    def append(self, tool: Union[Tool, Callable]) -> None:
        """
        Add a new tool to the Toolbox.

        :param tool: A Tool object or a callable function to be converted to a Tool.
        """
        if isinstance(tool, Tool):
            self.tools.append(tool)
            self.tool_dict[tool.schema["function"]["name"]] = tool
        elif callable(tool):
            T = Tool(tool)
            self.tools.append(T)
            self.tool_dict[T.schema["function"]["name"]] = T

    def remove(self, tool_name: str) -> None:
        """
        Remove a tool from the Toolbox by its name.

        :param tool_name: The name of the tool to remove.
        """
        if tool_name in self.tool_dict:
            tool = self.tool_dict.pop(tool_name)
            self.tools.remove(tool)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool from the Toolbox by its name.

        :param tool_name: The name of the tool to retrieve.
        :returns: The Tool object if found, None otherwise.
        """
        return self.tool_dict.get(tool_name, None)

    def describe_all(self) -> str:
        """
        Get a description of all tools in the Toolbox.

        :returns: A string containing descriptions of all tools.
        """
        return "\n".join([tool.describe() for tool in self.tools])

    def __call__(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call a tool in the Toolbox by its name.

        :param tool_name: The name of the tool to call.
        :param args: Positional arguments to pass to the tool.
        :param kwargs: Keyword arguments to pass to the tool.
        :returns: The result of calling the tool.
        :raises Exception: If the tool is not found in the Toolbox.
        """
        tool = self.get_tool(tool_name)
        if tool:
            return tool(*args, **kwargs)
        msg = f"Tool {tool_name} not found in toolbox"
        raise Exception(msg)

    def __getitem__(self, tool_name: str) -> Tool:
        """
        Get a tool from the Toolbox using square bracket notation.

        :param tool_name: The name of the tool to retrieve.
        :returns: The Tool object.
        :raises KeyError: If the tool is not found in the Toolbox.
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            msg = f"Tool {tool_name} not found in toolbox"
            raise KeyError(msg)
        return tool

    def __contains__(self, tool_name: str) -> bool:
        """
        Check if a tool is in the Toolbox.

        :param tool_name: The name of the tool to check.
        :returns: True if the tool is in the Toolbox, False otherwise.
        """
        return tool_name in self.tool_dict

    def __iter__(self) -> Iterator[Tool]:
        """
        Get an iterator over the tools in the Toolbox.

        :returns: An iterator over the Tool objects.
        """
        return iter(self.tools)

    def __len__(self) -> int:
        """
        Get the number of tools in the Toolbox.

        :returns: The number of tools.
        """
        return len(self.tools)

    def __repr__(self) -> str:
        """
        Get a string representation of the Toolbox.

        :returns: A string representation of the Toolbox.
        """
        return f"Toolbox({', '.join([tool.schema['function']['name'] for tool in self.tools])})"

def gatekeep(label: str, categories: Optional[List[str]] = None, raise_error: bool = True) -> Callable:
    """
    A decorator factory for gatekeeping function execution based on specified categories.

    :param label: A label describing the type of object (e.g., "python code").
    :param categories: List of categories to check (e.g., ["harmful", "bugged", "obscene", "misinformation", "oversize"]).
    :param raise_error: Whether to raise an error (True) or return a message (False) when gatekeeping fails.
    """
    if categories is None:
        categories = ["harmful", "bugged"]

    def decorator(func: Callable) -> Callable:
        @functools_wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not os.environ.get('OPENAI_API_KEY'):
                raise Exception('OpenAI API key must be in local environment for gatekeeping to run.')

            client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

            # Assume the first argument is the code/command to check
            object_to_check = args[0] if args else next(iter(kwargs.values()), "")
            # Determine if the following {label} is liable to harm the system running it, or has bugs that will cause it to crash. Programs that may return errors or loop continuously are not considered harmful, but are considered bugged if this is *definitely* the case; programs that may destroy data (beyond the scope of routine file operations) or freeze a computer are, as are actively malicious programs. Return a JSON object with booleans corresponding to whether the program is harmful or bugged, and a string explaining why (or the empty string if nothing's wrong
            response = client.chat.completions.create(
                messages=[
                    {'role': 'system', 'content': f"Determine if the following {label} falls into any of these categories: {', '.join(categories)}. Return a JSON object with boolean values for each category and an explanation string."},
                    {'role': 'user', 'content': object_to_check},
                ],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'harmful_code_flags',
                        'schema': {
                            'type': 'object',
                            'properties': {cat: {'type': 'boolean'} for cat in categories} | {'explanation': {'type': 'string'}},
                            'required': [*categories, "explanation"],
                            'additionalProperties': False,
                        },
                    },
                },
                model="gpt-4o-mini",
            )

            try:
                result = json_loads(response.choices[0].message.content)
            except Exception as e:
                raise Exception("Gatekeeper failed to process object.") from e

            if any(result[cat] for cat in categories):
                error_message = f"Gatekeeper caught issues with the {label}. Explanation: {result['explanation']}\nOriginal {label}:\n{object_to_check}"
                if raise_error:
                    raise Exception(error_message)
                return error_message

            return func(*args, **kwargs)
        return wrapper
    return decorator

@Tool
@gatekeep("python code", categories=["harmful", "bugged"])
def run_python(code: str) -> Any:
    """
    Executes python code and return the result. Make sure your code explicitly returns a result, rather than printing it.
    :param code: The python code to execute
    :returns: The result of the execution
    """
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None

    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id='_result', ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename='<ast>', mode='exec'), ns)
    return ns.get('_result')

@Tool
@gatekeep("shell command", categories=["harmful", "bugged"])
def run_shell(command: str) -> str:
    """
    Execute shell command and return the result.
    :param command: The shell command to execute
    :returns: The result of the execution
    """
    process = subprocess_Popen(command, shell=True, stdout=subprocess_PIPE, stderr=subprocess_PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode() + stderr.decode()

@Tool
@gatekeep("JavaScript code", categories=["harmful", "bugged"])
def run_javascript(code: str) -> str:
    """
    Execute JavaScript code and return the result.
    :param code: The JavaScript code to execute
    :returns: The result of the execution
    """
    # This is a placeholder. You'd need to implement actual JavaScript execution,
    # possibly using a tool like Node.js or a JavaScript engine for Python.
    return "JavaScript execution not implemented yet."

@Tool
def exa_search(query: str, num_results: int = 10, snippets: int = 3, snippet_sentences: int = 1) -> list[dict[str, str | list[str]]]:
    """
    Performs an Exa.ai neural search on a query. Useful for finding interesting long-form content (e.g. scientific papers, blog posts, articles) on a variety of topics.
    :param query: The search query string.
    :param num_results: The number of search results to retrieve. Default is 10.
    :param snippets: The number of highlight snippets to include per URL. Default is 3.
    :param snippet_sentences: The number of sentences per highlight snippet. Default is 1.
    :returns: A list of dictionaries containing formatted search results with keys 'url', 'title', 'date', 'snippet', and optionally 'author'.
    """
    results = regularize(clients_tools['exa'].search_and_contents(query=query, num_results=num_results, highlights={"num_sentences": snippet_sentences, "highlights_per_url": snippets}))['results']
    new_results = []
    for obj in results:
        new_obj = {'url': obj['url'], 'title': obj['title'], 'date': obj['published_date'], 'snippet': [x.strip() for x in obj['highlights']]}
        if obj['author'] and obj['author'].strip():
            new_obj['author'] = obj['author'].strip()
        new_results.append(new_obj)
    return new_results

@Tool
def google_search(query: str, num_results: int = 10) -> list[dict[str, str | None]]:
    """
    Performs a Google search on a query and returns a list of simplified search result objects.
    :param query: The search query string.
    :param num_results: The number of search results to retrieve, default is 10.
    :returns: A list of dictionaries containing the URL, title, and optionally snippet of each search result.
    """
    HTTP_SUCCESS = 200
    google_api_key, google_cx_id = keys_tools['google-search'], keys_tools['google-cx-id']
    req = requests.get(f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cx_id}&q={query}&num={num_results}")
    if req.status_code != HTTP_SUCCESS:
        msg = f"Google search failed with status code {req.status_code}"
        raise Exception(msg)
    results = json_loads(req.text)['items']
    new_results = []
    for obj in results:
        new_obj = {'url': obj['link'], 'title': obj['title']}
        if 'snippet' in obj:
            new_obj['snippet'] = obj['snippet']
        new_results.append(new_obj)
    return new_results

@Tool
def tokenize(content: str, tokenizer: str = "o200k_base", return_tokens: bool = False) -> dict:
    """
    Tokenize content using Jina AI's tokenization API.
    :param content: The text content to tokenize.
    :param tokenizer: The tokenizer to use (default: "o200k_base").
    :param return_tokens: Whether to return each individual token and its index (default: False).
    :returns: A dictionary containing tokenization results.
    """
    url = "https://segment.jina.ai/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {keys_tools['jina']}",
    }
    data = {
        "content": content,
        "tokenizer": tokenizer,
        "return_tokens": str(return_tokens).lower(),
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

@Tool
def get_contents(url: str) -> dict:
    """
    Fetch human-readable webpage contents using Jina AI's API.
    :param url: The URL of the webpage to analyze.
    :returns: A dictionary containing webpage information.
    """
    jina_url = "https://r.jina.ai/"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {keys_tools['jina']}",
        "Content-Type": "application/json",
    }
    data = {"url": url}

    response = requests.post(jina_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

@Tool
def get_html(url: str) -> dict:
    """
    Fetch raw HTML from a given URL. If you just want to read the page and don't need the HTML code, use `get_contents` instead.
    :param url: The URL from which to fetch HTML.
    :returns: A dictionary containing the processed contents, excluding 'id', 'url' keys and null/empty values.
    """
    try:
        results = regularize(clients_tools['exa'].get_contents([url]))
        contents = results['results'][0]
        contents = {k: v for k, v in contents.items() if k not in {'id', 'url'} and v not in {None, ''}}
        return contents
    except ValueError:  # exa couldn't find it, so use requests
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
        except requests.exceptions.HTTPError as http_err:
            ERRORS = {
                404: f"Error 404: The resource at {url} was not found.",
                403: f"Error 403: Access to the resource at {url} is forbidden.",
                500: f"Error 500: Internal Server Error at {url}.",
                503: f"Error 503: Service Unavailable at {url}.",
            }
            text = f"HTTP error occurred: {http_err} (Status Code: {response.status_code})"  # catch-all error message
            for (code, msg) in ERRORS.items():
                if response.status_code == code:
                    text = msg
        except requests.exceptions.RequestException as req_err:
            text = f"Request error occurred: {req_err}"
        return text

@Tool
def yield_control(self, message: str = '') -> str:
    '''
    Interrupts the present actor coroutine, yielding control to its parent routine. Call upon completing a task, requesting information, etc.
    :param message: An optional message to be returned along with control.
    '''
    self.set_state('loop', False)
    self.set_state('yield_control', True)
    return message

@Tool
def meditate(self, prompt: str = '') -> str:
    '''
    See how a clone of your present state, with tools removed, proceeds in this situation.
    :param prompt: An optional system prompt for your clone, e.g. "Create an outline of steps to achieving your goal.". Will be appended after all previous messages, including any prior system prompts.
    '''
    clone = self.clone(tools=None).undo()
    clone.debug = False
    if prompt:
        clone.system(prompt)
    return clone.next().last.content

@Tool
def ooda_planner(observation: str = '', orientation: str = '', decision: str = '', action_plan: str = '') -> None:
    '''
    This tool has no external effect, but you may call it to organize your thoughts into a well-structured OODA loop.
    :param observation: A description of the current situation.
    :param orientation: A description of your current goal or objective.
    :param decision: A description of a decision you're making based on the observation and orientation.
    :param action_plan: A description of the action(s) you will take to fulfill this decision.
    '''
    pass

@Tool
def ask_human(prompt: str) -> str:
    '''
    Flip the script by asking a human to complete a task that requires user input or real-world presence or etc.
    :param prompt: A description of the task.
    '''
    return input(prompt)
