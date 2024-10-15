"""
Methods for text and chat completions, and agent wrapper classes
"""
from __future__ import annotations

from .tool_calls import *

from datetime import datetime as datetime_datetime
from time import time as time_time, sleep as time_sleep
from os import path as os_path
from tiktoken import get_encoding

separator = "::"

# data/secrets.json format:
# { secrets : [{ id : "openai", name : "OpenAI API Key", value : "sk-abcd..." }] }
with open('data/secrets.json', encoding='utf-8') as f:
    secrets_table = Dict(json_load(f))

keys = Dict({item.id : item.value for item in secrets_table.secrets})

# { providers : [{ id : "openai", api: {url: "https://..."}, "models": [ {"id": "gpt-4o", "mode": "chat", ..., "parameters": [...]}, ...] }, ...] }
with open('data/providers.json', encoding='utf-8') as f:
    provider_table = Dict(json_load(f))

# keys = {k: v['api']['key'] for k, v in provider_table.items()}

clients = Dict({provider.id: OpenAI(api_key = keys.get(provider.id, ""), base_url = provider.api.url) for provider in provider_table.providers})

models_by_mode = Dict({'text': {}, 'chat': {}})
models = List()

for provider in provider_table.providers:
    prov_name = provider.id
    vars()[provider.id.replace('-','_')] = clients[provider.id]
    for model in provider.models:
        model_data = {"provider": provider.id} | model
        if model.get("disabled", False):
            continue
        duplex_id = separator.join([provider.id, model.id])
        models_by_mode[model.mode][duplex_id] = model_data
        models.append(model_data)

# for prov_name, prov_vals in provider_table.items():
#     for prov_model, model_details in prov_vals['models'].items():
#         if "disabled" in model_details:
#             continue
#         mfix = re.sub(r'^\d+(.*)$', r'\1', prov_model)  # to fix the 2deepseek problem
#         models_by_mode[model_details["mode"]][prov_name + separator + mfix] = model_details
#         models_by_mode[model_details["mode"]][prov_name + separator + mfix]['provider'] = prov_name
#     vars()[prov_name] = clients[prov_name]
# for t in ['chat', 'text']:
#     for k, prov_vals in models_by_mode[t].items():
#         models[k] = prov_vals | {"id": k, "mode": t}
        # will set deepseek-beta:deepseek-x to text
        # which is fine, since its chat side is useless

cl100k_base = get_encoding("cl100k_base")
token_count = Fun(lambda s: len(cl100k_base.encode(s)))

def resolve(id: str, mode: str = "") -> List[Tuple[str, str]]:
    """
    Resolves a model ID to a list of tuples containing provider and model names based on the specified mode.
    :param id: The model ID to resolve, which can include a provider prefix.
    :param mode: The mode in which to resolve the model ID, either 'text' or 'chat'. Defaults to an empty string, which triggers a combined resolution.
    :returns: A list of tuples where each tuple contains a provider name and a model name.
    """
    id = sub('^["\' ]*(.*?)[\'" ]*$', r'\1', id)
    if not mode:
        combined = resolve(id, "chat") + resolve(id, "text")
        return list(set(combined))
    # Define the providers for each mode
    parts = id.lower().replace(" ", "").split(separator)
    specified_provider = parts[0] if len(parts) > 1 else ""
    specified_model = parts[-1]
    relevant_providers = filter(lambda p: fuzzy_in(specified_provider, p), [models_by_mode[mode][m]['provider'] for m in models_by_mode[mode]])
    # relevant_providers = [p for p in mode_providers.get(mode,[]) if fuzzy_in(specified_provider, p)]
    matches = List()
    for provider in list(set(relevant_providers)):
        for model in query(provider_table.providers, "id", provider).models:
            if fuzzy_in(specified_model, model.id.lower()) and (provider, model.id) not in matches and separator.join([provider, model.id]) in models_by_mode[mode]:
                matches.append((provider, model.id))
    matches.sort(key=lambda x: len(x[1]))
    if len(matches) > 1:
        if all(fuzzy_in(matches[0][1].split("/")[-1].lower(), m[1].split("/")[-1].lower()) for m in matches[1:]):
            matches = [matches[0]]
        # elif "openrouter" in [x[0] for x in matches]:
        # 	# is openrouter adding a bunch of extra matches
        # 	matches2 = [x for x in matches if x[0] != "openrouter"]
        # 	if len(matches2) > 0 and all([fuzzy_in(matches2[0][1].split("/")[-1].lower(), m[1].split("/")[-1].lower()) for m in matches2[1:]]):
        # 		matches = [matches2[0]]
    return matches

def completion(options: Dict) -> Any:
    """
    Generates a text or chat completion from arguments.
    :param options: Arguments to the completion model.
    :returns: The generated completion.
    """
    # this function can take a LOT of arguments
    # some of them go to the API to influence what is generated (apiParams)
    # some of them influence how we deal with the generation (localParams)
    # since APIs are finicky about extra arguments, we're going to fill these dicts in with options
    # throwing away unrecognized keys
    apiParams = Dict.fromkeys(['model', 'prompt', 'messages', 'suffix', 'max_tokens', 'stream', 'n', 'logprobs', 'top_logprobs', 'logit_bias', 'temperature', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'top_p', 'min_p', 'top_k', 'top_a', 'tools', 'tool_choice', 'parallel_tool_calls', 'grammar', 'json_schema', 'response_format', 'seed'], None)
    localParams = Dict.fromkeys(['mode', 'provider', 'effect', 'callback', 'print_output', 'yield_output', 'return_output', 'debug', 'force_model', 'force_provider', 'return_raw', 'pretty_tool_calls', 'return_object'], None)
    for (k, v) in options.items():
        if k in apiParams and v is not None:
            apiParams[k] = v
        elif k in apiParams and v is None:
            del apiParams[k]
        if k in localParams and v is not None:
            localParams[k] = v
        elif k in localParams and v is None:
            del localParams[k]
    debug = localParams.get("debug", False)
    for k in filter(apiParams.has, ['grammar', 'json_schema']):
        if (val := apiParams.pop(k)) is not None:
            apiParams["response_format"] = {"type": k, k: val}
    if localParams.get("mode") == "chat" and 'prompt' in apiParams:
        prompt = apiParams.pop('prompt')
        apiParams['messages'] = apiParams.get('messages') or prompt
        if not isinstance(apiParams.get("messages"), list):
            apiParams["messages"] = [{"role": "user", "content": apiParams["messages"]}]
    if localParams.get("mode") == "text" and 'messages' in apiParams:
        messages = apiParams.pop('messages')
        apiParams['prompt'] = apiParams.get('prompt') or messages
    if 'suffix' in apiParams and not apiParams.get('suffix'):
        del apiParams['suffix']
    ap = apiParams.get("top_logprobs", None)
    max_logprobs: int = 5
    # if type(ap) == int and ap > 5, we can do fun stuff
    # but if type(ap) == NoneType, we can't even do a > comparison
    if isinstance(ap, int) and ap > max_logprobs:
        apiParams["top_logprobs"] = max_logprobs
    selections = resolve(apiParams.get("model"), localParams.get("mode"))
    if len(selections) != 1:
        if localParams.get("provider"):
            provided = filter(lambda x: x.startswith(localParams["provider"]), selections)
            if len(provided) >= 1:
                selections = provided
        if len(selections) > 1:
            raise Exception('No unique ' + localParams.get("mode") + ' model found for "' + apiParams.get("model", "") + '". (Possible: ' + ", ".join([separator.join(x) for x in selections]) + ")")
        if len(selections) == 0:
            raise Exception('No ' + localParams.get("mode") + ' model found for "' + apiParams.get("model", "") + '".')
    provider, apiParams["model"] = selections[0]
    model = apiParams["model"]
    effect, callback = localParams.get(["effect", "callback"])
    if not effect:
        effect = id
    print_output = localParams.get("print_output", False)
    yield_output = localParams.get("yield_output", False)
    return_output = localParams.get("return_output", False)
    return_raw = localParams.get("return_raw", False)

    # now we purge the sensitive (the ones in options) Nones from our dicts
    apiParams = apiParams.filter(lambda _, v: v is not None)
    localParams = localParams.filter(lambda _, v: v is not None)
    if force_model := localParams.get("force_model"):
        apiParams["model"] = force_model
    if force_provider := localParams.get("force_provider"):
        provider = force_provider
    if debug:
        print(apiParams, localParams, provider)
    base = clients[provider].completions.create if localParams.get('mode') == 'text' else clients[provider].chat.completions.create
    if localParams.get("mode") == "text":
        retrieve = lambda y: y.choices[0].text
    elif apiParams.get("stream"):
        retrieve = lambda y: y.choices[0].delta.content
    else:
        retrieve = lambda y: y.choices[0].message.content
    unpack = lambda y: effect(retrieve(y))
    if apiParams.get("stream"):
        if yield_output:
            def gen() -> Optional[str]:
                response = base(**apiParams)
                if localParams.get('return_object'):
                    return response
                for obj in response:
                    text = unpack(obj) or ''
                    if callback:
                        callback(text)
                    yield text
            if localParams.get('return_object'):
                return gen()
            if not return_output:
                for i in gen():
                    print(i, end='')
                return None
            return gen()
        result = ""
        response = base(**apiParams)
        for obj in response:
            text = unpack(obj) or ''
            result += text
            if print_output and text:
                print(text, end="")
            if callback:
                callback(text)
    else:
        obj = base(**apiParams)
        if localParams.get('return_object'):
            return obj
        result = unpack(obj)
        if print_output and result:
            print(result)
        if callback:
            callback(result)
        if obj.choices[0].finish_reason == 'tool_calls' and localParams.get("pretty_tool_calls", False):
            def pretty(s: Any) -> str:
                return str(s) if not isinstance(s, str) else f'"{s}"'
            for call in obj.choices[0].message.tool_calls:
                line = f'\ntool_call(id={pretty(call.id)}): {call.function.name}({", ".join([k + "=" + pretty(v) for k, v in json_loads(call.function.arguments).items()])})'
                result += line
                if print_output and line:
                    print(line, end='')
                if callback:
                    callback(result)
        if return_output and not return_raw:
            if debug:
                print("Returning result of length " + str(len(result)))
            return result
        if return_raw:
            return obj
    return None

def text_completion(prompt: str, options: Dict) -> Any:
    """
    Generates a text completion based on the provided prompt and arguments.
    :param prompt: Prompts to be provided to the chat completion model.
    :param options: Extra arguments to be provided.
    :returns: The generated text completion.
    """
    params = {"mode": "text", "prompt": prompt, "model": "gpt-3.5-turbo-instruct", "temperature": 1.0, "top_p": 1.0, "max_tokens": 1024, "stream": True, "yield_output": True, "return_output": not options.get('stream', False), "print_output": options.get("stream", False)} | options
    if params.get('messages') is not None:
        options['prompt'] = options.get('prompt') | str(params.get('messages'))
        del params['messages']
    # if options.get('messages')!=None:
    # 	options['prompt'] = options.get('prompt') | str(options.get('messages'))
    # 	del options['messages']

    response = completion(params)
    if response is None:
        return None
    if params.get('stream'):
        if params.get('yield_output'):
            return response  # Return the generator directly
        # Consume the generator safely
        result = list(response)
        if result and params.get('return_output'):
            return ''.join(result)
        return None
    return response

def chat_completion(messages: str | List[Dict[str, str]], options: Dict) -> Any:
    """
    Generates a chat completion based on the provided messages and arguments.
    :param messages: Message log to be provided to the chat completion model.
    :param options: Extra arguments to be provided.
    :returns: The generated chat completion.
    """
    params = {"mode": "chat", "messages": messages, "model": "openrouter" + separator + "gpt-4o", "temperature": 1.0, "top_p": 1.0, "max_tokens": 4096, "yield_output": True, "return_output": True} | options
    # "tools": None,"tool_choice":None,"parallel_tool_calls":None,"response_format":None
    if 'prompt' in params:
        del params['prompt']
    response = completion(params)
    if params.get('yield_output') and params.get('stream'):
        return response  # Return the generator directly
    # Consume the generator safely
    result = list(response)
    if result and params.get('return_output'):
        return ''.join(result)
    return None

def tools_object_from_string(tool_string: str) -> list[dict[str, Any]]:
    """
    Converts a string representation of tools into a list of dictionaries.
    :param tool_string: A string where each tool's details are separated by newlines, and nested details are indented with tabs.
    :returns: A list of dictionaries, each representing a tool's details parsed from the input string.
    """
    tools, current = [], []
    for i in tool_string.strip().split('\n'):
        if i.startswith('\t'):
            current.append(i)
        else:
            if (s := '\n'.join(current)):
                tools.append(schema_from_string(s))
            current = [i]
    if (s := '\n'.join(current)):
        tools.append(schema_from_string(s))
    return tools

class Role(Enum):
    """Enum class to represent different roles in the conversation."""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    FUNCTION = 'function'
    TOOL = 'tool'

# Usage example:
# async def main():
# 	logging.basicConfig(level=logging.INFO)

# 	provider, model = 'deepseek-beta', 'deepseek-coder'
# 	client = clients[provider]  # Assuming clients is defined elsewhere
# 	model_instance = ModelInstance(client, provider, model)
# 	history = History()

# 	await history.add_message("I want to build a unified, compositional framework for building self-organizing systems of AI agents; so far, all I've done is implement some ground-level functionalities as I figure out how to work with types and APIs and so on. Here's what I have:", Role.SYSTEM, model_instance)
# 	await history.add_message("It's obviously not very high-quality now, but I think it can be much better.", Role.USER, model_instance)
# 	await history.add_message("Can you work out an integrated plan for improving the codebase in these ways? For now, let's just start with error handling, logging, and documentation.", Role.USER, model_instance)

# 	response_message = await history.generate_response()
# 	print(response_message)

class Entity:
    """General utilities for networkable entities"""
    def __init__(self) -> None:
        self.id = gen_pseudoword(6).capitalize()

    def serialize(self) -> str:
        data = {}
        # ...
        return json_dumps(data)

    def deserialize(self, data: str) -> Entity:
        """
        Fills in an Entity instance from a JSON string.
        """
        data = json_loads(data)
        # ...
        return self

class Node(Entity):
    """Base class for network nodes."""
    def __init__(self,
        parents: Optional[List[Node]] = None,
        children: Optional[List[Node]] = None,
        callback: Optional[Callable] = None,
        **data: Any,
    ) -> None:
        self.parents: List[Node] = parents or []
        self.children: List[Node] = children or []
        self.data: Dict[str, Any] = data
        self.depth: int = self.get_depth()
        self.callback: Optional[Callable] = callback

    def get_depth(self) -> int:
        """Calculate the depth of the node in the network."""
        if not self.parents:
            return 0
        return min(parent.depth for parent in self.parents) + 1

    def add_parent(self, parent: Node) -> None:
        """Add a parent node to this node."""
        self.parents.append(parent)
        parent.children.append(self)

    def add_child(self, child: Node) -> None:
        """Add a child node to this node."""
        self.children.append(child)
        child.parents.append(self)

    @log_and_callback
    async def spawn(self) -> Node:
        """Create a new child node."""
        raise NotImplementedError("Spawn method must be implemented by subclasses")

    @log_and_callback
    async def update(self) -> None:
        """Update the node's content or state."""
        raise NotImplementedError("Update method must be implemented by subclasses")

class Relation(Node, Entity):
    """Represents a relation between nodes."""
    # To be implemented later
    pass

class Message(Node, Entity):
    """Represents a message in the conversation."""
    def __init__(self,
        content: str,
        role: Role,
        created_by: Optional[ModelInstance] = None,
        callback: Optional[Callable] = None,
        **data: Any,
    ) -> None:
        super(Node, self).__init__(callback=callback, **data)
        self.content: str = content
        self.role: Role = role
        self.created_by: Optional[ModelInstance] = created_by

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @log_and_callback
    async def spawn(self) -> Message:
        """Create a new message as a response to this message."""
        new_message = Message(content="", role=Role.ASSISTANT, created_by=self.created_by, callback=self.callback)
        self.add_child(new_message)
        return new_message

    @log_and_callback
    async def update(self) -> None:
        """Update the message content by querying the AI model."""
        if not self.created_by:
            raise ValueError("Message has no associated ModelInstance")

        messages = self._construct_message_chain()
        response = await self.created_by.client.chat.completions.create(
            model=self.created_by.model,
            messages=messages,
            max_tokens=8192,
            temperature=0.5,
        )
        r = regularize(response)  # Assuming regularize is defined elsewhere
        self.content = r.choices[0].message.content
        self.data['id'] = r.id
        self.data['usage'] = {
            'input': r.usage.prompt_tokens,
            'output': r.usage.completion_tokens,
            'cached_in': r.usage.get('prompt_cache_hit_tokens', 0),
        }
        self.data['refusal'] = r.choices[0].message.get('refusal', None)
        self.data['tool_calls'] = r.choices[0].message.get('tool_calls', None)
        self.data['datetime'] = datetime_datetime.utcfromtimestamp(r.created).strftime('%Y-%m-%dT%H:%M:%SZ')

        provider, model = self.created_by.provider, self.created_by.model
        self.data['usage']['cost'] = round(sum(
            self.data['usage'][k] * model_data[provider][model]['token_costs'][k]
            for k in self.data['usage']
        ), 10)

    def _construct_message_chain(self) -> List[Dict[str, str]]:
        """Construct the chain of messages for the conversation context."""
        messages = []
        current: Optional[Message] = self
        while current:
            messages.insert(0, {"role": current.role.value, "content": current.content})
            current = current.parents[0] if current.parents else None
        return messages

class Network(Entity):
    """Represents a network of nodes."""
    def __init__(self, root: Optional[Node] = None) -> None:
        self.root: Optional[Node] = root

    async def add_node(self, node: Node) -> Node:
        """Add a new node to the network."""
        if not self.root:
            self.root = node
            return self.root

        last_node = self._get_last_node()
        last_node.add_child(node)
        return self

    def _get_last_node(self) -> Node:
        """Get the last node in the network."""
        if not self.root:
            raise ValueError("Network is empty")
        current = self.root
        while current.children:
            current = current.children[-1]
        return current

class History(Network, Entity):
    """Represents a conversation history as a specialized network of messages."""
    async def add_message(self, content: str, role: Role, created_by: Optional[ModelInstance] = None, callback: Optional[Callable] = None) -> Message:
        """Add a new message to the conversation history."""
        new_message = Message(content, role, created_by, callback)
        return await self.add_node(new_message)

    async def generate_response(self) -> Message:
        """Generate a response using the AI model."""
        last_message = self._get_last_node()
        if not isinstance(last_message, Message):
            raise TypeError("Last node is not a Message")
        new_message = await last_message.spawn()
        await new_message.update()
        return new_message

class ModelInstance(Node, Entity):
    """Represents an instance of an AI model."""
    def __init__(self, client: Any, provider: str, model: str, callback: Optional[Callable] = None) -> None:
        super().__init__(callback=callback)
        self.client: Any = client
        self.provider: str = provider
        self.model: str = model
        self.token_costs: Dict[str, float] = model_data[provider][model]['token_costs']

    @log_and_callback
    async def spawn(self) -> ModelInstance:
        """Create a new instance of the model."""
        return ModelInstance(self.client, self.provider, self.model, self.callback)

    @log_and_callback
    async def update(self) -> None:
        """Update the model instance (placeholder for future implementation)."""
        pass

class StatefulChat(History, Entity):
    """Represents, and provides utilities for, stateful chat conversations."""
    def __init__(self, **kwargs) -> None:
        self.message_history: list = []
        self.tools: list = []
        self.debug: bool = False
        self.first_run: bool = True
        self.apiParams: dict = dict.fromkeys(['model', 'suffix', 'max_tokens', 'stream', 'n', 'logprobs', 'top_logprobs', 'logit_bias', 'temperature', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'top_p', 'min_p', 'top_k', 'top_a', 'tools', 'tool_choice', 'parallel_tool_calls', 'grammar', 'json_schema', 'response_format', 'seed'], None)
        self.localParams = {"mode": "chat", "return_raw": True, "pretty_tool_calls": False} | dict.fromkeys(['provider', 'force_model', 'force_provider', 'effect', 'callback', 'print_output', 'yield_output', 'return_output', 'debug', 'return_object'], None)
        self.stateParams: dict = {}
        for (k, v) in kwargs.items():
            if k in self.apiParams:
                if k == 'tools':
                    self.apiParams['tools'] = []
                    if isinstance(v, Toolbox):
                        v = v.tools
                    for i in v:
                        if isinstance(i, Tool):
                            self.tools.append(i)
                            self.apiParams['tools'].append(i.schema)
                        elif callable(i):
                            try:
                                self.tools.append(Tool(i))
                                self.apiParams['tools'].append(self.tools[-1].schema)
                            except Exception:
                                print('Warning: could not instantiate tool from callable ' + i.__name__ + '  (' + str(i) + '), discarding')
                        elif type(i) in {dict, Dict}:
                            self.apiParams['tools'].append(i)
                        else:
                            print('Warning: Could not recognize object ' + str(i) + ' of type ' + str(type(i)) + ' as a tool, discarding')
                    if self.apiParams['tools'] == []:
                        self.apiParams['tools'] = None
                    if self.tools == []:
                        self.tools = None
                else:
                    self.apiParams[k] = v
            elif k in self.localParams:
                if k == "debug":
                    self.debug = v
                else:
                    self.localParams[k] = v
            else:
                self.stateParams[k] = v

    def __getitem__(self, key) -> Any:
        if key in self.apiParams:
            return self.apiParams[key]
        if key in self.localParams:
            return self.localParams[key]
        return self.stateParams.get(key, None)

    def __setitem__(self, key, value) -> None:
        if key in self.apiParams:
            self.apiParams[key] = value
        elif key in self.localParams:
            self.localParams[key] = value
        else:
            self.stateParams[key] = value

    def set_state(self, key, value) -> StatefulChat:
        self.stateParams[key] = value
        return self

    def get_state(self, key, default=None) -> Any:
        return self.stateParams.get(key, default)

    @property
    def first(self) -> Any:
        if self.message_history == []:
            return None
        return self.message_history[0]

    @property
    def last(self) -> Any:
        if self.message_history == []:
            return None
        return self.message_history[-1]

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self.message_history

    def undo(self, n=1) -> StatefulChat:
        for _ in range(n):
            if len(self.message_history) > 0:
                self.message_history.pop()
        return self

    def redo(self) -> StatefulChat:
        pass

    def save(self, path: str) -> StatefulChat:
        with open(path, 'w') as f:
            json_dump({
                    'message_history': self.message_history,
                    'apiParams': self.apiParams,
                    'localParams': self.localParams,
                    'stateParams': self.stateParams,
                }, f)
        return self

    def save_string(self) -> str:
        return json_dumps({
                'message_history': self.message_history,
                'apiParams': self.apiParams,
                'localParams': self.localParams,
                'stateParams': self.stateParams,
            })

    def load(self, path: str) -> StatefulChat:
        with open(path) as f:
            data = json_load(f)
            self.message_history = data['message_history']
            self.apiParams = data['apiParams']
            self.localParams = data['localParams']
            self.stateParams = data['stateParams']
            return self

    def load_string(self, s: str) -> StatefulChat:
        data = json_loads(s)
        self.message_history = data['message_history']
        self.apiParams = data['apiParams']
        self.localParams = data['localParams']
        self.stateParams = data['stateParams']
        return self

    def clone(self, **kwargs) -> StatefulChat:
        new = StatefulChat()
        new.message_history = self.message_history.copy()
        new.apiParams = self.apiParams.copy()
        new.localParams = self.localParams.copy()
        new.stateParams = self.stateParams.copy()
        new.tools = self.tools.copy()
        for k, v in kwargs.items():
            new[k] = v
        return new

    def addMessage(self, role: str, content: str, **kwargs) -> StatefulChat:
        self.message_history.append({'role': role, 'content': content} | kwargs)
        return self

    def system(self, content: str, echo: bool = False) -> StatefulChat:
        if self.get_state("echo") or echo:
            print('<system>', content)
        return self.addMessage('system', content)

    def assistant(self, content: str, echo: bool = False) -> StatefulChat:
        if self.get_state("echo") or echo:
            print('<assistant>', content)
        return self.addMessage(
            'assistant', content)  # can add prefix = True for some models

    def user(self, content: str, echo: bool = False) -> StatefulChat:
        if self.get_state("echo") or echo:
            print('<user>', content)
        return self.addMessage('user', content)

    def tool(self, content: str, name: str, tool_call_id: str) -> StatefulChat:
        return self.addMessage('tool', content, name=name, tool_call_id=tool_call_id)

    def next(self, **kwargs) -> StatefulChat:
        if self.first_run:
            if self.debug:
                for message in self.message_history:
                    print(f'{message["role"]}: {message["content"]}')
            self.first_run = False
        # if self.apiParams.get('tools'):
        # 	tools = []
        response = completion(
            {"messages": self.message_history} |
            (self.apiParams | self.localParams | {"return_raw": True} | kwargs),
        )
        if kwargs.get('return_response') or kwargs.get('return_object'):
            return response
        response = regularize(response)
        content, usage, message = response['choices'][0], response[
            'usage'], response['choices'][0]['message']
        output = usage[{
            'completion_tokens': 'output_tokens',
            'prompt_tokens': 'input_tokens',
        }]
        output |= content[{
            'finish_reason': 'finish_reason',
            'logprobs': 'logprobs',
        }]
        output |= message[{
            'text': 'content',
            'role': 'role',
            'tool_calls': 'tool_calls',
            'function_call': 'function_call',
        }]
        if message['content'] and message['role'] and self.debug:
            print(f'{message["role"]}: {message["content"]}')
        self.message_history.append(message)
        if output['finish_reason'] == 'tool_calls':
            called = {
                y['id']: {
                    'name': y['function']['name'],
                    'arguments': json_loads(y['function']['arguments']),
                }
                for y in output['tool_calls']
            }
            for (k, v) in called.items():
                if self.localParams.get("print_output"):
                    print(f'Calling {v["name"]} with arguments {v["arguments"]}')
                called_tool = None
                if self.debug:
                    print("\t(calling " + v["name"] + " on arguments " +
                            str(v["arguments"]) + ')')
                for i in self.tools:
                    if (isinstance(i, Tool) and i.schema['function']['name'] == v['name']) or (isinstance(i, dict) and i['function']['name'] == v['name']):
                        called_tool = i
                        break
                if not isinstance(called_tool, Tool):
                    if self.debug:
                        print('\t(called tool has no implementation; exiting)')
                    return {"name": v["name"], "arguments": v["arguments"]}
                if ([*list(inspect_signature(called_tool.func).parameters), []])[0] == 'self':
                    tool_output = called_tool(self, **v['arguments'])
                else:
                    tool_output = called_tool(**v['arguments'])
                if self.debug:
                    print('\t(output: ' + str(tool_output) + ')')
                self.message_history.append({
                    'role': 'tool',
                    'tool_call_id': k,
                    'name': v['name'],
                    'content': json_dumps(tool_output),
                })
            if self.get_state("yield_control"):
                self.set_state("yield_control", False)
                return self
            return self.next()
        return self

    def run(self, **kwargs) -> None:
        self.set_state('loop', True)
        cycles = 0
        while self.get_state('loop') and (kwargs.get('max_cycles', 0) == 0 or cycles < kwargs.get('max_cycles')):
            self.next(**kwargs)
            cycles += 1
