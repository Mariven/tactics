r"""
Contains:
    class Role(Enum)

    class Entity
        serialize           (self) -> str
        deserialize         (self, data: str) -> Entity

    class Node(Entity)
        get_depth           (self) -> int
        add_parent          (self, parent: Node) -> None
        add_child           (self, child: Node) -> None
        spawn               (self) -> Node
        update              (self) -> None

    class Relation(Node, Entity)
        spawn               (self) -> Message
        update              (self) -> None
        _construct_message_chain     (self) -> List[Dict[str, str]]

    class Message(Node, Entity)
        add_node            (self, node: Node) -> Node
        _get_last_node      (self) -> Node

    class Network(Entity)
        add_message         (self, content: str, role: Role, created_by: Optional[ModelInstance], callback: Optional[Callable]) -> Message
        generate_response   (self) -> Message

    class ModelInstance(Node, Entity)
        spawn               (self) -> ModelInstance
        update              (self) -> None

    class StatefulChat(History, Entity)
        set_state           (self, key, value) -> StatefulChat
        get_state           (self, key, default=None) -> Any
        first               (self) -> Any
        last                (self) -> Any
        messages            (self) -> List[Dict[str, str]]
        undo                (self, n: int) -> StatefulChat
        redo                (self) -> StatefulChat
        save                (self, path: str) -> StatefulChat
        save_string         (self) -> str
        load                (self, path: str) -> StatefulChat
        load_string         (self, s: str) -> StatefulChat
        clone               (self, **kwargs) -> StatefulChat
        addMessage          (self, role: str, content: str, **kwargs) -> StatefulChat
        system              (self, content: str, echo: bool) -> StatefulChat
        assistant           (self, content: str, echo: bool) -> StatefulChat
        user                (self, content: str, echo: bool) -> StatefulChat
        next                (self, **kwargs) -> StatefulChat
        run                 (self, **kwargs) -> None
"""
from __future__ import annotations
from .completion import *

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
        return json.dumps(data)

    def deserialize(self, data: str) -> Entity:
        """
        Fills in an Entity instance from a JSON string.
        """
        data = json.loads(data)
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
        self.data['datetime'] = datetime.utcfromtimestamp(r.created).strftime('%Y-%m-%dT%H:%M:%SZ')

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
        return Dict(self.message_history[0])

    @property
    def last(self) -> Any:
        if self.message_history == []:
            return None
        return Dict(self.message_history[-1])

    @property
    def messages(self) -> List[Dict[str, str]]:
        return List(self.message_history)

    def undo(self, n: int = 1) -> StatefulChat:
        for _ in range(n):
            if len(self.message_history) > 0:
                self.message_history.pop()
        return self

    def redo(self) -> StatefulChat:
        pass

    def save(self, path: str) -> StatefulChat:
        with open(path, 'w') as f:
            json.dump({
                    'message_history': self.message_history,
                    'apiParams': self.apiParams,
                    'localParams': self.localParams,
                    'stateParams': self.stateParams,
                }, f)
        return self

    def save_string(self) -> str:
        return json.dumps({
                'message_history': self.message_history,
                'apiParams': self.apiParams,
                'localParams': self.localParams,
                'stateParams': self.stateParams,
            })

    def load(self, path: str) -> StatefulChat:
        with open(path) as f:
            data = json.load(f)
        self.message_history = data['message_history']
        self.apiParams = data['apiParams']
        self.localParams = data['localParams']
        self.stateParams = data['stateParams']
        return self

    def load_string(self, s: str) -> StatefulChat:
        data = json.loads(s)
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
                    'arguments': json.loads(y['function']['arguments']),
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
                if ([*list(inspect.signature(called_tool.func).parameters), []])[0] == 'self':
                    tool_output = called_tool(self, **v['arguments'])
                else:
                    tool_output = called_tool(**v['arguments'])
                if self.debug:
                    print('\t(output: ' + str(tool_output) + ')')
                if not kwargs.get("test"):
                    self.message_history.append({
                        'role': 'tool',
                        'tool_call_id': k,
                        'name': v['name'],
                        'content': json.dumps(tool_output),
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
