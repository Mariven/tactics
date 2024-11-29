r"""
Abstractions and frameworks for constructing text transformers.

Contains:
    pipe_factory
        (options_global: Options = None) -> Callable[[str, List[Tuple[str, str]], End[str], Options], Pipe]
    load_pipe
        (path: str) -> Pipe
"""

from .completion import *
from requests import get as requests_get
Options = Optional[Object]

def pipe_factory(
        options_global: Options = None
    ) -> Callable[[str, List[Tuple[str, str]], End[str], Options], Pipe]:
    """
    Creates a function that pipes an n-shot prompt (prompt + list of examples) to the chat_completion function.
    :param options_global: The global options to be used in the chat_completion function.
    :returns: A function that takes a prompt, examples, options, and an expander, and returns a function that takes a query and options and returns a response.
    """
    def pipe_constructor(
            prompt: str,
            examples: List[Tuple[str, str]],
            expander: End[str] = lambda x: x,
            options_local: Options = None
    ) -> Pipe:
        """
        Pre-fills the chat_completion function with a prompt and examples, creating a string transformer (pipe).
        :param prompt: The pipe's system prompt.
        :param examples: A list of examples to follow, formatted as user and assistant messages.
        :param options: The options to be sent to the chat_completion function.
        :param expander: A function that expands pipe inputs before transforming them. Useful in conjunction with the distribute decorator.
        :returns: A function that takes a query and optional options and returns a response.
        """
        messages = [{"role": "system", "content": prompt}]
        for (query, answer) in examples:
            messages.extend((
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ))

        @freeze_args(view=lambda _: None)(show_call)
        @distribute(after=lambda **obj: (obj["order"], obj["value"]), threads=25)
        def pipe(
                query: str,
                options_inst: Options = None
        ) -> str:
            """
            Transforms a query using the pre-filled prompt and examples.
            :param query: The query to be transformed.
            :param options_ex: The options to be sent to the chat_completion function.
            :returns: The transformed query.
            """
            options_base = {"stream": False, "mode": "chat"}
            return chat_completion(
                [*messages, {"role": "user", "content": expander(query)}],
                options_base | (options_global or {}) | (options_local or {}) | (options_inst or {})
            )
        return pipe
    return pipe_constructor

def load_pipe(path: str) -> Pipe:
    """
    Loads a pipe from a JSON file.
    :param path: The path to the JSON file.
    :returns: The pipe function.
    """
    with open(path) as file:
        data = Dict(json.load(file))
    if "options" not in data or not data.get("options") or not data.options.get("model"):
        raise ValueError("The JSON file must contain a non-empty 'options' object with a 'model' key.")
    if "prompt" not in data or not data.get("prompt"):
        raise ValueError("The JSON file must contain a non-empty 'prompt' key.")
    if "examples" not in data or not data.get("examples") or not data.examples.get("training"):
        data.examples = {"training": []}
    options, prompt, training_dict = data.options, data.prompt, data.examples.training
    training = [(pair.input, pair.output) for pair in training_dict]
    return pipe_factory(options)(prompt, training)
