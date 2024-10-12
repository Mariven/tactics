r"""
Abstractions and frameworks for constructing text transformers.

With:
    T, X, Y generic
    F         := Callable[..., Any]
    End       := Callable[[T], T]
    Object    := Dict[str, Any]
    Decorator := End[F]
    Pipe      := Callable[[str, Optional[Object]], str]
Contains:
    pipe_factory
        (options_global: Optional[Object] = None) -> Callable[[str, List[Tuple[str, str]], End[str], Optional[Object]], Pipe]
    load_pipe
        (path: str) -> Pipe
    make_lines
        (text: str, row_len: int, separators: List[str], newlines: List[str]) -> List[Tuple[int, str]]
"""

from .structure import *
from requests import get as requests_get


def pipe_factory(
        options_global: Optional[Object] = None
    ) -> Callable[[str, List[Tuple[str, str]], End[str], Optional[Object]], Pipe]:
    """
    Creates a function that pipes an n-shot prompt (prompt + list of examples) to the chat_completion function.
    :param options_global: The global options to be used in the chat_completion function.
    :return: A function that takes a prompt, examples, options, and an expander, and returns a function that takes a query and options and returns a response.
    """
    def pipe_constructor(
            prompt: str,
            examples: List[Tuple[str, str]],
            expander: End[str] = lambda x: x,
            options_local: Optional[Object] = None
    ) -> Pipe:
        """
        Pre-fills the chat_completion function with a prompt and examples, creating a string transformer (pipe).
        :param prompt: The pipe's system prompt.
        :param examples: A list of examples to follow, formatted as user and assistant messages.
        :param options: The options to be sent to the chat_completion function.
        :param expander: A function that expands pipe inputs before transforming them. Useful in conjunction with the distribute decorator.
        :return: A function that takes a query and optional options and returns a response.
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
                options_inst: Optional[Object] = None
        ) -> str:
            """
            Transforms a query using the pre-filled prompt and examples.
            :param query: The query to be transformed.
            :param options_ex: The options to be sent to the chat_completion function.
            :return: The transformed query.
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
    :return: The pipe function.
    """
    with open(path) as file:
        data = Dict(json_load(file))
    if "options" not in data or not data.get("options") or not data.options.get("model"):
        raise ValueError("The JSON file must contain a non-empty 'options' object with a 'model' key.")
    if "prompt" not in data or not data.get("prompt"):
        raise ValueError("The JSON file must contain a non-empty 'prompt' key.")
    if "examples" not in data or not data.get("examples") or not data.examples.get("training"):
        data.examples = {"training": []}
    options, prompt, training_dict = data.options, data.prompt, data.examples.training
    training = [(pair.input, pair.output) for pair in training_dict]
    return pipe_factory(options)(prompt, training)

def make_lines(text: str, row_len: int = 80, separators: List[str] = [" "], newlines: List[str] = ["\n"]) -> List[Tuple[int, str]]:
    """
    Parses text into lines, respecting row length and separators.
    Args:
      text (str): The text to be parsed.
      row_len (int): The maximum length of each line.
      separators (list): A list of characters considered separators.
      newlines (list): A list of characters that trigger a new line.
    Returns:
      list: A list of lines, where each line is a list containing its
              1-based index and the line content as a string.
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
