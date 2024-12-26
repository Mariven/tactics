"""
Methods for text and chat completions

Contains:
    resolve            (id: str, mode: str) -> list[tuple[str, str]]
    completion         (options: dict) -> Any
    text_completion    (prompt: str, options: dict) -> Any
    chat_completion    (messages: str | list[dict[str, str]], options: dict) -> Any

"""

from __future__ import annotations

from src.basetypes import *  # re, typing
from src.supertypes import *  # builtins, functools, inspect, itertools, operator, logging
from src.utilities import *  # datetime, json, os, sqlite3, time, types, random, requests
from src.tools import *  # ast

from tiktoken import get_encoding

separator = "::"

# data/secrets.json format:
# { secrets : [{ id : "openai", name : "OpenAI API Key", value : "sk-abcd..." }] }
with open("data/secrets.json", encoding="utf-8") as f:
    secrets_table = Dict(json.load(f))

keys = Dict({item.id: item.value for item in secrets_table.secrets})

# { providers : [{ id : "openai", api: {url: "https://..."}, "models": [ {"id": "gpt-4o", "mode": "chat", ..., "parameters": [...]}, ...] }, ...] }
with open("data/providers.json", encoding="utf-8") as f:
    provider_table = Dict(json.load(f))

# keys = {k: v['api']['key'] for k, v in provider_table.items()}
clients: Dict[str, OpenAI] = Dict()
for provider in provider_table.providers:
    p_key_id = provider.api.get("key-secret")
    provider.enabled = False

    if p_key_id and p_key_id in keys and keys[p_key_id]:
        provider.api.key = keys[p_key_id]
        clients[provider.id] = OpenAI(
            api_key=provider.api.key,
            base_url=provider.api.url
        )
        provider.enabled = True

models_by_mode = Dict({"text": {}, "chat": {}})
models = List()

for provider in provider_table.providers:
    if not provider.enabled:
        continue
    prov_name = provider.id
    vars()[provider.id.replace("-", "_")] = clients[provider.id]
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


def resolve(id: str, mode: str = "") -> list[tuple[str, str]]:
    """
    Resolves a model ID to a list of tuples containing provider and model names based on the specified mode.
    :param id: The model ID to resolve, which can include a provider prefix.
    :param mode: The mode in which to resolve the model ID, either 'text' or 'chat'. Defaults to an empty string, which triggers a combined resolution.
    :returns: A list of tuples where each tuple contains a provider name and a model name.
    """
    id = sub("^[\"' ]*(.*?)['\" ]*$", r"\1", id)
    if not mode:
        combined = resolve(id, "chat") + resolve(id, "text")
        return list(set(combined))
    # Define the providers for each mode
    parts = id.lower().replace(" ", "").split(separator)
    specified_provider = parts[0] if len(parts) > 1 else ""
    specified_model = parts[-1]
    relevant_providers = filter(
        lambda p: fuzzy_in(specified_provider, p),
        [models_by_mode[mode][m]["provider"] for m in models_by_mode[mode]],
    )
    # relevant_providers = [p for p in mode_providers.get(mode,[]) if fuzzy_in(specified_provider, p)]
    matches = List()
    for provider in list(set(relevant_providers)):
        for model in query(provider_table.providers, "id", provider).models:
            if (
                fuzzy_in(specified_model, model.id.lower())
                and (provider, model.id) not in matches
                and separator.join([provider, model.id]) in models_by_mode[mode]
            ):
                matches.append((provider, model.id))
    matches.sort(key=lambda x: len(x[1]))
    if len(matches) > 1:
        if all(
            fuzzy_in(matches[0][1].split("/")[-1].lower(), m[1].split("/")[-1].lower())
            for m in matches[1:]
        ):
            matches = [matches[0]]
        # elif "openrouter" in [x[0] for x in matches]:
        # 	# is openrouter adding a bunch of extra matches
        # 	matches2 = [x for x in matches if x[0] != "openrouter"]
        # 	if len(matches2) > 0 and all([fuzzy_in(matches2[0][1].split("/")[-1].lower(), m[1].split("/")[-1].lower()) for m in matches2[1:]]):
        # 		matches = [matches2[0]]
    return matches


def completion(options: dict) -> Any:
    """
    Generates a text or chat completion from arguments.
    :param options: Arguments to the completion model.
    :returns: The generated completion.
    """
    api_params_list = [
        "model",  #: str
        # The model identifier to use for generating the completion.
        "prompt",  #: str
        # The initial text or prompt to generate a completion from.
        "messages",  #: str | list[dict[str, str]]
        # The list of messages, or content of a single user message, to be passed to the API.
        "suffix",  #: str
        # A suffix to append to the generated text.
        "max_tokens",  #: int
        # The maximum number of tokens to generate.
        "stream",  #: bool
        # Whether to stream the output tokens as they are generated.
        "n",  #: int
        # The number of completions to generate.
        "logprobs",  #: int
        # The number of top log probabilities to return.
        "top_logprobs",  #: int
        # The number of top tokens to consider for log probabilities.
        "logit_bias",  #: dict[str, int]
        # A dictionary mapping token IDs to bias values for logits.
        "temperature",  #: float
        # The sampling temperature to use for randomness in generation.
        "presence_penalty",  #: float
        # A penalty applied for presence of certain tokens.
        "frequency_penalty",  #: float
        # A penalty applied for frequency of certain tokens.
        "repetition_penalty",  #: float
        # A penalty applied to repeated tokens.
        "top_p",  #: float
        # The cumulative probability threshold for nucleus sampling.
        "min_p",  #: float
        # The minimum probability threshold for sampling.
        "top_k",  #: int
        # The number of top tokens to consider for sampling.
        "top_a",  #: float
        # The alpha parameter for top-a sampling.
        "tools",  #: list[str]
        # A list of tools available for use in the generation.
        "tool_choice",  #: str
        # The choice of tool to use for generation.
        "parallel_tool_calls",  #: bool
        # Whether to allow parallel execution of tool calls.
        "grammar",  #: str
        # A grammar specification for the generated text.
        "json_schema",  #: dict
        # A JSON schema to validate the structure of the response.
        "response_format",  #: dict
        # The format in which the response should be returned.
        "seed",  #: int
        # A seed value for random number generation in the model.
    ]
    local_params_list = [
        "mode",  #: str
        # The mode of operation, either 'text' or 'chat'.
        "provider",  #: str
        # The provider of the model.
        "effect",  #: Callable
        # A function to apply to the generated text.
        "callback",  #: Callable
        # A callback function to execute with the generated text.
        "print_output",  #: bool
        # Whether to print the output to the console.
        "yield_output",  #: bool
        # Whether to yield the output as a generator.
        "return_output",  #: bool
        # Whether to return the generated output.
        "debug",  #: bool
        # Whether to print debug information.
        "force_model",  #: str
        # A model identifier to forcefully use, overriding other selections.
        "force_provider",  #: str
        # A provider to forcefully use, overriding other selections.
        "pretty_tool_calls",  #: bool
        # Whether to format tool calls in a human-readable way.
        "return_object",  #: bool
        # Whether to return the full response object instead of just the text.
        "use_parse",  #: bool
        # Whether to use the parse endpoint for chat completions.
    ]
    api_params = Dict.fromkeys(api_params_list, None)
    local_params = Dict.fromkeys(local_params_list, None)
    api_params = (api_params & Dict(options).strip()).strip()
    local_params = (local_params & Dict(options).strip()).strip()
    debug = local_params.get("debug", False)

    if local_params.get("mode") == "chat" and "prompt" in api_params:
        prompt = api_params.pop("prompt")
        api_params["messages"] = api_params.get("messages") or prompt
        if not isinstance(api_params.get("messages"), list):
            api_params["messages"] = [
                {"role": "user", "content": api_params["messages"]}
            ]
    if local_params.get("mode") == "text" and "messages" in api_params:
        messages = api_params.pop("messages")
        api_params["prompt"] = api_params.get("prompt") or messages

    for k in filter(api_params.has, ["grammar", "json_schema"]):
        if (val := api_params.pop(k)) is not None:
            api_params["response_format"] = {"type": k, k: val}
    if "suffix" in api_params and not api_params.get("suffix"):
        del api_params["suffix"]

    ap = api_params.get("top_logprobs", None)
    max_logprobs: int = 5
    # if type(ap) == int and ap > 5, we can do fun stuff
    # but if type(ap) == NoneType, we can't even do a > comparison
    if isinstance(ap, int) and ap > max_logprobs:
        api_params["top_logprobs"] = max_logprobs
    tool_choice = api_params.get("tool_choice")
    if tool_choice:
        if isinstance(tool_choice, str) and tool_choice not in {"none", "auto", "required"}:
            api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        if isinstance(tool_choice, Tool):
            api_params["tool_choice"] = {"type": "function", "function": {"name": tool_choice.name}}

    selections = resolve(api_params.get("model"), local_params.get("mode"))
    if len(selections) != 1:
        if local_params.get("provider"):
            provided = filter(
                lambda x: x.startswith(local_params["provider"]), selections
            )
            if len(provided) >= 1:
                selections = provided
        if len(selections) > 1:
            msg = f'No unique {local_params.get("mode")} model found for "{api_params.get("model", "")}". (Possible: {", ".join([separator.join(x) for x in selections])})'
            raise Exception(msg)
        if len(selections) == 0:
            msg = f'No {local_params.get("mode")} model found for "{api_params.get("model", "")}".'
            raise Exception(msg)
    provider, api_params["model"] = selections[0]
    model = api_params["model"]

    effect, callback = local_params.get(["effect", "callback"])
    if not effect:
        effect = id
    print_output = bool(local_params.get("print_output"))
    yield_output = bool(local_params.get("yield_output"))
    return_output = bool(local_params.get("return_output"))
    return_object = bool(local_params.get("return_object"))
    stream = bool(api_params.get("stream"))

    # now we purge the sensitive (the ones in options) Nones from our dicts
    api_params = api_params.strip()
    local_params = local_params.strip()
    if force_model := local_params.get("force_model"):
        api_params["model"] = force_model
    if force_provider := local_params.get("force_provider"):
        provider = force_provider
    if debug:
        print(api_params, local_params, provider)

    if local_params.get("mode") == "text":
        base = clients[provider].completions.create
        retrieve = lambda y: y.choices[0].text
    else:
        if local_params.get("use_parse"):
            base = clients[provider].beta.chat.completions.parse
        else:
            base = clients[provider].chat.completions.create
        retrieve = lambda y: y.choices[0].message.content
        if stream:
            retrieve = lambda y: y.choices[0].delta.content

    def unpack(y) -> str:
        content = effect(retrieve(y))
        if content:
            return content
        if (hasattr(y.choices[0], 'message') and (tool_calls := y.choices[0].message.tool_calls)):
            return '\n'.join([
                json.dumps({"id": call.id, "function": call.function.name, "arguments": json.loads(call.function.arguments)})
                for call in tool_calls
            ])
        return None

    if stream:
        if yield_output:

            def gen() -> Iterator[str]:
                response = base(**api_params)
                for obj in response:
                    if return_object:
                        yield obj
                    text = unpack(obj) or ""
                    if callback:
                        callback(text)
                    if print_output:
                        print(text, end="")
                    yield text

            return gen()
        result = ""
        response = base(**api_params)
        for obj in response:
            text = unpack(obj) or ""
            result += text
            if print_output and text:
                print(text, end="")
            if callback:
                callback(text)
    else:
        obj = base(**api_params)
        if local_params.get("return_object"):
            return obj
        result = unpack(obj)
        if print_output and result:
            print(result)
        if callback:
            callback(result)
        if obj.choices[0].finish_reason == "tool_calls" and local_params.get(
            "pretty_tool_calls", False
        ):

            def pretty(s: Any) -> str:
                return str(s) if not isinstance(s, str) else f'"{s}"'

            for call in obj.choices[0].message.tool_calls:
                line = f"\ntool_call(id={pretty(call.id)}): {call.function.name}({', '.join([k + '=' + pretty(v) for k, v in json.loads(call.function.arguments).items()])})"
                result += line
                if print_output and line:
                    print(line, end="")
                if callback:
                    callback(result)
        if return_output and not return_object:
            if debug:
                print("Returning result of length " + str(len(result)))
            return result
        return obj
    return None


def text_completion(prompt: str, options: dict = {}) -> Any:
    """
    Generates a text completion based on the provided prompt and arguments.
    :param prompt: Prompts to be provided to the chat completion model.
    :param options: Extra arguments to be provided.
    :returns: The generated text completion.
    """
    params = (
        Dict({
            "mode": "text",
            "prompt": prompt,
            "model": "gpt-3.5-turbo-instruct",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "stream": True,
            "yield_output": True,
            "return_output": not options.get("stream", False),
            "print_output": options.get("stream", False),
        })
        | options
    )
    if (m := params.pop("messages")) is not None:
        options["prompt"] = options.get("prompt") | str(m)

    response = completion(params)
    if response is None:
        return None
    if params.get("stream"):
        if params.get("yield_output"):
            return response  # Return the generator directly
        # Else, consume the generator
        result = list(response)
        if result and params.get("return_output"):
            return "".join(result)
        return None
    return response


def chat_completion(messages: str | list[dict[str, str]], options: dict = {}) -> Any:
    """
    Generates a chat completion based on the provided messages and arguments.
    :param messages: Message log to be provided to the chat completion model.
    :param options: Extra arguments to be provided.
    :returns: The generated chat completion.
    """
    params = {
        "mode": "chat",
        "messages": messages,
        "model": "openrouter" + separator + "gpt-4o",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "yield_output": True,
        "return_output": True,
    } | options
    if options.get("prompt") and not params.get("messages"):
        params["messages"] = [{"role": "user", "content": options["prompt"]}]
        del params["prompt"]
    if isinstance(messages, str):
        params["messages"] = [{"role": "user", "content": messages}]
    response = completion(params)
    if response is None:
        return None
    if params.get("stream"):
        if params.get("yield_output"):
            return response  # Return the generator directly
        # Else, consume the generator
        result = list(response)
        if result and params.get("return_output"):
            return "".join(result)
        return None
    return response
