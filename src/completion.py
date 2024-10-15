"""
Methods for text and chat completions, and agent wrapper classes
"""
from __future__ import annotations

from .tools import *

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
