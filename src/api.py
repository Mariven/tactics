"""
Set up a custom API
"""
from __future__ import annotations

from .completion import *

import json
import re
import httpx
import hashlib
import secrets
from typing import Optional
import datetime
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
local_token_hash = "80502743d54e8db2992a8b6cf7ccb1f1914358398ef1aa45d949c5957ee8edd6"
jina_secret = "\x01](\x14\x16^@\x03.r\n\x0e'\x17F\x12/PQpG\x04W)\tQr\x17~\x0bO\x00xw\t\\'J\x1b \x0c\r`\x1b \x07\x0c\x0e<\x00(0\x02\x0b\x15t\x19\x11\x0fXu;I!\x02"
security = HTTPBearer()

app = FastAPI()
HOST, PORT = "0.0.0.0", 11434
HTTP_CODES = {
    200: "Success",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}
HTTP_SUCCESS = 200


# Store for session tokens: {token: (user_id, expiration_time)}
session_store = {}
TOKEN_EXPIRATION = datetime.timedelta(hours=24)
def create_session_token(user_id: str) -> str:
    """
    Generate a unique session token for the given user.
    :param user_id: The ID of the user to generate a session token for.
    :returns: A unique session token.
    """
    token = secrets.token_urlsafe()
    expiration = datetime.datetime.utcnow() + TOKEN_EXPIRATION
    session_store[token] = (user_id, expiration)
    return token

def validate_session_token(token: str) -> bool:
    """
    Check if a session token is valid.
    :param token: The session token to validate.
    :returns: True if the token is valid, False otherwise.
    """
    if token in session_store:
        _, expiration = session_store[token]
        if datetime.datetime.utcnow() < expiration:
            return True
        del session_store[token]
    return False

def xor_strings(data: str, key: str) -> str:
    """
    Perform XOR encryption on two strings.
    :param data: The string to encrypt.
    :param key: The encryption key.
    :returns: The XOR encrypted string.
    """
    return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))

def verify_token(request: Request, response: Response, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify the provided authorization token.
    :param request: The incoming request object.
    :param response: The response object.
    :param credentials: Credentials provided by the authorization scheme.
    :returns: The user ID associated with a valid token.
    :raises: HTTPException: If the token is invalid.
    """
    if credentials and hashlib.sha256(credentials.credentials.encode()).hexdigest() == local_token_hash:
        new_token = create_session_token(credentials.credentials)
        response.set_cookie(key="session_token", value=new_token, samesite="strict")
        return credentials.credentials
    if credentials and validate_session_token(credentials.credentials):
        return session_store[credentials.credentials][0]

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

@app.get("/v1/protected-route")
async def protected_route(credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """protected_route"""
    return {"message": "Successfully authenticated"}

# @app.options("/{path:path}")
# async def options_route(request: Request):
# 	return {"message": "OK"}

def log_to_file(data, log_type, display=True) -> None:
    """log_to_file"""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "type": log_type, "data": data}
    if display:
        print(log_entry)
    with open("api_log.jsonl", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")


# use with JS as
# async function complete(args){
# 	const mode = 'prompt' in args ? 'text' : 'chat';
# 	const stream = args?.stream ?? false;
# 	const request = await fetch(
# 		`http://localhost:11434/v1/${mode}`,{
# 			method: 'POST',
# 			headers: {'Content-Type':'application/json'},
# 			body: JSON.stringify(args)
# 	}	);
# 	if (!stream) {
# 		const done = await request.json();
# 		return done;
# 	}
# 	const reader = request.body.getReader();
# 	let timeout = 0;
# 	while(timeout < 8192) {
# 		timeout += 1;
# 		const {done,value} = await reader.read();
# 		if(done) break;
# 		let chunk = new TextDecoder().decode(value);
# 		console.log(chunk);
# }	}
# # complete({messages: 'write a haiku about a tree', model: 'gpt-4o-mini', stream: true})
# or as
# async function complete(args){const r=await fetch(`http://localhost:11434/v1/${'prompt' in args?'text':'chat'}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)}); if(!(args?.stream??false)){const d=await r.json(); return d;} const re=r.body.getReader();let i=0; while(i<8192){i+=1;const {done,value}=await re.read(); if(done) break; let x=new TextDecoder().decode(value); console.log(x);}}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

global_params = {
    "text_model": "gpt-3.5-turbo-instruct",
    "chat_model": "gpt-4o-2024-08-06",
    "text_max_tokens": 256,
    "chat_max_tokens": 2048,
    "top_p": 1.0,
    "temperature": 0.7,
    "prompt": "Hi!",
    "messages": [{"role": "user", "content": "Hi!"}],
    "stream": True,
    } | dict.fromkeys(["model", "max_tokens", "seed", "stop", "n", "user", "logit_bias", "logprobs", "top_logprobs", "presence_penalty", "frequency_penalty", "suffix", "grammar", "json_schema", "response_format", "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call", "system_message", "service_tier", "stream_options", "echo"], None)

def fill(obj: BaseModel) -> Optional[BaseModel]:
    """
    Fill in missing values in the object with default values from global_params.
    :param obj: The object to fill.
    :returns: The filled object, or None if obj is not a BaseModel.
    """
    if not isinstance(obj, BaseModel):
        return None
    for i in obj.model_fields:
        if (hasattr(obj, i) and getattr(obj, i) is None) and (value := global_params.get(i, None)):
            if i not in {"prompt", "messages", "tools"}:
                setattr(obj, i, value)
    return obj

class TokenizeRequest(BaseModel):
    """
    A request for tokenization.
    :param content: The text to tokenize.
    :param return_tokens: Whether to return the tokens.
    :param tokenizer: The tokenizer to use.
    """
    content: Optional[str] = None
    return_tokens: Optional[bool] = False
    tokenizer: Optional[str] = "o200k_base"

class ScrapeRequest(BaseModel):
    """
    A request for web scraping.
    :param url: The URL to scrape.
    """
    url: Optional[str] = None

class ResolveRequest(BaseModel):
    """
    A request for resolving a model or tool.
    :param id: The ID of the model or tool.
    :param model: The model to resolve.
    :param mode: The mode to resolve (e.g. "text", "chat").
    """
    id: Optional[str] = None  # ambiguous whether to call id or model, so allow both
    model: Optional[str] = None
    mode: Optional[str] = ''

class Parameters(BaseModel):
    """
    Parameters for a request.
    :param model: The model to use.
    :param mode: The mode to use.
    :param temperature: The temperature to use.
    :param top_p: The top_p to use.
    :param max_tokens: The maximum number of tokens to generate.
    :param stream: Whether to stream the response (default: True).
        ...
    """
    model: Optional[str] = None
    mode: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    suffix: Optional[str] = None
    grammar: Optional[str] = None
    response_format: Optional[dict] = None
    force_model: Optional[str] = None
    force_provider: Optional[str] = None
    print_output: Optional[bool] = None
    yield_output: Optional[bool] = None
    return_output: Optional[bool] = None
    debug: Optional[bool] = None
    return_raw: Optional[bool] = None
    pretty_tool_calls: Optional[bool] = None
    return_object: Optional[bool] = None
    text_model: Optional[str] = None
    chat_model: Optional[str] = None
    text_max_tokens: Optional[int] = None
    chat_max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    n: Optional[int] = None
    user: Optional[str] = None
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    top_logprobs: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stream_options: Optional[dict] = None
    system_message: Optional[str] = None
    echo: Optional[bool] = None
    service_tier: Optional[str] = None

    prompt: Optional[str] = None
    messages: Optional[List[dict] | str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[dict] = None
    parallel_tool_calls: Optional[bool] = None

@app.post("/v1/text")
@app.post("/v1/completions")
async def api_text_completion(request: Parameters, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Any:
    """
    Takes a text completion request and returns the completion.
    :param request: The text completion request.
    :param credentials: The authorization credentials.
    :returns: The completion.
    """
    bare_request = {k: v for (k, v) in request.dict().items() if v is not None}
    b = str(bare_request.get("prompt"))
    max_log_length = 150
    if len(b) > max_log_length:
        b = b[:50] + f"...({len(b) - 100} additional characters)..." + b[-50:]
    bare_request["prompt"] = b
    log_to_file(bare_request, "text request")
    request.model = request.model or request.text_model or global_params['text_model']
    request.max_tokens = request.max_tokens or request.text_max_tokens or global_params['text_max_tokens']
    request.text_model = None
    request.text_max_tokens = None
    request = fill(request)
    params0 = {
        "print_output": False, "return_output": True, "stream": bool(request.stream), "yield_output": bool(request.stream),
        } | {k: v for (k, v) in regularize(request).items() if v is not None} | {"mode": "text"}
    params = {k: v for (k, v) in params0.items() if v is not None}
    # try:
    if request.stream:
        return StreamingResponse(
            content=gen_stream(text_completion(params['prompt'], params)),
            media_type="text/event-stream")
    result = text_completion(params['prompt'], params)
    log_to_file(result, "response", True)
    return result
    # except Exception as e:
    # 	print(e)
    # 	raise HTTPException(status_code=500, detail=str(e))

# @app.post("/v1/completions")
# async def api_text_completion2(request: TextCompletionRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
# 	return api_text_completion(request, credentials)

@app.post("/v1/chat")
@app.post("/v1/chat/completions")
async def api_chat_completion(request: Parameters, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Any:
    """
    Endpoint for chat completion requests.
    :param request: The chat completion request.
    :param credentials: The authorization credentials.
    :returns: The chat completion response.
    """
    bare_request = {k: v for (k, v) in request.dict().items() if v is not None}
    b = re.sub(r"\{(['\"])role['\"]: ['\"]([a-zA-Z]+)['\"], ['\"]content['\"]: ['\"]", r"{\1\2: ", str(bare_request.get("messages")))
    max_log_length = 150
    if len(b) > max_log_length:
        b = b[:50] + f"...({len(b) - 100} additional characters)..." + b[-50:]
    bare_request["messages"] = b
    log_to_file(bare_request, "chat request")
    fill(request)
    request.model = request.model or request.chat_model or global_params['chat_model']
    request.max_tokens = request.max_tokens or request.chat_max_tokens or global_params['chat_max_tokens']
    request.chat_model = None
    request.chat_max_tokens = None
    params0 = {
        "print_output": False, "return_output": True, "stream": bool(request.stream), "yield_output": bool(request.stream),
        } | regularize(request) | {"mode": "chat"}
    params = {k: v for (k, v) in params0.items() if v is not None}
    try:
        if request.stream:
            return StreamingResponse(
                content=gen_stream(chat_completion(params['messages'], params)),
                media_type="text/event-stream")
        result = chat_completion(params['messages'], params)
        log_to_file(result, "response", True)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/v1/params")
async def get_parameters(credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Retrieves the global parameters.
    :param credentials (HTTPAuthorizationCredentials): The authorization credentials.
    :returns: A dictionary containing the global parameters.
    """
    return global_params

@app.post("/v1/params")
async def set_parameters(request: Parameters, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Sets the global parameters.
    :param request: The parameters to be updated.
    :param credentials: The authorization credentials.
    :returns: A dictionary containing a success message.
    """
    for key, value in request.dict().items():
        if value is not None:
            global_params[key] = value
    return {"message": "Parameters updated successfully"}

@app.get("/v1/models")
async def get_models_query(
    id: str = Query(None), mode: str = Query(None),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Retrieves a list of models. If `id` or `mode` is provided, it will filter the list accordingly.
    :param id: The ID of the model (optional).
    :param mode: The mode of the model (optional).
    :param credentials: The authorization credentials.
    :returns: A dictionary containing a list of models.
    """
    if id or mode:
        return await do_resolve(ResolveRequest(id=id, model=id, mode=mode), credentials)
    L = []
    for t in ['chat', 'text']:
        for k, v in models_by_mode[t].items():
            L.append(v | {"id": k, "type": t})
    return {'object': 'list', 'data': L}

@app.post("/v1/models")
async def do_resolve(request: ResolveRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> str:
    """
    Resolves a model or tool based on the provided request.
    :param request (ResolveRequest): The request containing the ID, model, and mode.
    :param credentials: The authorization credentials.
    :returns: A string containing a list of possible models.
    """
    mode = str(request.mode) if request.mode else ''
    model = str(request.model) if request.model else str(request.id) if request.id else ''
    id = str(request.id) if request.id else str(request.model) if request.model else model
    return f'Possible models corresponding to \'{id}\': {", ".join([":".join(y) for y in resolve(id, mode)])}'
@app.post("/v1/tokenize")
@app.get("/v1/tokenize")
async def get_tokens(
    request: TokenizeRequest = None,
    content: str = Query(None), return_tokens: bool = Query(None), tokenizer: str = Query("o200k_base"),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Gives the tokens for a given text. If `request` is provided, it will use the `content`, `return_tokens`, and `tokenizer` from the request. Otherwise, it will use the query parameters.
    :param request: The tokenize request (optional).
    :param content: The text to tokenize (optional).
    :param return_tokens: Whether to return the list of tokens (optional).
    :param tokenizer: The tokenizer to use (optional).
    :param credentials: The authorization credentials.
    :returns: A dictionary containing the token count, list, etc.
    """
        # Combine query params and JSON body, prioritizing query params
    params = {
        "content": content or (request.content if request else None),
        "return_tokens": return_tokens if return_tokens is not None else (request.return_tokens if request else False),
        "tokenizer": tokenizer or (request.tokenizer if (request and request.tokenizer) else "o200k_base"),
    }

    # Validate that we have the necessary parameters
    if not params["content"]:
        raise HTTPException(status_code=400, detail="'content' is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + xor_strings(jina_secret, credentials),
    }
    content = request.content
    baseData = {"tokenizer": tokenizer, "return_tokens": return_tokens}
    output = {'token_count': 0, 'tokenizer': request.tokenizer}
    if request.return_tokens:
        output |= {'token_list': [], 'token_dict': {}}
    while len(content) > 0:
        data = {"content": content[:63000]} | baseData
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post("https://segment.jina.ai/", headers=headers, json=data)
        if response.status_code != HTTP_SUCCESS:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        res = response.json()
        output['token_count'] += res['num_tokens']
        if request.return_tokens:
            output['token_list'] += (w[0] for w in res["tokens"])
            output['token_dict'] |= {w[0]: min(w[1]) for w in res["tokens"]}
        content = content[63000:]
    return output

@app.post("/v1/scrape")
@app.get("/v1/scrape")
async def scrape_url(
    request: ScrapeRequest = None, url: str = Query(None),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Service for scraping text from a URL.
    :param request: The scrape request (optional).
    :param url: The URL to scrape (optional).
    :param credentials: The authorization credentials.
    :returns: A dictionary containing the scraped text.
    """
    scrape_url = url or (request.url if request else None)
    if not scrape_url:
        raise HTTPException(status_code=400, detail="'url' is required")
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + xor_strings(jina_secret, credentials),
        "Content-Type": "application/json",
    }
    data = {
        "url": scrape_url,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post("https://r.jina.ai/", headers=headers, json=data)

    if response.status_code != HTTP_SUCCESS:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    res = response.json()
    output = {"title": res["data"]["title"], "content": res["data"]["content"], "url": res["data"]["url"]}
    return output

@app.get("/v1/scrape/{url:path}")
async def scrape_url_direct(url: str, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Alias for `scrape_url` with the URL provided as a path parameter.
    :param url: The URL to scrape.
    :param credentials: The authorization credentials.
    :returns: A dictionary containing the scraped text.
    """
    return await scrape_url(url=url, credentials=credentials)

js_completer_prompt = """
### Instruction ###
You will be provided with an incomplete snippet of JavaScript code. Incomplete parts will be marked via block comments that describe the function of the missing code. These comments may provide code in another language (e.g., Python), use pseudocode, or describe the functionality in natural language, or a mix of these.

Your task is to complete the JavaScript code by pairing each incomplete line number with a list of new lines with the completed code for that number. You must provide the output via the line editor tool.

### Example ###
If the incomplete code is:
```
1 if(useCallback) {
2 	/* return map(lambda x: callback(x.params), objects) */
3 } else {
4 	/* reverse the order of x.params and break */
5 }
```
You would replace the Python code at line 2 and the pseudocode at line 4 with JavaScript code. The resultant JSON object should be:
```
{"edits": [
    {"id": 2, "lines": [
        "return objects.map(x => callback(x.params));"
    ]},
    {"id": 4, "lines": [
        "for (let i = 0; i < list.length; i++) {", 
        "list[i].params = list[i].params.reverse();",
        "break;",
        "}"
    ]}
]}
```
Do not call this tool several times in parallel, as a single call always suffices. For instance, instead of making two parallel calls
```
call #1: {"edits": [{"id": 4, "lines": [completed code]}]}
call #2: {"edits": [{"id": 4, "lines": [comment]}]}
```
You can make the single call
```
call #1: {"edits": [{"id": 4, "lines": [comment, completed code]}]}
```
### Guidelines ###
1. **Commenting:** You may include comment lines among your completions.
2. **Whitespace and Formatting:** Minimize the use of excessive whitespace, and adhere to existing naming and spacing conventions.
3. **Brackets:** Complete brackets where possible.
4. **Insufficient Information:** If you lack enough information to complete a line, do what you can, if anything, and concisely state the missing information that needs to be provided (e.g. the name of a certain function, the type of a certain variable). You must also reproduce the block comment in this case, since all block comments are replaced with your outputted lines.

By following these guidelines, use the line editor tool to provide the completed JavaScript code at each incomplete line number.
""".strip()

testJS = """
if (condition) {
    arr = arr.filter(num => num % 2 === 0).map(num => num * 2);
    } else {
        arr = arr.map(num => num ** 2).sort((a, b) => b - a);
    }
}
function processArray(arr, condition) {
    if (condition) {
        arr = arr.filter(num => num % 2 === 0).map(num => num * 2);
    } else {
        arr = arr.map(num => num * num).sort((a, b) => b - a);
    }
    return arr;
}

""".strip()

with open('data/structured_outputs/code_diff.json') as file:
    code_diff_schema = json.load(file)

def js_wand(incomplete_code: str, endpoint: str, complete_raw: Optional[bool] = None):
    """
    Takes incomplete JavaScript code and completes it using the provided endpoint.
    :param incomplete_code: The incomplete JavaScript code to be completed.
    :param endpoint: The endpoint to be used for completion.
    :param complete_raw: Whether to return the raw completion or not.
    :returns: The completed JavaScript code.
    """
    debug = True
    if debug:
        debug_file = open('../logs/debug.txt', 'a')
    try:
        def enumerate_lines(code):
            """
            Enumerate each line of the provided code snippet.
            """
            cleaned_code = re.sub('^\n*([\s\S]+?)\n*$', r'\1', code)
            code_lines = cleaned_code.split('\n')
            enumerated_lines = []
            for line_number, line_content in enumerate(code_lines):
                enumerated_lines.append(str(line_number + 1) + ' ' + line_content)
            return enumerated_lines

        enumerated_lines = enumerate_lines(incomplete_code)

        if not complete_raw:
            completion_request = requests.post(f'{endpoint}/v1/chat',
            json = {'messages': [
                {'role': 'system', 'content': js_completer_prompt},
                {'role': 'user', 'content': '\n'.join(enumerated_lines)}],
                'model': 'gpt-4o-2024-08-06', 'temperature': 0.3,
                'response_format': {'type': 'json_schema', 'json_schema': code_diff_schema}})
            complete_raw = json.loads(completion_request.content)['result']
            if debug:
                debug_file.write(str(complete_raw) + '\n')
        original_code_lines = {k: [v] for k, v in enumerate(re.sub('^\n*([\s\S]+?)\n*$', r'\1', incomplete_code).split('\n'))}
        completed_code_lines = {int(completion['index']) - 1: completion['lines'] for completion in json.loads(complete_raw)['completions']}

        if debug:
            debug_file.write(str(original_code_lines) + '\n')
            debug_file.write(str(completed_code_lines) + '\n')
        new_lines = []

        def get_indents(line):
            """
            Determine the indentation type and depth of a given line.
            """
            indent_info = {y: len(re.findall(f'^({y[0]}*)', line)[0]) // len(y) for y in ['   ', '\t']}
            indent_type, indent_depth = sorted(indent_info.items(), key=lambda x: x[1])[-1]
            return indent_type, indent_depth
        indent_type, indent_depth = get_indents(original_code_lines[0][0])
        for line_index, line_list in enumerate((original_code_lines | completed_code_lines).values()):
            for line_content in line_list:
                if line_content.strip()[0] in ')]}':
                    indent_depth -= 1
                if line_index in completed_code_lines.keys():
                    new_lines.append(indent_type * indent_depth + line_content)
                else:
                    if not (line_content.strip().startswith('/*') and line_content.strip().endswith('*/')):
                        new_lines.append(line_content)
                if line_content.strip()[-1] in '([{':
                    indent_depth += 1

        result = '\n'.join(new_lines)

    except Exception as e:
        result = ":("
        debug_file.write(str(e) + '\n')

    if debug:
        debug_file.close()

    return result

def text_wand(incomplete_text, raw_edits = None):
    def enumerate_lines(code):
        cleaned_code_lines = re.sub('^\n*([\s\S]+?)\n*$', r'\1', code).split('\n')
        return [str(idx + 1) + ' ' + text for idx, text in enumerate(cleaned_code_lines)]
    def obtain_edits(lines):
        response = completion(pretty_tool_calls=False, mode="chat", messages=[{'role': 'system', 'content': js_completer_prompt},
            {'role': 'user', 'content': '\n'.join(enumerated_lines)}],model='gpt-4o-2024-08-06', return_raw=True,tools=[line_replace_schema])
        if not response.choices[0].finish_reason == 'tool_calls':
            raise Exception('Model did not call the line_diff tool:\n'+str(response))
        raw_edits = response.choices[0].message.tool_calls
        if len(raw_edits) > 1:
            raise Exception('Model called line_diff multiple times in parallel:\n'+str(response))
        return raw_edits
    def apply_edits(old_text, raw_edits):
        old_lines = {k: [v] for k, v in enumerate(re.sub('^\n*([\s\S]+?)\n*$', r'\1', old_text).split('\n'))}
        line_edits = {int(edit['id']) - 1: edit['lines'] for edit in json.loads(raw_edits[0].function.arguments)['edits']}
        return [y for y_ in (old_lines | line_edits).values() for y in y_]

    enumerated_lines = enumerate_lines(incomplete_code)
    raw_edits = raw_edits or obtain_edits(enumerated_lines)
    new_lines = apply_edits(incomplete_code, raw_edits)
    return '\n'.join(new_lines)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=True)
