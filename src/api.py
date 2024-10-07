"""
Set up a custom API
"""
from __future__ import annotations

from .structure import *

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
    Args:
        user_id (str): The ID of the user to generate a session token for.
    Returns:
        str: A unique session token.
    """
    token = secrets.token_urlsafe()
    expiration = datetime.datetime.utcnow() + TOKEN_EXPIRATION
    session_store[token] = (user_id, expiration)
    return token

def validate_session_token(token: str) -> bool:
    """
    Check if a session token is valid.
    Args:
        token (str): The session token to validate.
    Returns:
        bool: True if the token is valid, False otherwise.
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
    Args:
        data: The string to encrypt.
        key: The encryption key.
    Returns:
        str: The XOR encrypted string.
    """
    return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))

def verify_token(request: Request, response: Response, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify the provided authorization token.
    Args:
        request (Request): The incoming request object.
        response (Response): The response object.
        credentials (HTTPAuthorizationCredentials, optional): Credentials provided by the authorization scheme.
    Returns:
        str: The user ID associated with a valid token.
    Raises:
        HTTPException: If the token is invalid.
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
    Args:
        obj (BaseModel): The object to fill.
    Returns:
        Optional[BaseModel]: The filled object, or None if obj is not a BaseModel.
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
    Attributes:
        content (Optional[str]): The text to tokenize.
        return_tokens (Optional[bool]): Whether to return the tokens.
        tokenizer (Optional[str]): The tokenizer to use.
    """
    content: Optional[str] = None
    return_tokens: Optional[bool] = False
    tokenizer: Optional[str] = "o200k_base"

class ScrapeRequest(BaseModel):
    """
    A request for web scraping.
    Attributes:
        url (Optional[str]): The URL to scrape.
    """
    url: Optional[str] = None

class ResolveRequest(BaseModel):
    """
    A request for resolving a model or tool.
    Attributes:
        id (Optional[str]): The ID of the model or tool.
        model (Optional[str]): The model to resolve.
        mode (Optional[str]): The mode to resolve (e.g. "text", "chat").
    """
    id: Optional[str] = None  # ambiguous whether to call id or model, so allow both
    model: Optional[str] = None
    mode: Optional[str] = ''

class Parameters(BaseModel):
    """
    Parameters for a request.
    Attributes:
        model (Optional[str]): The model to use.
        mode (Optional[str]): The mode to use.
        temperature (Optional[float]): The temperature to use.
        top_p (Optional[float]): The top_p to use.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        stream (Optional[bool]): Whether to stream the response (default: True).
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
    """api_text_completion"""
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
    Args:
        credentials (HTTPAuthorizationCredentials): The authorization credentials.
    Returns:
        Dict: A dictionary containing the global parameters.
    """
    return global_params

@app.post("/v1/params")
async def set_parameters(request: Parameters, credentials: HTTPAuthorizationCredentials = Depends(verify_token)) -> Dict:
    """
    Sets the global parameters.
    Args:
        request (Parameters): The parameters to be updated.
        credentials (HTTPAuthorizationCredentials): The authorization credentials.
    Returns:
        Dict: A dictionary containing a success message.
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
    Args:
        id (str): The ID of the model (optional).
        mode (str): The mode of the model (optional).
        credentials (HTTPAuthorizationCredentials): The authorization credentials.
    Returns:
        Dict: A dictionary containing a list of models.
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
    Args:
        request (ResolveRequest): The request containing the ID, model, and mode.
        credentials (HTTPAuthorizationCredentials): The authorization credentials.
    Returns:
        str: A string containing a list of possible models.
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
    get_tokens
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
    scrape_url
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
    scrape_url_direct
    """
    return await scrape_url(url=url, credentials=credentials)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=True)
