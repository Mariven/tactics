"""Abstract base classes and utilities for API clients."""
from __future__ import annotations

from src.basetypes import *  # re, typing
from src.supertypes import *  # builtins, functools, inspect, itertools, operator, logging
from src.utilities import *  # json, os, sqlite3, time, types, random, requests

import datetime
import threading

from pydantic import BaseModel
from typing import *
from abc import ABC, abstractmethod
from typing_extensions import Self


rapidapi_key = os.getenv("RAPIDAPI_API_KEY")
if not rapidapi_key:
    try:
        with open("secrets.json") as file:
            content = json.load(file)
            rapidapi_key = content["rapidapi-api-key"]
    except Exception as e:
        raise Exception("No RapidAPI API key found.") from e


class API(ABC):
    """
    An abstract base class for API clients.
    """

    @abstractmethod
    def build_headers(self) -> dict[str, str]:
        """
        Build the headers for an API request.
        """
        pass

    @abstractmethod
    def build_url(self, endpoint: str, params: dict[str, Any]) -> Any:
        """
        Build the URL for an API request to a given endpoint with given parameters.
        :param endpoint: The endpoint to request.
        :param params: The parameters to pass to the endpoint.
        :return: The URL for the request.
        """
        pass

    @abstractmethod
    def build_request(self, method: str, endpoint: str, params: dict[str, Any]) -> Any:
        """
        Build a request object for an API request to a given endpoint with given parameters.
        :param method: The HTTP method to use for the request.
        :param endpoint: The endpoint to request.
        :param params: The parameters to pass to the endpoint.
        :return: The request object.
        """
        pass

    def send(self, request: requests.Request) -> requests.Response:
        """
        Send a request object and return the response.
        :param request: The request object to send.
        :return: The response object.
        """
        with requests.Session() as session:
            prepared_request = session.prepare_request(request)
            response = session.send(prepared_request)
            response.raise_for_status()
            return response

    def parse_response(self,
        response: requests.Response,
        schema: BaseModel,
        with_headers: bool = False
    ) -> BaseModel | tuple[BaseModel, dict[str, str]]:
        """
        Parse a response object into a BaseModel.
        :param response: The response object to parse.
        :param schema: The schema to parse the response with.
        :param with_headers: Whether to return the headers in the response.
        :return: The parsed response, and optionally the headers.
        """
        try:
            if with_headers:
                return schema.parse_obj(response.json()), response.headers
            return schema.parse_obj(response.json())
        except Exception as e:
            msg = f"Error parsing response: {e}\n{response.text}"
            raise Exception(msg) from e

    def method(mt: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # noqa: N805
        """
        Decorator to convert an abstract method into an endpoint with a given HTTP method. This uses type hints to infer the parameters and return type.
        :param mt: The HTTP method to use for the endpoint.
        :return: A decorator that converts an abstract method into an endpoint with the given HTTP method.
        """
        def decorator(func: Callable) -> Callable:
            annotations = get_type_hints(func)
            ann_return = annotations.pop("return")
            # self.register_endpoint(func.__name__, mt, annotations, ann_return)
            if "self" in annotations:
                annotations.pop("self")
            endpoint = func.__name__
            params_schema = annotations
            response_schema = ann_return

            @functools.wraps(func)
            def endpoint_method(self, **kwargs) -> BaseModel:
                for param_name, param_type in params_schema.items():
                    if param_name not in kwargs:
                        if custom_generic_alias_repr(param_type).startswith("Optional"):
                            continue
                        msg = f"Missing required parameter: {param_name}"
                        raise TypeError(msg)
                    if not isinstance(kwargs[param_name], param_type):
                        msg = f"Parameter {param_name} must be of type {param_type}"
                        raise TypeError(msg)

                request = self.build_request(mt, endpoint, kwargs)
                response = self.send(request)
                return self.parse_response(response, response_schema)
            return endpoint_method
        return decorator

class RapidAPI(API):
    def __init__(self, rapidapi_key: str, api_id: str) -> None:
        self.rapidapi_key: str = rapidapi_key
        self.api_id: str = api_id
        self.endpoints: dict[str, dict[str, Any]] = {}
        self.param_types: dict[str, Any] = {}
        self.force_headers: bool = False

    def build_headers(self) -> dict[str, str]:
        return {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": f"{self.api_id}.p.rapidapi.com"
        }

    def build_url(self, endpoint: str, params: dict[str, Any]) -> str:  # noqa: ARG002
        return f"https://{self.api_id}.p.rapidapi.com/{endpoint}.php"

    def build_request(self, method: str, endpoint: str, params: dict[str, Any]) -> requests.Request:
        url = self.build_url(endpoint, params)
        headers = self.build_headers()
        if method == "GET":
            return requests.Request(method, url, headers=headers, params=params)
        if method == "POST":
            return requests.Request(method, url, headers=headers, json=params)
        msg = f"Unsupported method: {method}"
        raise ValueError(msg)

    def register_endpoint(self,
        endpoint: str,
        method: str,
        params_schema: list[str] | dict[str, Any],
        response_schema: BaseModel,
        with_headers: bool = False
    ) -> None:
        if isinstance(params_schema, list):
            params_schema = {k: self.param_types.get(k, Any) for k in params_schema}
        self.endpoints[endpoint] = {
            "method": method,
            "params_schema": params_schema,
            "response_schema": response_schema,
            "with_headers": with_headers
        }

    def register_endpoints(self, endpoints: list[tuple]) -> None:
        for args in endpoints:
            if len(args) == 4:
                endpoint, method, params_schema, response_schema = args
                with_headers = False
            else:
                endpoint, method, params_schema, response_schema, with_headers = args
            self.register_endpoint(endpoint, method, params_schema, response_schema, with_headers)

    def set_types(self, params: dict[str, Any]) -> None:
        self.param_types.update(params)

    def __getattr__(self, name: str) -> Callable:  # Allows calling registered endpoints as methods
        if name in self.endpoints:
            endpoint_data = self.endpoints[name]

            def endpoint_method(**kwargs) -> BaseModel:
                request = self.build_request(endpoint_data["method"], name, kwargs)
                response = self.send(request)
                with_headers = self.force_headers
                if "with_headers" in endpoint_data:
                    with_headers = endpoint_data["with_headers"]
                if "with_headers" in kwargs:
                    with_headers = kwargs["with_headers"]
                return self.parse_response(response, endpoint_data["response_schema"], with_headers)
            return endpoint_method
        try:
            return super().__getattr__(name)
        except AttributeError:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None


Locator: TypeAlias = Callable[[BaseModel], Any] | str
# Types a method to locate the cursor value in a paginated API response, either...
# -- a function that takes a BaseModel and returns the cursor value, or...
# -- a string key to access the cursor value

def cursor_traverse(
    object_fetcher: Callable[[dict[str, Any]], BaseModel],
    params: dict[str, Any],
    cursor_locator: Locator,
    data_locator: Locator,
    initial_object: BaseModel | None = None,
    max_fetches: int = 50,
    log_file: str | None = None
    ) -> dict[str, Any]:
    """
    Make consecutive calls to a paginated API to fetch data.
    :param object_fetcher: A function that takes a dictionary of parameters and returns a BaseModel via a call to the API.
    :param params: A dictionary of parameters to pass to the object fetcher.
    :param cursor_locator: A function that takes a BaseModel and returns the cursor value, or a string key to access the cursor value.
    :param data_locator: A function that takes a BaseModel and returns a list of data to fetch, or a string key to access the data.
    :param initial_object: An optional BaseModel to start the traversal with.
    :param max_fetches: The maximum number of fetches to make.
    :param log_file: An optional file path to log the traversal to.
    :return: A dictionary with keys "data", pointing to the data gathered from consecutive calls, and "cursor", pointing to the cursor value used to fetch the next page of data if max_fetches was reached, or None if the end of the data was reached.
    """
    if log_file is not None:
        try:
            file = open(log_file, "a")  # noqa: SIM115
        except FileNotFoundError:
            file = open(log_file, "w")  # noqa: SIM115
        file_open = True
    else:
        file_open = False

    def log(message: Any, verbatim: bool = False) -> None:
        if isinstance(message, BaseModel):
            message = message.model_dump_json()
        if isinstance(message, dict):
            message = json.dumps(message)
        if not verbatim:
            timestamp = round(time.time() - start, 2)
            message = f"\t({timestamp}): {message}"
        if file_open:
            file.write(message + "\n")

    if isinstance(cursor_locator, str):
        cursor_str = ''.join(cursor_locator.split())
        cursor_locator = lambda obj: obj[cursor_str]
    if isinstance(data_locator, str):
        data_str = ''.join(data_locator.split())
        data_locator = lambda obj: obj[data_str]
    fetched_num = 0
    fetched_data = []
    start = time.time()
    abs_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{abs_time}]: Starting traversal with params {params}"
    log(msg, verbatim = True)
    if initial_object is None:
        obj = regularize(object_fetcher(**params))
        fetched_num = 1
    else:
        obj = regularize(initial_object)
    data = data_locator(obj)
    fetched_data += data
    cursor = cursor_locator(obj)
    data_string = '\n\t\t'.join(json.dumps(d) for d in data)
    msg = f"Page {fetched_num}: {len(data)} objects fetched, {len(fetched_data)} total, next cursor: {cursor}. Fetched: {data_string}"
    log(msg, verbatim = False)
    try:
        while cursor is not None:
            if fetched_num >= max_fetches:
                log(f"Reached max fetches ({max_fetches}), stopping.")
                break
            obj = regularize(object_fetcher(**{**params, "cursor": cursor}))
            fetched_data += data_locator(obj)
            fetched_num += 1
            cursor = cursor_locator(obj)
            data_string = '\n\t\t'.join(json.dumps(d) for d in data)
            msg = f"Page {fetched_num}: {len(data)} objects fetched {len(fetched_data)} total, next cursor: {cursor}. Fetched: {data_string}"
            log(msg, verbatim = False)
        if log_file is not None:
            file.close()
        return {"data": fetched_data, "cursor": cursor}
    except Exception as e:
        log(f"Error: {e}", verbatim = False)
        if log_file is not None:
            file.close()
        raise


# FlowRate = tuple[int, Literal["s", "m", "h", "d"]]
# FlowLimit = tuple[FlowRate, Callable[..., float]]

class DependentLimiter:
    def __init__(self, rate_per_sec: float, capacity: float, shutdown_event: threading.Event) -> None:
        self.capacity = capacity
        self.volume = capacity
        self.rate_per_sec = rate_per_sec
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.shutdown_event = shutdown_event

    def _add_volume(self) -> None:
        now = time.monotonic()
        elapsed_secs = now - self.timestamp
        self.timestamp = now
        self.volume = min(self.capacity, self.volume + elapsed_secs * self.rate_per_sec)

    def consume(self, volume: float) -> None:
        with self.condition:
            while True:
                if self.shutdown_event.is_set():
                    raise RuntimeError("Executor is shutting down")
                self._add_volume()
                dV = volume - self.volume
                if self.volume >= volume:
                    self.volume = -dV
                else:
                    self.condition.wait(timeout=dV / self.rate_per_sec)

    def shutdown(self) -> None:
        with self.condition:
            self.condition.notify_all()

class RateLimitedExecutor:
    """
    A thread pool executor that enforces rate limits on both requests per minute
    and flow per minute of an arbitrary measure of a request's volume.
    :param max_workers: The maximum number of worker threads.
    :param requests_per_minute: The maximum number of requests allowed per minute.
    :param volume_per_minute: The maximum amount of some unit that can be used per minute.
    :param measure: A function that takes the same arguments as the submitted task and returns the volume required for that task.
    """
    def __init__(self,
        max_workers: int,
        requests_per_minute: float,
        measure_per_minute: float,
        measure: Callable[..., float]
        ) -> None:
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutdown_event: threading.Event = threading.Event()
        self.measure: Callable[..., float] = measure

        self.request_bucket: DependentLimiter = DependentLimiter(
            rate_per_sec=requests_per_minute / 60.0,
            capacity=requests_per_minute,
            shutdown_event=self.shutdown_event
        )
        self.flow_bucket: DependentLimiter = DependentLimiter(
            rate_per_sec=measure_per_minute / 60.0,
            capacity=measure_per_minute,
            shutdown_event=self.shutdown_event
        )
        self.started: bool = True

    def submit(self, func: Callable[..., T], *args, **kwargs) -> Future:
        """
        Submits a rate-limited task for execution.
        :param func: The function to execute.
        :return: A Future representing the task.
        """
        if self.shutdown_event.is_set():
            raise RuntimeError("Executor is shutting down")
        return self.executor.submit(self._execute_rate_limited, func, *args, **kwargs)

    def _execute_rate_limited(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Executes a function with rate limiting.
        :param func: The function to execute.
        :return: The result of the function.
        """
        if self.shutdown_event.is_set():
            raise RuntimeError("Executor is shutting down")
        volume_needed = self.measure(*args, **kwargs)
        self.request_bucket.consume(1)
        self.flow_bucket.consume(volume_needed)
        return func(*args, **kwargs)

    def map(self, func: Callable[..., T], *iterables) -> list[T | Exception]:
        """Applies a function to a list of items in parallel with rate limiting.
        Returns a list containing either the results of successful task executions
        or the exceptions raised by failed tasks.
        :param func: The function to apply to the items.
        :param iterables: The items to apply the function to.
        :return: A list of results or exceptions.
        """
        if self.shutdown_event.is_set():
            raise RuntimeError("Executor is shutting down")

        futures = [self.submit(func, *args) for args in zip(*iterables)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(e)
        return results

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Initiate shutdown and optionally wait for tasks to complete.
        :param wait: If True, wait for pending tasks to complete.
        :param cancel_futures: If True, cancel all pending futures.
        """
        self.shutdown_event.set()
        self.request_bucket.shutdown()
        self.flow_bucket.shutdown()
        self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.started = False

    def __enter__(self) -> Self:
        """Context manager entry point."""
        if self.shutdown_event.is_set():
            raise RuntimeError("Executor is shutting down")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self.shutdown()
