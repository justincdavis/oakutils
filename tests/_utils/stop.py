import concurrent
from typing import Callable, Any


def check_method_timout(method: Callable, name: str, timeout=5) -> Any:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(method)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"{name}, timed out after 5 seconds")
    return result
