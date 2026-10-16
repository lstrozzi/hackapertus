from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import jwt
import requests

from src.config.constants import BLAZOR_HOSTNAME, BLAZOR_PORT, BLAZOR_PROTOCOL, USE_REAL_SSL_CERTIFICATE
from src.core.base.Shutdown import shutdown_event
from src.utils.log import log, WARNING

RENEWAL_MARGIN_SECONDS: float = 120.0
MINIMUM_RENEWAL_MARGIN_RATIO: float = 0.2
RETRY_DELAY_SECONDS: float = 30.0

_service_token: str = os.getenv("THALABUS_PYTHON_INTERNAL_SERVICE_TOKEN", "").strip()
_token_times: Optional[Tuple[float, float]] = None
_token_lock = threading.RLock()
_renewal_thread: Optional[threading.Thread] = None
_missing_platform_token_warning_logged: bool = False


def _decode_token_times(token: str) -> Optional[Tuple[float, float]]:
    result: Optional[Tuple[float, float]] = None

    try:
        claims: Dict[str, Any] = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
        issued_at: float = float(claims["iat"])
        expires_at: float = float(claims["exp"])
        if expires_at > issued_at:
            result = (issued_at, expires_at)
    except (KeyError, TypeError, ValueError, jwt.PyJWTError):
        result = None

    return result


def _seconds_until_refresh(issued_at: float, expires_at: float, now: float) -> float:
    lifetime: float = max(expires_at - issued_at, 0.0)
    margin: float = min(RENEWAL_MARGIN_SECONDS, lifetime * MINIMUM_RENEWAL_MARGIN_RATIO)
    result: float = max(expires_at - now - margin, 0.0)

    return result


def _refresh_service_token_locked() -> bool:
    global _service_token, _token_times, _missing_platform_token_warning_logged

    platform_token: str = (os.getenv("PLATFORM_API_TOKEN") or os.getenv("THALABUS_TOKEN", "")).strip()
    refreshed: bool = False

    if not platform_token:
        if not _missing_platform_token_warning_logged:
            log(WARNING, "Cannot renew the Python service token: the platform API token is not configured.")
            _missing_platform_token_warning_logged = True
    else:
        api_url: str = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/api/python/service-token"
        headers: Dict[str, str] = {"Authorization": f"Bearer {platform_token}"}
        try:
            response = requests.post(
                api_url,
                headers=headers,
                timeout=5,
                verify=USE_REAL_SSL_CERTIFICATE,
            )
            if response.status_code == 200:
                response_data: Any = response.json()
                if isinstance(response_data, dict):
                    new_token: Any = response_data.get("token")
                    token_expiry: Any = response_data.get("expires_at_unix")
                    decoded_times: Optional[Tuple[float, float]] = _decode_token_times(new_token) if isinstance(new_token, str) else None

                    if decoded_times is not None and isinstance(token_expiry, (int, float)):
                        issued_at: float = decoded_times[0]
                        expires_at: float = decoded_times[1]
                        if abs(expires_at - float(token_expiry)) <= 2.0 and expires_at > time.time():
                            _service_token = new_token
                            _token_times = decoded_times
                            _missing_platform_token_warning_logged = False
                            refreshed = True
                        else:
                            log(WARNING, "The service-token endpoint returned inconsistent or expired token metadata.")
                    else:
                        log(WARNING, "The service-token endpoint returned an invalid token response.")
                else:
                    log(WARNING, "The service-token endpoint returned an invalid token response.")
            else:
                log(WARNING, f"Python service-token renewal returned HTTP {response.status_code}.")
        except (requests.RequestException, ValueError, TypeError) as exception:
            log(WARNING, f"Could not renew the Python service token: {exception}")

    return refreshed


def refresh_service_token() -> bool:
    with _token_lock:
        refreshed: bool = _refresh_service_token_locked()

    return refreshed


def get_service_token() -> str:
    global _token_times

    with _token_lock:
        if _token_times is None and _service_token:
            _token_times = _decode_token_times(_service_token)

        now: float = time.time()
        token_needs_refresh: bool = not _service_token or _token_times is None
        if _token_times is not None:
            token_needs_refresh = token_needs_refresh or _seconds_until_refresh(_token_times[0], _token_times[1], now) <= 0
        if token_needs_refresh:
            _refresh_service_token_locked()
        result: str = _service_token

    return result


def _renewal_loop() -> None:
    while not shutdown_event[0].is_set():
        with _token_lock:
            if _token_times is None and _service_token:
                current_token_times: Optional[Tuple[float, float]] = _decode_token_times(_service_token)
            else:
                current_token_times = _token_times

        delay: float = 0.0
        if current_token_times is not None:
            delay = _seconds_until_refresh(current_token_times[0], current_token_times[1], time.time())

        if delay > 0 and shutdown_event[0].wait(delay):
            break

        if not refresh_service_token() and shutdown_event[0].wait(RETRY_DELAY_SECONDS):
            break

    return


def start_service_token_renewal() -> None:
    global _renewal_thread

    with _token_lock:
        if _renewal_thread is None or not _renewal_thread.is_alive():
            _renewal_thread = threading.Thread(
                target=_renewal_loop,
                name="python-service-token-renewal",
                daemon=True,
            )
            _renewal_thread.start()

    return


def _send_with_token(method: str, url: str, token: str, request_options: Dict[str, Any]) -> Any:
    options: Dict[str, Any] = dict(request_options)
    headers: Dict[str, str] = dict(options.pop("headers", {}) or {})
    headers["Authorization"] = f"Bearer {token}"
    options["headers"] = headers
    response: Any = requests.request(method, url, **options)

    return response


def request_with_service_token(method: str, url: str, **request_options: Any) -> Any:
    token: str = get_service_token()
    response: Any = _send_with_token(method, url, token, request_options)

    if response.status_code == 401:
        previous_token: str = token
        if refresh_service_token():
            token = get_service_token()
            if token and token != previous_token:
                response = _send_with_token(method, url, token, request_options)

    return response


log_end_file(__file__)