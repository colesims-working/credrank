"""HTTP Basic Auth middleware for CredRank.

Gates all requests except `/health`. Behavior is controlled at runtime via env:

- `ALLOW_UNAUTH=1` disables auth entirely (for local dev and tests).
- Otherwise `BASIC_AUTH_PASSWORD` must be set; `BASIC_AUTH_USERNAME` is optional
  (defaults to `admin`). Credentials are compared with `secrets.compare_digest`
  to avoid timing leaks.

`validate_auth_config()` is called at app startup so a misconfigured deploy
fails fast instead of silently serving the world.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_UNPROTECTED_PATHS = {"/health"}
_TRUTHY = {"1", "true", "yes", "on"}


def _auth_disabled() -> bool:
    return os.environ.get("ALLOW_UNAUTH", "").strip().lower() in _TRUTHY


def _expected_credentials() -> tuple[str, str] | None:
    password = os.environ.get("BASIC_AUTH_PASSWORD")
    if not password:
        return None
    username = os.environ.get("BASIC_AUTH_USERNAME") or "admin"
    return username, password


def validate_auth_config() -> None:
    """Raise at startup if the app would be exposed without credentials."""
    if _auth_disabled():
        return
    if _expected_credentials() is None:
        raise RuntimeError(
            "Authentication is required but BASIC_AUTH_PASSWORD is not set. "
            "Set BASIC_AUTH_PASSWORD (and optionally BASIC_AUTH_USERNAME, "
            "default 'admin'), or set ALLOW_UNAUTH=1 for local/dev use."
        )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Enforces HTTP Basic Auth on every request except `/health`."""

    async def dispatch(self, request: Request, call_next):
        if _auth_disabled() or request.url.path in _UNPROTECTED_PATHS:
            return await call_next(request)

        creds = _expected_credentials()
        if creds is None:
            return _unauthorized()

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("basic "):
            return _unauthorized()

        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return _unauthorized()

        username, sep, password = decoded.partition(":")
        if not sep:
            return _unauthorized()

        expected_user, expected_pass = creds
        user_ok = secrets.compare_digest(
            username.encode("utf-8"), expected_user.encode("utf-8")
        )
        pass_ok = secrets.compare_digest(
            password.encode("utf-8"), expected_pass.encode("utf-8")
        )
        if not (user_ok and pass_ok):
            return _unauthorized()

        return await call_next(request)


def _unauthorized() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="credrank", charset="UTF-8"'},
    )
