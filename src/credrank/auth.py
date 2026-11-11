"""HTTP Basic Auth middleware for CredRank, with a session cookie on top.

Gates all requests except `/health`. Behavior is controlled at runtime via env:

- `ALLOW_UNAUTH=1` disables auth entirely (for local dev and tests).
- Otherwise `BASIC_AUTH_PASSWORD` must be set; `BASIC_AUTH_USERNAME` is optional
  (defaults to `admin`). Credentials are compared with `secrets.compare_digest`
  to avoid timing leaks.

After the first successful Basic Auth, the middleware sets a signed,
HttpOnly session cookie (`credrank_session`, HMAC-SHA256 over an expiry
timestamp) and accepts it on later requests — so returning visitors are
not re-prompted for credentials every session. The signing key is
derived from the configured credentials, which means rotating the
password invalidates all outstanding sessions. Lifetime defaults to 30
days; tune with `CREDRANK_SESSION_TTL_S` (0 disables cookie sessions).

`validate_auth_config()` is called at app startup so a misconfigured deploy
fails fast instead of silently serving the world.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_UNPROTECTED_PATHS = {"/health"}
_TRUTHY = {"1", "true", "yes", "on"}

_SESSION_COOKIE = "credrank_session"
_DEFAULT_SESSION_TTL_S = 30 * 24 * 3600


def _session_ttl_s() -> int:
    raw = os.environ.get("CREDRANK_SESSION_TTL_S", "").strip()
    if not raw:
        return _DEFAULT_SESSION_TTL_S
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SESSION_TTL_S


def _signing_key(creds: tuple[str, str]) -> bytes:
    username, password = creds
    return hashlib.sha256(f"credrank|{username}|{password}".encode()).digest()


def _make_session_token(creds: tuple[str, str], now: float | None = None) -> str:
    expires_at = int(now if now is not None else time.time()) + _session_ttl_s()
    sig = hmac.new(
        _signing_key(creds), str(expires_at).encode(), hashlib.sha256
    ).hexdigest()
    return f"{expires_at}.{sig}"


def _session_token_valid(token: str, creds: tuple[str, str]) -> bool:
    expires_str, sep, sig = token.partition(".")
    if not sep or not expires_str.isdigit():
        return False
    expected = hmac.new(
        _signing_key(creds), expires_str.encode(), hashlib.sha256
    ).hexdigest()
    if not secrets.compare_digest(sig, expected):
        return False
    return int(expires_str) > time.time()


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
    """Enforces HTTP Basic Auth (or a valid session cookie) on every
    request except `/health`."""

    async def dispatch(self, request: Request, call_next):
        if _auth_disabled() or request.url.path in _UNPROTECTED_PATHS:
            return await call_next(request)

        creds = _expected_credentials()
        if creds is None:
            return _unauthorized()

        # A valid session cookie from a previous successful login is
        # enough — no Basic Auth prompt on return visits.
        cookie = request.cookies.get(_SESSION_COOKIE, "")
        if cookie and _session_token_valid(cookie, creds):
            return await call_next(request)

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

        response = await call_next(request)
        ttl = _session_ttl_s()
        if ttl > 0:
            forwarded_proto = request.headers.get("x-forwarded-proto", "")
            response.set_cookie(
                _SESSION_COOKIE,
                _make_session_token(creds),
                max_age=ttl,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https"
                or forwarded_proto == "https",
            )
        return response


def _unauthorized() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="credrank", charset="UTF-8"'},
    )
