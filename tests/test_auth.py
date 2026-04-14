"""Tests for the Basic Auth middleware.

conftest.py sets ALLOW_UNAUTH=1 globally so other tests can hit endpoints
without auth headers. These tests flip it off (via monkeypatch so the
original value is restored after each test) and verify middleware behavior.

Uses `/api/nonexistent` as the canary protected path so the tests stay
free of route-specific mocking: if auth rejects, we see 401; if auth
passes, we see 404.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from credrank.api import app

client = TestClient(app)

TEST_USER = "tester"
TEST_PASS = "hunter2-super-secret"


def _basic(user: str, pwd: str) -> str:
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


@pytest.fixture
def auth_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_UNAUTH", "0")
    monkeypatch.setenv("BASIC_AUTH_USERNAME", TEST_USER)
    monkeypatch.setenv("BASIC_AUTH_PASSWORD", TEST_PASS)
    yield


# ── /health stays public in both modes ──


def test_health_public_when_unauth():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_public_when_auth_enabled(auth_enabled):
    resp = client.get("/health")
    assert resp.status_code == 200


# ── Protected paths gate on credentials ──


def test_protected_path_rejects_no_credentials(auth_enabled):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate", "").lower().startswith("basic")


def test_protected_path_rejects_wrong_password(auth_enabled):
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": _basic(TEST_USER, "wrong_password")},
    )
    assert resp.status_code == 401


def test_protected_path_rejects_wrong_username(auth_enabled):
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": _basic("wrong_user", TEST_PASS)},
    )
    assert resp.status_code == 401


def test_protected_path_accepts_correct_credentials(auth_enabled):
    # Auth passes → router returns 404 (no such route). 404 here is the
    # success signal: it means we got past the middleware.
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": _basic(TEST_USER, TEST_PASS)},
    )
    assert resp.status_code == 404


# ── Malformed auth headers all bounce to 401 ──


def test_rejects_non_basic_scheme(auth_enabled):
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": "Bearer sometoken"},
    )
    assert resp.status_code == 401


def test_rejects_malformed_base64(auth_enabled):
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": "Basic !@#$not-base64"},
    )
    assert resp.status_code == 401


def test_rejects_base64_without_colon(auth_enabled):
    bad = base64.b64encode(b"notacolonpair").decode("ascii")
    resp = client.get(
        "/api/nonexistent",
        headers={"Authorization": f"Basic {bad}"},
    )
    assert resp.status_code == 401


# ── ALLOW_UNAUTH=1 bypasses everything ──


def test_allow_unauth_lets_requests_through(monkeypatch):
    monkeypatch.setenv("ALLOW_UNAUTH", "1")
    resp = client.get("/api/nonexistent")
    # 404 from the router — auth was bypassed, no 401.
    assert resp.status_code == 404
