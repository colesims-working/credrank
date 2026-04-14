"""Pytest configuration.

Disables Basic Auth before any test module imports `credrank.api`, so
tests hit endpoints directly without needing auth headers. Must set the
env var here (at conftest load time) rather than inside fixtures — the
TestClient imports `app` at module load, which runs `validate_auth_config`.
"""

import os

os.environ.setdefault("ALLOW_UNAUTH", "1")
