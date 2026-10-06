import os

import pytest

# Set fake env vars BEFORE the app is imported, so tests never need real secrets.
os.environ.setdefault("API_KEY", "test-key")


from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the startup/shutdown events.
    with TestClient(app) as c:
        yield c
