from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import secrets
from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = get_settings().api_key
    if (
        not key
        or not expected
        or not secrets.compare_digest(key.encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
