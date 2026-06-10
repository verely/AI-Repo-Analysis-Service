from pydantic import BaseModel, field_validator
from typing import Literal

"""Request body for POST /summarize."""


class SummarizeRequest(BaseModel):
    github_url: str  # spec-required name
    # mode: Literal["quick", "deep"] = "quick"  # optional

    @field_validator("github_url")
    @classmethod
    def must_be_github(cls, v: str) -> str:
        v = v.strip()
        if "github.com" not in v:
            raise ValueError("URL must point to a GitHub repository.")
        return v
