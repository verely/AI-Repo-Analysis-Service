from pydantic import BaseModel

"""Internal payload from GitHub used to build the LLM prompt."""


class RepoMetadata(BaseModel):
    full_name: str
    description: str | None
    language: str | None
    stars: int
    topics: list[str]
    dependency_files: list[str]


"""Response for POST /summarize."""

class SummarizeResponse(BaseModel):
    summary: str                  # human-readable project description
    technologies: list[str]       # detected tech stack / dependencies
    structure: str                # description of project layout
