from datetime import datetime
from typing import List

from pydantic import BaseModel, field_validator


class Paper(BaseModel):
    """Schema for a single validated arXiv paper record."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: datetime
    url: str

    @field_validator("title", "abstract")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field cannot be empty")
        return v.strip()

    @field_validator("authors")
    @classmethod
    def at_least_one_author(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("must have at least one author")
        return v