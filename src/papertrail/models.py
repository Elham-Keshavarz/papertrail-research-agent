"""Stable contracts for research requests and evidence (no model calls)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=12, max_length=1000)
    max_papers: int = Field(default=5, ge=1, le=20)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class PaperMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    abstract: str = Field(min_length=1)
    published_at: datetime
    updated_at: datetime | None = None
    url: HttpUrl
    pdf_url: HttpUrl | None = None
    categories: list[str] = Field(default_factory=list)
    journal_reference: str | None = None
    is_peer_reviewed: bool | None = None

    @field_validator("published_at", "updated_at")
    @classmethod
    def dates_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Publication timestamps must include a timezone")
        return value


class PaperSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    start: int = Field(ge=0)
    total_results: int | None = Field(default=None, ge=0)
    papers: list[PaperMetadata]


class EvidencePassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    source_url: HttpUrl


class SupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
