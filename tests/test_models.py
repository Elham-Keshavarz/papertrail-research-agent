import pytest
from pydantic import ValidationError

from papertrail.models import EvidencePassage, ResearchRequest, SupportedClaim


def test_normalizes_question():
    request = ResearchRequest(question="  How do  RAG approaches\nreduce hallucinations?  ")
    assert request.question == "How do RAG approaches reduce hallucinations?"


@pytest.mark.parametrize("question", ["", "   ", "short", "x" * 1001])
def test_rejects_invalid_questions(question):
    with pytest.raises(ValidationError):
        ResearchRequest(question=question)


def test_rejects_unexpected_request_fields():
    with pytest.raises(ValidationError):
        ResearchRequest(question="Which papers study test-time scaling?", extra_param="unexpected")


def test_evidence_page_must_be_positive():
    with pytest.raises(ValidationError):
        EvidencePassage(
            evidence_id="E1", paper_id="2410.00001", text="example",
            source_url="https://arxiv.org/abs/2410.00001", page_number=0,
        )


def test_claim_requires_evidence():
    with pytest.raises(ValidationError):
        SupportedClaim(claim="Accuracy improves", evidence_ids=[])
