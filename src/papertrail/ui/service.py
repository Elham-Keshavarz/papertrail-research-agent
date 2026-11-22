"""Pure, testable UI orchestration. This module does not import Streamlit.

The Streamlit layer must not quietly turn on LLM or network access. A local
paper stays local in both Preview and Live modes; Live only enables the
explicit model-generation call. Experimental query styles remain Preview-only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from uuid import uuid4

from papertrail.agent.faithfulness import (
    FaithfulnessError, _read_json, save_audit_queue, summarize_manual_review,
)
from papertrail.agent.synthesis import OpenAICompatibleAnswerGenerator
from papertrail.agent.tool_execution import ExecutionOptions, PaperTrailTools, run_agent_research
from papertrail.config import Settings
from papertrail.discovery.arxiv_client import ArxivClient
from papertrail.ingestion.pdf_downloader import validate_paper_id
from papertrail.storage import atomic_write_bytes


class UIServiceError(ValueError):
    """Safe, user-facing validation error (never contains a credential)."""


@dataclass(frozen=True)
class LocalPaper:
    paper_id: str
    extracted: bool
    chunks: bool
    embeddings: bool
    index: bool

    @property
    def ready(self) -> bool:
        return self.chunks and self.embeddings and self.index


def paper_artifacts(paper_id: str, data_dir: Path) -> LocalPaper:
    paper_id = validate_paper_id(paper_id)
    name = paper_id.replace('/', '__')
    root = Path(data_dir)
    return LocalPaper(
        paper_id=paper_id,
        extracted=(root / 'extracted' / f'{name}.json').is_file(),
        chunks=(root / 'chunks' / f'{name}.json').is_file(),
        embeddings=(root / 'embeddings' / f'{name}.npy').is_file() and
                   (root / 'embeddings' / f'{name}.manifest.json').is_file(),
        index=(root / 'indexes' / f'{name}.faiss').is_file() and
              (root / 'indexes' / f'{name}.manifest.json').is_file(),
    )


def list_local_papers(data_dir: Path) -> list[LocalPaper]:
    """List known local papers without parsing untrusted cached document bodies."""
    root = Path(data_dir)
    stems = set()
    for directory in ('chunks', 'extracted'):
        stems.update(p.stem for p in (root / directory).glob('*.json') if p.is_file())
    stems.update(p.stem for p in (root / 'indexes').glob('*.faiss') if p.is_file())
    found = []
    for stem in sorted(stems):
        paper_id = stem.replace('__', '/')
        try:
            valid_id = validate_paper_id(paper_id)
        except (ValueError, TypeError, AttributeError):
            continue
        if valid_id.replace('/', '__') != stem:
            continue
        found.append(paper_artifacts(valid_id, root))
    return found


def discover_papers(search_terms: str, *, settings: Settings, consent: bool) -> list[dict]:
    if not consent:
        raise UIServiceError('Explicit network consent is required to search arXiv.')
    query = ' '.join(search_terms.split())
    if not 3 <= len(query) <= 150:
        raise UIServiceError('arXiv query must contain between 3 and 150 characters.')
    with ArxivClient(timeout_seconds=settings.request_timeout_seconds,
                     max_retries=settings.max_retries) as client:
        response = client.search(query, max_results=min(settings.max_papers, 5))
    return [
        {'paper_id': validate_paper_id(p.paper_id), 'title': p.title[:300],
         'authors': ', '.join(p.authors[:3]), 'url': f'https://arxiv.org/abs/{p.paper_id}'}
        for p in response.papers
    ]


def prepare_paper(paper_id: str, *, settings: Settings, consent: bool,
                  on_progress: Callable[[int, str], None] | None = None) -> LocalPaper:
    """Import one arXiv paper. The network/model-download consent is not implicit."""
    if not consent:
        raise UIServiceError('Explicit network/model-download consent is required to import papers.')
    paper_id = validate_paper_id(paper_id)
    options = ExecutionOptions(
        data_dir=settings.data_dir, allow_network=True, max_papers=settings.max_papers,
        max_pdf_pages=settings.max_pdf_pages, max_retries=settings.max_retries,
        timeout_seconds=settings.request_timeout_seconds,
    )
    tools = PaperTrailTools()
    def report(percent: int, stage: str) -> None:
        if on_progress is not None:
            on_progress(percent, stage)

    present = paper_artifacts(paper_id, settings.data_dir)
    report(0, 'Preparing paper import')
    if not present.extracted and not present.chunks:
        tools.ingest(paper_id, options)
    report(25, 'PDF processed or existing extraction reused')
    if not present.chunks:
        tools.chunk(paper_id, options)
    report(50, 'Text chunks prepared')
    if not present.embeddings:
        tools.embed(paper_id, options)
    report(80, 'Local embeddings prepared')
    tools.index(paper_id, options)
    report(95, 'FAISS index built or validated')
    ready = paper_artifacts(paper_id, settings.data_dir)
    if not ready.ready:
        raise UIServiceError('Preparation finished but one or more local artifacts are missing.')
    report(100, 'Paper import completed')
    return ready


def run_question(
    *, settings: Settings, paper_id: str, question: str,
    mode: str = 'preview', query_style: str = 'baseline', top_k: int = 3,
    live_confirmed: bool = False,
    on_progress: Callable[[int, str], None] | None = None,
) -> dict:
    """Run the actual LangGraph research/synthesis pipeline; never fake claims."""
    paper_id = validate_paper_id(paper_id)
    if mode not in {'preview', 'live'}:
        raise UIServiceError('Mode must be preview or live.')
    if query_style not in {'baseline', 'focused', 'multi', 'budgeted'}:
        raise UIServiceError('Unknown retrieval style.')
    if mode == 'live' and query_style != 'baseline':
        raise UIServiceError('Experimental query styles are Preview-only.')
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise UIServiceError('Top-k must be an integer between 1 and 10.')
    from papertrail.models import ResearchRequest
    request = ResearchRequest(question=question, max_papers=settings.max_papers)
    if not paper_artifacts(paper_id, settings.data_dir).ready:
        raise UIServiceError('This paper is not locally indexed. Import it or copy the previous data directory.')
    generator = None
    if mode == 'live':
        if not live_confirmed:
            raise UIServiceError('Explicit approval is required before the paid-capable model request.')
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
            raise UIServiceError('OPENAI_API_KEY must be configured in project-root .env for Live mode.')
        generator = OpenAICompatibleAnswerGenerator(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.model_name, base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
    options = ExecutionOptions(
        data_dir=settings.data_dir, allow_network=False, auto_select_first=False,
        top_k=top_k, context_window=1, max_tool_calls=12,
        max_papers=settings.max_papers, max_pdf_pages=settings.max_pdf_pages,
        max_retries=settings.max_retries, timeout_seconds=settings.request_timeout_seconds,
        min_hits_per_step=2, max_refinement_rounds=1, query_style=query_style,
    )
    return run_agent_research(
        request.question, paper_id=paper_id, options=options,
        planner_mode='deterministic', synthesize_mode=True,
        synthesis_provider=generator,
        on_progress=on_progress,
    )


def new_live_review_queue(result: dict, *, data_dir: Path) -> Path:
    """Persist the *already verified structurally* live claims for manual review.

    Uses the Phase 4.3 in-graph audit, NOT a re-validation against a fabricated
    evidence register. Does not claim PDF provenance or semantic entailment.
    """
    if result.get('generation_mode') != 'live' or not result.get('answer_generated'):
        raise UIServiceError('A generated Live answer is required for manual claim review.')
    synthesis = result.get('synthesis', {})
    audit = synthesis.get('faithfulness_audit')
    claims = synthesis.get('claims')
    if not isinstance(audit, dict) or not isinstance(claims, list):
        raise UIServiceError('No validated faithfulness audit is attached.')
    rows = audit.get('claims')
    if (not isinstance(rows, list) or len(rows) != len(claims) or
            len(rows) != audit.get('accepted_claim_count') or len(rows) > 9):
        raise UIServiceError('Inconsistent human-review metadata.')
    queue = []
    for pos, (row, claim) in enumerate(zip(rows, claims), start=1):
        if (row.get('claim_index') != pos or claim.get('step_id') != row.get('step_id') or
                not isinstance(claim.get('claim'), str) or not isinstance(claim.get('citations'), list) or
                [c.get('evidence_id') for c in claim['citations']] != row.get('evidence_ids')):
            raise UIServiceError('Review claim/citation identity mismatch.')
        queue.append({
            'claim_index': pos, 'step_id': row['step_id'], 'claim': claim['claim'],
            'citations': claim['citations'], 'mechanical_check': row['mechanical_check'],
            'reviewer_decision': 'pending', 'reviewer_note': '',
        })
    report = {
        'status': audit['status'], 'citation_validation_status': synthesis['status'],
        'audit': audit, 'rejections': synthesis.get('rejection_reasons', []),
        'review_queue': queue, 'annotation_status': 'unreviewed_live_agent_result',
        'llm_calls': result.get('llm_calls', 0),
        'network_opt_in': False,
        'warning': 'Structural quote checking is NOT PDF provenance or semantic confirmation.',
    }
    return save_audit_queue(report, Path(data_dir), f'ui_{uuid4().hex[:20]}')


def available_review_queues(data_dir: Path) -> list[Path]:
    base = Path(data_dir) / 'faithfulness_reviews'
    return sorted((p for p in base.glob('*_review_queue.json') if p.is_file() and not p.is_symlink()),
                  key=lambda p: p.name, reverse=True)


def _safe_queue_path(review_file: Path, data_dir: Path) -> Path:
    base = (Path(data_dir) / 'faithfulness_reviews').resolve()
    target = Path(review_file)
    if target.is_symlink() or target.parent.resolve() != base or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}_review_queue\.json', target.name):
        raise UIServiceError('Review file must be an existing local PaperTrail queue.')
    if not target.is_file():
        raise UIServiceError('Review file not found.')
    return target


def read_review_queue(review_file: Path, *, data_dir: Path) -> dict:
    return _read_json(_safe_queue_path(review_file, data_dir))


def update_review_decision(
    review_file: Path, *, data_dir: Path, claim_index: int,
    decision: str, note: str,
) -> dict:
    path = _safe_queue_path(review_file, data_dir)
    if decision not in {'pending', 'supported', 'unsupported', 'insufficient_context'}:
        raise UIServiceError('Unsupported human review decision.')
    if type(claim_index) is not int or not isinstance(note, str) or len(note) > 2_000:
        raise UIServiceError('Invalid claim index or review note.')
    report = _read_json(path)
    if not isinstance(report, dict) or not isinstance(report.get('audit'), dict):
        raise UIServiceError('Invalid review queue metadata.')
    rows = report.get('review_queue')
    if not isinstance(rows, list) or claim_index not in range(1, len(rows) + 1):
        raise UIServiceError('Claim not present in review queue.')
    if rows[claim_index - 1].get('claim_index') != claim_index:
        raise UIServiceError('Inconsistent review queue order.')
    report['review_queue'][claim_index - 1]['reviewer_decision'] = decision
    report['review_queue'][claim_index - 1]['reviewer_note'] = note
    # Validate the complete proposed report BEFORE persisting any change.
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if len(payload.encode('utf-8')) > 2_000_000:
        raise UIServiceError('Review queue exceeds local size limit.')
    # Validation shares the same schema as CLI review summary.
    # Avoid modifying the destination during validation by validating fields here.
    if (len(rows) > 9 or len(rows) != report.get('audit', {}).get('accepted_claim_count') or
            any(r.get('claim_index') != i or
                r.get('reviewer_decision') not in {'pending', 'supported', 'unsupported', 'insufficient_context'} or
                not isinstance(r.get('reviewer_note'), str) or len(r.get('reviewer_note')) > 2_000
                for i, r in enumerate(rows, 1))):
        raise UIServiceError('Review file is not internally consistent.')
    atomic_write_bytes(path, payload.encode('utf-8'))
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return summarize_manual_review(path)
