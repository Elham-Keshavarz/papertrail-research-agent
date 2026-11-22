"""Phase 4.3: auditable quote grounding and explicit semantic-review queue.

An exact source quote or extractive claim is not an entailment judgment.
The external source itself may be misleading or unverified. Everything here is
strictly local, deterministic and independent of LLM, FAISS and LangGraph.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .synthesis import (_compact, _numbers, validate_with_partial_recovery)


class FaithfulnessError(ValueError):
    """Bad local audit bundle or inconsistent already-validated citations."""


MAX_BUNDLE_BYTES = 2_000_000
SAFE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES:
        raise FaithfulnessError('Audit file missing or exceeds 2 MB')
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FaithfulnessError('Invalid UTF-8 JSON audit file') from exc


def _validate_bundle(bundle: Any) -> tuple[list[dict], list[dict], dict]:
    if not isinstance(bundle, dict) or set(bundle) != {'plan', 'register', 'draft'}:
        raise FaithfulnessError('Bundle needs exactly plan, register and draft')
    plan, register, draft = bundle['plan'], bundle['register'], bundle['draft']
    if (not isinstance(plan, list) or len(plan) > 20 or
            not isinstance(register, list) or len(register) > 12 or
            not isinstance(draft, dict) or len(json.dumps(draft, ensure_ascii=False)) > 100_000):
        raise FaithfulnessError('Invalid or oversized audit bundle')
    for step in plan:
        if not isinstance(step, dict) or not isinstance(step.get('id'), str) or not isinstance(step.get('proposed_tool'), str):
            raise FaithfulnessError('Invalid plan step')
    seen_ids: set[str] = set()
    for source in register:
        if not isinstance(source, dict) or not {'evidence_id', 'step_id', 'chunk_id', 'page_number', 'citation', 'passages'} <= set(source):
            raise FaithfulnessError('Invalid evidence register entry')
        rid = source['evidence_id']
        if (not isinstance(rid, str) or not re.fullmatch(r'E\d{2}', rid) or rid in seen_ids or
                not isinstance(source['step_id'], str) or
                not isinstance(source['page_number'], int) or isinstance(source['page_number'], bool) or
                source['page_number'] < 1 or not isinstance(source['chunk_id'], str) or
                len(source['chunk_id']) > 300 or not isinstance(source['citation'], str) or
                not isinstance(source['passages'], list) or not 1 <= len(source['passages']) <= 3 or
                any(not isinstance(p, str) or len(p) > 1_000 for p in source['passages'])):
            raise FaithfulnessError('Invalid evidence register metadata')
        # This is only an internal-consistency check, not verification against a PDF.
        url = source['citation']
        if not re.fullmatch(r'https://arxiv\.org/abs/\d{4}\.\d{4,5}(?:v\d+)?#page=\d+', url) or not url.endswith(f"#page={source['page_number']}"):
            raise FaithfulnessError('Invalid arXiv source citation')
        seen_ids.add(rid)
    if not isinstance(draft.get('claims'), list) or len(draft['claims']) > 9:
        raise FaithfulnessError('Draft needs bounded claims')
    return plan, register, draft


def audit_checked_answer(reviewed: dict, register: list[dict]) -> dict:
    """Recheck source-register consistency and classify literal vs paraphrase.

    Does NOT infer whether a claim is semantically entailed, even if extractive.
    Raises on corrupt/mismatched citations so graph can fail closed.
    """
    sources = {r['evidence_id']: r for r in register}
    rows: list[dict] = []
    for i, claim in enumerate(reviewed.get('claims', []), start=1):
        claim_text = claim['claim']
        quotes: list[str] = []
        ids: list[str] = []
        for citation in claim['citations']:
            rid = citation['evidence_id']
            src = sources.get(rid)
            if (src is None or src['step_id'] != claim['step_id'] or
                    src['citation'] != citation['citation'] or
                    src['page_number'] != citation['page_number'] or
                    not any(_compact(citation['quote']) in _compact(p) for p in src['passages'])):
                raise FaithfulnessError('Citation no longer matches registered source')
            quotes.append(citation['quote'])
            ids.append(rid)
        if not _numbers(claim_text).issubset(set().union(*(_numbers(q) for q in quotes))):
            raise FaithfulnessError('Claim numbers no longer grounded in quotes')
        # Stronger *literal* test, NOT semantic validation. No token-overlap
        # confidence score, which is unsafe for negation and contradictions.
        literal = any(_compact(claim_text) in _compact(quote) for quote in quotes)
        rows.append({
            'claim_index': i,
            'step_id': claim['step_id'],
            'evidence_ids': ids,
            'mechanical_check': 'verbatim_within_quote' if literal else 'quote_anchored_nonverbatim',
            'semantic_support': 'not_evaluated',
            'human_review_required': True,
        })
    counts = Counter(row['mechanical_check'] for row in rows)
    return {
        'status': 'needs_human_semantic_review' if rows else 'no_accepted_claims',
        'accepted_claim_count': len(rows),
        'verbatim_within_quote_count': counts['verbatim_within_quote'],
        'quote_anchored_nonverbatim_count': counts['quote_anchored_nonverbatim'],
        'semantic_entailment_verified': False,
        'independent_pdf_provenance_verified': False,
        'human_review_required': True,
        'claims': rows,
        'interpretation': ('Citation identity and text anchoring were checked within the registered evidence. '
                           'A verbatim match does not independently validate the PDF or scientific truth. '
                           'Nonverbatim claims require manual semantic entailment review.'),
    }


def audit_bundle(bundle: Any) -> dict:
    """Audit a local manually provided bundle, with strict per-claim recovery."""
    plan, register, draft = _validate_bundle(bundle)
    checked = validate_with_partial_recovery(draft, plan, register)
    reviewed = checked['reviewed']
    audit = audit_checked_answer(reviewed, register) if reviewed is not None else {
        'status': 'no_accepted_claims', 'accepted_claim_count': 0,
        'verbatim_within_quote_count': 0, 'quote_anchored_nonverbatim_count': 0,
        'semantic_entailment_verified': False, 'independent_pdf_provenance_verified': False,
        'human_review_required': True, 'claims': [],
    }
    # Locally stored queue intentionally contains source quotes and claim text
    # to make human judgment possible. Never print these contents to the CLI.
    queue: list[dict] = []
    for row, claim in zip(audit['claims'], reviewed['claims'] if reviewed else []):
        queue.append({
            'claim_index': row['claim_index'], 'step_id': row['step_id'],
            'claim': claim['claim'],
            'citations': claim['citations'],
            'mechanical_check': row['mechanical_check'],
            'reviewer_decision': 'pending',
            'reviewer_note': '',
        })
    return {
        'status': audit['status'],
        'citation_validation_status': reviewed['status'] if reviewed else 'citation_validation_failed',
        'audit': audit,
        'rejections': checked['rejections'],
        'review_queue': queue,
        'annotation_status': 'unreviewed_user_supplied_bundle',
        'llm_calls': 0, 'network_opt_in': False,
        'warning': ('Bundle-level checks do NOT authenticate claims against a stored PDF. '
                    'Human verification of source provenance and semantic support is mandatory.'),
    }


def load_faithfulness_suite(path: Path) -> dict:
    suite = _read_json(path)
    if (not isinstance(suite, dict) or set(suite) != {'suite_id', 'annotation_status', 'cases'} or
            not isinstance(suite['suite_id'], str) or not SAFE_NAME.fullmatch(suite['suite_id']) or
            suite['annotation_status'] != 'synthetic_regression_not_independent' or
            not isinstance(suite['cases'], list) or not 1 <= len(suite['cases']) <= 50):
        raise FaithfulnessError('Invalid synthetic faithfulness suite')
    ids = set()
    for case in suite['cases']:
        if (not isinstance(case, dict) or set(case) != {'case_id', 'bundle', 'expected'} or
                not isinstance(case['case_id'], str) or not SAFE_NAME.fullmatch(case['case_id']) or
                case['case_id'] in ids or not isinstance(case['expected'], dict) or
                set(case['expected']) != {'accepted', 'rejected', 'verbatim'} or
                any(type(case['expected'][k]) is not int or not 0 <= case['expected'][k] <= 9
                    for k in ('accepted', 'rejected', 'verbatim'))):
            raise FaithfulnessError('Invalid synthetic test case')
        _validate_bundle(case['bundle'])
        ids.add(case['case_id'])
    return suite


def run_faithfulness_suite(suite: dict) -> dict:
    rows = []
    for case in suite['cases']:
        actual = audit_bundle(case['bundle'])
        got = {
            'accepted': actual['audit']['accepted_claim_count'],
            'rejected': len(actual['rejections']),
            'verbatim': actual['audit']['verbatim_within_quote_count'],
        }
        rows.append({'case_id': case['case_id'], 'expected': case['expected'],
                     'observed': got, 'passed': got == case['expected'],
                     'semantic_entailment_verified': False})
    passed = sum(r['passed'] for r in rows)
    return {
        'phase': '4.3', 'status': 'synthetic_regression_complete' if passed == len(rows) else 'synthetic_regression_failed',
        'suite_id': suite['suite_id'], 'annotation_status': suite['annotation_status'],
        'case_count': len(rows), 'checks_passed': passed,
        'checks_failed': len(rows) - passed,
        'llm_calls': 0, 'network_opt_in': False,
        'semantic_entailment_verified': False,
        'independently_labeled_ground_truth': False,
        'per_case': rows,
        'interpretation': ('Synthetic validator regression tests only: this does not measure '
                           'real-world faithfulness, semantic entailment accuracy, or retrieval quality.'),
    }


def _private_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit exclusive creation avoids silently overwriting a human review.
    # Atomic replacement is deliberately avoided for potentially sensitive queue files.
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    try:
        path.chmod(0o600)
    except OSError:
        pass


def save_audit_queue(report: dict, data_dir: Path, name: str) -> Path:
    if not SAFE_NAME.fullmatch(name):
        raise FaithfulnessError('Audit name must be lowercase alphanumeric, hyphen or underscore')
    output = Path(data_dir) / 'faithfulness_reviews' / f'{name}_review_queue.json'
    try:
        _private_json(output, report)
    except FileExistsError as exc:
        raise FaithfulnessError('Review queue already exists; choose a new --name') from exc
    except OSError as exc:
        raise FaithfulnessError('Could not write local review queue') from exc
    return output


def save_faithfulness_suite_report(report: dict, data_dir: Path) -> tuple[Path, Path]:
    base = Path(data_dir) / 'agent_evaluations'
    base.mkdir(parents=True, exist_ok=True)
    jpath = base / f"{report['suite_id']}_phase4_3_faithfulness.json"
    mpath = base / f"{report['suite_id']}_phase4_3_faithfulness.md"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Phase 4.3 — Faithfulness guard regression (synthetic only)', '',
             '**Not an empirical faithfulness/entailment benchmark.**', '',
             '| Test case | Accepted | Rejected | Verbatim | Pass |',
             '|---|---:|---:|---:|---|']
    for row in report['per_case']:
        x = row['observed']
        lines.append(f"| {row['case_id']} | {x['accepted']} | {x['rejected']} | {x['verbatim']} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ['', report['interpretation'], '',
              'Every accepted claim still needs human review of meaning and source provenance.']
    mpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return jpath, mpath


def summarize_manual_review(review_file: Path) -> dict:
    """Count *user-entered*, unverified decisions; never equate them with gold labels."""
    report = _read_json(review_file)
    if not isinstance(report, dict) or not isinstance(report.get('review_queue'), list) or not isinstance(report.get('audit'), dict):
        raise FaithfulnessError('Invalid human review queue file')
    rows = report['review_queue']
    if len(rows) > 9 or len(rows) != report['audit'].get('accepted_claim_count'):
        raise FaithfulnessError('Human review queue count mismatch')
    counts = Counter()
    allowed = {'pending', 'supported', 'unsupported', 'insufficient_context'}
    for i, row in enumerate(rows, start=1):
        if (not isinstance(row, dict) or row.get('claim_index') != i or
                row.get('reviewer_decision') not in allowed or
                not isinstance(row.get('reviewer_note'), str) or
                len(row['reviewer_note']) > 2_000):
            raise FaithfulnessError('Invalid reviewer decision or note')
        counts[row['reviewer_decision']] += 1
    return {
        'phase': '4.3', 'status': 'user_supplied_review_decisions_summarized',
        'total_claims': len(rows),
        'pending': counts['pending'], 'supported': counts['supported'],
        'unsupported': counts['unsupported'],
        'insufficient_context': counts['insufficient_context'],
        'reviewed': len(rows) - counts['pending'],
        'semantics_checked_by_software': False,
        'independent_pdf_provenance_verified': False,
        'independently_adjudicated': False,
        'llm_calls': 0, 'network_opt_in': False,
        'interpretation': ('Counts only decisions entered in this local file; identity, reviewer '
                           'independence, PDF provenance and factual accuracy are not authenticated.'),
    }
