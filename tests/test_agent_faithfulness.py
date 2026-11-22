"""Phase 4.3: mechanical quote claims must never masquerade as semantic proof."""
import json
from pathlib import Path

import pytest

from papertrail.agent.faithfulness import (
    FaithfulnessError, _read_json, audit_bundle, audit_checked_answer,
    load_faithfulness_suite, run_faithfulness_suite,
    save_audit_queue, save_faithfulness_suite_report,
)
from papertrail.agent.synthesis import validate_and_render_draft
from papertrail import cli

SUITE_PATH = Path(__file__).resolve().parents[1] / 'benchmarks/faithfulness_synthetic_v1.json'
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / 'benchmarks/faithfulness_manual_bundle_example.json'


def test_synthetic_suite_all_expected_guards_pass():
    report = run_faithfulness_suite(load_faithfulness_suite(SUITE_PATH))
    assert report['case_count'] == 10
    assert report['checks_passed'] == 10
    assert report['checks_failed'] == 0
    assert report['semantic_entailment_verified'] is False
    assert report['independently_labeled_ground_truth'] is False


def test_quote_anchored_paraphrase_still_needs_review():
    bundle = _read_json(EXAMPLE_PATH)
    report = audit_bundle(bundle)
    assert report['audit']['accepted_claim_count'] == 2
    assert report['audit']['verbatim_within_quote_count'] == 1
    assert report['audit']['quote_anchored_nonverbatim_count'] == 1
    assert report['audit']['claims'][1]['semantic_support'] == 'not_evaluated'
    assert all(row['reviewer_decision'] == 'pending' for row in report['review_queue'])
    assert report['audit']['independent_pdf_provenance_verified'] is False


def test_negation_trap_remains_semantically_unverified():
    suite = load_faithfulness_suite(SUITE_PATH)
    case = next(x for x in suite['cases'] if x['case_id'] == 'negation_inversion_not_detected')
    result = audit_bundle(case['bundle'])
    assert result['audit']['status'] == 'needs_human_semantic_review'
    assert result['audit']['claims'][0]['mechanical_check'] == 'quote_anchored_nonverbatim'
    assert result['audit']['semantic_entailment_verified'] is False


def test_forged_quote_and_unsupported_numbers_are_rejected():
    suite = load_faithfulness_suite(SUITE_PATH)
    for name in ('invented_quote_rejected', 'unsupported_number_rejected', 'cross_facet_rejected'):
        case = next(x for x in suite['cases'] if x['case_id'] == name)
        result = audit_bundle(case['bundle'])
        assert result['audit']['accepted_claim_count'] == 0
        assert len(result['rejections']) == 1
        assert result['review_queue'] == []


def test_per_claim_recovery_does_not_promote_rejected_text():
    suite = load_faithfulness_suite(SUITE_PATH)
    case = next(x for x in suite['cases'] if x['case_id'] == 'partial_recovery')
    report = audit_bundle(case['bundle'])
    assert report['audit']['accepted_claim_count'] == 1
    assert len(report['rejections']) == 1
    assert len(report['review_queue']) == 1
    assert '95 percent' not in json.dumps(report['review_queue'])


def test_auditor_fails_closed_on_changed_quote_or_citation():
    bundle = _read_json(EXAMPLE_PATH)
    checked = validate_and_render_draft(bundle['draft'], bundle['plan'], bundle['register'])
    checked['claims'][0]['citations'][0]['quote'] = 'A quote that is not in the registered source.'
    with pytest.raises(FaithfulnessError):
        audit_checked_answer(checked, bundle['register'])


def test_auditor_fails_closed_on_changed_page():
    bundle = _read_json(EXAMPLE_PATH)
    checked = validate_and_render_draft(bundle['draft'], bundle['plan'], bundle['register'])
    checked['claims'][0]['citations'][0]['page_number'] += 1
    with pytest.raises(FaithfulnessError):
        audit_checked_answer(checked, bundle['register'])


def test_missing_claims_abstention_has_no_semantic_proof():
    suite = load_faithfulness_suite(SUITE_PATH)
    case = next(x for x in suite['cases'] if x['case_id'] == 'empty_abstention')
    report = audit_bundle(case['bundle'])
    assert report['audit']['status'] == 'no_accepted_claims'
    assert report['audit']['semantic_entailment_verified'] is False


@pytest.mark.parametrize('unsafe', ['../traversal', 'UPPER', 'a.b', 'a/b', ''])
def test_review_names_sanitized(unsafe, tmp_path):
    with pytest.raises(FaithfulnessError):
        save_audit_queue({}, tmp_path, unsafe)


def test_review_queue_created_only_once(tmp_path):
    report = audit_bundle(_read_json(EXAMPLE_PATH))
    p = save_audit_queue(report, tmp_path, 'synthetic_review')
    assert p.exists() and len(json.loads(p.read_text())['review_queue']) == 2
    with pytest.raises(FaithfulnessError):
        save_audit_queue(report, tmp_path, 'synthetic_review')


def test_invalid_bundle_rejected_before_any_audit():
    bundle = _read_json(EXAMPLE_PATH)
    bundle['register'][0]['citation'] = 'https://attacker.example.net/#page=2'
    with pytest.raises(FaithfulnessError):
        audit_bundle(bundle)


def test_bounded_json_reader_rejects_too_large(tmp_path):
    p = tmp_path / 'too_large.json'
    p.write_bytes(b' ' * (2_000_001))
    with pytest.raises(FaithfulnessError):
        _read_json(p)


def test_duplicate_source_id_fails_closed():
    bundle = _read_json(EXAMPLE_PATH)
    bundle['register'].append(dict(bundle['register'][0]))
    with pytest.raises(FaithfulnessError):
        audit_bundle(bundle)


def test_reports_saved_without_real_claim_text(tmp_path):
    report = run_faithfulness_suite(load_faithfulness_suite(SUITE_PATH))
    jp, mp = save_faithfulness_suite_report(report, tmp_path)
    assert 'semantic_entailment_verified' in jp.read_text()
    assert 'synthetic' in mp.read_text()
    assert 'The proposed method combines' not in mp.read_text()


def test_cli_synthetic_evaluation_no_key_no_network(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert cli.main(['agent-faithfulness-evaluate', '--suite', str(SUITE_PATH)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['checks_passed'] == 10 and data['llm_calls'] == 0
    assert 'per_case' not in data


def test_cli_review_file_contains_no_raw_claims_in_stdout(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(['agent-faithfulness-audit', '--input', str(EXAMPLE_PATH), '--name', 'testcase']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['accepted_claims'] == 2
    assert data['semantic_entailment_verified'] is False
    assert 'The proposed method' not in json.dumps(data)
    assert (tmp_path / data['review_queue_file']).exists()


def test_cli_bad_suite_returns_error_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / 'broken.json'
    p.write_text('not json', encoding='utf-8')
    assert cli.main(['agent-faithfulness-evaluate', '--suite', str(p)]) == 2
    err = capsys.readouterr().err
    assert 'Validation error' in err and 'Traceback' not in err


def test_manual_adjudication_counts_not_automatic_support(tmp_path):
    from papertrail.agent.faithfulness import summarize_manual_review
    p = save_audit_queue(audit_bundle(_read_json(EXAMPLE_PATH)), tmp_path, 'human_review')
    data = json.loads(p.read_text(encoding='utf-8'))
    data['review_queue'][0]['reviewer_decision'] = 'supported'
    data['review_queue'][1]['reviewer_decision'] = 'unsupported'
    p.write_text(json.dumps(data), encoding='utf-8')
    summary = summarize_manual_review(p)
    assert (summary['supported'], summary['unsupported'], summary['pending']) == (1, 1, 0)
    assert not summary['semantics_checked_by_software']
    assert not summary['independently_adjudicated']


def test_unknown_manual_verdict_rejected(tmp_path):
    from papertrail.agent.faithfulness import summarize_manual_review
    p = save_audit_queue(audit_bundle(_read_json(EXAMPLE_PATH)), tmp_path, 'decision_test')
    data = json.loads(p.read_text(encoding='utf-8'))
    data['review_queue'][0]['reviewer_decision'] = 'automatically_proven'
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(FaithfulnessError):
        summarize_manual_review(p)


def test_review_summary_cli_no_raw_text(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = save_audit_queue(audit_bundle(_read_json(EXAMPLE_PATH)), tmp_path, 'local')
    assert cli.main(['agent-faithfulness-review-summary', '--review-file', str(p)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['pending'] == 2
    assert 'The proposed method combines' not in json.dumps(data)
