"""Check early package import defaults in a fresh interpreter (no real LLM calls)."""
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CODE = "import json, os; from papertrail.cli_bootstrap import configure_numeric_threads; configure_numeric_threads(); print(json.dumps({k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TOKENIZERS_PARALLELISM')}))"


def _startup(monkeypatch, **environment):
    variables = os.environ.copy()
    variables.update({"PYTHONPATH": str(PROJECT / 'src'),
                      "OPENBLAS_NUM_THREADS": "32", "OMP_NUM_THREADS": "32",
                      "MKL_NUM_THREADS": "16", "NUMEXPR_NUM_THREADS": "16"})
    variables.pop('PAPERTRAIL_MATH_THREADS', None)
    variables.pop('PAPERTRAIL_LIMIT_MATH_THREADS', None)
    variables.pop('TOKENIZERS_PARALLELISM', None)
    variables.update(environment)
    result = subprocess.run([sys.executable, '-c', CODE], cwd=PROJECT, env=variables,
                            capture_output=True, text=True, timeout=20, check=True)
    import json
    return json.loads(result.stdout)


def test_startup_caps_inherited_math_threads(monkeypatch):
    got = _startup(monkeypatch)
    assert {got[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')} == {'1'}
    assert got['TOKENIZERS_PARALLELISM'] == 'false'


def test_startup_supports_bounded_explicit_overrides(monkeypatch):
    got = _startup(monkeypatch, PAPERTRAIL_MATH_THREADS='2', TOKENIZERS_PARALLELISM='true')
    assert {got[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')} == {'2'}
    assert got['TOKENIZERS_PARALLELISM'] == 'true'


def test_startup_opt_out_does_not_touch_inherited_settings(monkeypatch):
    got = _startup(monkeypatch, PAPERTRAIL_LIMIT_MATH_THREADS='0')
    assert [got[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')] == ['32','32','16','16']
    assert got['TOKENIZERS_PARALLELISM'] is None


def test_startup_invalid_thread_override_falls_back_to_one(monkeypatch):
    got = _startup(monkeypatch, PAPERTRAIL_MATH_THREADS='unbounded')
    assert got['OPENBLAS_NUM_THREADS'] == '1'


def test_importing_library_does_not_modify_process_threads():
    variables = os.environ.copy()
    variables.update({"PYTHONPATH": str(PROJECT / 'src'), "OPENBLAS_NUM_THREADS": "19", "OMP_NUM_THREADS": "21"})
    program = "import os, papertrail; assert os.environ['OPENBLAS_NUM_THREADS']=='19'; assert os.environ['OMP_NUM_THREADS']=='21'"
    subprocess.run([sys.executable, '-c', program], cwd=PROJECT, env=variables,
                   capture_output=True, text=True, timeout=20, check=True)


def test_python_m_entrypoint_guard_runs_before_imports(tmp_path):
    import json
    variables = os.environ.copy()
    variables.update({"PYTHONPATH": str(PROJECT / 'src'), "OPENBLAS_NUM_THREADS": "40", "OMP_NUM_THREADS": "40"})
    variables.pop('PAPERTRAIL_LIMIT_MATH_THREADS', None)
    variables.pop('PAPERTRAIL_MATH_THREADS', None)
    completed = subprocess.run([sys.executable, '-m', 'papertrail', 'doctor'], cwd=tmp_path,
                               env=variables, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(completed.stdout)['project'] == 'PaperTrail'


def test_console_script_uses_resource_bootstrap():
    import tomllib
    metadata = tomllib.loads((PROJECT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert metadata['project']['scripts']['papertrail'] == 'papertrail.cli_bootstrap:main'
