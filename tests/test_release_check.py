"""Release scanner: independently build tiny repos to test all required-file guards."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('papertrail_check_release', SOURCE / 'tools' / 'check_release.py')
scanner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scanner)


def minimal_repo(tmp_path: Path) -> Path:
    for filename in scanner.REQUIRED_FILES:
        p = tmp_path / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('placeholder', encoding='utf-8')
    for dirname in scanner.REQUIRED_DIRS:
        (tmp_path / dirname).mkdir(exist_ok=True)
    (tmp_path / '.gitignore').write_text('.env\ndata/\n.venv/\n.streamlit/secrets.toml\n', encoding='utf-8')
    (tmp_path / '.env.example').write_text('OPENAI_API_KEY=\n', encoding='utf-8')
    return tmp_path


def test_release_scanner_accepts_minimal_clean_tree(tmp_path):
    assert scanner.run_checks(minimal_repo(tmp_path)) == []


def test_release_scanner_requires_readme_and_docs(tmp_path):
    root = minimal_repo(tmp_path)
    (root / 'README.md').unlink()
    assert 'Missing required release file: README.md' in scanner.run_checks(root)


def test_release_scanner_finds_credential_in_python_file(tmp_path):
    root = minimal_repo(tmp_path)
    (root / 'test_keys.py').write_text('token = "sk-' + 'a' * 30 + '"\n', encoding='utf-8')
    assert any('API credential' in x for x in scanner.run_checks(root))


def test_release_scanner_scans_notebook_source_and_outputs(tmp_path):
    root = minimal_repo(tmp_path)
    file = root / 'notebook.ipynb'
    file.write_text(json.dumps({
        'cells': [{'cell_type': 'code', 'source': ['sk-' + 'X' * 30],
                   'outputs': [{'output_type': 'stream', 'text': ['ghp_' + 'z' * 36]}]}]
    }), encoding='utf-8')
    issues = scanner.run_checks(root)
    assert any('API credential' in x and 'notebook.ipynb' in x for x in issues)
    assert any('GitHub classic token' in x and 'notebook.ipynb' in x for x in issues)


def test_release_scanner_ignores_local_venv_and_data(tmp_path):
    root = minimal_repo(tmp_path)
    for folder in ('.venv', 'data', 'outputs'):
        p = root / folder / 'x.py'
        p.parent.mkdir()
        p.write_text('sk-' + 'x' * 30, encoding='utf-8')
    assert scanner.run_checks(root) == []


def test_release_scanner_detects_env_and_private_key(tmp_path):
    root = minimal_repo(tmp_path)
    (root / '.env').write_text('OPENAI_API_KEY=fake\n', encoding='utf-8')
    (root / 'key.pem').write_text('test', encoding='utf-8')
    issues = scanner.run_checks(root)
    assert any('.env' in issue and 'Unsafe file' in issue for issue in issues)
    assert any('key.pem' in issue and 'Unsafe file' in issue for issue in issues)


def test_release_scanner_rejects_secret_in_env_example(tmp_path):
    root = minimal_repo(tmp_path)
    (root / '.env.example').write_text('OPENAI_API_KEY=some-value\n', encoding='utf-8')
    assert any('blank OPENAI_API_KEY' in x for x in scanner.run_checks(root))


def test_release_scanner_detects_tracked_ignored_file(tmp_path):
    if shutil.which('git') is None:
        return
    root = minimal_repo(tmp_path)
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    (root / 'data').mkdir(exist_ok=True)
    (root / 'data' / 'secret.json').write_text('{"x":1}', encoding='utf-8')
    subprocess.run(['git', 'add', '-f', 'data/secret.json'], cwd=root, check=True)
    issues = scanner.run_checks(root)
    assert any('Unsafe file tracked by Git: data/secret.json' in x for x in issues)


def test_streamlit_does_not_use_deprecated_width_argument():
    app = (SOURCE / 'streamlit_app.py').read_text(encoding='utf-8')
    assert 'use_container_width' not in app
    assert 'width="stretch"' in app
