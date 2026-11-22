"""Offline release hygiene checks for a PaperTrail source checkout or extracted ZIP.

No network calls. This is a guardrail, not a complete secret/security audit.
Use `python tools/check_release.py` from any directory.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    'README.md', 'LICENSE', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md',
    'SECURITY.md', 'CHANGELOG.md', '.env.example', '.gitignore',
    '.github/workflows/ci.yml', 'pyproject.toml', 'streamlit_app.py',
    'tools/check_release.py', 'docs/GITHUB_RELEASE.md',
)
REQUIRED_DIRS = ('src', 'tests', 'docs', 'benchmarks')
TEXT_EXTENSIONS = {
    '.py', '.ipynb', '.md', '.txt', '.toml', '.json', '.jsonl',
    '.yaml', '.yml', '.cfg', '.ini', '.sh', '.ps1', '.xml', '.csv',
}
EXCLUDE_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', '.tox', '.nox', 'data', 'outputs',
    'dist', 'build', '.ipynb_checkpoints',
}
MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024
# Only recognize credible marker formats, not generic variable names or examples.
SECRET_PATTERNS = {
    'OpenAI-style API credential': re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
    'GitHub classic token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'GitHub fine-grained token': re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    'AWS key ID': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'Private key block': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
}
BANNED_NAMES = {'.env', 'secrets.toml', 'id_rsa', 'id_ed25519'}
BANNED_EXTENSIONS = {'.pem', '.p12', '.pfx', '.key'}


def _is_excluded(relative: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in relative.parts[:-1])


def _git_tracked_files(root: Path) -> list[str] | None:
    if not (root / '.git').exists():
        return None
    try:
        done = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '--cached', '-z'],
            capture_output=True, check=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [name.decode('utf-8', errors='replace') for name in done.stdout.split(b'\0') if name]


def _is_disallowed(relative: Path) -> bool:
    return (
        relative.name.lower() in BANNED_NAMES
        or relative.suffix.lower() in BANNED_EXTENSIONS
        or (relative.name.startswith('.env.') and relative.name != '.env.example')
        or 'data' in relative.parts or 'outputs' in relative.parts
        or '.venv' in relative.parts
    )


def run_checks(root: Path) -> list[str]:
    root = Path(root).resolve()
    issues: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            issues.append(f'Missing required release file: {relative}')
    for relative in REQUIRED_DIRS:
        if not (root / relative).is_dir():
            issues.append(f'Missing required release directory: {relative}')

    gitignore = root / '.gitignore'
    if gitignore.is_file():
        ignore_text = gitignore.read_text(encoding='utf-8')
        for pattern in ('.env', 'data/', '.venv/', '.streamlit/secrets.toml'):
            if pattern not in ignore_text:
                issues.append(f'Missing .gitignore safeguard: {pattern}')

    example = root / '.env.example'
    if example.is_file():
        for line in example.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith('OPENAI_API_KEY='):
                if line.split('=', 1)[1].strip():
                    issues.append('.env.example must have a blank OPENAI_API_KEY')

    # Prune large local model caches / virtualenvs *before* walking them.
    files: list[Path] = []
    for parent, dirs, names in os.walk(root, followlinks=False):
        base = Path(parent)
        for dirname in dirs[:]:
            directory = base / dirname
            if dirname in EXCLUDE_DIRS:
                dirs.remove(dirname)
            elif directory.is_symlink():
                issues.append(f'Symbolic link requires manual review: {directory.relative_to(root).as_posix()}')
                dirs.remove(dirname)
        files.extend(base / name for name in names)
    for file in files:
        relative = file.relative_to(root)
        if file.is_symlink():
            issues.append(f'Symbolic link requires manual review: {relative.as_posix()}')
            continue
        if _is_disallowed(relative):
            issues.append(f'Unsafe file in release tree: {relative.as_posix()}')
            continue
        if file.suffix.lower() not in TEXT_EXTENSIONS and file.name not in {'.env.example', '.gitignore'}:
            continue
        if file.stat().st_size > MAX_TEXT_FILE_BYTES:
            issues.append(f'Text file exceeds safety-scan size limit: {relative.as_posix()}')
            continue
        content = file.read_text(encoding='utf-8', errors='replace')
        for description, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                issues.append(f'Potential {description} in: {relative.as_posix()}')

    tracked = _git_tracked_files(root)
    if tracked is not None:
        for file in tracked:
            relative = Path(file)
            if _is_disallowed(relative) or _is_excluded(relative):
                issues.append(f'Unsafe file tracked by Git: {file}')
    return sorted(set(issues))


def main() -> int:
    issues = run_checks(PROJECT_ROOT)
    print(json.dumps({
        'project': 'PaperTrail',
        'check': 'offline_release_hygiene',
        'status': 'pass' if not issues else 'fail',
        'issues': issues,
        'note': 'Heuristic checks only: review Git diff/history and dependencies manually.',
    }, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == '__main__':
    raise SystemExit(main())
