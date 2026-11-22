# PaperTrail v1.6.2 — GitHub Release Checklist

**Status: a GitHub-ready candidate, NOT yet pushed to any account.** This ZIP does not contain downloaded PDFs, `.env`, models, local indexes, `data/` or a `.git` directory. Publishing is a separate manual action.

## 1. Preflight on Windows PowerShell

Extract the v1.6.2 archive and open its **inner** `papertrail-research-agent` directory in VS Code. Keep your v1.6.1 directory as a backup. If you want your local cached papers in the new version, copy **only** `data/` (not `.venv`, credentials or private notes into Git):

```powershell
Copy-Item -Path "..\..\papertrail_phase6_github_release_candidate_full_v1_6_1\papertrail-research-agent\data" -Destination ".\data" -Recurse -Force
```

If `data/` was already copied, skip that line. Verify your working directory contains `pyproject.toml`, then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,search,agent,ui]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_release.py
.\.venv\Scripts\python.exe -m papertrail doctor
```

## 2. Check the local UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Visit http://127.0.0.1:8501. If local `data/` is present, choose indexed paper `2510.22344`, **Preview**, **Baseline** and `top-k=3`, ask the FAIR-RAG comparison question, and inspect passages/citations. **No LLM answer is generated in Preview.** Stop the server with `Ctrl+C`.

Check the About and Human review tabs. Do not turn on Live or arXiv import merely to verify this release; those workflows may incur API cost or download model weights.

## 3. Manual checklist before Git

- [ ] README describes Preview (offline evidence, no LLM response) and optional Live (paid-capable synthesis).
- [ ] `.env.example` contains an empty API key; no real `.env`, PDF, data cache, index, personal output or review queue is staged.
- [ ] No token or sensitive content appears in notebooks (including cell outputs), screenshots, Git history or generated logs.
- [ ] CI regression tests pass, and limitations are clear: citation validation is not factual entailment; retrieval annotations are provisional.
- [ ] The repository license covers original code only, not third-party papers, model weights or datasets.
- [ ] An optional Live smoke test is done only if intentionally approved, with results labeled as provisional.

The scanner does **not** guarantee security, check external services, certify scientific accuracy or inspect historical deleted commits.

## 4. Publish to your GitHub account

Create an **empty** public repository named `papertrail-research-agent` on GitHub. Avoid creating a remote README or LICENSE (already supplied locally). From the inner project directory:

```powershell
git init
git branch -M main
git add .
git status --short
git diff --cached --name-only
.\.venv\Scripts\python.exe tools\check_release.py
git commit -m "Release PaperTrail v1.6.2"
git remote add origin https://github.com/YOUR_USERNAME/papertrail-research-agent.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual account name. If the project is **already a Git repository**, inspect `git remote -v` and use `git remote set-url origin ...` only when necessary. Do not blindly create a new repository or rewrite remote history. Never embed tokens in the remote URL.

After push and CI verification, optionally create a version tag:

```powershell
git tag v1.6.2
git push origin v1.6.2
```

A GitHub Release can be created on the GitHub website from that tag. The ZIP itself does **not** publish the project.

## Suggested GitHub metadata

**Description:** Evidence-first AI research agent for arXiv papers using LangGraph, hybrid RAG, traceable citations, optional LLM synthesis and Streamlit.

**Topics:** `ai-agent`, `rag`, `langgraph`, `streamlit`, `faiss`, `bm25`, `research-assistant`, `citation-validation`, `python`.

## Release acceptance

- [ ] Local scanner and full tests pass on the exact candidate folder.
- [ ] Streamlit opens in a browser; Preview retrieves actual local evidence.
- [ ] Git staging was audited for secrets/data, and no unexpected history is pushed.
- [ ] GitHub Actions is green on the pushed commit.
- [ ] The description and research accuracy claims match tested capabilities.

After GitHub publishing, keep this document as a **repeatable checklist for future releases**, not as a statement that publication is pending forever.
