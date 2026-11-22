# Security Policy

## Reporting vulnerabilities

Please use GitHub **Security → Report a vulnerability** if private vulnerability reporting is available in the repository. Otherwise contact the maintainer using a private channel published on their profile. Never post passwords, API keys, private PDFs or exploit details in public issues. There is no invented project security email address.

## Deployment and privacy boundaries

- The Streamlit frontend is **localhost-only and single-user**. It has no authentication or multi-user isolation. Bind it to `127.0.0.1`; do not deploy it as a public website without additional security engineering.
- Preview retrieval uses locally cached artifacts and does not send an LLM request. Live sends the question and bounded source excerpts to the chosen OpenAI-compatible provider **only after explicit opt-in**; the provider may charge fees and retain data according to its own policy.
- Public arXiv search/import and initial model-weight downloads require separate network permission. Do not use the import path on confidential source documents.
- `.env`, `data/`, review queues and PDFs are intentionally ignored by Git. Git ignore protects only **untracked** files: audit `git ls-files` and staged files before publishing. A file deleted in a later commit may still exist in Git history.
- Retrieved PDF content and model-generated text are untrusted data. Citation anchoring is structural; it does not provide semantic truth, prompt-injection immunity, or validated original PDF provenance.
- The offline release scanner recognizes a limited set of credential patterns and release mistakes. It does **not** audit dependency supply chains, guarantee secret detection or inspect prior Git history.

## Before publishing

Run `python tools/check_release.py` and `python -m pytest -q`; review `git status --short`, `git diff --cached --name-only` and historical commits if the repository was used before. If a token was exposed, rotate it immediately and clean Git history appropriately before publication.

Only the latest candidate may receive fixes; no security-response SLA is promised for this portfolio prototype.
