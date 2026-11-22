# Contributing to PaperTrail

Thank you for improving evidence-first research tooling.

Use Python 3.11 or newer. From the project root, install development dependencies, run tests, and inspect the release check:

```bash
python -m pip install -e '.[dev,search,agent,ui]'
python -m pytest -q
python tools/check_release.py
```

The embeddings extra is optional for development (`pip install -e '.[embeddings]'`). Most tests use synthetic fixtures and do not need model downloads.

Before opening a pull request, include a clear problem description, tests, and any changes to network or LLM permissions. Preserve the default offline Preview behavior; experimental retrieval strategies should not silently become production defaults. Do not add dependencies or change evaluation labels without explaining the tradeoffs.

Never commit `.env`, `data/`, PDFs, downloaded model weights, embeddings, FAISS indexes, reviewer notes or notebooks with credentials. Confirm the actual staged changes (`git diff --cached --name-only`) and read [SECURITY.md](SECURITY.md).

Do not claim that synthetic tests establish semantic entailment, independent PDF provenance, or statistically generalizable research performance. Original project code uses the MIT license; downloaded academic articles and models have separate usage terms.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.
