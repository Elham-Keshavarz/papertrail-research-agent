"""Frontend start-up smoke test without a Streamlit installation.

This checks Python wiring only; it does NOT claim native Streamlit rendering was tested.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


class Surface:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeStreamlit:
    sidebar = Surface()

    def __init__(self):
        self.session_state = {}
        self.messages = []

    def __getattr__(self, name):
        if name == 'progress':
            class FakeProgress:
                def progress(inner_self, percent, *, text=None):
                    self.messages.append(('progress_update', (percent, text)))
                    return inner_self
            self.messages.append(('progress_init', (name,)))
            return lambda *args, **kwargs: FakeProgress()
        if name == 'tabs':
            return lambda names: [Surface() for _ in names]
        if name in {'button', 'checkbox'}:
            return lambda *args, **kwargs: False
        if name == 'text_input':
            return lambda *args, **kwargs: kwargs.get('value', '')
        if name == 'stop':
            return lambda: (_ for _ in ()).throw(AssertionError('UI stopped unexpectedly'))
        return lambda *args, **kwargs: self.messages.append((name, args))


def test_ui_starts_in_empty_library_without_streamlit(monkeypatch, tmp_path):
    fake = FakeStreamlit()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'))
    assert any(name == 'title' and args[0] == '📚 PaperTrail' for name, args in fake.messages)
    assert any(name == 'info' and 'No fully indexed papers' in args[0] for name, args in fake.messages)

class ResearchStreamlit(FakeStreamlit):
    def __init__(self):
        super().__init__()
        self.ran = False

    def __getattr__(self, name):
        if name == 'spinner':
            return lambda *args, **kwargs: Surface()
        if name == 'form':
            return lambda *args, **kwargs: Surface()
        if name == 'columns':
            class MetricSurface(Surface):
                def metric(self, *_args, **_kwargs):
                    return None
            return lambda n: [MetricSurface() for _ in range(n if isinstance(n, int) else len(n))]
        if name == 'expander':
            return lambda *args, **kwargs: Surface()
        if name == 'selectbox':
            return lambda label, options, **kwargs: options[kwargs.get('index', 0)]
        if name == 'select_slider':
            return lambda *args, **kwargs: kwargs.get('value', 3)
        if name == 'radio':
            return lambda label, options, **kwargs: options[0]
        if name == 'text_area':
            return lambda *args, **kwargs: kwargs.get('value', '')
        if name == 'form_submit_button':
            def submit(*args, **kwargs):
                if not self.ran:
                    self.ran = True
                    return True
                return False
            return submit
        return super().__getattr__(name)


def test_preview_research_app_calls_service_and_renders_evidence(monkeypatch, tmp_path):
    fake = ResearchStreamlit()
    root = tmp_path / 'data'
    for folder, filename in (
        ('chunks', '2510.22344.json'),
        ('embeddings', '2510.22344.npy'),
        ('embeddings', '2510.22344.manifest.json'),
        ('indexes', '2510.22344.faiss'),
        ('indexes', '2510.22344.manifest.json'),
    ):
        path = root / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test', encoding='utf8')
    import papertrail.ui.service as svc
    seen = []
    def fake_research(**kwargs):
        seen.append(kwargs)
        kwargs['on_progress'](35, 'Local index ready')
        kwargs['on_progress'](100, 'Workflow finished')
        return {'status': 'ready_for_llm_synthesis', 'llm_calls': 0, 'tool_calls': 3,
                'primary_evidence_hits': 1, 'generation_mode': 'preview',
                'synthesis': {'claims': [], 'answer': None},
                'evidence': [{'step_id': 'retrieve_01', 'query': 'research', 'hits': [
                    {'rank': 1, 'page_number': 2, 'chunk_id': '2510.22344:chunk1',
                     'text': 'Excerpt from PDF', 'context': []}]}]}
    monkeypatch.setattr(svc, 'run_question', fake_research)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'))
    assert len(seen) == 1
    assert seen[0]['mode'] == 'preview'
    assert seen[0]['paper_id'] == '2510.22344'
    assert seen[0]['live_confirmed'] is False
    assert any(name == 'text' and args[0] == 'Excerpt from PDF' for name, args in fake.messages)
    assert ('progress_update', (35, '35% · Local index ready')) in fake.messages
    assert ('progress_update', (100, '100% · Workflow finished')) in fake.messages
