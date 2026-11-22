"""Optional native Streamlit smoke test, executed after installing the UI extra."""
import pytest


def test_streamlit_initial_render(tmp_path, monkeypatch):
    st = pytest.importorskip('streamlit', reason='Install [ui] for Streamlit-native smoke test')
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).resolve().parents[1] / 'streamlit_app.py'
    app = AppTest.from_file(str(script)).run(timeout=30)
    assert not app.exception
    assert 'PaperTrail' in app.title[0].value
