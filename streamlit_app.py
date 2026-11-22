"""PaperTrail — Streamlit interface for the existing LangGraph engine.

Run from the project root: python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

from papertrail.cli_bootstrap import configure_numeric_threads

configure_numeric_threads()

import json
import re
from pathlib import Path

import streamlit as st

from papertrail.config import Settings
from papertrail.agent.faithfulness import FaithfulnessError
from papertrail.ui.service import (
    UIServiceError, available_review_queues, discover_papers, list_local_papers,
    new_live_review_queue, paper_artifacts, prepare_paper,
    read_review_queue, run_question, update_review_decision,
)


st.set_page_config(page_title='PaperTrail | Evidence-first research', page_icon='📚', layout='wide')


def status_line(value: str) -> str:
    return value.replace('_', ' ').capitalize() if value else 'Unknown'


def progress_reporter(bar):
    """Return a callback for actual completed workflow stages (no fake timer)."""
    def report(percent: int, label: str) -> None:
        bar.progress(percent, text=f'{percent}% · {label}')
    return report


def display_citations(citations: list[dict], paper_id: str) -> None:
    for cite in citations:
        page = cite.get('page_number')
        if type(page) is not int or page < 1:
            continue
        with st.expander(f"{cite.get('evidence_id', 'Source')} · Page {page} · Exact quote"):
            st.code(str(cite.get('quote', ''))[:550], language=None, wrap_lines=True)
            st.link_button('Open source on arXiv', f'https://arxiv.org/abs/{paper_id}#page={page}')
            st.caption('A matching quote is not proof that the claim is logically entailed.')


def display_result(result: dict, paper_id: str, query: str) -> None:
    st.divider()
    st.subheader('Research result')
    st.caption(f'Paper: {paper_id} · Question: {query}')
    status = result.get('status', 'unknown')
    stats = st.columns(4)
    stats[0].metric('Primary evidence hits', result.get('primary_evidence_hits', 0))
    stats[1].metric('Tool calls', result.get('tool_calls', 0))
    stats[2].metric('LLM calls', result.get('llm_calls', 0))
    stats[3].metric('Claims', len(result.get('synthesis', {}).get('claims', [])))
    st.info(f'Status: **{status_line(status)}**. Scientific correctness and entailment are NOT automatically verified.')
    if result.get('errors'):
        st.warning('Research did not complete. Review the safe diagnostic events below.')
    answer_tab, evidence_tab, trace_tab = st.tabs(['Claims & citations', 'Retrieved evidence', 'Execution diagnostics'])
    with answer_tab:
        synthesis = result.get('synthesis') or {}
        claims = synthesis.get('claims') or []
        if not claims:
            if result.get('generation_mode') == 'preview':
                st.info('Preview mode does not generate an answer or call an LLM. Examine the retrieved evidence instead.')
            else:
                st.warning('No validated claims were generated. The Agent did not invent an answer.')
        else:
            st.warning('Citation and quote-anchoring checks passed for these claims. Semantic support is still UNVERIFIED and requires human review.')
            audit = synthesis.get('faithfulness_audit') or {}
            mechanics = {row['claim_index']: row['mechanical_check'] for row in audit.get('claims', [])}
            for pos, claim in enumerate(claims, start=1):
                st.markdown(f'**Claim {pos} · {claim.get("step_id", "")}**')
                st.write(claim.get('claim', ''))
                st.caption('Mechanical check: ' + mechanics.get(pos, 'not evaluated'))
                display_citations(claim.get('citations', []), paper_id)
            if result.get('generation_mode') == 'live' and audit:
                if st.button('Create local human-review queue', key='create_live_review', type='secondary'):
                    try:
                        path = new_live_review_queue(result, data_dir=SETTINGS.data_dir)
                    except (UIServiceError, OSError, ValueError) as exc:
                        st.error(f'Could not create review queue: {type(exc).__name__}')
                    else:
                        st.session_state['latest_review'] = path.name
                        st.success('Review queue saved locally. Open the Human review tab to record decisions.')
        if synthesis.get('missing_step_ids'):
            st.warning('Facets without an accepted claim: ' + ', '.join(synthesis['missing_step_ids']))
        if synthesis.get('validation_rejections') or synthesis.get('rejection_reasons'):
            with st.expander('Rejected claim diagnostics (safe reason codes)'):
                st.json(synthesis.get('validation_rejections') or synthesis.get('rejection_reasons'))
    with evidence_tab:
        rows = result.get('evidence', [])
        if not rows:
            st.info('No passages available. The source might be missing or structural retrieval may have failed.')
        for facet in rows:
            st.markdown(f'**Facet {facet.get("step_id", "unknown")}** · {len(facet.get("hits", []))} primary hits')
            st.caption('Search query: ' + str(facet.get('query', ''))[:550])
            for hit in facet.get('hits', []):
                page = hit.get('page_number')
                rank = hit.get('rank')
                if type(page) is not int or page < 1:
                    continue
                with st.expander(f'#{rank} · Page {page} · {str(hit.get("chunk_id", ""))[-16:]}'):
                    st.text(str(hit.get('text', ''))[:4000])
                    st.link_button('Read paper on arXiv', f'https://arxiv.org/abs/{paper_id}#page={page}')
                    for c in hit.get('context', [])[:2]:
                        st.caption('Adjacent same-page context:')
                        st.text(str(c.get('text', ''))[:1000])
    with trace_tab:
        st.caption('These are structural diagnostics, not scientific validation.')
        st.json({
            'status': result.get('status'), 'planner': result.get('planner'),
            'source_audit': result.get('source_audit'),
            'assessment': result.get('assessment'),
            'tool_events': result.get('tool_events', []),
            'trace': result.get('trace', []),
        }, expanded=False)
    safe_export = {
        'paper_id': paper_id, 'question': query, 'status': status,
        'generation_mode': result.get('generation_mode'), 'llm_calls': result.get('llm_calls'),
        'synthesis': result.get('synthesis', {}),
        'assessment': result.get('assessment'),
        'source_audit': result.get('source_audit'),
        'evidence': result.get('evidence', []),
        'limitations': result.get('limitations', []),
    }
    st.download_button('Export this result as JSON', json.dumps(safe_export, ensure_ascii=False, indent=2),
                       file_name='papertrail_research_result.json', mime='application/json')


try:
    SETTINGS = Settings()
except Exception:
    st.error('Invalid project environment settings. Check your project-root .env configuration.')
    st.stop()

st.title('📚 PaperTrail')
st.caption('Evidence-first AI Research Agent · local RAG + LangGraph + transparent citations')
st.warning('Research assistant, not a scientific fact checker. Citation matching does NOT establish semantic entailment or PDF provenance.')

library = list_local_papers(SETTINGS.data_dir)
ready_papers = [p.paper_id for p in library if p.ready]

with st.sidebar:
    st.header('Workspace')
    st.caption('Local data directory: ' + str(SETTINGS.data_dir))
    st.caption('Local library readiness is file-presence only; source integrity is checked when researching.')
    st.metric('Papers with index files', len(ready_papers))
    st.caption('Network access is disabled during research. arXiv imports and Live LLM generation require separate explicit actions.')
    if SETTINGS.openai_api_key and SETTINGS.openai_api_key.get_secret_value():
        st.success('Live LLM credential configured (value hidden).')
    else:
        st.info('No API key configured; Preview remains available.')
    st.caption('Running locally? Keep the server bound to 127.0.0.1 and do not publish your .env or data directory.')

research_tab, library_tab, review_tab, about_tab = st.tabs(['Research', 'Paper library', 'Human review', 'About'])

with research_tab:
    if not ready_papers:
        st.info('No fully indexed papers found. Copy your previous data/ folder into this project or import a paper from the Paper library tab.')
    else:
        selected = st.selectbox('Indexed paper', ready_papers, key='ui_paper_selection')
        st.caption('Source: https://arxiv.org/abs/' + selected)
        with st.form('research_form'):
            question = st.text_area('Research question',
                value='Compare FAIR-RAG with traditional RAG and discuss their evaluation metrics and limitations.',
                height=110, max_chars=1000)
            left, mid, right = st.columns([1, 1, 1])
            with left:
                mode = st.radio('Answer mode', ['preview', 'live'], horizontal=True,
                                format_func=lambda x: 'Preview · no LLM' if x == 'preview' else 'Live · LLM may cost money')
            with mid:
                style = st.selectbox('Retrieval policy', ['baseline', 'focused', 'multi', 'budgeted'],
                    help='Experimental policies are Preview-only and not deployed as the default.')
            with right:
                top_k = st.select_slider('Primary hits per facet', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=3)
            live_ok = st.checkbox('I explicitly approve a potentially paid LLM request.', value=False,
                                  help='The app will not send an LLM request until you press Run in Live mode with this selected.')
            submitted = st.form_submit_button('Run PaperTrail', type='primary', width="stretch")
        if submitted:
            if mode == 'live' and style != 'baseline':
                st.error('Live mode only supports the production baseline retrieval policy.')
            elif mode == 'live' and not live_ok:
                st.error('Live mode requires explicit approval of a potentially paid LLM call.')
            else:
                st.caption('Progress shows completed workflow checkpoints, not elapsed time or model-weight percentages.')
                progress_bar = st.progress(0, text='0% · Starting research workflow')
                with st.spinner('Running research tasks (model loading may take longer)…'):
                    try:
                        result = run_question(settings=SETTINGS, paper_id=selected,
                            question=question, mode=mode, query_style=style, top_k=top_k,
                            live_confirmed=live_ok, on_progress=progress_reporter(progress_bar))
                    except (UIServiceError, ValueError) as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f'Execution failed ({type(exc).__name__}). Check your local caches, dependencies, and provider settings.')
                    else:
                        # The backend reaches 100% only after LangGraph finalize.
                        st.session_state['last_result'] = result
                        st.session_state['last_request_label'] = (selected, question)
                        st.session_state.pop('latest_review', None)
    if 'last_result' in st.session_state:
        old_paper, old_query = st.session_state['last_request_label']
        display_result(st.session_state['last_result'], old_paper, old_query)

with library_tab:
    st.subheader('Local research papers')
    if library:
        st.dataframe([{'arXiv ID': p.paper_id, 'Ready': p.ready, 'Text chunks': p.chunks,
                        'Embeddings': p.embeddings, 'FAISS index': p.index} for p in library],
                     hide_index=True, width="stretch")
    else:
        st.info('No cached paper data found.')
    st.divider()
    st.subheader('Discover and import from arXiv')
    st.caption('Both buttons below can contact public arXiv. Import may also download a Hugging Face embedding model if it is not already cached.')
    permission = st.checkbox('Allow arXiv network requests and model-weight downloads in this tab', value=False)
    terms = st.text_input('Paper search keywords', value='retrieval augmented generation')
    if st.button('Search public arXiv'):
        try:
            items = discover_papers(terms, settings=SETTINGS, consent=permission)
        except (UIServiceError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f'arXiv search unavailable ({type(exc).__name__}).')
        else:
            st.session_state['arxiv_matches'] = items
            st.success(f'Found {len(items)} papers. Review their titles before importing.')
    matches = st.session_state.get('arxiv_matches', [])
    if matches:
        choice = st.selectbox('Choose a paper to prepare', range(len(matches)),
                              format_func=lambda i: f"{matches[i]['paper_id']} · {matches[i]['title'][:100]}")
        st.write('Authors: ' + matches[choice]['authors'])
        st.link_button('Review on arXiv', matches[choice]['url'])
        if st.button('Download, chunk, embed and index selected paper'):
            st.caption('Import progress tracks completed stages; download and model-loading times vary.')
            import_progress = st.progress(0, text='0% · Preparing paper import')
            with st.spinner('Preparing source PDF and local RAG artifacts…'):
                try:
                    new_paper = prepare_paper(matches[choice]['paper_id'], settings=SETTINGS, consent=permission,
                                              on_progress=progress_reporter(import_progress))
                except (UIServiceError, ValueError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f'Import failed ({type(exc).__name__}). Check arXiv access and local model cache.')
                else:
                    st.success(f"Prepared {new_paper.paper_id}. Open the Research tab to query it.")
                    st.rerun()
    st.caption('Downloaded papers, embeddings, and index files stay inside data/ and are excluded from Git.')

with review_tab:
    st.subheader('Human semantic review')
    st.caption('Read the original PDF yourself before classifying a claim. Reviewer decisions are user-supplied, not independently adjudicated.')
    queue_files = available_review_queues(SETTINGS.data_dir)
    if not queue_files:
        st.info('No review queue yet. Generate a Live answer and select “Create local human-review queue”, or use the Phase 4.3 CLI audit command.')
    else:
        saved_name = st.session_state.get('latest_review')
        initial = next((i for i, p in enumerate(queue_files) if p.name == saved_name), 0)
        choice = st.selectbox('Local review queue', queue_files, index=initial,
                              format_func=lambda p: p.name)
        try:
            report = read_review_queue(choice, data_dir=SETTINGS.data_dir)
        except (UIServiceError, ValueError, OSError) as exc:
            st.error(f'Invalid review queue ({type(exc).__name__}).')
        else:
            for row in report.get('review_queue', []):
                index = row.get('claim_index')
                if type(index) is not int:
                    continue
                st.markdown(f'**Claim {index}**')
                st.write(row.get('claim', ''))
                st.caption('Mechanical check: ' + str(row.get('mechanical_check', 'unknown')))
                # Original-page hyperlinks were validated when the claim was generated.
                for citation in row.get('citations', []):
                    st.code(str(citation.get('quote', ''))[:550], language=None, wrap_lines=True)
                    uri = citation.get('citation', '')
                    if isinstance(uri, str) and re.fullmatch(r'https://arxiv\.org/abs/\d{4}\.\d{4,5}(?:v\d+)?#page=[1-9]\d*', uri):
                        st.link_button('Check the original source page', uri)
                with st.form(f'review_{choice.name}_{index}'):
                    decisions = ['pending', 'supported', 'unsupported', 'insufficient_context']
                    current = row.get('reviewer_decision', 'pending')
                    decision = st.selectbox(f'Your decision for claim {index}', decisions,
                                            index=decisions.index(current) if current in decisions else 0)
                    note = st.text_area(f'Notes for claim {index}', value=row.get('reviewer_note', ''), max_chars=2000)
                    save = st.form_submit_button('Save review decision')
                if save:
                    try:
                        summary = update_review_decision(choice, data_dir=SETTINGS.data_dir,
                            claim_index=index, decision=decision, note=note)
                    except (UIServiceError, FaithfulnessError, ValueError, OSError) as exc:
                        st.error(f'Could not save review ({type(exc).__name__}).')
                    else:
                        st.success(f"Review saved. Pending claims: {summary['pending']}")
                        st.rerun()
            st.caption('This review is local and does not automatically validate the original PDF or verify scientific truth.')

with about_tab:
    st.subheader('How PaperTrail works')
    st.markdown('''
    **PaperTrail** uses locally stored arXiv PDF text, page-aware chunks,
    sentence-transformer embeddings, FAISS + BM25 retrieval, and a LangGraph
    workflow for evidence collection and structural citation checking.

    **Preview** is offline, makes zero LLM calls, and deliberately does **not**
    generate an answer. **Live** requires an explicit approval and a configured
    model provider; responses have mechanically checked citations but are not
    semantically fact-checked.

    The **Human review** workflow stores user decisions locally. It does not
    claim independent verification or scientific recall/accuracy scores.
    ''')
    st.caption('Phase 6 · Streamlit UI · Experimental query styles remain Preview-only')
