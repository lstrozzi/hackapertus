"""Knowledge graph extraction box: one document, ontology ``Parlament``, one run.

The markup is server-rendered; the behaviour lives in ``index.html`` (delegated
change/click handlers) and talks to the endpoints backed by
``operations/kg_extraction_runner.py``. The ontology is hard-coded by design —
it is the demo's subject — and the dropdown only offers documents that can
actually be extracted: status ``Ready`` or ``NeedsCategorization``, with text.
"""


def render() -> str:
    """Render the document-to-graph extraction box."""
    return '''<article class="operation-box kg-extraction" aria-labelledby="kg-extraction-title">
  <header><div><p class="operation-kicker">07 | Extract</p><h3 id="kg-extraction-title">Knowledge graph extraction</h3></div><span class="operation-status" id="kg-extraction-status">Not run</span></header>
  <p>Pick one extracted document; the pipeline discovers entities per ontology type, deduplicates them and materialises the declared relationships, writing them into this store under community <code>Parlament</code>.</p>
  <div class="operation-field"><label for="kg-extraction-document">Document (Ready or NeedsCategorization)</label>
    <select id="kg-extraction-document" disabled><option>Loading documents…</option></select></div>
  <div class="operation-empty" id="kg-extraction-preview">No extraction run yet.</div>
  <div class="operation-footer"><span id="kg-extraction-note">Ontology: Parlament. Entities become :Entity nodes, every created entity gets an IS_DOCUMENTED link to the source document.</span><button type="button" id="kg-extract-button" data-kg-extract disabled>Extract to graph</button></div>
</article>'''
