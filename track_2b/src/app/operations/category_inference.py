"""Category inference box: propose categories for one document without storing them."""


def render() -> str:
    """Render the category proposal box, wired by index.html to /api/operations/categories/infer."""
    return '''<article class="operation-box category-inference" aria-labelledby="category-inference-title">
  <header><div><p class="operation-kicker">03 | Propose</p><h3 id="category-inference-title">Category inference</h3></div><span class="operation-status" id="category-inference-status">Proposal</span></header>
  <p>Pick a sample document whose platform counterpart is awaiting categorisation; the categorisation skill proposes ontology categories for review, without storing them.</p>
  <div class="operation-field"><label for="category-inference-document">Document (awaiting categorisation on the platform)</label>
    <select id="category-inference-document" disabled><option>Loading documents…</option></select></div>
  <div class="operation-empty" id="category-inference-results">No proposals yet.</div>
  <div class="operation-footer"><span id="category-inference-note">Ontology: Parlament. Proposals are stored on the document for review, not assigned.</span><button type="button" id="category-infer-button" data-infer-categories disabled>Propose categories</button></div>
</article>'''
