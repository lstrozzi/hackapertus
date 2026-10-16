"""Document categorization box: run the skill on one selected document and store its categories."""


def render() -> str:
    """Render the categorisation box, wired by index.html to /api/operations/categories/apply."""
    return '''<article class="operation-box categorization" aria-labelledby="categorization-title">
  <header><div><p class="operation-kicker">05 | Apply</p><h3 id="categorization-title">Categorization</h3></div><span class="operation-status" id="categorization-status">Not run</span></header>
  <p>Select one document; the categorisation skill assigns ontology categories to it explicitly.</p>
  <div class="operation-field"><label>Document</label>
    <ul class="categorization-document-list" id="categorization-document-list"><li>Loading documents…</li></ul></div>
  <div class="operation-empty" id="categorization-results">Select a document to categorise.</div>
  <div class="operation-footer"><span>Assignments are written to the graph on click</span><button type="button" id="categorize-apply-button" data-apply-categories disabled>Categorize document</button></div>
</article>'''
