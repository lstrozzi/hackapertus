"""Ontology extension box: let one document inspire NEW category proposals."""


def render() -> str:
    """Render the ontology-extension box, wired by index.html to /api/operations/categories/extend."""
    return '''<article class="operation-box category-extension" aria-labelledby="category-extension-title">
  <header><div><p class="operation-kicker">06 | Extend</p><h3 id="category-extension-title">Ontology extension</h3></div><span class="operation-status" id="category-extension-status">Not run</span></header>
  <p>Pick a sample document; the extension prompt proposes new ontology categories for review. Proposals join the ontology flagged <em>proposed</em> and are inert for categorisation until approved here.</p>
  <div class="operation-field"><label for="category-extension-document">Document (awaiting categorisation on the platform)</label>
    <select id="category-extension-document" disabled><option>Loading documents…</option></select></div>
  <div class="operation-empty" id="category-extension-results">No ontology proposals yet.</div>
  <div class="operation-footer"><span id="category-extension-note">Ontology: Parlament. Proposals are stored as flagged definitions, never assigned.</span><button type="button" id="category-extend-button" data-extend-ontology disabled>Propose new categories</button></div>
</article>'''
