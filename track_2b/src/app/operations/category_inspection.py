"""Read-only category inspection box: list the ontology categories available for categorisation."""


def render() -> str:
    """Render the read-only categories list, wired by index.html to /api/operations/categories."""
    return '''<article class="operation-box category-inspection" aria-labelledby="category-inspection-title">
  <header><div><p class="operation-kicker">04 | Inspect</p><h3 id="category-inspection-title">Categories</h3></div><span class="operation-status operation-status--readonly">Read-only</span></header>
  <p>Inspect the categories available to documents. This view will not modify them.</p>
  <div class="operation-table-wrap"><table><thead><tr><th>Category</th><th>Description</th><th>Documents</th></tr></thead><tbody id="category-inspection-body"><tr><td colspan="3">No categories loaded.</td></tr></tbody></table></div>
  <div class="operation-footer"><span id="category-inspection-note">Inspection only</span><button type="button" id="category-inspection-button" data-refresh-categories>Refresh list</button></div>
</article>'''
