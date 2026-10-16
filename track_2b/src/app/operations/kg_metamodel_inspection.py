"""Read-only knowledge graph metamodel inspection box.

The markup is only the shell: the table body is filled by the GUI from
`GET /api/operations/kg/metamodel`, whose server side lives in
`kg_metamodel.py` (ontology id `Parlament`, hard-coded). Nothing in this box
ever edits the schema; "Refresh" is a re-read.
"""


def render() -> str:
    """Render the metamodel inspection shell for the workspace's Knowledge tab."""
    return '''<article class="operation-box kg-metamodel-inspection" aria-labelledby="kg-metamodel-title">
  <header><div><p class="operation-kicker">06 | Inspect</p><h3 id="kg-metamodel-title">Knowledge graph metamodel</h3></div><span class="operation-status operation-status--readonly">Read-only</span></header>
  <p>Browse the ontology's entity types, attributes, and relationships without editing its schema. Inspected ontology: <strong>Parlament</strong>.</p>
  <div class="operation-table-wrap"><table><thead><tr><th>Element</th><th>Type</th><th>Property</th><th>Description</th></tr></thead><tbody id="kg-metamodel-rows"><tr><td colspan="4">Metamodel not loaded.</td></tr></tbody></table></div>
  <div class="operation-footer"><span id="kg-metamodel-source">Schema inspection only</span><button type="button" data-refresh-metamodel>Refresh metamodel</button></div>
</article>'''
