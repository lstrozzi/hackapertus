"""Document ingestion box."""


def render() -> str:
    """Render the uploaded document list and ingestion status."""
    return '''<article class="operation-box document-ingestion" aria-labelledby="document-ingestion-title">
  <header><div><p class="operation-kicker">02 | Processing</p><h3 id="document-ingestion-title">Document ingestion</h3></div></header>
  <p>Platform status is shown in the table; the counters track ingestion in this app.</p>
  <dl class="operation-metrics"><div><dt>Ready</dt><dd id="sample-document-ready-count">0</dd></div><div><dt>Not yet processed</dt><dd id="sample-document-pending-count">0</dd></div><div><dt>In progress</dt><dd id="sample-document-processing-count">0</dd></div><div><dt>Failed</dt><dd id="sample-document-failed-count">0</dd></div></dl>
  <div class="operation-table-wrap sample-document-table" aria-live="polite"><table><thead><tr><th></th><th>Document</th><th title="Identifier the platform gave this document (:Document.id), matched by filename">Doc ID</th><th>Size</th><th title="The Document.status property in the matched platform graph">Platform status</th></tr></thead><tbody id="sample-document-list"><tr><td colspan="5">No documents uploaded.</td></tr></tbody></table></div>
  <div class="operation-footer"><span id="ingestion-selection-status">0 selected for ingestion</span><button id="start-ingestion-button" type="button" disabled>Start ingestion</button></div>
</article>'''
