"""Document upload box."""


def render() -> str:
    """Render the document upload controls."""
    return '''<article class="operation-box document-upload" aria-labelledby="document-upload-title">
  <header><div><p class="operation-kicker">01 | Documents</p><h3 id="document-upload-title">Document upload</h3></div></header>
  <p>Add source files to the sample folder and document list.</p>
  <label class="upload-dropzone" for="operation-upload-files"><span class="upload-mark" aria-hidden="true">+</span><span><strong>Choose files</strong><small>Accepted: .txt, .md, .pdf, .html</small></span></label>
  <input id="operation-upload-files" type="file" accept=".txt,.md,.pdf,.html" multiple>
  <div class="operation-footer"><span id="operation-upload-status" role="status">No files selected</span><button type="button" data-upload-documents disabled>Upload documents</button></div>
</article>'''
