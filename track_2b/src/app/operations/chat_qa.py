"""Chat Q&A verification box — the live single-pass RAG answer surface."""


def render() -> str:
    """Render the chat box wired to POST /api/operations/chat by the guide script."""
    return '''<article class="operation-box chat-qa" aria-labelledby="chat-qa-title">
  <header><div><p class="operation-kicker">08 | Verify</p><h3 id="chat-qa-title">Chat Q&amp;A</h3></div><span class="operation-status" id="chat-status">Ready</span></header>
  <p>Ask a question: the pipeline embeds it, searches the graph, reranks the fragments, and the model answers only from fragments scoring at least 50%.</p>
  <div class="chat-transcript" id="chat-transcript" aria-label="Conversation" aria-live="polite"><p class="chat-empty">The conversation will appear here.</p></div>
  <div class="operation-field"><label for="chat-question">Question</label><textarea id="chat-question" rows="3" placeholder="Ask about the available documents..."></textarea></div>
  <div class="operation-field"><label for="chat-mode">Search mode</label><select id="chat-mode"><option value="simple" selected>Simple — embed, search, rerank, answer</option><option value="advanced">Advanced — expand seeds, category and entity affinity, pair-checked</option></select><span class="muted">Advanced re-ranks the enlarged set with the same 50% floor — slower, opt-in per question.</span></div>
  <div class="operation-footer"><span id="chat-note">Ctrl+Enter sends; each citation links to the source document and shows its rerank score.</span><button type="button" id="chat-ask-button" disabled>Ask question</button></div>
</article>'''
