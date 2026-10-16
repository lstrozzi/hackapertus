"""Demo rehearsal: run the five questions of Kleine Anfrage 2016/8 through the deployed pipeline.

Uses the real endpoint function (retrieval, rerank, threshold, generation) and prints one
verdict line per question plus the answer's opening, so the demo script rests on evidence.
"""

from typing import Any, Dict, List

QUESTIONS: List[str] = [
    "Erachtet die Regierung den zitierten Bericht der Sonntagszeitung als glaubwuerdig?",
    "Hat eine der genannten Firmen vom Kanton Schaffhausen einzelbetriebliche Foerdermittel oder Steuererleichterungen erhalten?",
    "Weiss die Regierung um welches geistige Eigentum es sich bei der Schweizer Niederlassung von Rising Tide Ungarn handelt und kann sie darueber Auskunft geben?",
    "Welche Massnahmen ergreift der Kanton Schaffhausen, damit keine Firmen angesiedelt und unterstuetzt werden, die zwielichtige Geschaefte betreiben?",
    "Wie wird verhindert, dass ueber gemeinnuetzige Stiftungen zwielichtige Geschaefte abgewickelt werden?",
]

from app.operations.chat_rag import answer_question

for index, question in enumerate(QUESTIONS, start=1):
    result: Dict[str, Any] = answer_question(question)
    if not result["ok"]:
        print(f"Q{index} FAILED {result['error']}")

        continue

    citations: List[Dict[str, Any]] = result["citations"]
    best: float = max((c["score"] for c in citations), default=0.0)
    print(f"Q{index} {result['answer_source']:13} cites={len(citations)} best={best:.0%} "
          f"retr={result['candidates']} passed={result['passed_threshold']} "
          f"gen={result['timings_ms'].get('generate')}ms")
    print("    " + (result["answer"] or "").replace("\n", " ")[:320])
