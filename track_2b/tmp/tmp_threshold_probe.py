"""How high does the right chunk actually score? Measures the top rerank score for several phrasings.

Read-only against the model services; writes nothing.
"""

from typing import Any, Dict, List

from app.graphstore.neo4j import load_neo4j_config
from app.model_services import load_model_services_config
from app.operations import chat_rag

QUESTIONS: List[str] = [
    "Wer hat die Kleine Anfrage 2016/8 eingereicht?",
    "Wer hat die Kleine Anfrage 2016/8 eingereicht und welchen Vorwurf enthaelt sie?",
    "Transparenz von Stiftungen und angesiedelten Firmen",
    "Rising Tide Foundation Schaffhausen",
    "Wie entwickelte sich der Gewinn von Rising Tide Budapest?",
    "Wie viele Mitarbeiter hatte die Rising Tide GmbH in Herrenacker 15?",
    "Wurden die Bundesbahnen saniert?",
]

config = load_neo4j_config()
services = load_model_services_config()

for question in QUESTIONS:
    vector: Any = chat_rag._embed_texts(services, [question])[0]
    candidates: List[Dict[str, Any]] = chat_rag._rerank(services, question, chat_rag._vector_search(config, vector))
    ranked: List[Dict[str, Any]] = sorted(candidates, key=lambda c: -c["rerank_score"])
    top: Dict[str, Any] = ranked[0]
    runner: Dict[str, Any] = ranked[1] if len(ranked) > 1 else top
    print(f"{top['rerank_score']:.3f} | runner-up {runner['rerank_score']:.3f} | {str(top.get('document_name'))[:34]:34} | {question[:52]}")
