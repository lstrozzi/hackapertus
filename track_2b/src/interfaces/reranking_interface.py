from __future__ import annotations

from typing import List, Optional, Dict, Any


class RerankingInterface:
    """Interface for reranking services.

    Concrete implementations should provide a `rerank` method which accepts a
    query (string), a list of fragment ids (or fragment dicts), an endpoint
    and model identifier, and an optional top_n limit. The method should
    return an ordered list of dicts with at least 'fragmentid' and 'score'/
    'ranking' keys, or None if reranking cannot be performed.
    """

    def rerank(self, query: str, fragments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Rerank provided fragments for the given query.

        Args:
            query: Query text to compare against fragments.
            fragments: List of fragment dictionaries. Each dictionary must follow this
                specification:
                  - "id": string, mandatory, same as fragment_id
                  - "score": float, optional -- the RAG embeddings similarity score
                  - "text": string, optional -- the entire fragment's text

        Returns:
            Ordered list of dicts with keys such as 'fragmentid' and 'score'/'ranking',
            or None if reranking cannot be performed.
        """
        raise NotImplementedError()

    def rerank_multilang(self, query: str, fragments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Rerank fragments using multi-language queries.
        
        Translates the query into multiple languages (EN, DE, FR, IT) and performs
        reranking against each translation. For each fragment, the highest score
        across all translations is used.
        
        Args:
            query: Query text to compare against fragments.
            fragments: List of fragment dictionaries with "id", "score", "text" fields.
            
        Returns:
            Ordered list of reranked fragments, or None if reranking fails.
        """
        raise NotImplementedError()

    def rerank_documents(self, query: str, document_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Rerank documents by their chapters using parallel processing.

        This method fetches all chapters for the provided documents from the documents
        service, reranks them in parallel using ThreadPoolExecutor, and returns the
        MAX ranking score achieved by any chapter within each document.

        Args:
            query: Query text to compare against document chapters.
            document_ids: List of document IDs to rerank.

        Returns:
            Ordered list of dicts with keys: id (document id), ranking (max chapter ranking),
            score, or None if reranking cannot be performed.
        """
        raise NotImplementedError()

    def rerank_strings(self, query: str, strings: List[str], must_return_top_n: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Rerank provided strings for the given query.

        Args:
            query: Query text to compare against strings.
            strings: List of strings to rerank.
            must_return_top_n: When True (default), forces at least MIN_RERANKING_OUTPUT_DOC results
                even if no string meets the minimum score threshold — useful for RAG where some
                result is always required. When False, returns None if no string passes the
                threshold — appropriate for hints where irrelevant results should be suppressed.
        Returns:
            Ordered list of dicts with keys such as 'string' and 'score'/'ranking',
            or None if reranking cannot be performed.
        """
        raise NotImplementedError()