from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union


class RagInterface(ABC):
    """Abstract interface for retrieval-augmented generation (RAG) operations."""

    @abstractmethod
    def contains_semantic_query_parts(self, query: str) -> bool:
        """
        Determine if the query contains semantic components that warrant reranking.
        :param query: The query string to analyze.
        :return: True if the query contains semantic parts, False otherwise.
        """
        pass

    @abstractmethod
    def is_thalabus_query(self, text: str) -> bool:
        """
        Detect if a string appears to be a thalabus query.
        
        A thalabus query typically contains:
        - Category expressions: CATEGORY_NAME or CATEGORY_NAME=VALUE1,VALUE2
        - Boolean operators: AND, OR, NOT
        - Parentheses for grouping
        - Quoted strings for semantic/exact text search
        - Patterns like UPPERCASE_WITH_UNDERSCORES=value
        
        This method helps distinguish queries from regular conversation text,
        enabling automatic /query command prepending.
        
        :param text: The text to check
        :return: True if the text appears to be a thalabus query, False otherwise
        """
        pass

    @abstractmethod
    def thalabus_query(self, query: str, effective_user_id: Optional[str], ontology_id: Optional[str] = None, subset_of_datasetids: Optional[List[str]] = None, automatically_rerank: bool = True, create_dataset: Optional[Union[bool, str]] = None, session_id: Optional[str] = None, userid: Optional[str] = None, limit_to_documents: Optional[List[str]] = None, limit_to_dossier_id: Optional[str] = None, return_documents: bool = True, llm_filter_prompt: Optional[str] = None, llm_filter_score_threshold: Optional[float] = None) -> Tuple[List[dict], Optional[List[str]]]:
        """
        Search for categories within a given ontology based on a query.
        :param query: The query string in thalabus query language.
        :param ontology_id: The ID of the ontology to search within. If None, searches across any ontology.
        :param subset_of_datasetids: Optional list of dataset IDs to restrict search (currently uses first one).
        :param automatically_rerank: If True, implementations may call a reranking service
                                     after retrieval to reorder/filter results.
        :param create_dataset: If True, create a dataset with auto-generated name. If string, use as dataset name.
        :param session_id: Session ID for dataset creation. Required if create_dataset is set.
        :param userid: User ID for dataset creation. If not provided and session_id is set, extracted from session_id.
        :param effective_user_id: Neo4j User.id of the requesting user. When provided, only documents the effective user
                         holds READ permission on are included in the result — no unauthorized document
                         appears in the returned list (ACL guarantee). When None, ACL filtering is skipped
                         (safe fallback for callers that do not yet supply it).
        :param limit_to_documents: Optional list of document IDs to restrict results to. When provided, only
                                   documents whose ID appears in this list are searched and returned.
                                   If None or empty, no restriction is applied (default behaviour).
        :param limit_to_dossier_id: Optional dossier ID. When provided, search is restricted to documents
                                    owned by that dossier. If limit_to_documents is also provided, the
                                    effective set is the intersection of both.
        :param return_documents: If True (default), aggregate fragments to document-level and return document list.
                                 If False, return raw fragments directly (skipping document aggregation and
                                 Neo4j detail fetch). Dataset is still created when create_dataset is set.
        :param llm_filter_prompt: Optional relevance question for the LLM relevance gate (T33). When
                                  llm_filter_score_threshold is set, every candidate whose final reranked
                                  score is STRICTLY greater than that threshold is evaluated by the
                                  configured DEFAULT model (medium reasoning) against this question:
                                  Documents by full text, kg:// entities by Thalabus serialization.
                                  One request is used when the model's is_reasoning capability supports
                                  self-reasoning, otherwise reasoning is collected first and a second,
                                  separate request converts it into a YES/NO verdict. Requests for
                                  different candidates are issued concurrently, bounded by
                                  MAX_PARALLELISM. Candidates judged
                                  NO are removed from the results and from any dataset created by this
                                  call; candidates judged YES are kept with dataset relationship weight
                                  exactly 1.0. Supplying a prompt without a threshold leaves the gate
                                  disabled (valid, warning logged).
        :param llm_filter_score_threshold: Optional strict lower bound (finite number) on the final
                                  reranked score of candidates to be LLM-filtered (T33). None (default)
                                  disables the gate entirely. Candidates at or below the threshold are
                                  never evaluated and never modified. Gate failures (invalid arguments,
                                  DB errors, oversized content, LLM failures, malformed verdicts) are
                                  never interpreted as NO and never yield an unfiltered success: nothing
                                  is written to the dataset, the query returns an empty result list with
                                  no dataset ids, and the failure is logged as an error.
        :return: A tuple of (results, dataset_ids) where:
                 - results: List of document dicts (return_documents=True) or fragment dicts (return_documents=False),
                   sorted by score in descending order. When effective_user_id is set, every entry has been drawn
                   exclusively from ACL-permitted fragments; no unauthorized document appears in this list.
                 - dataset_ids: List with single dataset ID if dataset was created, None otherwise
        """
        pass

    @abstractmethod
    def thalabus_query_analyser(self, query: str, ontology_id: Optional[str] = None, subset_of_datasetids: Optional[List[str]] = None, automatically_rerank: bool = True) -> Tuple[Dict[str, Any], List[dict]]:
        """
        Analyze the given query and return an analysis of its components, performed in this way:
        1. runs the query and identifies all results
        2. gathers all CATEGORIES from the results
        3. for each category, counts how many results belong to it
        4. for each category where the values are multiple, aggregates the counts
        5. returns a dictionary with the following structure:
           {
               "total_results": int,  # total number of results found
               "categories": {
                   "category_name_1": {
                       "count": int,  # number of results in this category
                       "values": {
                           "value_1": int,  # count of results with this value
                           "value_2": int,
                           ...
                        }
                    },
                    "category_name_2": {
                        "count": int,
                        "values": {
                        "value_1": int,
                        "value_2": int
                        }
                    },
                   ...
               }
           }

        :param query: The query string to analyze.
        :return: A tuple of:
            - dictionary containing the analysis results,
            - results list as returned by thalabus_query.
        """
        pass

    @abstractmethod
    def extend_dataset_with_linked_entities(self, dataset_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Extend a document-fragment dataset with KG entities linked to its documents.

        For every document fragment in the dataset, walks the IS_DOCUMENTED chain backwards
        (paragraph -> chapter -> document -> entity) and merges each linked entity into the
        dataset as a kg:// fragment: entities not yet present are added with the fragment's
        weight (the maximum weight among fragments of the same document); entities already
        present are updated only when the new weight is higher.

        :param dataset_id: The id of the dataset to extend.
        :param effective_user_id: Optional Neo4j user id reserved for ACL alignment with other RAG calls.
        :return: Dict with 'success', 'candidates', 'added', 'bumped', 'linked' and 'dataset_id'.
        """
        pass
