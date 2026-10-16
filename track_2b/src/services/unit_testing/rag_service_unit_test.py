from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import os
import uuid
import json
import datetime
import traceback
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports to avoid circular module import at import-time
from src.services.rag_service import get_rag_service
from src.services.ontology_service import get_ontology_service
from src.services.dossiers_service import get_dossiers_service
from src.core.base.InputOutput import execute_neo4j_query
from src.core.base.enums import AgentReasoningEnum
from src.core.llm import LlmManager, MAX_COMPLETION_TOKENS_MEDIUM
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL
from src.config.constants import MAX_PARALLELISM


def _fetch_documents(dossier_id: str):
    dossier_service = get_dossiers_service()
    docs = dossier_service.get_documents_of_dossier(dossier_id, include_details=True)
    if not docs:
        log(ERROR, f"No documents found in dossier {dossier_id}")
        return []
    log(INFO, f"Found {len(docs)} documents in dossier")
    return docs


def _fetch_categories(ontology_service, ontology_id: str):
    categories = {}
    if not ontology_service:
        log(ERROR, "Ontology service not available")
        return categories

    categories_result = ontology_service.get_categories_for_ontology(ontology_id, exclude_proposed=True)
    if not categories_result:
        return categories

    for record in categories_result:
        cat_type = None
        values = []
        if isinstance(record, dict):
            if record.get('text_en'):
                cat_type = record.get('text_en')
            else:
                for key in ('category_type', 'category', 'name', 'text_it', 'text_de', 'text_fr', 'description', 'id'):
                    if key in record and record.get(key):
                        cat_type = record.get(key)
                        break

            if 'values' in record and record.get('values'):
                values = record.get('values') or []
            elif 'values_list' in record and record.get('values_list'):
                values = record.get('values_list') or []
            elif 'lov' in record and record.get('lov'):
                lov_val = record.get('lov')
                if isinstance(lov_val, (list, tuple)):
                    values = [str(v).strip() for v in lov_val if v is not None]
                else:
                    values = [v.strip() for v in str(lov_val).split(',') if v.strip()]

        if cat_type:
            categories[cat_type] = values or None
            if values:
                log(INFO, f"  Category: {cat_type} ({len(values)} values)")
            else:
                log(INFO, f"  Category: {cat_type} (no values)")

    if not categories:
        log(WARNING, "No categories found in ontology, will only generate content-based tests")

    return categories


def _build_categories_text(categories: dict) -> str:
    return "\n".join([
        f"- {cat_type}: {', '.join([str(v) for v in (values or ['<no specific values>'])[:20]])}"
        for cat_type, values in categories.items()
    ])


def _select_documents(documents: list, documents_limit: int) -> list:
    total_docs = len(documents)
    if documents_limit == -1:
        to_process_count = total_docs
    else:
        to_process_count = max(0, min(int(documents_limit), total_docs))

    log(INFO, f"Generating AI-based test questions for up to {to_process_count} documents...")

    if to_process_count <= 0:
        return []
    if to_process_count >= total_docs:
        return documents.copy()
    try:
        return random.sample(documents, to_process_count)
    except Exception:
        return documents[:to_process_count]


def _filter_existing_tests(docs_to_check: list, questions_per_category: int) -> list:
    try:
        doc_ids = [d.get('document_id') or d.get('id') for d in docs_to_check if (d.get('document_id') or d.get('id'))]
        if not doc_ids:
            return docs_to_check

        cypher_check = """
            MATCH (d:Document)
            WHERE d.id IN $ids OR d.documentid IN $ids
            OPTIONAL MATCH (d)<-[:HAS_TEST]-(t:TestQuestionRag)
            RETURN d.id AS id, count(t) AS test_count, collect(t.id) AS test_ids
        """
        params = {"ids": doc_ids}
        check_results = execute_neo4j_query(cypher_check, params)

        expected_tests = max(1, int(questions_per_category)) * 3

        docs_to_skip = set()
        test_ids_to_delete = []

        if check_results:
            for rec in check_results:
                docid = rec.get('id')
                test_count = rec.get('test_count', 0) or 0
                test_ids = rec.get('test_ids') or []

                if test_count >= expected_tests:
                    # Enough tests: skip document
                    if docid:
                        docs_to_skip.add(docid)
                elif 0 < test_count < expected_tests:
                    # Partial tests: delete existing tests and re-generate
                    # Collect test ids to delete (filter out falsy values)
                    for tid in test_ids:
                        if tid:
                            test_ids_to_delete.append(tid)
                else:
                    # test_count == 0 -> leave document to be processed
                    pass

        # Delete partial test nodes if any
        if test_ids_to_delete:
            # Remove duplicates
            unique_tids = list({str(tid) for tid in test_ids_to_delete})
            try:
                delete_cypher = """
                    UNWIND $tids AS tid
                    MATCH (t:TestQuestionRag {id: tid})
                    DETACH DELETE t
                """
                execute_neo4j_query(delete_cypher, {"tids": unique_tids})
                log(INFO, f"Deleted {len(unique_tids)} partial TestQuestionRag nodes so documents will be regenerated")
            except Exception as e:
                log(ERROR, f"Failed to delete partial TestQuestionRag nodes: {e}")

        # Return only documents that do NOT have enough tests (i.e., to be processed)
        if docs_to_skip:
            original_count = len(docs_to_check)
            docs_to_check = [d for d in docs_to_check if ((d.get('document_id') or d.get('id')) not in docs_to_skip)]
            skipped = original_count - len(docs_to_check)
            log(INFO, f"Skipping {skipped} documents with >= {expected_tests} existing tests")

    except Exception as e:
        log(ERROR, f"Error checking existing TestQuestionRag links: {e}")

    return docs_to_check


def _generate_single_query(llm_manager, doc, doc_index, total_docs, query_type, categories_text, excluded_queries=None):
    if excluded_queries is None:
        excluded_queries = []
    
    doc_id = doc.get('document_id') or doc.get('id')
    title = doc.get('title', '')
    description = doc.get('description', '')
    text = doc.get('text', '')

    doc_context = f"""Title: {title}\n\nDescription: {description}\n\nText: {text}"""
    log(INFO, f"Processing document {doc_index}/{total_docs}: {doc_id} - {query_type}")

    system_prompt = f"""You are an expert at generating test queries for a document search system using the Thalabus Query Language.

THALABUS QUERY LANGUAGE SYNTAX (KEY POINTS):

1. Use Category filters, Semantic Search (double quotes), exact text (single quotes), @'case' for case-sensitive, boolean operators (`AND`, `OR`, `NOT`), parentheses, and the `TRUE` special keyword.
2. When combining terms, ALWAYS include an explicit boolean operator (`AND` or `OR`) or use parentheses. NEVER place two quoted terms or standalone terms adjacent without an operator (that is invalid syntax).
3. Prefer clear, well-formed queries that would be parsable by the Thalabus query parser.

AVAILABLE CATEGORIES FOR THIS ONTOLOGY:
{categories_text}

Your task is to generate a single test query that would retrieve the provided document or similar documents. Follow the rules precisely and return only a valid Thalabus query string in the `query` field of the structured response."""

    if query_type == "SEMANTIC":
        user_prompt = ("Generate a single SEMANTIC-only Thalabus Query Language query. "
                       "Return a JSON object with fields: description and query.")
    elif query_type == "CATEGORIES":
        user_prompt = ("Generate a single CATEGORIES-only Thalabus Query Language query. "
                       "Return a JSON object with fields: description and query.")
    else:
        user_prompt = ("Generate a single MIXED Thalabus Query Language query combining semantic/text with categories. "
                       "Return a JSON object with fields: description and query.")

    # Strong syntax requirements to avoid invalid bare adjacency of terms
    user_prompt += (
        "\n\nIMPORTANT SYNTAX RULES:\n"
        "- When combining multiple terms, always use explicit boolean operators `AND` or `OR`, or group with parentheses.\n"
        "- Do NOT place quoted terms or standalone words side-by-side without an operator (for example, the following is INVALID: \"\"cat\" \"spilled milk\"\"; instead use: \"\"cat\" AND \"spilled milk\"\" or \"\"cat\" OR \"spilled milk\"\").\n"
        "- Use `@'text'` for case-sensitive exact matches when needed.\n"
        "- If the query is semantic-only, wrap the phrase in double quotes (\"...\") and avoid inserting raw category tokens without operators.\n"
        "\nProvide a brief `description` explaining the intent."
    )
    
    # Add instruction to avoid duplicate queries if we have previously generated ones
    if excluded_queries:
        excluded_str = '\n'.join([f"- {q}" for q in excluded_queries[:10]])  # Show first 10 to avoid token overload
        excluded_note = f"\n\nIMPORTANT: Do NOT generate any of these previously generated queries:\n{excluded_str}"
        if len(excluded_queries) > 10:
            excluded_note += f"\n(and {len(excluded_queries) - 10} more...)"
        user_prompt += excluded_note

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"DOCUMENT CONTEXT:\n{doc_context}"},
            {"role": "user", "content": user_prompt}
        ]

        json_output_example = {
            "description": f"{query_type} query for document",
            "query": '"example query"' if query_type == "SEMANTIC" else "CATEGORY=VALUE"
        }

        json_schema = {
            "name": "schema_query_test",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Thalabus query string"},
                    "description": {"type": "string", "description": "Brief explanation of the query intent"}
                },
                "required": ["query", "description"],
                "additionalProperties": False
            },
            "strict": True
        }

        response = llm_manager.chat_completions(
            messages=messages,
            temperature=0.7,
            max_completion_tokens=MAX_COMPLETION_TOKENS_MEDIUM,
            structured_output=json_schema,
            response_format=json_output_example,
            maximum_reasoning_required=AgentReasoningEnum.SIMPLE
        )

        if response.response is None or not isinstance(response.response, dict):
            log(ERROR, f"LLM returned invalid response for document {doc_id} - {query_type}")
            return None

        content: dict = response.response
        query_value = content.get('query') or content.get('thalabus_query')
        if not query_value:
            log(ERROR, f"Structured LLM response missing query field for {doc_id} - {query_type}: {content}")
            return None

        return {
            'query': query_value,
            'description': content.get('description', f'AI-generated {query_type} query'),
            'type': 'AI_GENERATED',
            'base_doc_id': doc_id
        }

    except Exception as e:
        log(ERROR, f"Failed to generate query for {doc_id} - {query_type}: {e}")
        log(ERROR, f"Traceback: {traceback.format_exc()}")
        return None


def _create_tasks(docs_to_check: list, questions_per_category: int):
    tasks = []
    query_types = ["SEMANTIC", "CATEGORIES", "MIXED"]
    per_cat = max(1, int(questions_per_category))
    for i, doc in enumerate(docs_to_check):
        doc_num = i + 1
        total = len(docs_to_check)
        for qtype in query_types:
            for _ in range(per_cat):
                tasks.append((doc, doc_num, total, qtype))
    return tasks


def _run_query_tasks(tasks: list, llm_manager, categories_text: str, excluded_queries: list = None) -> list:
    if excluded_queries is None:
        excluded_queries = []
    
    test_questions = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_task = {}
        for task in tasks:
            doc, doc_num, total, qtype = task
            future = executor.submit(_generate_single_query, llm_manager, doc, doc_num, total, qtype, categories_text, excluded_queries)
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    test_questions.append(result)
            except Exception as e:
                doc_id = task[0].get('document_id')
                query_type = task[3]
                log(ERROR, f"Task failed for {doc_id} - {query_type}: {e}")

    return test_questions


def _execute_single_test(service, test_q: dict, idx: int, total: int, ontology_id: str, dossier_id: str) -> dict:
    """Execute a single test query and store results in Neo4j.
    
    Args:
        service: RAG service instance
        test_q: Test question dict with 'query', 'description', 'type', 'base_doc_id'
        idx: Test index (1-based for logging)
        total: Total number of tests
        ontology_id: Ontology ID
        dossier_id: Dossier ID
        
    Returns:
        Test result dict with status, results, and Neo4j node IDs
    """
    query = test_q['query']
    description = test_q['description']
    test_type = test_q['type']
    base_doc_id = test_q['base_doc_id']

    log(INFO, f"Test {idx}/{total}: {query}")

    try:
        results, _ = service.thalabus_query(query, None, ontology_id, dossier_id)
        if results is None:
            # Query is invalid (e.g., unterminated quotes); create test node with invalid=true flag
            log(ERROR, f"  Query is invalid, marking node as invalid: {query}")
            
            test_id = f"test-rag-{uuid.uuid4().hex[:12]}"
            
            cypher_create = """
                MERGE (t:TestQuestionRag {id: $test_id})
                SET t.thalabus_query = $query,
                    t.description = $description,
                    t.test_type = $test_type,
                    t.status = 'INVALID',
                    t.invalid = true,
                    t.result_count = 0,
                    t.created_at = datetime()
            """
            execute_neo4j_query(cypher_create, {
                "test_id": test_id,
                "query": query,
                "description": description,
                "test_type": test_type
            })
            
            if base_doc_id:
                cypher_base_rel = """
                    MATCH (t:TestQuestionRag {id: $test_id})
                    MATCH (d:Document)
                    WHERE d.id = $base_doc_id OR d.documentid = $base_doc_id
                    MERGE (t)<-[:HAS_TEST]-(d)
                """
                execute_neo4j_query(cypher_base_rel, {
                    "test_id": test_id,
                    "base_doc_id": base_doc_id
                })
            
            return {
                'test_id': test_id,
                'query': query,
                'description': description,
                'test_type': test_type,
                'status': 'INVALID',
                'result_count': 0,
                'results': [],
                'base_doc_id': base_doc_id,
                'exception': 'Invalid query (unterminated quotes or parse error)'
            }

        num_results = len(results)

        if num_results == 0:
            status = "NEW"
        elif base_doc_id and any(r['id'] == base_doc_id for r in results):
            status = "CORRECT_ANSWER"
        elif base_doc_id:
            status = "WRONG_ANSWER"
        else:
            status = "NEW"

        test_id = f"test-rag-{uuid.uuid4().hex[:12]}"

        cypher_create = """
            MERGE (t:TestQuestionRag {id: $test_id})
            SET t.thalabus_query = $query,
                t.description = $description,
                t.test_type = $test_type,
                t.status = $status,
                t.result_count = $result_count,
                t.created_at = datetime()
        """
        execute_neo4j_query(cypher_create, {
            "test_id": test_id,
            "query": query,
            "description": description,
            "test_type": test_type,
            "status": status,
            "result_count": num_results
        })

        if base_doc_id:
            cypher_base_rel = """
                MATCH (t:TestQuestionRag {id: $test_id})
                MATCH (d:Document)
                WHERE d.id = $base_doc_id OR d.documentid = $base_doc_id
                MERGE (t)<-[:HAS_TEST]-(d)
            """
            execute_neo4j_query(cypher_base_rel, {
                "test_id": test_id,
                "base_doc_id": base_doc_id
            })

        valid_doc_ids = [r['id'] for r in results]
        for valid_doc_id in valid_doc_ids:
            cypher_valid_rel = """
                MATCH (t:TestQuestionRag {id: $test_id})
                MATCH (d:Document {id: $valid_doc_id})
                MERGE (t)-[:VALID_ANSWER]->(d)
            """
            execute_neo4j_query(cypher_valid_rel, {
                "test_id": test_id,
                "valid_doc_id": valid_doc_id
            })

        log(INFO, f"  Status: {status}, Results: {num_results}")

        return {
            'test_id': test_id,
            'query': query,
            'description': description,
            'test_type': test_type,
            'status': status,
            'result_count': num_results,
            'results': results,
            'base_doc_id': base_doc_id,
            'exception': None
        }

    except Exception as e:
        log(ERROR, f"  Error executing test: {e}")
        return {
            'test_id': None,
            'query': query,
            'description': description,
            'test_type': test_type,
            'status': 'ERROR',
            'result_count': 0,
            'results': [],
            'base_doc_id': base_doc_id,
            'exception': str(e)
        }


def _execute_and_store_tests(service, test_questions: list, ontology_id: str, dossier_id: str) -> list:
    """Execute and store test questions in parallel using ThreadPoolExecutor.
    
    Skips any tests marked with invalid=true to avoid executing invalid queries.
    
    Args:
        service: RAG service instance
        test_questions: List of test question dicts
        ontology_id: Ontology ID
        dossier_id: Dossier ID
        
    Returns:
        List of test result dicts
    """
    test_results = []
    
    # Filter out tests marked as invalid before executing
    valid_tests = [q for q in test_questions if not q.get('invalid', False)]
    invalid_tests = [q for q in test_questions if q.get('invalid', False)]
    
    if invalid_tests:
        log(INFO, f"Skipping {len(invalid_tests)} invalid test(s) marked with invalid=true")
    
    total = len(valid_tests)
    
    if total == 0:
        log(INFO, f"No valid tests to execute (all {len(test_questions)} tests were marked invalid)")
        return test_results
    
    # Use ThreadPoolExecutor with MAX_PARALLELISM workers
    max_workers = min(MAX_PARALLELISM, total)
    log(INFO, f"Executing {total} valid tests with {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, test_q in enumerate(valid_tests):
            future = executor.submit(_execute_single_test, service, test_q, idx + 1, total, ontology_id, dossier_id)
            future_to_idx[future] = idx
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                test_results.append(result)
            except Exception as e:
                test_q = valid_tests[idx]
                log(ERROR, f"Task failed for test {idx + 1}: {e}")
                test_results.append({
                    'test_id': None,
                    'query': test_q.get('query', ''),
                    'description': test_q.get('description', ''),
                    'test_type': test_q.get('type', ''),
                    'status': 'ERROR',
                    'result_count': 0,
                    'results': [],
                    'base_doc_id': test_q.get('base_doc_id'),
                    'exception': str(e)
                })
    
    # Sort results by original test index for consistent ordering
    # Create mapping of query to original index to preserve order (use valid_tests for mapping)
    query_to_orig_idx = {test_q['query']: idx for idx, test_q in enumerate(valid_tests)}
    test_results.sort(key=lambda r: query_to_orig_idx.get(r['query'], 999))
    
    return test_results


def _deduplicate_test_questions(test_questions: list) -> list:
    """Remove duplicate queries while preserving order and keeping first occurrence.
    
    Two queries are considered duplicates if they have the same thalabus_query string.
    
    Args:
        test_questions: List of test question dicts with 'query', 'description', 'type', 'base_doc_id'
        
    Returns:
        List of deduplicated test questions, preserving original order
    """
    if not test_questions:
        return test_questions
    
    seen_queries = set()
    deduplicated = []
    duplicate_count = 0
    
    for test_q in test_questions:
        query = test_q.get('query', '')
        if query not in seen_queries:
            seen_queries.add(query)
            deduplicated.append(test_q)
        else:
            duplicate_count += 1
    
    if duplicate_count > 0:
        log(INFO, f"Removed {duplicate_count} duplicate queries ({len(deduplicated)} unique remain)")
    
    return deduplicated


def _write_report(report_file: str, dossier_id: str, ontology_id: str, test_results: list) -> None:
    report_dir = os.path.dirname(report_file)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAG Test Generation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dossier: {dossier_id}\n")
        f.write(f"Ontology: {ontology_id}\n")
        f.write(f"Total Tests: {len(test_results)}\n\n")

        status_counts = {}
        for result in test_results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        f.write("Summary by Status:\n")
        f.write("-" * 40 + "\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"  {status}: {count}\n")
        f.write("\n")

        type_counts = {}
        for result in test_results:
            test_type = result['test_type']
            type_counts[test_type] = type_counts.get(test_type, 0) + 1

        f.write("Summary by Test Type:\n")
        f.write("-" * 40 + "\n")
        for test_type, count in sorted(type_counts.items()):
            f.write(f"  {test_type}: {count}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Detailed Test Results\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(test_results):
            f.write(f"Test {idx + 1}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test ID: {result['test_id']}\n")
            f.write(f"Query: {result['query']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"Type: {result['test_type']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Result Count: {result['result_count']}\n")

            if result['base_doc_id']:
                f.write(f"Base Document: {result['base_doc_id']}\n")

            if result['exception']:
                f.write(f"Exception: {result['exception']}\n")

            if result['results']:
                f.write(f"\nReturned Documents:\n")
                for doc in result['results'][:10]:
                    doc_id = doc.get('id')
                    title = doc.get('title', 'N/A')
                    score = doc.get('score', 0.0)
                    f.write(f"  - {doc_id} (score: {score:.4f}): {title}\n")
                if len(result['results']) > 10:
                    f.write(f"  ... and {len(result['results']) - 10} more\n")

            f.write("\n")


def _filter_documents_with_categories(documents: list) -> list:
    """Return only documents that have at least one :HAS_CATEGORY -> (c:Category) relationship.

    Uses the `categories` field returned by `get_documents_of_dossier(include_details=True)`
    which contains collected category dicts when categories exist.
    """
    if not documents:
        return []

    filtered = []
    for d in documents:
        cats = d.get('categories')
        if not cats:
            continue

        # Ensure there is at least one non-empty category entry
        has_valid = False
        if isinstance(cats, (list, tuple)) and len(cats) > 0:
            for c in cats:
                if not c:
                    continue
                if isinstance(c, dict):
                    # Accept if any expected category fields are present
                    if c.get('text') or c.get('category_text_en') or c.get('category_id'):
                        has_valid = True
                        break
                else:
                    # Non-dict truthy entry
                    has_valid = True
                    break

        if has_valid:
            filtered.append(d)

    log(INFO, f"Filtering documents by categories: {len(filtered)} of {len(documents)} have categories")
    return filtered


def generate_rag_test_questions(dossier_id: str = "dossier-jurispub",
                                ontology_id: str = "Juris",
                                report_file: str = "reports/rag_test_generation_report.txt",
                                documents_limit: int = 20,
                                questions_per_category: int = 3,
                                target_questions: int = 12):
    """Orchestrator that generates RAG test queries and stores results.

    This function coordinates fetching documents and categories, generating
    AI-based queries, executing them, storing results, and writing the report.
    
    Args:
        dossier_id: ID of the dossier to generate tests for
        ontology_id: ID of the ontology
        report_file: Path to output report file
        documents_limit: Maximum number of documents to process
        questions_per_category: Number of questions per category type per document
        target_questions: Target number of valid test questions to generate (default: 12)
    """
    service = get_rag_service()
    ontology_service = get_ontology_service()

    log(INFO, f"{'=' * 80}\nGenerating RAG Test Questions\n{'=' * 80}")
    log(INFO, f"Dossier: {dossier_id}")
    log(INFO, f"Ontology: {ontology_id}")
    log(INFO, f"Target valid questions: {target_questions}")

    documents = _fetch_documents(dossier_id)
    if not documents:
        return

    categories = _fetch_categories(ontology_service, ontology_id)
    categories_text = _build_categories_text(categories)

    llm_manager = LlmManager.instance()

    docs_to_check = _select_documents(documents, documents_limit)
    docs_to_check = _filter_existing_tests(docs_to_check, questions_per_category)

    # Keep only documents that have been categorized (i.e., have a :HAS_CATEGORY -> (c:Category) link)
    docs_to_check = _filter_documents_with_categories(docs_to_check)
    if not docs_to_check:
        log(INFO, "No categorized documents found in the selected dossier after filtering; nothing to generate")
        return

    # Ensure we do not process more documents than requested after filtering
    try:
        if documents_limit != -1 and len(docs_to_check) > int(documents_limit):
            log(INFO, f"Limiting documents to {documents_limit} after filtering existing tests (was {len(docs_to_check)})")
            # preserve original order where possible
            docs_to_check = docs_to_check[:int(documents_limit)]
    except Exception:
        pass

    # Regeneration loop to reach target valid questions
    all_test_questions = []
    excluded_queries = []
    max_regeneration_attempts = 5
    regeneration_attempt = 0
    
    while len(all_test_questions) < target_questions and regeneration_attempt < max_regeneration_attempts:
        regeneration_attempt += 1
        log(INFO, f"\n--- Question Generation Attempt {regeneration_attempt} ---")
        
        tasks = _create_tasks(docs_to_check, questions_per_category)
        log(INFO, f"Created {len(tasks)} tasks ({max(1,int(questions_per_category))} per type) for {len(docs_to_check)} documents")

        # Pass excluded_queries to avoid regenerating the same ones
        test_questions = _run_query_tasks(tasks, llm_manager, categories_text, excluded_queries)
        log(INFO, f"Generated {len(test_questions)} test questions from LLM")

        # Remove duplicate queries that may have been generated in parallel
        test_questions_before_dedup = len(test_questions)
        test_questions = _deduplicate_test_questions(test_questions)
        log(INFO, f"After deduplication: {len(test_questions)} questions (removed {test_questions_before_dedup - len(test_questions)} duplicates)")

        # Add new questions to collection
        all_test_questions.extend(test_questions)
        
        # Update excluded queries with all generated ones so far (only valid ones)
        excluded_queries = [q.get('query', '') for q in all_test_questions if q.get('query')]
        
        valid_count = len([q for q in all_test_questions if not q.get('invalid', False)])
        log(INFO, f"Progress: {valid_count} valid questions generated (target: {target_questions})")
        
        if valid_count >= target_questions:
            log(INFO, f"Reached target of {target_questions} valid questions")
            break
        elif regeneration_attempt < max_regeneration_attempts:
            log(INFO, f"Need {target_questions - valid_count} more questions, regenerating with deduplication...\n")

    test_results = _execute_and_store_tests(service, all_test_questions, ontology_id, dossier_id)

    log(INFO, "\n--- Step 5: Generating report ---")
    _write_report(report_file, dossier_id, ontology_id, test_results)
    log(INFO, f"Report generated: {report_file}")
    log(INFO, f"\n{'=' * 80}\nTest generation complete\n{'=' * 80}")


def _fetch_test_questions_with_new_status(dossier_id: str, limit: int = -1) -> list:
    """Fetch all TestQuestionRag nodes with 'NEW' status for a dossier.
    
    Args:
        dossier_id: Dossier ID to fetch tests for
        limit: Maximum number of tests to fetch (-1 for unlimited)
        
    Returns:
        List of test question dicts with fields: test_id, query, description, test_type, base_doc_id
    """
    try:
        # Get all documents in the dossier
        dossier_service = get_dossiers_service()
        docs = dossier_service.get_documents_of_dossier(dossier_id) or []
        
        if not docs:
            log(INFO, f"No documents found in dossier '{dossier_id}'")
            return []
        
        doc_ids = [d.get('id') or d.get('document_id') for d in docs if (d.get('id') or d.get('document_id'))]
        
        if not doc_ids:
            log(INFO, f"No valid document IDs in dossier '{dossier_id}'")
            return []
        
        # Fetch all TestQuestionRag nodes with 'NEW' status linked to these documents
        cypher_query = """
            MATCH (d:Document)-[:HAS_TEST]->(t:TestQuestionRag)
            WHERE (d.id IN $doc_ids OR d.documentid IN $doc_ids)
            AND t.status = 'NEW'
            RETURN t.id AS test_id, t.thalabus_query AS query, t.description AS description, 
                   t.test_type AS test_type, d.id AS base_doc_id
            ORDER BY t.created_at ASC
        """
        
        params = {"doc_ids": doc_ids}
        results = execute_neo4j_query(cypher_query, params)
        
        if not results:
            log(INFO, f"No test questions with 'NEW' status found in dossier '{dossier_id}'")
            return []
        
        # Convert Neo4j records to dictionaries
        test_questions = []
        for record in results:
            test_q = {
                'test_id': record.get('test_id'),
                'query': record.get('query'),
                'description': record.get('description'),
                'test_type': record.get('test_type'),
                'base_doc_id': record.get('base_doc_id')
            }
            test_questions.append(test_q)
        
        # Apply limit if specified
        if limit != -1:
            test_questions = test_questions[:limit]
        
        log(INFO, f"Fetched {len(test_questions)} test questions with 'NEW' status from dossier '{dossier_id}'")
        return test_questions
        
    except Exception as e:
        log(ERROR, f"Error fetching test questions: {e}")
        return []


def _execute_single_test_question(service, test_q: dict, idx: int, total: int, ontology_id: str, dossier_id: str) -> dict:
    """Execute a single test question using thalabus_query.
    
    Args:
        service: RAG service instance
        test_q: Test question dict with 'test_id', 'query', 'description', 'test_type', 'base_doc_id'
        idx: Test index (1-based for logging)
        total: Total number of tests
        ontology_id: Ontology ID for query
        dossier_id: Dossier ID for query
        
    Returns:
        Test result dict with test_id, status, and results
    """
    test_id = test_q['test_id']
    query = test_q['query']
    base_doc_id = test_q['base_doc_id']
    
    log(INFO, f"Executing test {idx}/{total}: {test_id}")
    
    try:
        # Execute the query
        results, _ = service.thalabus_query(query, None, ontology_id, dossier_id)
        
        if results is None or not isinstance(results, list):
            log(WARNING, f"  Query returned invalid results: {results}")
            return {
                'test_id': test_id,
                'status': 'EXECUTION_FAILED',
                'result_count': 0,
                'results': [],
                'exception': 'Query returned invalid results'
            }
        
        result_count = len(results)
        result_ids = [r.get('id') for r in results if r.get('id')]
        
        # Determine matching status
        if result_count == 0:
            status = 'MATCHING_FAILED'
        elif base_doc_id and result_ids[0] == base_doc_id:
            status = 'MATCHING'
        elif base_doc_id and base_doc_id in result_ids[:5]:
            status = 'MATCHING_IN_TOP_5'
        else:
            status = 'MATCHING_FAILED'
        
        log(INFO, f"  Status: {status}, Results: {result_count}")
        
        return {
            'test_id': test_id,
            'status': status,
            'result_count': result_count,
            'results': results,
            'exception': None
        }
        
    except Exception as e:
        log(ERROR, f"  Error executing test: {e}")
        return {
            'test_id': test_id,
            'status': 'EXECUTION_FAILED',
            'result_count': 0,
            'results': [],
            'exception': str(e)
        }


def _update_test_question_status(test_id: str, status: str) -> bool:
    """Update the status of a test question in Neo4j.
    
    Args:
        test_id: ID of the test question node
        status: New status to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cypher_query = """
            MATCH (t:TestQuestionRag {id: $test_id})
            SET t.status = $status, t.executed_at = datetime()
            RETURN t.id AS id
        """
        
        result = execute_neo4j_query(cypher_query, {"test_id": test_id, "status": status})
        return bool(result)
        
    except Exception as e:
        log(ERROR, f"Error updating test status for {test_id}: {e}")
        return False


def execute_rag_tests(dossier_id: str, ontology_id: str, test_limit: int = -1) -> dict:
    """Execute RAG tests from TestQuestionRag nodes with 'NEW' status.
    
    Runs tests in parallel using MAX_PARALLELISM workers. Updates Neo4j node status
    based on whether the top result matches the expected document.
    
    Args:
        dossier_id: Dossier ID
        ontology_id: Ontology ID for query execution
        test_limit: Maximum number of tests to run (-1 for unlimited)
        
    Returns:
        Dict with summary stats: {
            'total_tests': int,
            'executed': int,
            'matching': int,
            'matching_in_top_5': int,
            'matching_failed': int,
            'execution_failed': int
        }
    """
    log(INFO, f"{'=' * 80}\nExecuting RAG Tests\n{'=' * 80}")
    log(INFO, f"Dossier: {dossier_id}")
    log(INFO, f"Ontology: {ontology_id}")
    log(INFO, f"Test limit: {test_limit}")
    
    service = get_rag_service()
    
    # Fetch tests with NEW status
    test_questions = _fetch_test_questions_with_new_status(dossier_id, test_limit)
    
    if not test_questions:
        log(INFO, "No tests to execute")
        return {
            'total_tests': 0,
            'executed': 0,
            'matching': 0,
            'matching_in_top_5': 0,
            'matching_failed': 0,
            'execution_failed': 0
        }
    
    total = len(test_questions)
    log(INFO, f"\nExecuting {total} tests with {min(MAX_PARALLELISM, total)} parallel workers")
    
    # Execute tests in parallel
    test_results = []
    max_workers = min(MAX_PARALLELISM, total)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, test_q in enumerate(test_questions):
            future = executor.submit(_execute_single_test_question, service, test_q, idx + 1, total, ontology_id, dossier_id)
            future_to_idx[future] = idx
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                test_results.append(result)
            except Exception as e:
                log(ERROR, f"Error in test execution: {e}")
    
    # Sort results by original test index for consistent ordering
    test_results.sort(key=lambda r: next((i for i, q in enumerate(test_questions) if q['test_id'] == r['test_id']), 999))
    
    # Update Neo4j with results
    log(INFO, f"\nUpdating test question statuses in Neo4j...")
    for result in test_results:
        _update_test_question_status(result['test_id'], result['status'])
    
    # Calculate summary statistics
    summary = {
        'total_tests': total,
        'executed': len(test_results),
        'matching': len([r for r in test_results if r['status'] == 'MATCHING']),
        'matching_in_top_5': len([r for r in test_results if r['status'] == 'MATCHING_IN_TOP_5']),
        'matching_failed': len([r for r in test_results if r['status'] == 'MATCHING_FAILED']),
        'execution_failed': len([r for r in test_results if r['status'] == 'EXECUTION_FAILED'])
    }
    
    # Print summary
    log(INFO, f"\n{'=' * 80}\nTest Execution Summary\n{'=' * 80}")
    log(INFO, f"Total tests: {summary['total_tests']}")
    log(INFO, f"Executed: {summary['executed']}")
    log(INFO, f"  - MATCHING: {summary['matching']}")
    log(INFO, f"  - MATCHING_IN_TOP_5: {summary['matching_in_top_5']}")
    log(INFO, f"  - MATCHING_FAILED: {summary['matching_failed']}")
    log(INFO, f"  - EXECUTION_FAILED: {summary['execution_failed']}")
    
    success_rate = (summary['matching'] + summary['matching_in_top_5']) / summary['executed'] * 100 if summary['executed'] > 0 else 0
    log(INFO, f"Success rate (MATCHING + MATCHING_IN_TOP_5): {success_rate:.1f}%")
    log(INFO, f"{'=' * 80}\n")
    
    return summary


log_end_file(__file__)
