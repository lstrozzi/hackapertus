# Slice of `PythonServer/src` (RagService + KgFromDocument skill), prepared for release

**Status: open source is the intent; the licence itself is still to be chosen.**
See *Before publishing*.

| Field | Value |
| ----- | ----- |
| Source branch | `feat-link-kg-to-doc` |
| Source commit | `4d4698f82f5816a9d955152fbdd6e2a956545eda` |
| Working tree at extraction | 0 changed file(s) under `PythonServer/src` |
| Extracted | 2026-10-14 04:15 UTC |
| Extracted on | Windows-10-10.0.26300-SP0 |
| Seeds | `src/services/rag_service.py`, `src/interfaces/rag_interface.py`, `src/core/extensions/skillsets/Documents/KgFromDocument.py` |
| Copied verbatim | yes, except the publication scrub recorded below |

## Import roots

Files keep their original positions, so both paths belong on the import path:

- `track_2b` &mdash; for the `src.x.y` spellings the service uses
- `track_2b/src` &mdash; for the bare `x.y` spellings used elsewhere

The slice therefore occupies the `src`, `api`, `core`, `services`,
`interfaces`, `utils` and `config` namespaces: nothing else in this
submission may claim them. The layout is deliberately not renamed, because a
verbatim copy stays diffable against the original repository, apart from the
single rewrite recorded next.

## Publication scrub applied on copy

Upstream, `get_neo4j_driver()` in `core/base/neo4j_queries.py` falls back to a hardcoded
default password, and local development relies on that default. In this distributed
copy the fallback is replaced by an explicit `RuntimeError` when `NEO4J_PASSWORD` is
unset, so no credential ships. Upstream keeps its behaviour. The rewrite runs inside
the slicer and refuses to pass if the line ever moves, so re-extraction can never
silently restore the secret.

## Before publishing

1. Add a licence file. Open-source intent is confirmed; without a chosen licence a
   public release defaults to all rights reserved.
2. Reconcile the dependency split below against `requirements.txt`, which still lists
   only the app's four packages and none of these.
3. Run the verifier and require zero unresolved first-party imports.
4. Search the distributed tree for secrets once more before pushing.

Updating the slice: re-run the slicer after service changes. It reads the live source,
and the scrub is part of the copy step, so it survives every re-extraction.

## Files

| Path | SHA-256 | Bytes |
| ---- | ------- | ----- |
| `__init__.py` | `e3b0c44298fc1c14` | 0 |
| `api/auth/service_token.py` | `a127823fa781d567` | 6978 |
| `config/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `config/app_config.py` | `7e941486a80341ca` | 720 |
| `config/constants.py` | `c9b38d5cb7e2d360` | 7802 |
| `core/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `core/agents/Agent.py` | `d77999f1af6f1b85` | 30880 |
| `core/agents/AgentPlanner.py` | `ce1d399d7643ec01` | 97529 |
| `core/agents/__init__.py` | `ef2a7fb6739c95cf` | 214 |
| `core/base/InputOutput.py` | `ae1c2fab663529f9` | 41822 |
| `core/base/OpenaiUtilities.py` | `a0e6f6aac1f7b529` | 12292 |
| `core/base/ResourceMgmt.py` | `2b37e41b7a1165a6` | 783 |
| `core/base/Shutdown.py` | `20e694309813a4b9` | 141 |
| `core/base/__init__.py` | `ef2a7fb6739c95cf` | 214 |
| `core/base/_pydantic.py` | `6790ec456166585d` | 3166 |
| `core/base/enums.py` | `6ea8e4a1f56dc173` | 5753 |
| `core/base/hints.py` | `b38f13e1a34c0de5` | 91282 |
| `core/base/llm_response.py` | `900bc8b549f0f89f` | 3003 |
| `core/base/neo4j_indexes.py` | `5b2ead51c8a09509` | 8335 |
| `core/base/neo4j_queries.py` | `9fc770f219fa868d` | 10937 |
| `core/base/reasoning/ReasoningDivision.py` | `59526c2347b2e593` | 58814 |
| `core/base/reasoning/ReasoningDocument.py` | `db702eb6de27655d` | 102483 |
| `core/base/reasoning/ReasoningIssue.py` | `d10164e7b124f2e8` | 6953 |
| `core/base/reasoning/ReasoningLanguage.py` | `5567c543ef3f4f7c` | 3254 |
| `core/base/reasoning/ReasoningResult.py` | `f48d8d52df71e526` | 3591 |
| `core/base/reasoning/ReasoningSummary.py` | `a0f689612fa704a9` | 4976 |
| `core/base/reasoning/ReasoningTask.py` | `9839106b2148aed4` | 7020 |
| `core/base/reasoning/ReasoningTitle.py` | `b67bbf45e8d86889` | 3237 |
| `core/base/reasoning/xml_utils.py` | `0ae7729a71e141fb` | 6369 |
| `core/extensions/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `core/extensions/classes/BaseExtension.py` | `3e5560820828135a` | 250 |
| `core/extensions/classes/Innovator.py` | `f13d9951e575f580` | 7879 |
| `core/extensions/mcp_client/mcp_client_global.py` | `f9ea760cd6826339` | 80 |
| `core/extensions/skillset_management.py` | `afc1f47566ef3c18` | 14667 |
| `core/extensions/skillsets/Documents/KgFromDocument.py` | `2b13d1a05be7c0d4` | 38709 |
| `core/extensions/skillsets/Documents/KgFromDocument_llm.py` | `801b00d84a9b8fb2` | 19202 |
| `core/extensions/skillsets/KnowledgeGraph/KnowledgeGraphEditor.py` | `d88b9cab54c11163` | 83744 |
| `core/extensions/skillsets/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `core/llm/__init__.py` | `cbe33d43201dbd33` | 2729 |
| `core/llm/agent_rag_settings.py` | `1aa5397157f6a902` | 10295 |
| `core/llm/cache.py` | `79c489a8497ab566` | 1864 |
| `core/llm/client.py` | `38c56f9e6a4c69fc` | 4332 |
| `core/llm/config.py` | `37aa9c8dc5eacf76` | 4542 |
| `core/llm/embeddings.py` | `d4ccecf6ff1f359d` | 14559 |
| `core/llm/init.py` | `109c46375fe98ad2` | 2418 |
| `core/llm/interface.py` | `9ab26ab187ad5226` | 94608 |
| `core/llm/manager.py` | `4f6f98ac331475be` | 56703 |
| `core/llm/providers/__init__.py` | `f81acd230ecc596c` | 639 |
| `core/llm/providers/provider_apertus.py` | `9e8d5603e154bf57` | 2046 |
| `core/llm/providers/provider_endpoint.py` | `0f0b9a3402c408d9` | 1587 |
| `core/llm/providers/provider_groq.py` | `bce709dc474bd879` | 7211 |
| `core/llm/providers/provider_http.py` | `0b8754bb942714d3` | 20738 |
| `core/llm/providers/provider_mistral.py` | `adb798f211aaabc0` | 17926 |
| `core/llm/providers/provider_openai.py` | `a76dd07f5b6a5684` | 11191 |
| `core/llm/providers/provider_qwen.py` | `4c0fb42f6f5e4f1a` | 27921 |
| `core/llm/providers/provider_thalabus.py` | `8a34eda607494fc0` | 8287 |
| `core/llm/reranking.py` | `e0baaa6b5ddff93d` | 18273 |
| `core/llm/utils.py` | `ebe0d10b748f384c` | 69348 |
| `core/llm/utils_yaml.py` | `18c861c663771235` | 20953 |
| `core/thalaplan/Plan.py` | `c586d7ba951e353f` | 69145 |
| `core/thalaplan/goal/Goal.py` | `7a132fbaad7c78f6` | 305289 |
| `core/thalaplan/goal/__init__.py` | `fd1b52589f091256` | 19 |
| `core/thalaplan/goal/enums.py` | `8498d79d5c95dfd0` | 885 |
| `core/thalaplan/goal/exceptions.py` | `76cad7694e196af6` | 439 |
| `core/thalaplan/plan_utils.py` | `5995454e21935f83` | 1832 |
| `interfaces/__init__.py` | `316500a8041130c0` | 458 |
| `interfaces/cache_entry_interface.py` | `f3eb88e42bf00c3c` | 4539 |
| `interfaces/dataset_interface.py` | `d03e83a696a33fad` | 18016 |
| `interfaces/documents_interface.py` | `7c9300bafe58cafc` | 38979 |
| `interfaces/dossiers_interface.py` | `7e0b713527996a1f` | 7742 |
| `interfaces/hints_interface.py` | `4071954f6f0c7e62` | 7137 |
| `interfaces/memory_interface.py` | `5b3913680ad6e34e` | 972 |
| `interfaces/ontology_interface.py` | `cbcf90f773914a56` | 24538 |
| `interfaces/rag_interface.py` | `50b8f39e53f1d7fc` | 9386 |
| `interfaces/reasoning_cache_interface.py` | `a496a6ca20c173b2` | 990 |
| `interfaces/reranking_interface.py` | `21160ee913f5318e` | 3699 |
| `interfaces/translation_interface.py` | `f7cc80102f59729b` | 2471 |
| `interfaces/user_interface.py` | `db420194516fb7e3` | 1957 |
| `services/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `services/cache_entry.py` | `a9e10ac8a6e7b16a` | 13897 |
| `services/dataset_exception.py` | `c0d5ff3113454ecb` | 327 |
| `services/dataset_service.py` | `ebe321484428e667` | 103794 |
| `services/documents_service.py` | `0cec3e3ba70b5911` | 176335 |
| `services/dossiers_service.py` | `066c4869b1734ff1` | 32723 |
| `services/hints_service.py` | `bbf84db8aa0cfd0b` | 26500 |
| `services/ontology_service.py` | `2d9fcfdcc96f99bf` | 180962 |
| `services/ontology_source_helper.py` | `d4932f72b616aa7d` | 4606 |
| `services/rag_service.py` | `af28f1d7fa15aeea` | 351482 |
| `services/reasoning_cache_service.py` | `f490ffd3f3cd6e05` | 4067 |
| `services/reranking_service.py` | `ad77996442e593c3` | 41029 |
| `services/runpod/__init__.py` | `8a33a962160b789e` | 21 |
| `services/runpod/reservations.py` | `31b70f19adb12376` | 5780 |
| `services/runpod/runpod.py` | `b57416f92148a6cf` | 13773 |
| `services/translation_service.py` | `09cc0088531d5fde` | 22013 |
| `services/unit_testing/rag_service_unit_test.py` | `0df0fca8d2f48449` | 41204 |
| `services/user_service.py` | `b420657ab0312dbd` | 9458 |
| `utils/__init__.py` | `e3b0c44298fc1c14` | 0 |
| `utils/acl_cypher_utils.py` | `cffef172c0c49ccd` | 26835 |
| `utils/dataset_helpers.py` | `2670d7526d419544` | 5042 |
| `utils/document_partition.py` | `de322ed55de855b8` | 266 |
| `utils/json_utils.py` | `05f786aeaa272a5d` | 2058 |
| `utils/log.py` | `b2c560254df445c0` | 36734 |
| `utils/log_fileload.py` | `8af3dd13c3a17db0` | 566 |
| `utils/ontology_label_validator.py` | `5fd51b0367b00c1a` | 653 |
| `utils/performance.py` | `5d9b83b1344e5760` | 1348 |
| `utils/system.py` | `6d4666508d918947` | 1463 |
| `utils/text_utils.py` | `82201d00853b6646` | 73892 |
| `utils/unit_test.py` | `e6536748f582d324` | 327 |

## Skill descriptors (not Python, so absent from the table above)

Copied verbatim beside their modules; they declare the arguments a skill is
called with.

- `core/extensions/skillsets/Documents/KgFromDocument.skill` (3,618 bytes)
- `core/extensions/skillsets/KnowledgeGraph/KnowledgeGraphEditor.skill` (6,742 bytes)

## Third-party imports

The slice vendors no third-party code. Split by how strongly it is needed:

- **Required to import** (22): `bs4`, `click`, `elasticsearch`, `flask`, `groq`, `httpx`, `jsonschema`, `jwt`, `lxml`, `markdown`, `neo4j`, `numpy`, `openai`, `pycountry`, `pydantic`, `pytz`, `requests`, `tiktoken`, `torch`, `typing_extensions`, `tzlocal`, `yaml`
- **Guarded by `try`/`except ImportError`** (1): `mistralai`
- **Imported inside a function** (1): `datasets`

## Added `__init__.py` (absent in source)

Empty, so they carry no vendored implementation. Upstream these directories
are namespace packages; the init exists to make the copy importable.

- `api/auth/__init__.py`
- `core/base/reasoning/__init__.py`
- `core/extensions/classes/__init__.py`
- `core/extensions/mcp_client/__init__.py`
- `core/extensions/skillsets/Documents/__init__.py`
- `core/extensions/skillsets/KnowledgeGraph/__init__.py`
- `core/thalaplan/__init__.py`
- `services/unit_testing/__init__.py`
