# AviationBot API Documentation

BASE URL: https://beta.aviationbot.ai/api/v1

## Authentication

Authentication is done using API keys as bearer tokens.
See .env for AVIATION_BOT_API_KEY.

```python
import requests

headers = {
    "Authorization": f"Bearer {os.getenv('AVIATION_BOT_API_KEY')}"
}

response = requests.get(
    "https://beta.aviationbot.ai/api/v1/EASA/easa-reference-to-ear",
    headers=headers,
)

print(response.json())
```

# EASA API Endpoints Documentation

This document provides comprehensive documentation for all API endpoints in the EASA module.

**Router Prefix:** `/EASA`  
**Base URL:** `/api/v1/EASA` (assuming standard FastAPI routing)

---

## 1. EASA Reference to EAR Lookup

**Endpoint:** `POST /EASA/easa-reference-to-ear`

**Description:**  
Lookup metadata for EASA Easy Access Rules (EAR) references and return matching regulatory details. This endpoint performs reference normalization (handling AMC/GM prefixes, parenthetical suffixes, CS variants) and returns detailed metadata for each matched reference.

**Rate Limits:**
- 20 requests per 60 seconds
- 1000 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Request Body:**
```json
{
  "references": ["145.A.30", "AMC1 145.A.30", "CS 29.801(e)"],
  "include_base_without_parentheses": true,
  "include_without_amc_gm_prefix": true,
  "fallback_search_for_amc_gm_cs": true
}
```

**Request Schema:** `EASAEAReferenceLookupRequest`
- `references` (List[str], required, min_items=1): List of EASA reference titles to lookup
- `include_base_without_parentheses` (bool, default=True): Also lookup hardlaw from references with parenthetical suffixes (e.g., `145.A.30(a)` → `145.A.30`)
- `include_without_amc_gm_prefix` (bool, default=True): Also lookup hardlaw from references with AMC/GM prefixes (e.g., `AMC1 145.A.30` → `145.A.30`)
- `fallback_search_for_amc_gm_cs` (bool, default=True): Retry unmatched references by prefixing them with AMC/GM/CS during lookups

**Response:** `List[EASAEAReferenceLookupResponse]`

**Response Schema:** `EASAEAReferenceLookupResponse`
- `input_reference` (str): Original reference provided
- `input_reference_transformed` (List[str], optional): Transformed variants used for lookup
- `input_reference_transformed_matched` (str, optional): The transformed variant that matched
- `occurrence_count` (int, optional): Number of occurrences (when used with document analysis)
- `occurrences` (List[EASAEAReferenceOccurrence], optional): List of occurrence positions
- `parent_titles` (List[str], optional): Parent title path in EAR hierarchy
- `domain` (List[str], optional): Regulatory domain
- `doc_title` (str, optional): Document title
- `regulatory_source` (str, optional): Regulatory source identifier
- `parent_perma_ids` (List[str], optional): Permanent IDs of parent elements
- `sdt_id` (str, optional): SDT identifier
- `file_id` (str, optional): File identifier
- `erules_id` (str, optional): ERules identifier
- `perma_id` (str, optional): Permanent identifier
- `regulatory_subject` (List[str], optional): Regulatory subject categories
- `display_name` (str, optional): Display name of the document
- `short_name` (str, optional): Short name of the document
- `version` (str, optional): Document version
- `source_title` (str, optional): Source title
- `more_info_url` (str, optional): URL for more information
- `reference_matched_to_suggested_articles` (List[EASAEALLMMatch], optional): LLM-suggested article matches

---

## 2. EASA Reference to EAR Hierarchy (Structured)

**Endpoint:** `POST /EASA/easa-reference-to-ear-hierarchy`

**Description:**  
Lookup metadata for EASA Easy Access Rules references and return grouped filters/subdivisions per document. This endpoint provides a structured view organized by document short name, with filters (regulatory subjects) and subdivisions (parent title paths).

**Rate Limits:**
- 20 requests per 60 seconds
- 1000 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Request Body:** Same as endpoint #1 (`EASAEAReferenceLookupRequest`)

**Response:** `Dict[str, EASAEAReferenceStructuredItem]`

**Response Schema:** `EASAEAReferenceStructuredItem`
- Key: Document `short_name`
- Value:
  - `filters` (List[str]): Regulatory subject filters
  - `subdivision` (List[EASAEAReferenceSubdivisionItem]): List of subdivisions with parent title paths
    - `parent_title_path` (List[str]): Path of parent titles in hierarchy
    - `ear_child_elements` (List[EASAEARChildElementsPerFile], optional): Child elements (not included in this endpoint)
    - `reference_matched_to_suggested_articles` (List[EASAEALLMMatch], optional): LLM-suggested matches (not included in this endpoint)

---

## 3. EASA EAR Reference Lookup from Document

**Endpoint:** `POST /EASA/easa-ear-reference-lookup-from-document`

**Description:**  
Extract regulatory references from an uploaded document (text or Markdown) and return comprehensive lookup and structured metadata. This endpoint:
1. Extracts EASA references from the document text
2. Performs reference lookup and metadata retrieval
3. Optionally includes EAR hierarchy child elements
4. Optionally uses LLM-assisted matching to align references with EAR child elements
5. Caches results based on document content hash

**Rate Limits:**
- 1000 requests per 60 seconds
- 5000 requests per 3600 seconds (1 hour)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `False` (both authenticated users and API keys can use)

**Request:**
- **File Upload:** `file` (multipart/form-data, required): Text or Markdown file containing potential EASA references
- **Query Parameters:**
  - `return_EAR_hierarchy_child_elements` (bool, default=True): Include EAR child elements for each matched hierarchy subdivision
  - `resolve_child_elements_with_llm` (bool, default=True): Use LLM-assisted matching to align references with EAR child elements
  - `llm_context_chars` (int, default=30, min=0, max=200): Number of characters before and after a reference to include as context for LLM matching

**Response:** `EASAEAReferenceDocumentResponse`

**Response Schema:** `EASAEAReferenceDocumentResponse`
- `input_references_found` (List[str]): Unique references found in the document
- `input_references_transformed` (List[str], optional): Transformed reference variants used for lookup
- `EAR_hierarchy_matched` (Dict[str, EASAEAReferenceStructuredItem]): Structured hierarchy results (same format as endpoint #2, but may include child elements if requested)
- `references_matched_to_EAR_section` (List[EASAEAReferenceLookupResponse]): Detailed lookup results for each reference (includes occurrence positions)
- `cache_metadata` (Dict[str, Any], optional): Cache information including cache key, fetched timestamp, source, and schema version
- `reference_matched_to_suggested_articles` (List[EASAEALLMMatch], optional): Aggregated LLM-suggested article matches

**Notes:**
- Results are cached based on document content hash
- LLM matching uses context snippets around each reference occurrence
- References are grouped by proximity (within 1 line gap) for LLM processing

---

## 4. EASA EAR Child Elements Lookup

**Endpoint:** `POST /EASA/easa-ear-child-elements`

**Description:**  
Find EAR child elements for a specific document short name and parent title path. This endpoint retrieves the hierarchical child elements (articles, AMCs, GMs, etc.) under a specific parent title path in the EAR structure.

**Rate Limits:**
- 20 requests per 60 seconds
- 100 requests per 3600 seconds (1 hour)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Request Body:**
```json
{
  "short_name": "CS-145",
  "parent_title_path": ["Part-145", "145.A.30"],
  "include_all_descendants": false,
  "metadata": ["RegulatorySource", "Domain"],
  "truncate_metadata_chars_max": 100
}
```

**Request Schema:** `EASAEARChildElementsRequest`
- `short_name` (str, optional): Short name of the EAR document to limit the search. When omitted, all matching documents are returned.
- `parent_title_path` (List[str], required, min_items=1): Exact parent_titles path to match in the EAR hierarchy
- `include_all_descendants` (bool, default=False): When true, recursively load all descendant elements using children_perma_ids metadata
- `metadata` (List[str], optional): Optional list of metadata keys to include on each child element in the response
- `truncate_metadata_chars_max` (int, optional, gt=0): When provided, truncate returned metadata values to this many characters and append a truncated notice

**Response:** `EASAEARChildElementsResponse`

**Response Schema:** `EASAEARChildElementsResponse`
- `found_in_EARs` (List[EASAEARChildElementsPerFile]): List of documents containing matching child elements
  - `display_name` (str, optional): Display name of the document
  - `version` (str, optional): Document version
  - `id` (str): Document identifier
  - `short_name` (str, optional): Short name of the document
  - `supersedes_id` (str, optional): ID of superseded document
  - `children` (List[EASAEARChildElementItem]): Hierarchical child elements
    - `source_title` (str, optional): Title of the element
    - `ERulesId` (str, optional): ERules identifier
    - `RegulatorySource` (str, optional): Regulatory source
    - `Domain` (str, optional): Regulatory domain
    - `TypeOfContent` (str, optional): Type of content (e.g., "Hard Law", "AMC", "GM")
    - `document_order` (str, optional): Document order
    - `sdt_id` (str, optional): SDT identifier
    - `perma_id` (str, optional): Permanent identifier
    - `RegulatorySourceUpdated` (bool, optional): Whether regulatory source was updated
    - `more_info_url` (str, optional): URL for more information
    - `metadata` (Dict[str, Any]): Additional metadata fields
    - `children` (List[EASAEARChildElementItem]): Nested child elements (recursive)
    - `reference_matched_to_suggested_articles` (List[EASAEALLMMatch], optional): LLM-suggested matches

---

## 5. Analyse Own Document

**Endpoint:** `GET /EASA/analyse-own-doc`

**Description:**  
Analyze the user's own uploaded document using AI tools. This endpoint uses the user's project-specific prompts and persona settings to analyze documents in the context of their organization and project.

**Rate Limits:**
- 5 requests per 60 seconds
- 20 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for document analysis

**Response:** `str` (content from the analysis tool)

**Notes:**
- Uses user-specific state including organization prompt, user prompt, project prompt, and persona settings
- Requires authenticated user session

---

## 6. Document Retrieval

**Endpoint:** `GET /EASA/document-retrieval`

**Description:**  
Get EASA aviation regulations based on a query. This endpoint retrieves relevant EASA documents using semantic search, optionally filtered by specific ERules IDs.

**Rate Limits:**
- 20 requests per 60 seconds
- 100 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for document retrieval
- `erules_ids` (List[str], optional): The list of ERulesIds to filter on

**Response:** Tool result object (structure depends on the tool implementation)

**Notes:**
- Uses user-specific state for personalized results
- Can filter results to specific ERules documents when `erules_ids` is provided

---

## 7. Document Finder

**Endpoint:** `GET /EASA/doc-finder`

**Description:**  
Find EASA document names for a query. This endpoint helps identify the correct EASA document name based on a natural language query.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for document finding

**Response:** `str` (content from the document finder tool)

---

## 8. Meta Model

**Endpoint:** `GET /EASA/meta-model`

**Description:**  
Get information about EASA regulatory structure. This endpoint provides metadata about the EASA regulatory framework, document organization, and structural relationships.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for meta model information

**Response:** `str` (content from the meta model tool)

---

## 9. NPA Updates

**Endpoint:** `GET /EASA/npa-updates`

**Description:**  
Get EASA Notices of Proposed Amendment (NPA). This endpoint retrieves information about proposed regulatory amendments and updates.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for NPA updates

**Response:** `str` (content from the NPA updates tool)

---

## 10. Part-145 MOE Cross-Reference

**Endpoint:** `GET /EASA/part-145-moe-xref`

**Description:**  
Get cross-references between EASA Part-145 and MOE (Maintenance Organization Exposition). This endpoint helps find relationships and mappings between Part-145 regulations and MOE requirements.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for Part-145 MOE cross-references

**Response:** `str` (content from the Part-145 MOE cross-reference tool)

---

## 11. Part-21G Yellow Pages

**Endpoint:** `GET /EASA/part-21g-yellow-pages`

**Description:**  
Get Part-21G yellow pages information. This endpoint retrieves directory-style information about Part-21G organizations, contacts, and related data.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for Part-21G yellow pages

**Response:** `str` (content from the Part-21G yellow pages tool)

---

## 12. Part-145 Yellow Pages

**Endpoint:** `GET /EASA/part-145-yellow-pages`

**Description:**  
Get Part-145 yellow pages information. This endpoint retrieves directory-style information about Part-145 organizations, contacts, and related data.

**Rate Limits:**
- 20 requests per 60 seconds
- 50 requests per 86400 seconds (24 hours)

**Permissions:**
- Allowed for API keys: `False`
- Blocked for users: `True` (only authenticated users, not API keys)

**Query Parameters:**
- `query` (str, required): The query string for Part-145 yellow pages

**Response:** `str` (content from the Part-145 yellow pages tool)

---

## Common Response Models

### EASAEALLMMatch
- `references` (List[str]): List of references that matched
- `perma_id` (str): Permanent identifier of the matched element
- `source_title` (str): Title of the matched source
- `reason` (str): Justification for the match
- `matched` (bool, default=True): Whether the match was successful
- `merged` (bool, optional): Whether this match was merged with another

### EASAEAReferenceOccurrence
- `line` (int): Line number where the reference occurs
- `char_start` (int): Starting character position
- `char_end` (int): Ending character position

---

## Notes

1. **Reference Normalization:** The API automatically handles various reference formats:
   - AMC/GM prefixes (e.g., `AMC1 145.A.30` → `145.A.30`)
   - Parenthetical suffixes (e.g., `145.A.30(a)` → `145.A.30`)
   - CS/E variants (e.g., `CS 29.801`, `CS-29.801`, `E 29.801`)

2. **Caching:** The document analysis endpoint (`/easa-ear-reference-lookup-from-document`) caches results based on document content hash to improve performance for repeated requests.

3. **LLM Matching:** When enabled, the document analysis endpoint uses LLM-assisted matching to align references with specific EAR child elements, providing more accurate and context-aware results.

4. **Rate Limiting:** All endpoints have rate limits applied. Exceeding limits will result in HTTP 429 (Too Many Requests) responses.

5. **Authentication:** Most endpoints require authenticated user sessions (not API keys). Check the permissions section for each endpoint.
