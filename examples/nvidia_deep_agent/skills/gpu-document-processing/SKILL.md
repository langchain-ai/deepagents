---
name: gpu-document-processing
description: Use when processing large PDFs, document collections, or bulk text extraction tasks that benefit from GPU-accelerated processing. Triggers when the user provides large documents or needs bulk document analysis.
---

# GPU Document Processing Skill

Process large documents and document collections using GPU-accelerated tools. This skill uses the sandbox-as-tool pattern: the agent runs on CPU for reasoning, and sends document processing work to a GPU-equipped environment.

## When to Use This Skill

Use this skill when:
- Processing large PDF files (50+ pages)
- Analyzing collections of documents (10+ files)
- Extracting structured data from unstructured documents
- Performing bulk text extraction and chunking
- Generating embeddings for large document sets
- The user uploads or references large documents for analysis

## Architecture: Sandbox as Tool

This skill follows the **sandbox-as-tool pattern** for GPU execution:

1. **Agent reasons on CPU** - planning, synthesis, report writing
2. **Processing sent to GPU sandbox** - document parsing, embedding, extraction
3. **Results returned to agent** - structured output for further analysis

This separation ensures:
- API keys stay outside the sandbox (security)
- Agent state persists independently of processing jobs
- Processing can be parallelized across documents
- Cost-efficient: GPU used only during processing, not during reasoning

## Capabilities

### PDF Text Extraction
Extract text content from PDF documents with layout preservation:
- Headers, paragraphs, lists, and tables detected separately
- Page numbers and section boundaries preserved
- Multi-column layout handling

### Tabular Data Extraction
Extract tables from documents into structured formats:
- PDF tables to CSV/DataFrames using GPU-accelerated parsing
- Automatic column type detection
- Handles merged cells and multi-row headers

### Document Chunking
Split large documents into meaningful chunks for analysis:
- Semantic chunking (by topic/section boundaries)
- Fixed-size chunking with overlap for embedding
- Configurable chunk sizes (default: 512 tokens)

### Embedding Generation
Generate vector embeddings for document chunks:
- Uses NVIDIA NeMo Retriever NIM for GPU-accelerated embedding
- Supports batch processing for large document sets
- Compatible with standard vector stores (Milvus, ChromaDB)

## Workflow

1. **Receive document reference** from the orchestrator
2. **Determine processing type** (extraction, analysis, embedding)
3. **Send to GPU sandbox** for processing
4. **Collect structured results** (text, tables, embeddings)
5. **Write findings** to /shared/ for the orchestrator to synthesize

## Processing Large Document Collections

For multiple documents:
1. Process documents in parallel batches (3-5 concurrent)
2. Extract key metadata first (title, date, author, page count)
3. Generate per-document summaries
4. Cross-reference findings across documents
5. Write consolidated findings with per-document citations

## Output Format

When reporting document processing results:
- Include document metadata (filename, pages, size)
- Structure extracted content by section/chapter
- Format tables as markdown tables
- Include page references for all extracted content
- Note any extraction quality issues (scanned images, corrupted pages)

## Integration with NVIDIA NIM

For production deployments, GPU document processing can leverage:
- **NVIDIA NeMo Retriever**: GPU-accelerated embedding and retrieval
- **NVIDIA RAPIDS cuDF**: Tabular data processing from extracted tables
- **NVIDIA Triton**: Scalable inference for document classification models

See NVIDIA's NIM documentation for self-hosted deployment options.
