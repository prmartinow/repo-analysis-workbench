# repo-analysis: Overview and Use Cases

## What repo-analysis is

`repo-analysis` is a local code-intelligence and retrieval toolkit for large source repositories.

Its job is to turn a repository into a structured, queryable analysis surface that is useful for:

- human code exploration
- LLM coding agents such as Claude or Codex
- symbol and file lookup
- graph-based code navigation
- evidence-backed answer-bundle preparation
- retrieval benchmarking and evaluation

It is designed for **fast repeated local queries**, not for multi-user remote serving.

---

## What it does

repo-analysis builds multiple analysis layers over a repository:

### 1. Raw inventory
It first scans the repository and records the files, directories, language mix, and relevant source roots.

### 2. Parsed metadata
It extracts structured code information such as:

- files
- symbols
- imports
- references
- statements
- symbol bodies

This becomes the canonical exact-access metadata layer.

### 3. Search index
It builds a lexical search index so queries like symbol names, file paths, API names, and code phrases can be resolved quickly.

### 4. Code graph
It builds a graph representation of the repository, including relationships such as:

- calls
- imports
- reads
- writes
- references
- inheritance / implementation
- containment
- control flow
- data flow

### 5. Summaries
It generates higher-level summaries for:

- the whole repo
- packages
- directories
- files
- symbols

These are precomputed in the summaries phase, synced into the metadata layer, and used for compressed navigation and LLM context preparation.

### 6. Evaluation and benchmarking
It can prepare benchmark prompts, score answer bundles, and measure retrieval quality and interactive latency.

---

## Current storage model

repo-analysis is now centered around three main runtime systems:

### Tantivy
Used for:

- lexical search
- identifier search
- fuzzy file/path lookup
- retrieval seed generation

### LMDB
Used for:

- exact metadata lookup
- symbol-by-id lookup
- file-by-path lookup
- symbol bodies
- summaries
- artifact metadata
- evaluation cache

### RyuGraph
Used for:

- graph storage
- graph traversal
- path search
- neighborhood expansion
- statement slices
- summary graph augmentation

This means the runtime architecture is optimized for:

- fast local reads
- repeated interactive queries
- deterministic artifacts
- low operational overhead

---

## What repo-analysis is helpful for

### 1. Fast repo navigation
It helps answer questions like:

- Where is this symbol defined?
- Which files mention this concept?
- What does this function body look like?
- What directories or packages matter for this subsystem?

### 2. Relationship discovery
It is useful for questions like:

- Who calls this function?
- What does this symbol read or write?
- Which module imports this symbol?
- What is the path between these two code elements?

### 3. LLM grounding
It is especially useful for giving an LLM agent compact, evidence-backed context instead of forcing the model to open many raw files blindly.

Typical workflow:

1. lexical retrieval
2. graph expansion
3. metadata hydration
4. reranking
5. answer-bundle preparation

### 4. Repository summarization
It helps compress large repositories into more digestible layers, so an agent or human can start from repo/package/file/symbol summaries before diving into raw code.

### 5. Retrieval quality evaluation
It helps compare retrieval strategies and benchmark whether the current indexing and context assembly are actually good enough.

---

## Main command categories

repo-analysis includes commands for:

### Build pipeline
- parsing repos
- building metadata
- building graph artifacts
- building lexical search
- building summaries
- building optional embeddings
- running benchmarks

### Lookup and retrieval
- symbol search
- file search
- lexical search
- semantic search (optional)
- repo overview
- path summarization
- symbol signatures and bodies
- enclosing context

### Graph exploration
- callers
- callees
- imports
- reads
- writes
- references
- implementations
- inheritance
- bounded neighborhood expansion
- path search
- statement slicing

### LLM-oriented preparation
- retrieval planning
- context preparation
- answer-bundle preparation
- iterative retrieval refinement

### Evaluation
- benchmark prompt export
- answer-bundle scoring
- external-answer scoring
- benchmark execution
- interactive latency benchmarking

---

## Why this exists instead of just grepping the repo

Simple grep or plain-text search is often not enough for serious code understanding.

repo-analysis adds:

- symbol awareness
- statement awareness
- cross-reference structure
- graph relationships
- package/file/directory summaries
- more efficient retrieval for large codebases
- better context packaging for LLMs

This makes it more useful for questions that depend on structure and relationships, not just text matching.

---

## Intended usage model

repo-analysis is best used when:

- the repository is non-trivial in size
- you need repeated queries
- you want consistent local artifacts
- you want an LLM agent to reason over the repo faster and with less noise
- you care about evidence-backed answers rather than vague summaries

It is not primarily intended to be:

- a cloud code search service
- a collaborative multi-user database server
- a general analytics warehouse

---

## Practical examples

repo-analysis is useful when you want to answer questions like:

- What are the parser entry points in this repo?
- Which functions are upstream of this handler?
- What symbols sit on the path between A and B?
- Which files and symbols should I show an LLM for this task?
- Which summary should I read before opening this directory?
- Is the retrieval pipeline actually finding the right evidence?

---

## In one sentence

`repo-analysis` is a local, structured code-retrieval system that turns a repository into a fast search + metadata + graph + summary surface for humans and LLM coding agents.
