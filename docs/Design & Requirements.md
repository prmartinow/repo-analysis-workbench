Yes. Here is the merged **final locked design / requirements discovery**.
# Repo-Analysis design and requirements

## Workload and operating model

This system is for an **LLM coding agent** such as Claude or Codex to query repository artifacts repeatedly, with the main priority being **interactive speed** under many requests, not one-off batch analytics.

The repository access pattern includes:

* file-level retrieval
* identifier / lexical search
* AST-derived structure access
* graph traversal queries such as `callers-of`, `callees-of`, `who-imports`, `path-between`, and `statement-slice`
* hybrid retrieval combining lexical search, optional embeddings, graph expansion, summaries, and reranking
* batch rebuilds with freshness based on rebuild pipeline and per-file cache, not a live daemon/watch service

The code/data representations include:

* raw code / text
* symbol records
* statement records
* AST-derived structure
* code graph
* control-flow edges
* data-flow edges
* Rust type / trait / impl semantics where available

The retrieval/indexing strategy includes:

* sparse retrieval / BM25
* Tantivy-backed lexical search
* hierarchical retrieval across repo/package/directory/file/symbol/statement levels
* logical chunking such as file, symbol, function body, type body, and statement
* metadata-rich indexing with path, package, module, visibility, tags, hashes
* caching via LMDB and in-process caches
* optional embeddings as a sidecar, not the default core path

The long-context strategy includes:

* RAG-style retrieval
* iterative retrieval
* answer-bundle preparation
* summaries / abstraction layers
* tool-driven exploration

The optimization style includes:

* query classification / routing
* multi-stage retrieval planning
* selective retrieval gating
* reranking
* static-analysis-informed retrieval
* tool-use architecture

## Hardware and scale assumptions

Per repo, design for roughly:

* **250k symbols**
* **500k statements**
* **3–5 million graph edges**

Available hardware assumptions:

* **~500 GB RAM**
* **~100 GB fast local NVMe disk**
* more disk later

That means:

* memory is not the bottleneck
* disk footprint is secondary to latency
* the system should optimize for **lowest warm-query latency**
* duplication across engines is acceptable when it removes lookup hops
* this scale still fits comfortably into a **single-machine, in-process, per-repo engine stack**
* no sharding, remote serving, or client/server database is required

---

# Final locked architecture

## Keep only 3 runtime systems

### 1. Tantivy

Use Tantivy for:

* lexical search
* identifier search
* file/path fuzzy search
* retrieval seed generation
* stored hit payloads for top-k results

### 2. LMDB

Use LMDB for:

* exact metadata
* symbol-by-id
* qname/name/path resolution
* file records
* symbol bodies
* summaries
* eval cache
* internal artifact/version metadata

### 3. RyuGraph

Use RyuGraph for:

* full code graph
* statement graph
* control-flow and data-flow edges
* graph traversal
* path search
* neighborhood expansion
* graph-backed symbol summaries


---

# Final answers to the design questions

## 1. Runtime language

Keep the orchestration layer in **Python**, but treat **Tantivy and RyuGraph as native engines** and push as much hot-path work into native code as possible.

The current retrieval and planning layers are already Python coordinators over backend interfaces, so that is the natural shape to keep.

## 2. RyuGraph integration

Use **native bindings / in-process integration**, not a separate service.

The workload is local, latency-sensitive, and many-request oriented, so an extra RPC/service hop would work against the goal.

## 3. Latency priority

Optimize for **warm repeated-query latency first**, cold start second.

The expected user is an LLM coding agent issuing many requests, and the system already supports iterative retrieval and repeated answer-bundle preparation.

## 4. Build model

Keep **batch rebuild as the default**, with optional future incremental rebuild by file hash/content hash.

Do **not** design around a live watcher or index daemon first.

## 5. Tantivy stored fields

Store **enough payload to satisfy top-k retrieval results without extra disk hits for every hit**.

At minimum, store:

* title
* path
* kind
* repo
* name
* qualified name
* symbol id
* preview/snippet
* searchable chunk text
* lightweight metadata needed for rerank and routing

This avoids a second lookup on every lexical hit.

## 6. Symbol bodies

Keep them **duplicated**:

* **canonical exact body payload in LMDB** for `get-symbol-body` and exact symbol access
* **indexed/stored chunks in Tantivy** for lexical retrieval

That is the fastest split for this workload.

## 7. Graph scope

Put the **full graph** in RyuGraph, and duplicate **minimal node metadata** there so graph queries can return useful neighbors directly.

Keep **rich symbol/file/summary payloads canonical in LMDB**.

This matches the current logical retrieval pattern: graph expansion first, then metadata hydration.

## 8. Summaries

Keep summaries **precomputed** and store them in **LMDB**.

They are part of answer-bundle preparation and summary bonus scoring, so they should be fast reads, not on-demand computations.

## 9. Statement granularity

**Keep statement-level records and graph structure.**

The workload explicitly includes `statement-slice` and control-flow/data-flow usage, so removing statement granularity would weaken core functionality.

## 10. `path-between` semantics

Default to **directed shortest path with optional edge-type filters**.

Keep an optional `"both"` mode for exploratory use, but directed should be the default because it is more meaningful and cheaper.

## 11. `find-file` strategy

Use a **hybrid**:

* **LMDB** for exact and prefix path lookup
* **Tantivy** for fuzzy/lexical path search

That is better than forcing one engine to do both jobs.

## 12. Repo layout

Use **one isolated Tantivy + LMDB + RyuGraph set per repo**.

Cross-repo composition should happen at the orchestration layer, not in shared physical storage.

## 13. Memory budget

Treat RAM as **effectively generous**.

Design assumptions:

* keep hot working sets memory-resident
* keep per-repo engines open
* allow duplication across engines when it improves latency
* optimize for latency, not compactness

## 14. Observability / metadata

**Small internal metadata records** inside LMDB / RyuGraph / Tantivy for:

* schema versioning
* artifact fingerprinting
* rebuild provenance
* generated-at timestamps
* internal consistency checks

## 15. Optimization priority

Optimize in this order:

1. `prepare-answer-bundle` / `retrieve_context`
2. `find-symbol` and `get-symbol-body`
3. `callers-of`, `callees-of`, `path-between`, `statement-slice`
4. benchmark/eval reuse

That matches the actual expected LLM-agent interaction pattern.

## 16. Deployment assumption

Assume **single machine, local process family, no remote shared service**.

---

# What each engine should store

## Tantivy should store

At minimum:

* repo/package/directory/file/symbol/statement docs
* function-body chunks
* type-body chunks
* path
* kind
* repo
* name
* qualified name
* symbol id
* title
* preview/snippet
* lightweight metadata used for rerank/routing

The purpose is that the first retrieval stage returns useful candidates with very few follow-up reads.

## LMDB should store

Canonical exact-access buckets:

* `symbol_by_id`
* `symbol_ids_by_qname`
* `symbol_ids_by_name`
* `file_by_path`
* `body_by_symbol_id`
* `summary_by_id`
* `summary_by_path`
* `summary_by_symbol_id`
* `eval_case_cache`
* `artifact_metadata`

LMDB is the canonical store for exact reads and cached enriched payloads.

## RyuGraph should store

Canonical graph entities:

* repo nodes
* package nodes
* directory nodes
* file nodes
* symbol nodes
* statement nodes

Canonical relationships:

* `CALLS`
* `IMPORTS`
* `READS`
* `WRITES`
* `REFS`
* `IMPLEMENTS`
* `INHERITS`
* `CONTAINS`
* `CONTROL_FLOW`
* `DATA_FLOW`

Plus enough node properties to let graph queries return useful neighbors without immediate LMDB hydration on every hop.

---

# Hot-path query plan

For the main agent workload, the target path should be:

**Tantivy lexical seed search → RyuGraph expansion/traversal → LMDB hydration of selected winners → rerank → answer bundle**

This is the final target execution path.

---

