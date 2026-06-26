# Multi-Language Code Intelligence Implementation Plan

Date: 2026-06-26

## Goal

Implement the repo-analysis multi-language expansion without turning the
workbench into a loose collection of external tools.

The implementation is complete only when repo-analysis can:

- benchmark retrieval quality with explicit expected evidence;
- index non-Rust symbols through provider-neutral artifacts;
- ingest broad symbols from Universal Ctags;
- enrich common languages with tree-sitter tags;
- import precise definitions and references from SCIP;
- compare Tantivy and Zoekt search quality without replacing Tantivy by default;
- produce benchmark reports for the AI-tooling candidate repositories.

## Full Completion Checklist

| Step | Milestone | Status | Completion evidence |
|---|---|---|---|
| 1 | Retrieval benchmark cases and metrics | Done | `run-benchmarks` loads JSON/JSONL cases, rejects empty case sets, and reports recall@k, MRR, and NDCG. |
| 2 | Provider-neutral symbol records | Done | Rust symbols are normalized through provider-neutral records with `provider`, `language`, `kind`, `path`, `name`, `qualified_name`, `range`, `scope`, and `confidence`. |
| 3 | Universal Ctags provider | Done | Non-Rust repos produce provider-normalized symbols from `ctags --output-format=json`; provider metadata records ctags provenance. |
| 4 | Generic non-Rust summary fallback | Done | Repos with zero symbols get project, directory, and file summaries from raw inventory, marked `inventory_fallback` and `shallow`. |
| 5 | Tree-sitter tag provider | Done | Python, Go, JS/TS, shell, and YAML/config files can produce provider-normalized `tree_sitter_tags` symbols where grammars are available. |
| 6 | SCIP JSON importer | Done | `.scip`/`.scip.json` indexes can be imported into symbols, references, and graph edges through `build-index --scip-index`. |
| 7 | Zoekt sidecar comparison | Done | CLI can build/query Zoekt sidecar indexes and compare Zoekt path results against Tantivy lexical search. |
| 8 | AI-tooling benchmark reports | Not started | Benchmark reports exist for the selected `/mnt/workspace/AI tooling` repos and show per-mode quality. |

## Implementation Order

### 1. Add Retrieval Benchmark Case Loading And Metrics

Why first:

Search and ranking have failed qualitatively in the past. Before changing the
retrieval stack, add a way to prove whether a change improves or damages
retrieval.

Implementation:

- Add benchmark case loading from JSON or JSONL under `data/eval/cases/`.
- Keep `src/evaluation/harness.py` as the metric owner.
- Add metrics:
  - `recall_at_k`;
  - `mrr`;
  - `ndcg`;
  - expected path hit;
  - expected symbol hit;
  - latency per retrieval mode.
- Keep existing semantic modes, but allow explicit lexical-only and backend
  comparison modes for evaluation even if production answer-bundle paths still
  require embeddings.

Suggested commit:

`Add retrieval benchmark case loading and metrics`

Verification:

```bash
python3 src/cli/main.py run-benchmarks --repo agent-kit --limit 5
```

Done when:

- benchmark cases can be declared outside Python code;
- runs produce a deterministic report under `data/eval/`;
- empty benchmark sets fail or warn clearly instead of silently passing.

### 2. Generalize Symbol Artifacts For Multiple Providers

Why second:

Ctags, tree-sitter, and SCIP should not be forced through Rust-only data
assumptions. The workbench needs a normalized symbol record before adding new
providers.

Implementation:

- Introduce provider-neutral symbol normalization in `src/symbols/indexer.py`
  or a new `src/symbols/schema.py`.
- Preserve current Rust parser output as provider `rust_static`.
- Required common fields:
  - `provider`;
  - `language`;
  - `kind`;
  - `path`;
  - `name`;
  - `qualified_name`;
  - `range`;
  - `scope`;
  - `confidence`.
- Keep Rust-specific fields optional.
- Update graph, search, summary, and LMDB persistence code to tolerate missing
  Rust-only fields.

Suggested commit:

`Generalize symbol records for multiple providers`

Verification:

```bash
python3 src/cli/main.py build-index --repo <rust-repo>
python3 src/cli/main.py build-search --repo <rust-repo>
python3 src/cli/main.py build-summaries --repo <rust-repo>
```

Done when:

- existing Rust behavior still works;
- symbol records can represent shallow non-Rust symbols without fake crate or
  module fields;
- downstream graph/search/summary builders do not assume every symbol came
  from the Rust parser.

### 3. Add Universal Ctags Symbol Provider

Why third:

Universal Ctags gives the fastest broad language coverage and directly fixes
the pilot failure where non-Rust repos produced zero symbols.

Implementation:

- Add `src/parsers/ctags_backend.py`.
- Invoke Universal Ctags as an external tool:

```bash
ctags --output-format=json --fields=+neK --extras=+q <files>
```

- Normalize `_type=tag` rows into provider-neutral symbols.
- Preserve ctags fields when available:
  - `name`;
  - `path`;
  - `language`;
  - `kind`;
  - `scope`;
  - `scopeKind`;
  - `line`;
  - `end`;
  - `pattern`.
- Add generated/vendor/binary exclusions using the existing inventory filters.
- Add timeouts and per-repo progress events.
- Cache raw ctags output or normalized provider output per repo.

Suggested commit:

`Add Universal Ctags symbol provider`

Verification:

```bash
python3 src/cli/main.py build-index --workspace-root "/mnt/workspace/AI tooling/repos" --repo <non-rust-repo>
python3 src/cli/main.py find-symbol --repo <non-rust-repo> "<known symbol>"
```

Done when:

- Go, Python, JS/TS, Java, C/C++, and shell repos produce symbols;
- ctags provider failures do not break Rust indexing;
- provider provenance is visible in parsed metadata.

### 4. Add Generic Summary Fallback For Non-Rust Repos

Why fourth:

The pilot showed summary builds with zero files and zero symbols for non-Rust
repos. Even before precise parsing, repo-analysis should summarize project,
directory, and file surfaces from raw inventory.

Implementation:

- Update `src/summaries/builder.py`.
- If no parsed file records exist, derive file summaries from `repo_map.json`.
- Derive directory summaries from language mix, file counts, names, README
  presence, package manifests, docs, config, tests, and obvious entrypoints.
- Mark fallback summaries with `provider=inventory_fallback` and
  `confidence=shallow`.

Suggested commit:

`Add generic summary fallback for non-Rust repos`

Verification:

```bash
python3 src/cli/main.py build-summaries --repo <non-rust-repo>
python3 src/cli/main.py repo-overview --repo <non-rust-repo>
```

Done when:

- non-Rust repos get non-empty file and directory summaries;
- fallback summaries are clearly marked as shallow;
- exact parsed summaries still win when provider-backed symbols exist.

### 5. Add Tree-Sitter Tag Provider

Why fifth:

Ctags is broad but shallow. Tree-sitter adds syntax-aware ranges, nesting, and
better function/class summaries.

Implementation:

- Extend `src/parsers/tree_sitter_backend.py` beyond Rust.
- Add provider output compatible with the normalized symbol schema.
- Start with languages most relevant to AI-tooling repos:
  - Python;
  - Go;
  - JavaScript;
  - TypeScript;
  - TSX/JSX;
  - shell;
  - YAML/config;
  - Rust.
- Prefer language packages and tagging queries where available.
- Let tree-sitter enrich or supersede ctags records by stable path/name/range
  matching.

Suggested commit:

`Add tree-sitter tag provider`

Verification:

```bash
python3 src/cli/main.py build-index --repo <python-or-go-repo>
python3 src/cli/main.py build-summaries --repo <python-or-go-repo>
```

Done when:

- common languages produce range-aware tag records where grammars exist;
- ctags remains a fallback for unsupported languages;
- summaries can use tree-sitter ranges and nesting.

### 6. Add SCIP JSON Importer

Why sixth:

SCIP is the strategic path to precise definitions, references, occurrences,
symbol metadata, and relationships across languages.

Implementation:

- Add `src/parsers/scip_backend.py` or `src/codeintel/scip_importer.py`.
- First version may shell out to:

```bash
scip print --json < index.scip
```

- Map SCIP objects:
  - `Document` to file records;
  - `Occurrence` to references and definition ranges;
  - `SymbolInformation` to symbol records and docs;
  - `Relationship` to graph edges.
- Import `.scip` files discovered in repo roots or provided by CLI arg.
- Keep precise SCIP records higher confidence than ctags/tree-sitter.

Suggested commit:

`Add SCIP JSON importer`

Verification:

```bash
python3 src/cli/main.py build-index --repo <repo-with-scip> --scip-index index.scip
python3 src/cli/main.py refs-of --repo <repo-with-scip> "<known symbol>"
```

Done when:

- a SCIP index can populate symbols and references;
- graph edges can answer definition/reference questions from SCIP data;
- missing `scip` binary or missing index files fail clearly.

### 7. Add Zoekt Sidecar Backend Comparison

Why seventh:

Search quality and ranking need direct comparison. Zoekt should be tested as a
sidecar, not swapped in prematurely.

Implementation:

- Add optional Zoekt backend code under `src/backends/zoekt/` or
  `src/search/zoekt_backend.py`.
- Add CLI commands:
  - `build-zoekt`;
  - `search-zoekt`;
  - `compare-search-backends`.
- Index the same repos as Tantivy.
- Compare:
  - recall@k;
  - MRR;
  - NDCG;
  - latency;
  - symbol-aware ranking;
  - generated/vendor pollution.

Suggested commit:

`Add Zoekt sidecar backend comparison`

Verification:

```bash
python3 src/cli/main.py build-zoekt --repo <repo>
python3 src/cli/main.py compare-search-backends --repo <repo> "<query>"
```

Done when:

- Tantivy remains the default search backend;
- Zoekt can be built and queried when installed;
- benchmark reports can compare both backends.

### 8. Add Benchmark Reports For AI Tooling Repos

Why last:

Only after coverage, summaries, SCIP, and Zoekt comparison exist can the
AI-tooling evaluation say whether the workbench is actually better.

Implementation:

- Add benchmark cases for selected repos under `/mnt/workspace/AI tooling`.
- Include questions around:
  - session/state sync;
  - agent coordination;
  - MCP gateway behavior;
  - ZCode bridge/supervisor logic;
  - rules/skills sync;
  - durable memory;
  - auth/filtering/observability.
- Run benchmark modes:
  - Tantivy lexical;
  - Tantivy plus ctags;
  - Tantivy plus tree-sitter;
  - SCIP graph where available;
  - Zoekt;
  - embedding;
  - rerank.
- Publish reports under `/mnt/workspace/code-intel/` and keep repo-local
  templates under `data/eval/cases/` or `docs/`.

Suggested commit:

`Add benchmark reports for AI tooling repos`

Verification:

```bash
python3 src/cli/main.py run-benchmarks --workspace-root "/mnt/workspace/AI tooling/repos" --limit 10
```

Done when:

- benchmark reports identify retrieval wins and regressions;
- non-Rust repos no longer look like plain file-search-only targets;
- reports include enough evidence to guide future ranking changes.

## Goal Check Protocol

At the start and end of each implementation session:

1. Open this document.
2. Check which milestone is next.
3. Run the milestone verification command or explain why it cannot run yet.
4. Update the milestone status in this file only when the implementation and
   verification evidence both exist.
5. Commit each completed milestone separately with a body derived from the diff.
6. Do not mark the goal complete until all eight milestones are done.

## Current Goal Status

Overall status: not complete.

Current next milestone: AI-tooling benchmark reports.

Current risk: sidecar search comparison works, but the selected AI-tooling repos
still need benchmark reports that show retrieval quality per mode.

Latest verification:

```bash
python3 -m unittest discover tests/unit
python3 src/cli/main.py run-benchmarks --search-root /mnt/workspace/code-intel/repo-analysis-pilot/search --graph-root /mnt/workspace/code-intel/repo-analysis-pilot/graph --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --eval-root /mnt/workspace/code-intel/repo-analysis-pilot/eval --cases-root data/eval/cases --repo agent-kit --mode lexical_only --limit 5
python3 -m py_compile src/symbols/schema.py src/symbols/indexer.py src/graph/builder.py src/search/indexer.py src/summaries/builder.py
sudo apt-get install -y universal-ctags
python3 src/cli/main.py build-index --workspace-root "/mnt/workspace/AI tooling" --raw-root /mnt/workspace/code-intel/repo-analysis-pilot/raw --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --graph-root /mnt/workspace/code-intel/repo-analysis-pilot/graph --repo agent-kit --progress-interval 1
python3 src/cli/main.py build-search --workspace-root "/mnt/workspace/AI tooling" --raw-root /mnt/workspace/code-intel/repo-analysis-pilot/raw --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --search-root /mnt/workspace/code-intel/repo-analysis-pilot/search --repo agent-kit
python3 src/cli/main.py find-symbol --search-root /mnt/workspace/code-intel/repo-analysis-pilot/search --repo agent-kit mustExist --limit 5
python3 src/cli/main.py build-summaries --raw-root /mnt/workspace/code-intel/repo-analysis-pilot/raw --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --graph-root /mnt/workspace/code-intel/repo-analysis-pilot/graph --repo agent-kit
python3 -m unittest discover tests/unit
python3 -m py_compile src/parsers/tree_sitter_backend.py src/symbols/indexer.py tests/unit/test_tree_sitter_backend.py
/mnt/workspace/code-intel/repo-analysis-tree-sitter-venv/bin/python src/cli/main.py build-index --workspace-root "/mnt/workspace/AI tooling" --raw-root /mnt/workspace/code-intel/repo-analysis-pilot/raw --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --graph-root /mnt/workspace/code-intel/repo-analysis-pilot/graph --repo agent-kit --progress-interval 1
python3 src/cli/main.py repo-overview --parsed-root /mnt/workspace/code-intel/repo-analysis-pilot/parsed --repo agent-kit
python3 src/cli/main.py search-lexical --search-root /mnt/workspace/code-intel/repo-analysis-pilot/search --repo agent-kit tree_sitter_tags --limit 5
python3 -m unittest tests.unit.test_scip_backend
python3 -m unittest tests.unit.test_symbol_index tests.unit.test_symbol_snapshot tests.unit.test_symbol_semantics tests.unit.test_parser_fusion tests.unit.test_search_and_summaries
python3 -m py_compile src/parsers/scip_backend.py src/symbols/indexer.py src/graph/builder.py src/cli/main.py tests/unit/test_scip_backend.py
python3 src/cli/main.py parse-repos --workspace-root /mnt/workspace/code-intel/scip-fixture-workspace --output-root /mnt/workspace/code-intel/scip-fixture-artifacts/raw --repo scip-demo
python3 src/cli/main.py build-index --workspace-root /mnt/workspace/code-intel/scip-fixture-workspace --raw-root /mnt/workspace/code-intel/scip-fixture-artifacts/raw --parsed-root /mnt/workspace/code-intel/scip-fixture-artifacts/parsed --graph-root /mnt/workspace/code-intel/scip-fixture-artifacts/graph --repo scip-demo --scip-index index.scip.json --progress-interval 1
python3 src/cli/main.py build-search --raw-root /mnt/workspace/code-intel/scip-fixture-artifacts/raw --parsed-root /mnt/workspace/code-intel/scip-fixture-artifacts/parsed --search-root /mnt/workspace/code-intel/scip-fixture-artifacts/search --repo scip-demo
python3 src/cli/main.py find-symbol --search-root /mnt/workspace/code-intel/scip-fixture-artifacts/search --repo scip-demo run --limit 5
python3 src/cli/main.py build-summaries --raw-root /mnt/workspace/code-intel/scip-fixture-artifacts/raw --parsed-root /mnt/workspace/code-intel/scip-fixture-artifacts/parsed --graph-root /mnt/workspace/code-intel/scip-fixture-artifacts/graph --summary-root /mnt/workspace/code-intel/scip-fixture-artifacts/summaries --repo scip-demo
python3 src/cli/main.py repo-overview --parsed-root /mnt/workspace/code-intel/scip-fixture-artifacts/parsed --repo scip-demo
python3 -m unittest tests.unit.test_zoekt_backend
python3 -m py_compile src/search/zoekt_backend.py src/cli/main.py tests/unit/test_zoekt_backend.py
sudo apt-get install -y golang-go
GOBIN=/mnt/workspace/code-intel/bin go install ./cmd/zoekt-index ./cmd/zoekt
python3 src/cli/main.py build-zoekt --workspace-root "/mnt/workspace/AI tooling" --zoekt-root /mnt/workspace/code-intel/repo-analysis-pilot/zoekt --repo agent-kit --zoekt-index-bin /mnt/workspace/code-intel/bin/zoekt-index
python3 src/cli/main.py search-zoekt --zoekt-root /mnt/workspace/code-intel/repo-analysis-pilot/zoekt --repo agent-kit mustExist --limit 5 --zoekt-bin /mnt/workspace/code-intel/bin/zoekt
python3 src/cli/main.py compare-search-backends --search-root /mnt/workspace/code-intel/repo-analysis-pilot/search --zoekt-root /mnt/workspace/code-intel/repo-analysis-pilot/zoekt --repo agent-kit mustExist --limit 5 --zoekt-bin /mnt/workspace/code-intel/bin/zoekt
```
