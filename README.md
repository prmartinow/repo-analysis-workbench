# Repo Analysis Workbench

Local code-intelligence and retrieval workbench for large source repositories.

The system turns a repository into structured local artifacts that an LLM coding agent or human operator can query repeatedly:

- raw repository inventory
- parsed files, symbols, imports, references, statements, and symbol bodies
- LMDB exact metadata and summary storage
- Tantivy lexical, identifier, and path search
- RyuGraph structural traversal
- deterministic summaries
- mandatory local-Qwen embedding sidecars
- retrieval planning, answer-bundle preparation, and evaluation commands

This is best described as a **code-intelligence and retrieval workbench**. It includes a repository-memory layer, but memory is not the whole system; the core value is the combination of exact parsed metadata, lexical retrieval, graph-backed expansion, summaries, and evaluation for agentic code understanding.

## Architecture

- **Python orchestration** for CLI commands, planning, evaluation, and artifact coordination.
- **Tantivy** for fast lexical, identifier, and path retrieval.
- **LMDB** for exact metadata, symbol bodies, summaries, and evaluation cache.
- **RyuGraph** for graph storage and traversal.
- **Mandatory local-Qwen embeddings** as a batch-built semantic sidecar.
- **Native Rust worker** for hot-path parsing and indexing support.

## Basic Workflow

Assume your target repositories live next to this repo under one workspace directory:

```bash
workspace/
  repo-analysis-workbench/
  target-repo/
```

Bootstrap artifact folders:

```bash
./scripts/bootstrap.sh
```

Build artifacts for one repository:

```bash
./scripts/parse_repos.sh --repo target-repo
./scripts/build_index.sh --repo target-repo
./scripts/build_search.sh --repo target-repo
./scripts/export_summaries.sh --repo target-repo
./scripts/build_embeddings.sh --repo target-repo
```

Embedding retrieval expects the batch-built sidecar to exist. `prepare-context`, `prepare-answer-bundle`, and benchmark retrieval always query this sidecar; missing embeddings are treated as an artifact build failure. The default provider is the local RPC Qwen inference service:

```bash
./scripts/build_embeddings.sh --repo target-repo --provider qwen --model text
```

The default local endpoint is `http://127.0.0.1:18200/v1/embeddings`; override it with `REPO_ANALYSIS_QWEN_EMBEDDINGS_URL` if needed.

The local Qwen reranker is enabled by default for the top retrieval candidates:

```bash
python3 src/cli/main.py prepare-answer-bundle --repo target-repo "How does retry handling work?"
```

The reranker uses `http://127.0.0.1:18200/rerank` by default and reranks at most five top candidates. If the local service is unavailable, retrieval fails loudly unless `REPO_ANALYSIS_ALLOW_HEURISTIC_RERANK_FALLBACK=1` is set.

Query the repository:

```bash
python3 src/cli/main.py repo-overview --repo target-repo
python3 src/cli/main.py find-symbol --repo target-repo "SomeSymbol"
python3 src/cli/main.py search-lexical --repo target-repo "retry logic"
python3 src/cli/main.py callers-of --repo target-repo "SomeSymbol"
python3 src/cli/main.py prepare-context --repo target-repo "How does retry handling work?"
python3 src/cli/main.py prepare-answer-bundle --repo target-repo "How does retry handling work?"
```

## Repository Scope

This repo intentionally does not vendor target repositories and does not include project-specific submodules. Point the workbench at local repositories with `--workspace-root` and `--repo`.

Generated artifacts live under `data/` and are gitignored.
