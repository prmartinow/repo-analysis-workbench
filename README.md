# Repo Analysis Workbench

Local code-intelligence and retrieval workbench for large source repositories.

The system turns a repository into structured local artifacts that an LLM coding agent or human operator can query repeatedly:

- raw repository inventory
- parsed files, symbols, imports, references, statements, and symbol bodies
- LMDB exact metadata and summary storage
- Tantivy lexical, identifier, and path search
- RyuGraph structural traversal
- deterministic summaries
- optional embedding sidecars
- retrieval planning, answer-bundle preparation, and evaluation commands

This is best described as a **code-intelligence and retrieval workbench**. It includes a repository-memory layer, but memory is not the whole system; the core value is the combination of exact parsed metadata, lexical retrieval, graph-backed expansion, summaries, and evaluation for agentic code understanding.

## Architecture

- **Python orchestration** for CLI commands, planning, evaluation, and artifact coordination.
- **Tantivy** for fast lexical, identifier, and path retrieval.
- **LMDB** for exact metadata, symbol bodies, summaries, and evaluation cache.
- **RyuGraph** for graph storage and traversal.
- **Optional embeddings** as a sidecar, not the source of truth.
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
