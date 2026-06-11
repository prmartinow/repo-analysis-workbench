# Repo-Analysis: Target Architecture and Phased Migration

## Purpose

`repo-analysis` should become a **repository intelligence layer for coding agents**, not just a local search tool.

The goal is to help an agent:

- reach a correct mental model of a repository faster
- spend fewer tokens on blind file reads
- retrieve smaller but more sufficient evidence sets
- stay grounded while answering, documenting, planning, and editing
- improve over repeated work on the same repository

The design notes behind this prototype point to one central conclusion:

**the problem is not mainly "better chunk retrieval."**
It is **cost-aware evidence acquisition over structured repository state**.

That means the long-term system should be optimized as a controller over:

- structured repository artifacts
- multi-view graph traversal
- selective retrieval and stopping
- evidence-set construction
- repository-native memory
- execution-derived signals
- evaluation of both quality and efficiency

This document translates that design direction into a staged architecture for the current repository.

## What The Current Prototype Already Gets Right

The current implementation already matches a meaningful part of the literature.

- It is **parser-first**, not embedding-first.
- It uses a **three-engine runtime** with Tantivy, LMDB, and RyuGraph.
- It already supports **symbols, statements, summaries, and graph traversal**.
- It already has an **agent-facing retrieval path** in `retrieve_context`, `plan_query`, and `prepare_answer_bundle`.
- It already treats embeddings as a **sidecar**, not the primary truth source.
- It already includes an early form of **selective retrieval gating**.

That is a strong base. The architecture in [Overview.md](Overview.md) and [Design & Requirements.md](Design%20&%20Requirements.md) is directionally correct.

## Where The Prototype Is Still Thin

The current prototype gaps are also fairly clear.

### 1. Retrieval is still mostly heuristic

The current retrieval stack has lexical search, exact symbol lookup, graph expansion, reranking, and summaries, but it does **not yet estimate evidence sufficiency** or optimize for the **smallest sufficient evidence set**.

It also does not yet explicitly defend against **textual bias** in code retrieval, where names, docstrings, and surface wording can outrank structurally central code that matters more for the task.

### 2. The graph is useful but not yet multi-view enough

The graph covers code structure well enough for the current slice, but the research strongly suggests that future gains will come from adding more views:

- code-to-test relations
- build/dependency/environment relations
- change-impact relations
- history-derived relations
- runtime/execution-derived relations

### 3. Summaries are present but shallow

The summary layer is useful for routing, but it is still mostly deterministic rollup text. It is not yet a richer abstraction layer for subsystem intent, dependency boundaries, change surfaces, or execution behavior.

It also does not yet provide short **semantic file descriptions** that capture:

- file purpose
- core components
- relationships to other files

That matters because repository trees and filenames alone can be too weak for coarse localization when lexical signals are misleading.
Those descriptions should be **precomputed and cached** as routing metadata, not generated ad hoc during the hot path.

### 4. There is no repository-native memory layer

The system currently indexes repository state, but it does not preserve **project memory** across tasks:

- prior successful retrieval paths
- design commitments
- active subsystem notes
- commit-history abstractions
- prior failure and regression signatures
- evolving task/specification state

### 5. Execution feedback is mostly outside the architecture

The research now makes it clear that build/test/runtime evidence should affect retrieval, ranking, and stopping. The current repo mostly stops before that layer.

### 6. Evaluation is too narrow for the next stage

The evaluation harness measures retrieval modes and interactive command latency, which is useful, but the next architecture needs additional metrics:

- explored vs utilized evidence
- prompt efficiency
- evidence sufficiency
- retrieval noise rate
- impacted-test ranking quality
- long-horizon memory reuse quality

## Target System

The best target for this repository is a **layered repository intelligence system** with one cross-cutting controller.

### Cross-cutting controller

This is the main architectural change in emphasis.

`repo-analysis` should stop acting like a passive collection of indexes and start acting like a **budgeted evidence controller** that decides:

- whether retrieval is needed
- which task profile and latency budget applies
- which retrieval stage to use
- how far to expand
- how much marginal evidence gain the next retrieval step is likely to add relative to its cost
- which evidence items belong together
- when enough evidence has been collected
- when the evidence is still insufficient and the system should abstain, defer, or ask for verification
- when to run targeted verification
- when to stop

That controller should primarily live in the retrieval/planning surface, not in the storage engines.

The agent-facing tool surface should stay **minimal, non-overlapping, and token-efficient** so the model is not forced to choose between redundant commands or carry unnecessary tool schema overhead.

### Layer 1. Artifact Ingestion

Purpose:

- scan repositories
- parse source code
- collect exact symbols and statements
- collect package/build/test metadata
- fingerprint artifacts for rebuilds

Current base:

- `src/symbols/indexer.py`
- `src/parsers/*`
- adapter inventory

Target extension:

- keep the current parser-first approach
- preserve per-repo isolated artifacts
- add language-specific parsers behind one normalized repository IR
- when a language exposes a compiler-native semantic API, prefer it over per-symbol LSP RPC for large-scale semantic resolution workloads
- make logical code units such as files, classes, methods, variables, and selected statements the primary index units, with contiguous text chunks kept as secondary packaging or fallback only
- use fast shallow indexing for broad coverage and then complete cross-file edges and relations in follow-up passes where needed
- update syntax artifacts incrementally after edits and remain useful even when the current file contains syntax errors
- materialize core static-analysis artifacts directly, including CFG/call/dependency facts, instead of asking an LLM to infer them on demand
- add normalized build/test/config metadata collection
- add optional license and provenance metadata for files and snippets when the repository exposes it
- add optional issue/spec/documentation sidecars when local repository artifacts make those links available
- add optional project-rule sidecars from files such as `CLAUDE.md`, `AGENTS.md`, and cursor rules, normalized into project info, conventions, guidelines, workflow directives, and examples
- keep project-rule artifacts separate from generic documentation and code overviews, and deduplicate them against already available docs before surfacing them
- add change/history sidecars as optional artifacts
- add optional entrypoint, test-trace, and interface-spec artifacts when commands or tests can expose executable feature boundaries
- add stable file identities and rename-aware alias mapping so commit-history evidence can be projected back onto the current repository state
- add hierarchical project, directory, and file summary artifacts so the agent can narrow context top-down before reading raw code
- support compressed placeholder summaries for deep subtrees, with on-demand expansion when retrieval needs deeper detail
- let summary artifacts carry both high-level purpose and selected member-level details when that helps natural-language retrieval
- keep generated descriptions concise and task-oriented, focusing on purpose, interfaces, behavior, and important relationships rather than long prose
- add a consistent filtering pipeline for generated content, secrets, binaries, oversized files, and low-value paths
- persist cached raw file content, content hashes, and symbol slice metadata so exact source can be served without reparsing whole files

### Layer 2. Repository IR And Multi-View Graph

Purpose:

- store the repository as connected program entities rather than disconnected text
- support graph navigation, path search, slicing, impact propagation, and neighborhood expansion
- make explicit repository relations first-class so retrieval and generation stay consistent with evolving codebase context

Current base:

- `src/graph/builder.py`
- `src/backends/ryugraph/*`

Target graph views:

- file and directory hierarchy
- package/module ownership
- symbol/signature ownership
- statement-level nodes and h-hop dependency slices when statement granularity matters
- call/reference edges
- selected control-flow and control-dependence edges
- type/inheritance/implementation edges
- selected data-flow / dependence edges
- code-to-test edges
- build/dependency/environment edges
- optional issue/spec/documentation/config traceability edges
- optional requirement/spec concept nodes and requirement-to-code mapping edges when those artifacts exist or can be derived reliably
- optional history and execution evidence edges

Design rule:

Keep the current graph runtime, but widen the graph schema only when an added edge family is queryable, benchmarkable, and cheap enough to justify its build/query cost.

Parser-derived graph facts should remain the primary structural truth. LLM-extracted graph edges can exist as optional hints or backfill signals, but they should carry lower trust and explicit provenance.

Agent-facing graph access should prefer task-shaped operations such as slicing, caller expansion, impact tracing, and dependency-flow lookup over exposing a raw graph-query DSL as the default interface.

When optional higher-order relation families are induced rather than parsed directly, build them through bounded candidate pools and sparse validation instead of dense all-pairs inference.

When choosing richer semantic structure, prefer compact relations with high retrieval value per cost, such as data-flow or dependence facts, over deep tree expansion by default.

Dependency artifacts should support **incremental affected-region updates** after repository changes instead of forcing full recomputation whenever a file changes.

### Layer 3. Retrieval And Localization

Purpose:

- localize candidate repository regions cheaply
- avoid early prompt flooding
- promote exact identity and structural relevance over text overlap
- make symbol-first exploration cheap enough that agents do not need to jump to whole-file reads early

Current base:

- `src/search/indexer.py`
- `src/backends/tantivy/search.py`
- `src/retrieval/engine.py`

Target behavior:

1. summaries and coarse scopes first
2. exact symbol/file localization next
3. graph expansion inside the narrowed scope
4. body hydration only for likely winners
5. optional semantic sidecar only when lexical/structural paths are weak, ideally as short cached file descriptions rather than large generated prose
6. reranking that can correct for superficial textual matches when structural evidence points elsewhere
7. semantic file descriptions can augment tree/path localization when filenames and docstrings are weak or misleading
8. exact retrieval results should surface freshness or verification status when the underlying content may have changed
9. query planning should adapt to the evidence actually available, mixing lexical, structural, execution, and history signals rather than assuming one fixed retrieval route
10. a query-understanding step should extract likely files, symbols, entity names, and term variants before search when the request is phrased in natural language
11. query understanding should also extract likely subrequirements, latent API needs, and semantically similar intents, not only literal entity names
12. issue and edit tasks should allow broad file/module search first and then deeper function or statement search as the frontier narrows
13. commit-history retrieval should be a distinct route over commit messages and diffs, with rename-aware mapping from historical paths back to current repository entities
14. semantic hits on documentation, comments, or generated descriptions should resolve back to concrete code entities before graph expansion
15. retrieval should support line-level, API-level, function-body, and larger edit-step granularity depending on the task
16. retrieval policy should change by task profile, since chat, autocomplete, editing, and test generation need different context and latency behavior
17. vague bug reports and natural-language issue descriptions should be routed through the project/directory/file summary tree before raw body retrieval
18. explicit code-entity queries should have a direct graph-backed fast path, with deeper graph search only when the direct path is empty or ambiguous
19. candidate selection should preserve diversity across repository regions so the evidence set does not fill with near-duplicate local hits
20. language-aware retrieval should be able to combine broad semantic filtering with deterministic structural queries when overloaded identifiers, namespace shadowing, or similar ambiguities make lexical search unreliable
21. read and search tools should accept an optional focus question or goal hint so coarse outputs can be filtered against the agent’s current intent before they enter the main context
22. for code-generation tasks, the system should be able to reshape structural evidence into caller-usage context and callee-dependency context, not only detached snippet bundles
23. once a first draft or hypothesized API set exists, retrieval can use those provisional signals to refine downstream dependency lookup
24. cheap lexical retrieval should support fuzzy path/file lookup plus regex search with directory or path scoping, since strong agents can narrow repository structure effectively before semantic expansion
25. for code-generation tasks, hybrid retrieval should combine required dependency definitions with representative in-repo usage examples, since both building blocks and usage patterns matter
26. caller-centric exploration should be a first-class retrieval path for generation and reconstruction tasks, since caller files often expose the most useful usage patterns and integration constraints
27. query planning can rewrite raw requests into a compact task schema containing goals, operations, artifacts, constraints, and high-value keywords before hybrid seeding
28. for multi-step tasks, retrieval should be able to infer semantic input/output roles and search for dependency paths that connect them

Design rule:

Default to **coarse-to-fine hybrid retrieval**. Do not let embeddings become the default route.

Agent-facing lookups should prefer a **symbol-first, outline-first interaction style** before escalating to raw file bodies.

Cold-start discovery should expose repo outlines, file outlines, and suggested initial queries so agents can orient before they know repository vocabulary.

Agent-facing lookup tools should support **fold, preview, and full-detail modes**, and graph traversals should be rendered as compact tree-shaped text with explicit relation labels.

Project rule files such as `AGENTS.md` should usually be surfaced as targeted directives or examples, not naively injected as full repository overviews.

Exact source retrieval on hot paths should come from **stored offsets into cached file content** with optional verification, not repeated reparsing or whole-file rereads.

If a semantic sidecar is used for natural-language queries, prefer **precomputed code representations with runtime query encoding** over recomputing code-side semantics on the hot path.

When graph-backed evidence is rendered for the model, preserve node types, relation types, and a small amount of selected property metadata instead of flattening everything into unlabeled snippets.

Embedding indexes, when present, should be treated as **approximate retrieval infrastructure** with explicit recall/latency/memory tradeoffs, predicate filtering, and offline build/update behavior.

Default retrieval cadence should be **per answer/edit step**, not per generated token, unless a task has a very specific need for token-level retrieval.

When multiple cheap retrieval channels are available, combine their rankings with a simple fusion strategy instead of forcing one channel to win globally.

For low-latency task profiles, cheap local retrieval and deeper cross-file retrieval should be able to run speculatively in parallel so the system can return fast when enough cheap evidence exists and fall through to richer evidence only when needed.

For read-heavy workflows, a lightweight goal-conditioned skimming layer should be able to sit between coarse file reads and the main model context.

The retrieval surface should support a small **reason-act-observe loop**: search, inspect, expand, verify, and stop.

Tool-call repair and retry should happen in short side contexts so formatting errors do not pollute the main retrieval context.

### Layer 4. Evidence-Set Construction

Purpose:

- build a small, coherent evidence coalition rather than a top-k bag of hits
- reduce harmful and redundant context
- preserve dependency paths that matter

Current base:

- `prepare_answer_bundle`
- reranking and stage scoring

Target extension:

- utility-aware filtering
- coalition-aware evidence scoring
- bridge-node preservation
- ambiguity reduction scoring
- path-complete evidence bundles for explanation and edits
- reranking and filtering that explicitly counter name/docstring bias when semantic and structural evidence disagree
- token-budgeted context bundles with explicit inclusion/exclusion accounting
- operational metadata on bundles so an agent can tell what was truncated, what was verified, and what budget was actually consumed
- claim-to-evidence alignment so major answer claims can be backed by specific file-path and line-range evidence
- execution-derived filters that can clip candidates not exercised in the relevant trace or scenario when such evidence is available
- soft robustness to mixed relevant and irrelevant evidence, so noisy bundles are downweighted and trimmed without blindly discarding useful context
- task-specific evidence types such as existing test-file exemplars and local test conventions when the task is test generation
- topology-preserving pruning that keeps signatures, summaries, and dependency structure while dropping low-value implementation detail from dependent files when budgets are tight
- goal-conditioned line- or block-level skimming on coarse file reads, while preserving enough syntax and local structure to remain usable
- separate wide analysis memory from the final synthesis bundle so the system can gather broadly during investigation and then filter to the smallest useful evidence set before answer or patch generation
- frontier-pruning during iterative search so unproductive branches can be dropped instead of repeatedly revisited
- evidence items should be attributable as positive, neutral, or negative contributors when training or evaluating context filters
- semantic verification passes should be able to reject semantically incongruent top candidates and trigger candidate-pool expansion or diversification instead of repeatedly reranking the same local neighborhood
- for generation tasks, evidence construction should support caller-context inlining and direct callee materialization when that preserves semantics better than detached top-k snippets
- offline distillation of minimal sufficient contexts should be used to train or calibrate context compressors, rather than relying only on relevance scoring proxies
- if the host runtime applies further context or KV-cache compression, structurally critical anchors such as call sites, branch conditions, and assignments should be preserved instead of relying on attention-only pruning

This is where the research on CODEFILTER, RepoShapley, selective retrieval, and caller-centric exploration fits most directly.

### Layer 5. Repository-Native Memory

Purpose:

- preserve repository knowledge across tasks and long sessions
- reduce repeated blind exploration
- carry forward trustworthy project state

This layer does not exist yet and should be added explicitly.

Memory objects should include:

- subsystem summaries refined through repeated use
- design commitments and architectural invariants
- recent active-module summaries
- active multi-step edit plans and already-applied change state
- prior successful localization trails
- prior failing retrieval trails
- structured action/result traces from prior research trajectories
- multi-facet experience summaries for issue comprehension, localization strategy, and modification strategy
- structured note/checkpoint objects and compaction summaries that preserve decisions, unresolved bugs, open questions, and current plan state
- code-centric session memories that track validated edits, later revisions, and possible forgetting or contradiction across turns
- small working-set snapshots for the currently active files or entities so compaction can preserve the hot local context of an ongoing task
- synchronized project-level specification state and local task state, stored separately so durable commitments are not mixed with short-lived edit context
- commit-history abstractions
- time-filtered commit exemplars and linked-issue notes
- hotspot functionality summaries for frequently edited files and modules
- distilled project rules and collaboration constraints from repository rule files
- regression signatures and impacted-test mappings
- evolving specification notes tied to repo entities

Memory retention should balance **recency and actual retrieval usefulness**, not rely on FIFO-only eviction.

Memory should support a wider analysis view and a smaller synthesis view, rather than replaying every stored item during generation.

Historical memory lookup should be time-filtered relative to the task or benchmark instance so future commits do not leak into retrieval.

Compaction should preserve high-value state and clear stale raw tool outputs instead of replaying whole traces.

Compaction tuning should bias toward recall before precision so critical state is not dropped early.

When live context grows noisy, the controller should be able to restart from a fresh session seeded with rendered project state instead of dragging a polluted transcript forward.

When related-task reuse is available, concise summaries of prior trajectories should usually be preferred over raw full traces unless the full trace is clearly needed.

This memory should be **repository-native**, not generic chat transcript replay.

### Layer 6. Execution And Verification Signals

Purpose:

- turn build/test/runtime evidence into retrieval and planning signals
- reduce false confidence from static-only reasoning

This layer should eventually capture:

- failing test identifiers
- crash reports and reproducer scripts
- impacted-test candidates
- build/dependency failures
- CI job outcomes and failed-step summaries
- unresolved imports/references
- static-analysis diagnostics, warnings, and editor-visible errors
- security test outcomes and sanitizer/fuzzer signals
- execution-free verifier scores for candidate answers, patches, or trajectories when full execution is unavailable, too slow, or too noisy
- hybrid verification signals that combine execution-based outcomes with execution-free ranking when both are available
- graph-derived test maps and regression-risk signals for targeted verification
- mutation-derived test-adequacy signals and mutation-guided test-augmentation candidates
- stack trace anchors
- trace summaries
- entrypoint-level execution traces, including executed-line and executed-file coverage where available
- trace configuration parameters such as target file, trace scope, target function, and trace depth
- structured execution workflow reports distilled from raw traces, not raw trace dumps
- runtime state transition summaries
- regression signatures
- scenario-specific execution traces that can be used as ranking and filtering signals during retrieval, not only after edits
- isolated sandbox verification results for target functions plus local dependencies when full-repository execution is too expensive or fragile
- coverage signals from targeted verification harnesses when they are available
- environment-alignment state that tracks external dependency satisfaction separately from repository-internal reference resolution
- execution-evidence attribution that maps failures to the smallest revisable environment or code region before the next repair step

Design rule:

Execution should not be only a final gate. It should feed back into ranking, memory, and stopping.

Dynamic traces should be narrowed adaptively when trace volume exceeds budget, rather than pasted raw into context or abandoned entirely.

### Layer 7. Evaluation And Observability

Purpose:

- measure whether the system is actually improving the agent
- separate retrieval quality from generation quality
- protect against "more context, same or worse outcome"

Current base:

- `src/evaluation/harness.py`
- telemetry and artifact metadata

Target metrics:

- retrieval recall and precision
- retrieval acc@k where the task has a known gold context target
- explored vs utilized evidence
- evidence bundle size and token cost
- sufficiency estimator quality
- harmful-context rate
- query latency
- verification-hit rate
- stale-result rate and freshness-wait cost
- actual context occupancy, including system/tool-schema overhead, reserved history, and cached-token occupancy
- impacted-test ranking quality
- memory reuse lift
- end-task success under fixed budget
- completion/generation quality measured separately from retrieval quality when the evaluation task includes a generation step
- retrieval-only, generation-only, and full-pipeline scores when the benchmark exposes those phases separately
- retrieval quality across diverse task types and benchmark families, not only one leaderboard-style niche
- retrieval robustness under normalization-style stress tests where names, variables, and docstrings are weakened or removed
- retrieval robustness under mixed relevant and irrelevant evidence bundles
- coverage across text-to-code, code-to-code, code-to-text, and hybrid retrieval tasks
- standardized retrieval-task adapters so internal evaluations and external benchmarks can be compared through one consistent schema
- retrieval granularity slices for function-, file-, module-, and repository-level tasks, including change-request-driven retrieval
- file-, definition-block-, and line-level retrieval metrics against gold contexts, with blocks standardized as reusable semantic units rather than arbitrary AST nodes
- language-aware benchmark slices, since retrieval behavior and context usefulness differ across repository languages
- context-depth sensitivity, since long-context retrieval quality changes with how deeply the needed code is buried
- context-utilization slices such as caller coverage, masked-file reconstruction quality, and usage-aware integration tests that isolate whether repository context was actually used
- long-horizon conversational context-management slices that separate topic awareness, information item extraction, and final function generation
- Oracle/Empty/Full-style normalized context-management scores so systems are compared against matched upper and lower bounds
- task-profile slices for chat, autocomplete, editing, test generation, and executable issue-resolution settings
- IDE-native tool-interface slices and full-stack workflow slices, not only terminal-only or static-context settings
- issue-workflow stage metrics for discovery, localization, and fix generation when the benchmark exposes those phases
- intermediate-reasoning slices for issue understanding, file localization, implementation task identification, and step decomposition when the benchmark exposes them
- abstention quality when context is insufficient, not just answer quality when a response is forced
- repository-QA answer quality dimensions such as factual accuracy, completeness, relevance, clarity, reasoning quality, and source correctness when answers mix prose and code
- repository-QA citation correctness for file-path and line-range evidence
- repository-QA slices built from real developer-authored questions and accepted answers when those artifacts are available
- documentation-sidecar utility measured through functionality detection, functionality localization, functionality completion, and downstream feature implementation success rather than judge-only scoring
- question-type slices for repository QA, including `what`, `why`, `where`, and `how`, plus finer-grained intents such as dependency tracing, design rationale, feature location, and API usage
- marginal evidence gain per retrieval step, stop-policy quality, and search-trajectory diversity during multi-step retrieval
- feature-development task slices that span multiple commits, PRs, or interfaces, not only bug-fix tasks
- related-task sequence evaluation for experience reuse, measuring accuracy, wall-clock time, and token cost under full-trajectory reuse versus compact-summary reuse
- emergent-specification versus single-shot control slices for long-horizon coding
- implementation-faithfulness metrics that pair semantic component faithfulness with structural integration, since test pass rate can miss specification loss
- execution-trace complexity slices, such as trace length and unique files touched, for codebase-understanding tasks that depend on runtime behavior
- temporal-leakage controls for history-aware evaluation, including masking future commits and future repository state when replaying historical tasks
- matched time-consistent A/B evaluation in which the same agent is compared with and without repository-derived knowledge under identical historical task instances
- fresh post-cutoff same-repository tasks and outside-repository tasks to detect benchmark or repository memorization
- unpublished or never-publicly-exposed repository slices when possible, especially for IDE-style evaluation
- context-free diagnostic slices, such as file-path or patch-memory probes, to separate real repository reasoning from memorization
- standalone function-level control tasks so repo-aware retrieval overhead is not mistaken for general code ability
- security-sensitive task slices that require both correctness tests and dynamic security tests when available
- regression-aware coding metrics such as test-level pass-to-pass failures, catastrophic regression counts, and net resolution-versus-regression tradeoffs
- mutation score or mutant-kill coverage for benchmark regression suites, so weak tests are not mistaken for solved tasks
- repository-health metrics across sequential tasks, including cognitive-complexity drift, technical-debt accumulation, and contamination from residual task state
- static-analysis success rate, type-check success rate, integration-test pass rate, and selective human review for high-value tasks
- verifier discrimination and calibration metrics such as AUC and ECE when learned reward or ranking models are used
- tool-call sequence telemetry, read/edit/execute round distribution, and intent-to-outcome alignment for agent workflows

The evaluation harness should be able to generate repository-specific QA slices from seed question templates and repository metadata when a target repository needs custom understanding checks.

Benchmark setups should let the agent use its own retrieval policy when the goal is to measure repository intelligence, instead of forcing one fixed context provider.

## Phased Migration

Each phase below covers the active focus for this repository: core / must-have first, then important / later. The research backlog / optional track remains deferred to the end of this document.

## Priority Classification

The architecture is broad. The priority split below defines what this repository should treat as required for the problem at hand versus what should wait until the core retrieval loop is already working well.

### Core / Must-Have

These are the parts that most directly improve agent accuracy, reduce token waste, and make the tool usable as a real repository-intelligence layer now.

- parser-first artifact ingestion with incremental updates and exact symbol/statement identities
- hierarchical project, directory, and file summaries for top-down narrowing
- cached exact-source retrieval from stored offsets and content hashes
- summary-first, symbol-first, and outline-first retrieval flows
- coarse-to-fine retrieval that starts with cheap lexical and structural routes before deeper expansion
- task-aware query planning, query typing, and retrieval presets
- direct graph-backed fast paths for explicit entity requests
- task-shaped graph operators such as caller expansion, slicing, path lookup, and impact tracing
- anti-textual-bias reranking and semantic verification that can reject misleading high-surface-form matches
- evidence-bundle construction that optimizes for small coherent sets rather than top-k bags
- token-budget accounting, bundle diagnostics, and actual context-occupancy telemetry
- sufficiency estimation and stop/continue control for retrieval loops
- evaluation that separates retrieval quality from generation quality and measures prompt efficiency, harmful-context rate, and evidence sufficiency

If these pieces are weak, the rest of the architecture does not matter much because the agent still spends too many tokens exploring and still risks building on the wrong local evidence.

### Important / Later

These are high-value extensions, but they depend on the core retrieval and evidence loop already being solid.

- richer multi-view graph coverage for code-to-test, build, dependency, environment, and history relations
- rename-aware history mapping and commit-history retrieval
- project-rule sidecars, issue/spec/doc sidecars, and requirement-to-code mapping where those artifacts exist locally
- repository-native memory for subsystem summaries, localization trails, design commitments, spec state, and reusable traces
- execution-derived ranking and filtering signals from tests, traces, CI, diagnostics, and coverage
- graph-derived regression-risk signals and impacted-test ranking
- isolated verification and execution-free verifier calibration
- documentation-sidecar evaluation and repository-specific QA generation
- broader benchmark adapters and task-family slices for IDE, long-horizon, and workflow-heavy evaluations

These items should come after the retrieval controller and evidence bundling are already reliable, because otherwise they mostly add more data and more surface area without enough control over how that extra evidence is used.

### Research Backlog / Optional

These are useful ideas to keep in scope, but they should stay optional unless a concrete benchmark or repository workload proves their value.

- optional license and provenance metadata
- optional runtime-level structure-aware compaction hooks for hosts that support KV-cache or post-prompt compression
- induced higher-order graph relations that require sparse validation or learned inference
- requirement/spec concept nodes when the source artifacts are noisy or only weakly grounded
- mutation-guided test augmentation beyond targeted evaluation use
- compact semantic sidecars beyond short task-oriented file descriptions
- very broad evaluation slices that do not yet affect implementation decisions
- benchmark-specific features that do not transfer to the repositories this tool actually needs to support

These are worth keeping visible so the architecture stays extensible, but they should not compete with the retrieval, evidence, and control work that solves the immediate problem.

### Phase 0. Stabilize The Current Core

Outcome:

- treat the current three-engine architecture as the stable baseline
- make it easy for a coding agent to use immediately

Work:

- preserve the current CLI and artifact layout
- harden the current summary and answer-bundle path
- make summary-first, symbol-first, and exact-lookup-first usage the default agent workflow
- add a hierarchical project/directory/file summary tree for top-down repository narrowing
- ingest repository rule files and surface them separately from code evidence
- deduplicate project-rule content against existing repository docs before surfacing it
- add optional goal-hint parameters on read/search tools so downstream pruning can be task-aware without breaking existing workflows
- add lightweight outline/tree/query-suggestion surfaces for cold-start repository exploration
- surface freshness and verification metadata consistently in agent-facing responses
- improve artifact metadata and telemetry so retrieval decisions can be inspected
- instrument actual context occupancy and compaction savings so hidden overhead is visible

Why this first:

The current system is already good enough to complement an agent today if it is used as a **routing surface** rather than as a general-purpose semantic search box.

### Phase 1. Retrieval Quality Before New Complexity

Outcome:

- improve localization, reranking, and prompt efficiency without changing the runtime shape

Work:

- add stronger query typing and route selection
- add task-profile-aware retrieval presets for chat, autocomplete, editing, and test generation
- add query decomposition into literal entities plus latent subrequirements
- add compact task-schema rewriting before hybrid retrieval seeding
- strengthen summary-scope localization
- add direct graph-backed fast paths for explicit code-entity requests
- add task-shaped graph operators for slicing, caller expansion, impact tracing, and dependency-path lookup
- add short cached semantic file descriptions for coarse localization, especially where filenames or tree structure are weak
- add language-aware structural lookup paths for repositories where lexical ambiguity is common
- add generation-oriented caller/callee context shaping after localization when the task is implementation, not just question answering
- add dependency-plus-usage-example retrieval for generation tasks
- improve suppression of low-value artifacts like locals/tests/scaffolding when not requested
- add reranking features that downweight superficial name/docstring matches when structural evidence is weak
- add symbol-first and outline-first response modes for agent-facing lookup commands
- add fold/preview/full-detail response modes and compact tree-form graph outputs for agent-facing lookup commands
- add exact-source retrieval backed by stored offsets and optional content verification
- add lightweight rank-fusion across lexical, structural, local-recency, and semantic channels where available
- add diversity-aware candidate selection so bundles cover distinct useful regions before repeating similar hits
- add topology-preserving context pruning modes for dependent-file evidence
- add semantic verification that can reject incongruent hits and widen the candidate pool
- add short-side-context repair for malformed tool calls in iterative retrieval loops
- add token-budget reports for context-bundle construction
- add evidence-bundle diagnostics
- add retrieval-noise and lexical-bias stress metrics to evaluation
- add question-type-aware QA evaluation and common benchmark adapters in the harness

Code focus:

- `src/retrieval/engine.py`
- `src/retrieval/planner.py`
- `src/backends/tantivy/search.py`
- `src/evaluation/harness.py`

Why this phase:

The research strongly suggests that better control over the current layers is higher leverage than immediately adding more storage or more embeddings.

### Phase 2. Multi-View Graph Expansion

Outcome:

- move from a code graph toward a repository evidence graph

Work:

- add code-to-test edges
- add build/dependency/environment nodes and edges
- add change-impact-oriented edges where feasible
- add rename-aware commit-history retrieval and current-path mapping
- add optional requirement-to-code mapping artifacts for issue/spec-guided retrieval
- add code-to-test impact maps and graph-derived regression-risk views
- expose task-shaped graph operators and internal graph queries that support impacted-test and execution-aware reasoning

Code focus:

- `src/graph/builder.py`
- `src/backends/ryugraph/queries.py`
- `src/agents/toolkit.py`
- `src/evaluation/harness.py`

Why this phase:

Once the core retrieval controller is reliable, richer repository relations become worth the added storage and query complexity.

### Phase 3. Sufficiency And Evidence-Set Control

Outcome:

- stop optimizing for "more relevant hits"
- start optimizing for "smallest sufficient evidence set"

Work:

- add a sufficiency estimator
- add stop/continue decisions to retrieval loops
- log retrieval traces so stop decisions can be inspected and tuned
- add coalition-aware evidence filtering
- add evaluated prompt-budget targets
- separate retrieval quality from generation quality in evaluation
- add prompt-efficiency, harmful-context, and evidence-sufficiency metrics
- keep bundle-level diagnostics tied to measurable stop-policy outcomes

Code focus:

- `src/retrieval/engine.py`
- `src/retrieval/planner.py`
- new `src/evidence/` or `src/control/` package
- `src/evaluation/harness.py`

Why this phase:

This phase most directly targets better accuracy with fewer tokens by teaching the system when it already has enough evidence.

### Phase 4. Repository-Native Memory

Outcome:

- support repeated work on the same repository without restarting from zero

Work:

- add a repository memory store
- define schemas for project-state objects
- define schemas that separate durable project state from short-lived local task state
- store reusable subsystem and localization memories
- store structured analysis traces and filtered synthesis memories
- store time-filtered commit exemplars, hotspot summaries, and multi-facet experience memories
- add structured note-taking and compaction artifacts outside the prompt
- add code-centric session memory and AST-based forgetting or contradiction checks across turns
- store spec deltas and design commitments
- add retrieval routes that can consult memory before broad repo search

Code focus:

- new `src/memory/` package
- LMDB buckets for memory artifacts
- retrieval/planner integration

Why this phase:

The tool starts paying back repeated use only when it can preserve trusted project state and successful prior discovery paths.

### Phase 5. Execution-Aware Loop

Outcome:

- close the loop between repository understanding and executable verification

Work:

- ingest targeted build/test/runtime evidence
- ingest reproducer scripts and security-test signals alongside build/test/runtime evidence
- add isolated sandbox verification for target functions and local dependencies when full-repo execution is not worth the cost
- ingest CI outcomes and targeted coverage signals
- add execution-free verifier signals and calibrate them against execution outcomes where both are available
- capture entrypoint-derived execution traces and executed-line/file coverage for runtime-aware understanding
- add dynamic-trace granularity controls and trace-report summarization
- model environment alignment explicitly, separating external dependency setup from internal reference-resolution failures
- rank and route by impacted tests
- turn trace and failure evidence into reusable retrieval signals
- support execution-aware refinement of answer bundles and edit plans

Code focus:

- new `src/execution/` package
- graph and memory integration
- evaluation harness expansion

Why this phase:

This is where the system becomes behavior-aware instead of relying only on static repository structure.

## Recommended Immediate Operating Mode

Until later phases exist, the best way to use the current repo with a coding agent is:

1. `repo-overview` and `summarize-path` first
2. `find-symbol`, `find-file`, `where-defined`, `get-symbol-signature` next
3. `callers-of`, `callees-of`, `who-imports`, `path-between`, `statement-slice` only after localization
4. `plan-query` or `prepare-answer-bundle` before asking the model for a final explanation
5. targeted tests/builds outside the tool after the retrieval phase

That makes the current prototype act like a **repo router** rather than a last-mile answer engine.

## Concrete Recommendation For This Repository

The best next implementation order is:

1. stabilize the current routing surface and retrieval telemetry
2. improve localization, reranking, and evidence-bundle construction
3. expand repository sidecars and the multi-view graph
4. add sufficiency-aware retrieval control and evaluation
5. add repository-native memory
6. add execution-aware verification and ranking

In other words:

**keep the current three-engine core, make it good at finding the smallest sufficient evidence set, and then extend it into graph, memory, and execution-aware repository intelligence.**

That is the migration path most consistent with both the current codebase and the research folder.

The priority split above should govern tradeoffs inside those phases:

- if a feature improves localization quality, evidence sufficiency, prompt efficiency, or stop decisions, it belongs toward the front
- if a feature mostly adds new artifact families, new graph edges, or broader benchmarks without improving the control loop, it should wait
- optional research ideas should enter only when a target repository, benchmark, or measured failure mode justifies them

## Future Roadmap

Everything below is intentionally outside the active focus. It should start only after Phases 0 through 5 are working well enough that a concrete workload or benchmark justifies the extra complexity.

### Research Backlog / Optional

- add optional license and provenance metadata
- add optional runtime-level structure-aware compaction hooks for runtimes that support KV-cache or post-prompt compression
- explore induced higher-order graph relations that require sparse validation or learned inference
- explore requirement/spec concept nodes when the source artifacts are noisy or only weakly grounded
- add mutation-guided regression-suite diagnosis where benchmark or repository tests are suspiciously weak
- use distilled minimal-sufficient contexts to train or calibrate compression and pruning modules
- expand semantic sidecars beyond short task-oriented file descriptions only if the current routing layer still misses enough relevant code
- broaden evaluation slices only when they change implementation decisions or are needed for target repositories and benchmarks
