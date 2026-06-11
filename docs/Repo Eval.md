## How to review a repository before scoring it

Use these evidence layers first.

| Evidence layer            | What to inspect                                                                                          | Why it matters                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Docs / claims**         | README, architecture docs, RFCs, examples, API docs, guarantee statements                                | Shows what the repo claims                         |
| **Model / planning**      | core types, logical plan, IR, DAG, AST, operators, planner, optimizer, schemas, type system              | Shows what the repo thinks it is                   |
| **Runtime / state**       | engine, scheduler, checkpoints, state backends, partitioning, shuffle, watermarking, replay logic        | Shows how it actually behaves                      |
| **Recovery / boundaries** | offsets, commits, idempotency, dedup, sink protocol, crash restore, DLQ/quarantine                       | Shows whether guarantees are credible              |
| **Tests / ops**           | restart tests, replay tests, late-data tests, schema drift tests, overload tests, metrics, logs, tracing | Shows whether guarantees are verified and operable |

---

## Evaluation matrix

| Dimension                                           | Master evaluation question                                                             | Architecture Path                        | What it means                                                                                        |
| --------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **1. Model clarity**                                | Is the unit of data and computation explicit?                                          | Ingestion, Transformation, Processing    | Rows, files, events, CDC, relations, mappings, windows, bounded jobs, continuous queries             |
| **2. Semantic clarity**                             | Does the framework define what results mean?                                           | Ingestion, Transformation, Processing    | Meaning under duplicates, nulls, ordering, windows, mappings, refinement                             |
| **3. Correctness**                                  | Does the output match intended meaning?                                                | Ingestion, Transformation, Processing    | Load correctness, semantic equivalence, correct stateful results                                     |
| **4. Capture model correctness**                    | How is data acquired, and are source semantics preserved?                              | Ingestion                                | Polling, CDC, snapshot+tail, gap handling, source transaction boundaries, snapshot-to-stream handoff |
| **5. Completeness / consistency / freshness**       | Is the expected data present, valid, and timely?                                       | Ingestion                                | Especially important at ingestion boundary                                                           |
| **6. Heterogeneity handling**                       | Can it handle diverse sources and representations?                                     | Ingestion, Transformation                | Source autonomy, schema mismatch, mediation                                                          |
| **7. Mapping expressiveness**                       | Can source-target or schema-schema relationships be expressed well?                    | Ingestion, Transformation                | Strongest in integration/transformation systems                                                      |
| **8. Declarativity / composability**                | Can logic be expressed and combined systematically?                                    | Transformation, Processing               | Algebraic operators, composable transforms, non-ad-hoc structure                                     |
| **9. Logical-to-physical mapping**                  | Is there a clear bridge from logic to execution?                                       | Transformation, Processing               | Planner, optimizer, operator selection, runtime strategy                                             |
| **10. Result semantics**                            | Are outputs final, incremental, revisable, retractable, or eventually complete?        | Transformation, Processing               | Especially important for transformations and streams                                                 |
| **11. Time semantics**                              | Are event time, processing time, windows, and watermarks explicit?                     | Ingestion, Processing                    | Core for stream ingestion and processing                                                             |
| **12. Ordering / lateness semantics**               | Is behavior under out-of-order and late data defined?                                  | Ingestion, Processing                    | Buffering, window closure, disorder tolerance                                                        |
| **13. Determinism / convergence**                   | Do reruns, replays, and backfills converge to the same logical result?                 | Ingestion, Processing                    | Key for correctness under disorder and recovery                                                      |
| **14. State model**                                 | Is state first-class, durable, and inspectable?                                        | Processing                               | Keyed state, operator state, checkpoints, TTL, restore                                               |
| **15. Incrementality / maintenance**                | Can results be maintained incrementally?                                               | Transformation, Processing               | IVM, deltas, changelogs, continuous stateful updates                                                 |
| **16. Delivery semantics**                          | What delivery guarantees apply, and where?                                             | Ingestion, Processing                    | At-most-once, at-least-once, effectively-once, exactly-once scope                                    |
| **17. Fault tolerance / recovery semantics**        | What survives crashes, retries, partitions, and restarts?                              | Ingestion, Processing                    | Restore correctness, durable state, safe checkpoint/ack ordering                                     |
| **18. Replay / backfill support**                   | Can historical and live processing use the same logic safely?                          | Ingestion, Processing                    | Replay mode, backfill mode, historical/live equivalence                                              |
| **19. DLQ / poison-message handling**               | How are bad, malformed, or unprocessable records isolated?                             | Ingestion                                | DLQ, quarantine, operator-level failure isolation                                                    |
| **20. Disorder / skew / overload handling**         | How does the runtime behave when data is messy or load is uneven?                      | Processing                               | Hot keys, backlog growth, bursty input, slow sinks                                                   |
| **21. Flow control / backpressure**                 | Can the engine remain stable under sustained load?                                     | Processing                               | Backpressure, bounded buffering, graceful overload behavior                                          |
| **22. Materialization / reuse**                     | Can intermediates, views, or derived state be reused intentionally?                    | Transformation, Processing               | Persisted intermediates, reusable subplans, checkpointed derived artifacts                           |
| **23. Optimization / planner quality**              | Can equivalent logic be transformed into better physical execution?                    | Transformation, Processing               | Rewrite rules, pushdown, pruning, fusion, cost reasoning                                             |
| **24. Boundary discipline**                         | Are ingestion, transformation, and execution cleanly separated?                        | Ingestion                                | Connectors should not own business logic; source quirks should not leak into core                    |
| **25. Extensibility transparency**                  | Can new connectors/operators/UDFs be added without breaking semantics or optimization? | Transformation, Processing               | Extensions stay visible to planner/runtime and remain inspectable                                    |
| **26. Schema evolution**                            | What happens when fields are added, renamed, deleted, or drift?                        | Ingestion, Transformation                | Compatibility, migrations, mapping adaptation, safe failure                                          |
| **27. Provenance / observability / inspectability** | Can users see why outputs exist and whether the system is healthy right now?           | Ingestion, Transformation, Processing    | Lineage, explain plans, lag, checkpoint health, state growth, retries, freshness, completeness       |
| **28. Data quality / distortion**                   | Do cleaning and normalization preserve downstream meaning?                             | Transformation                           | Validity vs statistical distortion tradeoff                                                          |
| **29. Testing and proof of semantics**              | Are guarantees proven in code, not just described?                                     | Ingestion, Transformation, Processing    | Restart tests, replay tests, late-data tests, equivalence tests, fuzz/property tests                 |
| **30. Benchmark / evidence quality**                | Are performance and correctness claims measured rigorously?                            | Ingestion, Transformation, Processing    | Real workloads, precise metrics, sustainable load, failure injection                                 |
| **31. Performance / cost model clarity**            | Are performance tradeoffs explicit and architecturally honest?                         | Processing                               | Throughput, latency, memory, spill, checkpoint frequency, state backend tradeoffs                    |
| **32. Architectural honesty**                       | Does the implementation really match the repo’s claims?                                | Ingestion, Transformation, Processing    | No fake “exactly-once,” no fake “unified,” no fake “streaming”                                       |

---

## Fast interpretation guide

### If a repo scores high on these, it is likely robust

* model clarity
* semantic clarity
* correctness
* capture model correctness
* determinism / convergence
* state model
* fault tolerance / recovery
* schema evolution
* provenance / observability / inspectability
* testing and proof of semantics
* architectural honesty

### If a repo scores high on speed but low on these, it is probably one of:

* connector toolkit
* orchestration layer
* convenience pipeline library
* stream wrapper without strong semantics
* batch engine with streaming bolted on
* marketing-level “unified” system

---

## Strong red flags

| Red flag                                         | Why it matters                           |
| ------------------------------------------------ | ---------------------------------------- |
| “Exactly-once” without scoped implementation     | Usually marketing, not a real guarantee  |
| Checkpoints advance before sink durability       | Breaks recovery correctness              |
| Replay and live paths use different logic        | Convergence is untrustworthy             |
| “Streaming” is just polling + append             | Weak execution semantics                 |
| No event-time model or late-data policy          | Results become semantically vague        |
| State lives only in local memory                 | Recovery claims are weak                 |
| Connectors contain business transformation logic | Boundary discipline is broken            |
| UDFs make plans opaque                           | Optimization and inspectability collapse |
| No restart/replay/schema-drift tests             | Guarantees are unproven                  |
| No lag/checkpoint/state metrics                  | Operators cannot verify correctness      |
| Repo claims unification, code shows dual engines | Architectural dishonesty                 |

---

## Final condensed decision rule

A strong repository should be able to answer **yes** to most of these:

* Is the native data unit explicit?
* Are semantics precise?
* Is source capture correct and explicit?
* Is there a real logical transformation model?
* Are time, ordering, and result semantics defined?
* Is state first-class and recoverable?
* Do replay and backfill converge?
* Are delivery guarantees scoped and justified?
* Can it handle disorder, skew, and overload deliberately?
* Are boundaries clean between capture, transform, and execute?
* Are schema changes handled intentionally?
* Can operators inspect lineage, lag, and progress?
* Do tests prove semantics under failure?
* Does the implementation match the claims?