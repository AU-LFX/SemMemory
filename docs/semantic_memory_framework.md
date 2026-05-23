# EB Habitat Agent Semantic Memory Framework

> Target repo: `embodiedbench/evaluator/meta_flat_habitat_agent.py`
>
> Mission: extend the flat-feedback Habitat agent with a unified, auditable semantic memory that raises semantic understanding for objects, space, time, and rules/constraints without blowing up prompt size or WM capacity.

---

## 0. Engineering goals & constraints

| Goal | Practical meaning |
| --- | --- |
| Closed-loop semantic state | Every perception → memory → retrieval → planning → execution → feedback cycle produces an executable, verifiable semantic subgraph that can be updated next step and condensed into transferable knowledge after the episode. |
| Cross-episode transfer | Stable knowledge migrates into long-term structures (KG + schema registry) with provenance so it can seed the next episode. |
| Minimal-evidence policy | Any semantic conclusion must link to at least one `Clip` (evidence reference) for audits and rollback. |
| Structured read/write | No stuff everything into prompts. WM stays a capacity-bounded graph with operator-mediated updates. |

| Constraint | Enforcement strategy |
| --- | --- |
| WM capacity bounded / real-time | WM graph limited by utility-based TTL + simplify operator; operator engine runs in < 10 ms / step aside from VLM calls. |
| LTM auditable & versioned | KG nodes/edges and schema revisions carry git-style version IDs, timestamps, and Clip references; consolidation pipeline writes append-only logs. |
| No raw-history prompts | Only planner queries receive structured WM.Read outputs + pointer to missing fields. |

---

## 1. System architecture (online + offline)

```
┌──────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ Perception & │→→│ WM Store +         │→→│ WM.Read          │
│ Abstraction  │   │ Operator Engine    │   │ (Planner query) │
└──────┬───────┘   └─────────┬──────────┘   └────────┬────────┘
       │                     │                        │
       │ feedback delta      │ LTM search trigger     │ subgraph + gaps
       ▼                     ▼                        ▼
┌──────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ Feedback &   │←←│ LTM Retrieval/Write │←→│ Planner/Executor │
│ Validation   │   │ (KG + Schema)      │   │                 │
└──────────────┘   └────────┬──────────┘   └─────────────────┘
                            │ offline
                            ▼
                  ┌────────────────────────────┐
                  │ Consolidation & KG upkeep │
                  └────────────────────────────┘
```

### Module responsibilities

1. **Perception & Abstraction**: generate candidate semantic units (U) for objects/places/events/rules with initial evidence clips. Normalise geometry + semantics.
2. **WM Store + Operators**: maintain bounded semantic graph; operators implement write/read/correct/abstract/simplify/integrate.
3. **LTM Retrieval (KG/Schema)**: answer WM gap queries with structured concepts, rules, schema templates, and validation perspectives.
4. **Planner**: VLM planner receives WM.Read subgraph + gap list + constraints and outputs grounded action plan.
5. **Executor**: runs Habitat action IDs, emits structured success/failure codes and sensor deltas into WM.
6. **Feedback & Validation**: predicts expectations, compares with env feedback, attributes errors, triggers WM updates.
7. **Consolidation Pipeline** (offline/episodic): distils stable patterns/diagnostics into LTM.
8. **KG/Schema Maintenance**: version, split, merge, retire long-term knowledge.

---

## 2. Core data model — semantic unit `U = Symbol + Clip`

### 2.1 Unified schema

| Field | Description |
| --- | --- |
| `id: UUID` | Stable identity across WM updates and optional links to LTM concept IDs. |
| `kind` | `object | place | link | surface | event | phase | rule | schema_hint | perspective`. |
| `symbol.type_candidates` | List of `(concept_id, label, confidence)` representing hypotheses; may map to KG nodes. |
| `symbol.state_vars` | Arbitrary key-value store for numeric/boolean states (e.g., `open_ratio`, `risk_level`, `occupancy`, `phase_progress`). |
| `symbol.relations` | Array of typed edges pointing to other `U` IDs with attributes `{relation, confidence, recency}` (e.g., `connected-to`, `inside`, `before`, `triggers`). |
| `symbol.conditions` | Machine-checkable expressions describing preconditions/trigger/termination (JSONLogic or mini-DSL). |
| `symbol.constraints` | Hard constraints or policy guards (force limit, passable == True). |
| `metadata.confidence` | Posterior belief after fusion (0-1). |
| `metadata.recency` | Last update step + decay. |
| `metadata.ttl` | Steps before auto-prune absent access. |
| `metadata.provenance` | `enum`: perception | retrieval | planner | executor | feedback | consolidation. |
| `clip_refs` | List of Clip IDs that justify current fields. |

### 2.2 Clip structure (lightweight evidence)

```
Clip = {
  "id": hash(uri + timestamp),
  "kind": obs | geom | interaction | temporal | rule,
  "uri": pointer to stored frame/point-cloud/log entry,
  "digest": small textual/metric summary,
  "timestamps": {"first_seen": t0, "last_seen": t1},
  "annot": optional structured metrics (e.g., contact_force, slip_code)
}
```

Clips stay tiny (references + digest). Raw frames/logs remain in object storage (existing `outputs/<date>/...`).

### 2.3 Working Memory graph

- Maintained as `SemanticWMGraph` = `(U nodes, relation edges, indexes)`.
- Indexes:
  - `by_kind[k] -> deque[U_id]` (fast read per semantics).
  - `vector_index` on concatenated embeddings of symbol fields for semantic similarity.
  - `state_index` for numeric ranges (e.g., surfaces with `risk_level > 0.5`).
- Capacity control: `max_nodes` (~256) + `utility_score` (recent read, risk relevance, correctness hits) feeding Simplify operator.

### 2.4 Long-Term Memory (LTM)

1. **Knowledge Graph (KG)**
   - Nodes: concept, rule, skill, place-type, connector-type, temporal phase, failure-mode, perspective.
   - Edges: `synonymy`, `inheritance`, `association`, `metaphor`, `causal`, `compatibility`.
   - Attributes: `confidence`, `recency`, `centrality`, `fuzziness`, `perspective_bias`.
2. **Schema Registry**
   - Structured objects: `{name, applicability_expr, parameters, preconditions, steps, termination, failure_modes, evidence_policy}`.
   - Stores parameter ranges + evidence requirements per schema variant.

---

## 3. Working Memory operator engine

| Operator | Inputs | Outputs | Highlights |
| --- | --- | --- | --- |
| **Write** | Candidate `U` list from perception/feedback/retrieval; WM graph | Updated WM graph + write log | `match()` uses multi-signal similarity (embedding, geometry, track ID, relations). `allocate()` creates nodes if no match. `update_fields()` fuses per-field confidence. `attach_clip()` keeps only clips that justify delta fields. |
| **Read** | Query struct `{goal, focus_kinds, risk_tags, missing_fields}` from planner | `(subgraph, gap_list)` | Multi-stage selection: (1) filter by `kind/relations` (e.g., tasks needing navigation request `place/link`). (2) rank by `score = confidence * centrality * recency`. (3) return subgraph with minimal context (neighbors within 2 hops). Gap list enumerates unresolved fields/triggers LTM search. |
| **Correctness Check** | `plan_step`, executor feedback (action logs, failure codes), predicted expectations | Error annotations, field updates | `predict_expectations()` compiles checkable assertions (e.g., `link.passable == True`). `compare()` sets `error = observed - expected`. `attribute()` maps error to `(U_id, field)`; triggers `apply_fix()` (confidence decay, new clip, TTL refresh). |
| **Abstract** | WM deltas or repeated clips | Compressed invariants | Multi-frame smoothing to stabilize type candidates, derive constraints (e.g., `surface.grip_coeff < 0.3`). Updates `conditions/constraints` + prunes noisy clips. |
| **Simplify** | Current WM graph | Pruned WM graph | Maintains `utility_score`; removes low-utility clips first, merges high-similarity nodes (`object` duplicates, overlapping `place`). Converts expired nodes into summary notes stored in WM log for consolidation. |
| **Integrate** | WM graph + new/updated nodes | Consistent WM graph | Enforces cross-view identity (same object seen twice), ensures topology coherence (links referencing valid places), orders events/phase transitions, resolves conflicting fields by storing conditional values or multi-hypothesis with `fuzziness`. |

Implementation: operators share `OperatorContext` (access to indexes, scoring functions, TTL policy, clip store). Each invocation emits structured logs for audit + offline consolidation.

---

## 4. LTM services & operators

1. **Search**
   - Trigger: WM gap list with `goal`, `subgraph summary`, `uncertain fields`, `risk tags`.
   - Output: `{concept_edges, schema_candidates, constraints, perspectives}` with weights `(confidence, recency, centrality)`.
   - Implementation: vector + graph search; returns both symbolic nodes and actionable hints (e.g., inspect underside of shelf).

2. **Consolidate**
   - Trigger: end-of-episode or periodic stability check.
   - Sources: WM write logs + correctness confirmations.
   - Output: strengthened KG edges, refined schema parameter ranges, updated `graduality/fuzziness` metrics.
   - Policy: only promote fields that were validated multiple times or linked to consistent clips.

3. **Decompose**
   - Trigger: persistent fuzziness or systematic failures where one concept behaves differently across contexts.
   - Output: new sub-concepts or rule variants with scoped applicability; schema split into branches with separate failure modes.

4. **Add/Remove**
   - Add: create concept/rule/schema with supporting clips + provenance (perception, planner reasoning, or consolidation). Version recorded.
   - Remove: decay priority for unused/contradicted knowledge; once below threshold, mark as deprecated but keep history for rollback.

---

## 5. Step-by-step pipeline integration

### Online loop (per environment step)

1. **Perception → candidate U**
   - Extend `PerceptionModule` to emit structured candidates for object/place/link/event/rule triggers with initial clips referencing saved RGB/depth patches and interaction metrics.
2. **WM.Write + WM.Integrate**
   - Replace current `WorkingMemory` list appends with `SemanticWMGraph`. Write operator attaches new nodes or updates existing ones; integrate ensures consistent topology.
3. **WM.Abstract**
   - Run after integration to compute task-ready fields: `conditions`, `constraints`, `state_vars`. Output flagged unknowns to `gap_list`.
4. **LTM.Search (conditional)**
   - If `gap_list` contains high-risk unknowns (e.g., `link.passable` missing), send structured query to LTM; merge returned schema hints back via Write (`provenance=retrieve`).
5. **WM.Read → Planner**
   - Planner receives subgraph + risk list. Flattened action reasoning now references node IDs and constraints instead of prose.
6. **Executor**
   - Execute action IDs; log `ExecutionStepTrace` + structured failure codes.
7. **Feedback → WM.Correctness**
   - Compare predicted vs observed, attribute errors, update nodes (decay confidence, add conditional values) and possibly trigger replan/reperceive decisions.

### Offline / episode end

1. **Consolidation pipeline**
   - Scan WM logs for fields validated multiple times, stable schemas, failure patterns.
2. **LTM.Consolidate**
   - Strengthen KG edges, shrink schema parameter ranges, annotate risk heuristics.
3. **LTM.Decompose/Add/Remove**
   - Split fuzzy concepts, add new schemas, retire contradicted entries.
4. **Audit outputs**
   - Produce human-readable report linking promoted knowledge to Clip IDs for review.

---

## 6. Semantic specialisation strategies

| Semantic type | WM representation | Operator hooks | Planner usage |
| --- | --- | --- | --- |
| **Object semantics** | `kind=object`, `state_vars` (pose, affordance, held_by), `relations` (on/inside/attached). | Write matches via instance masks + track IDs; Abstract fuses multi-view attributes; Simplify merges duplicates; Integrate ensures same object across steps. | Planner gets object affordance (e.g., `graspable`, `contains`). Failure codes (slip/no_effect) feed Correctness to adjust `risk_level`. |
| **Spatial semantics** | `kind=place/link/surface`, `state_vars` include `passable`, `clearance`, `surface_normal`. | Write from perception + geometry; Integrate maintains graph connectivity; Abstract creates `constraints` (forbidden zones). | Planner treats constraints as hard limits in search cost (distance + risk penalty). LTM perspectives may recommend extra sensing ("check doorway height"). |
| **Temporal semantics** | `kind=event/phase`, `relations` with `before/after/during`, `state_vars.phase_progress`, `conditions` for transitions. | Correctness updates when phase triggers succeed/fail; Abstract smooths progress; Integrate enforces ordering and TTL. | Planner ensures actions respect phase boundaries (e.g., `phase=pickup_completed` before `place`). Temporal nodes also store TTL policies for Simplify. |
| **Rules & constraints semantics** | `kind=rule/schema_hint`, `constraints` storing JSONLogic expressions, `evidence_policy` linking to required clips. | LTM.Search injects rule nodes; Correctness re-validates, and Simplify drops inactive ones after TTL. | Planner ingests them as hard constraints (force limits, `impact_budget`) or conditional branches in schema steps. |

---

## 7. Evidence handling & auditing

- **Clip storage**: maintain `clip_index` per WM node referencing artifact URIs from `outputs/<date>/<time>/...`.
- **Minimal evidence policy**: any change to `symbol.state_vars`, `conditions`, or `constraints` must specify clip IDs explaining why (`attach_clip`).
- **Traceability**: Operator logs record `(U_id, field, clip_ids, provenance, timestamp)`; consolidation pipeline uses these logs to justify KG/schema updates.
- **Rollback**: WM snapshots stored every N steps; LTM updates go through versioned commit objects with rollback pointers.

---

## 8. Implementation roadmap

1. **Data structures**
   - Introduce `SemanticUnit`, `Clip`, `SemanticWMGraph`, and `OperatorContext` classes alongside existing `WorkingMemory`. Provide adapters to read/write legacy structures during migration.
2. **Perception upgrades**
   - Extend `PerceptionModule.perceive` to output candidate U records (object/place/link/event/rule) plus clip metadata (image URI, geometry digest, timestamp).
3. **Operator engine**
   - Build `MemoryOperatorEngine` with six WM operators and pluggable policies (matching thresholds, TTL, scoring). Integrate into `meta_flat_habitat_agent` loop.
4. **Planner contract change**
   - Update VLM planner prompts to reference WM.Read JSON (subgraph + gap list + constraints). Ensure failure codes map back to node IDs.
5. **Executor & feedback instrumentation**
   - Standardize `ExecutionStepTrace.env_feedback` to include `failure_code`, contact metrics, slip flags.
6. **LTM services**
   - Stand up graph store (e.g., networkx/Neo4j or structured JSON) with APIs `search`, `consolidate`, `decompose`, `mutate`. Use provenance logs for auditing.
7. **Consolidation tasks**
   - Batch process WM logs at episode end, generate KG/schema patches, push through review hooks.
8. **Monitoring & tooling**
   - Provide dashboards (textual/JSON) summarizing WM graph state, operator events, and pending LTM changes for debugging.

---

## 9. Next steps

1. Prototype `SemanticWMGraph` alongside current `WorkingMemory` to maintain backward compatibility during rollout.
2. Implement operator stubs with logging first, then add scoring/fusion logic.
3. Define JSON schemas for WM.Read output and LTM.Search reply to align with planner API changes.
4. Create evaluation harness that tracks semantic coverage metrics (object/spatial/temporal/rule) per episode to verify gains.
