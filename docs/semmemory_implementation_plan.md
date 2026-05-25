# SemMemory for MemVerse + ALFRED: Code Implementation Plan

This document is the implementation plan for adding SemMemory to the current
runnable path:

```text
embodiedbench/evaluator/memverse_alfred_agent.py
MemVerse/
```

Legacy semantic-memory files under `embodiedbench/evaluator/` are reference
material only. Do not extend them for this implementation.

## 1. Core Decision

Use MemVerse's existing LightRAG insertion logic, but replace MemVerse's generic
memory builder for the ALFRED SemMemory path.

Current generic MemVerse write path:

```text
POST /insert
  -> handle_insert()
  -> update_long_term_memory()
  -> build_memory.process_memory()
       LLM: core_memory_agent.txt
       LLM: episodic_memory_agent.txt
       LLM: semantic_memory_agent.txt
  -> insert_chunks_from_json()
  -> LightRAG.ainsert(text)
       LLM: LightRAG entity/relation extraction
```

New ALFRED SemMemory write path:

```text
POST /alfred/end_episode
  -> event_parser LLM
       full episode -> raw_case + event-centric semantic experience graph
  -> consolidator LLM
       graph + existing schemas -> schema ops from D/A/L/X
  -> SemMemory export code
       raw_case -> episodic output_text chunks
       graph/schema/hypothesis/correction -> semantic output_text chunks
  -> insert_chunks_from_json()
  -> LightRAG.ainsert(text)
       LightRAG extracts final KG entities/relations
```

So SemMemory replaces:

```text
build_memory.process_memory()
```

It does not replace LightRAG's KG construction in the first implementation.

This is closer to the existing MemVerse design and keeps LightRAG as the final
KG/RAG builder. If LightRAG's default entity/relation extraction weakens the
event-centric structure or the four semantic dimensions, then modify LightRAG's
entity extraction prompt instead of bypassing LightRAG with `ainsert_custom_kg()`.

## 2. What Must Not Happen

ALFRED SemMemory should not call the old generic memory builder:

- no generic `/insert` during or after the episode;
- no `build_memory.process_memory()` from `/alfred/end_episode`;
- no `semantic_memory_agent.txt` or `episodic_memory_agent.txt` for ALFRED
  SemMemory;
- no `reflect_and_summarize()` before long-term memory writing.

But ALFRED SemMemory may reuse:

- `insert_chunks_from_json()`;
- `LightRAG.ainsert(text)`;
- LightRAG's entity/relation extraction prompt and graph storage.

This means the total write path still has LightRAG LLM calls during insertion.
That is intentional. The duplicate part we remove is the old MemVerse
`build_memory.process_memory()` LLM stage.

## 3. Runtime Flow

```text
MemVerseAlfredAgent.run_single_episode()
  -> start_episode()
  -> loop:
       perceive / observe
       retrieve_long_term()
       planner.act()
       env.step()
       update local working_memory
       append local step_records
  -> end_episode()
```

During the episode:

- keep `working_memory` local to the agent;
- keep `step_records` local to the agent;
- do not call `ingest_step()`;
- do not call generic `/insert`;
- do not write long-term memory.

At episode end:

- send the complete episode payload to `/alfred/end_episode`;
- save raw episode JSON artifacts;
- run SemMemory parser and consolidator;
- write SemMemory-generated JSONL chunks;
- call `insert_chunks_from_json()` for `mem_epi` and `mem_sem`.

## 4. Simplified File Layout

First implementation:

```text
MemVerse/MemoryKB/SemMemory/
  __init__.py
  contracts.py
  store.py
  event_parser.py
  consolidator.py
  exporter.py
  retriever.py
  service.py
  prompts/
    event_extraction.txt
    schema_consolidation.txt
```

Do not add these files in the first implementation:

```text
episode_buffer.py
rules.py
kg_mapper.py
lightrag_bridge.py
formatter.py
```

Reason:

- `episode_buffer.py`: the ALFRED agent already keeps local `step_records`.
- `rules.py`: rule semantics are extracted by the LLM parser and used by the LLM
  consolidator; first version does not need a deterministic rule engine.
- `kg_mapper.py`: not needed because we are not using `ainsert_custom_kg()` in
  the first implementation.
- `lightrag_bridge.py`: export logic is small enough to live in `exporter.py`
  and `service.py`.
- `formatter.py`: retrieval formatting can live in `retriever.py` first.

## 5. Module Responsibilities

### `contracts.py`

- Validate endpoint payloads with lightweight checks.
- Define required semantic dimension keys:

```python
SEMANTIC_DIMENSION_KEYS = [
    "entity_semantics",
    "spatial_semantics",
    "temporal_semantics",
    "rule_semantics",
]
```

- Provide helpers:

```python
validate_episode_payload(payload: dict) -> None
validate_experience_graph(graph: dict) -> list[str]
validate_consolidation_report(report: dict) -> list[str]
```

Validation should be permissive in the first version: collect errors in reports
instead of crashing the episode whenever possible.

### `store.py`

Owns JSON artifacts under:

```text
MemVerse/MemoryKB/Long_Term_Memory/semmemory/
  active_episodes/
  raw_episodes/
  cases/
  graphs/
  schemas/
  chunks/
  prompts/
  llm_outputs/
  reports/
  indexes/
```

Required methods:

```python
class SemanticStore:
    def start_episode(self, episode_id: str, payload: dict) -> None: ...
    def save_raw_episode(self, episode_id: str, payload: dict) -> None: ...
    def save_case(self, case: dict) -> None: ...
    def save_experience_graph(self, graph: dict) -> None: ...
    def load_schemas(self) -> list[dict]: ...
    def upsert_schemas(self, schemas: list[dict]) -> None: ...
    def append_schema_update(self, update: dict) -> None: ...
    def save_chunk_jsonl(self, name: str, rows: list[dict]) -> str: ...
    def save_prompt_and_output(self, episode_id: str, stage: str, prompt: str, output: str) -> None: ...
    def save_report(self, episode_id: str, report: dict) -> None: ...
```

The JSON artifact store is not a third conceptual memory system. It is for
debugging, provenance, replay, and rebuilding `mem_epi` / `mem_sem`.

### `event_parser.py`

LLM stage 1.

Input:

- instruction;
- full `step_records`;
- action history;
- feedback history;
- final working memory;
- final result;
- available actions if present.

Output:

```json
{
  "episode_id": "alfred_base_12",
  "raw_case": {},
  "experience_graph": {},
  "parser_debug": {}
}
```

The experience graph is event-centric because event nodes anchor local subgraphs.
However, the four semantic dimensions belong to every KG node, not just event
nodes.

Each node must have:

```json
{
  "id": "entity:mug",
  "node_type": "entity",
  "label": "mug",
  "semantic_dimensions": {
    "entity_semantics": {},
    "spatial_semantics": {},
    "temporal_semantics": {},
    "rule_semantics": {}
  },
  "provenance": {}
}
```

The parser prompt should ask the LLM to identify:

- event nodes for key actions, state changes, failures, and phase transitions;
- entity/object nodes;
- receptacle/location nodes;
- phase nodes;
- rule/constraint nodes;
- event-centered edges such as `acts_on`, `located_at`, `before`, `requires`,
  `causes`, `violates`, and `supports`;
- a raw episodic case summary for `mem_epi`.

### `consolidator.py`

LLM stage 2.

Input:

- parsed `experience_graph`;
- `raw_case`;
- existing schema records from JSON store;
- optionally a small context-only retrieval from `mem_sem`.

Output:

```json
{
  "episode_id": "alfred_base_12",
  "ops": [
    {
      "op": "update",
      "target_type": "schema",
      "target_id": "schema:pickup_visible_reachable_object",
      "source_operator": "lifting",
      "support_event_ids": ["event:alfred_base_12:3"],
      "payload": {}
    }
  ],
  "schemas": [],
  "hypotheses": [],
  "corrections": [],
  "invalidated": [],
  "debug": {}
}
```

The four operators are implemented conceptually inside this one consolidator
prompt:

- Decoupling: split multi-role or over-entangled nodes into role-specific schema
  candidates.
- Abstraction: remove episode-specific details and keep reusable task semantics.
- Lifting: merge repeated compatible structures into stable schema memory.
- Extension: apply existing schemas to new entities/tasks as unverified
  hypotheses.

### `exporter.py`

No LLM.

Converts SemMemory JSON records into JSONL rows compatible with existing
`insert_chunks_from_json()`:

```json
{
  "id": "semmemory:sem:graph:alfred_base_12",
  "timestamp": "...",
  "input_text": "episode_id=alfred_base_12",
  "output_text": "..."
}
```

Required methods:

```python
def case_to_epi_chunks(case: dict) -> list[dict]: ...
def graph_to_sem_chunks(graph: dict) -> list[dict]: ...
def schemas_to_sem_chunks(records: list[dict]) -> list[dict]: ...
```

Export targets:

```text
raw_case -> semmemory/chunks/<episode_id>_episodic.jsonl -> mem_epi
experience_graph -> semmemory/chunks/<episode_id>_semantic_graph.jsonl -> mem_sem
schemas/hypotheses/corrections -> semmemory/chunks/<episode_id>_semantic_schema.jsonl -> mem_sem
```

`output_text` must preserve SemMemory structure explicitly, because LightRAG will
extract the final KG from this text. Each JSONL row should carry a stable `id`,
and `insert_chunks_from_json()` should pass that id plus the JSONL file path into
`LightRAG.ainsert(text, ids=..., file_paths=...)` so later KG nodes and edges can
be traced back to SemMemory chunks. Use stable labels such as:

```text
[SemMemory]
layer: experience_graph
episode_id: alfred_base_12
node_id: event:alfred_base_12:3
node_type: event
label: pick up mug
entity_semantics: participants entity:mug; holding_before none; holding_after mug
spatial_semantics: mug on counter; reachable assumed
temporal_semantics: step 3; phase acquire_object; previous event event:alfred_base_12:2
rule_semantics: preconditions object_visible, hand_empty; effects agent_holding_object
edges: event:alfred_base_12:3 acts_on entity:mug
relationship: event:alfred_base_12:3 acts_on entity:mug; relationship_keywords: acts_on, event_centric_edge
relationship: schema:pickup_visible_reachable_object has_precondition object_visible; relationship_keywords: has_precondition, requires, rule_semantics
```

This makes the SemMemory structure visible to LightRAG's entity/relation
extractor.

### `retriever.py`

No memory writes.

Builds a semantic-state query from:

- instruction;
- current phase;
- latest perception;
- visible objects;
- holding state;
- active entities and relationships;
- recent feedback/failure.

Queries both memories:

```text
mem_sem -> semantic graph, schemas, hypotheses, corrections, failure schemas
mem_epi -> prior cases and concrete trajectories
```

Use LightRAG in context-only mode when possible:

```python
QueryParam(mode="hybrid", only_need_context=True)
```

This avoids MemVerse's generic final-answer LLM. The retriever then applies
lightweight deterministic filtering:

- reject invalidated records;
- prefer same phase;
- prefer matching action type;
- prefer matching entity role;
- reject clearly contradicted preconditions;
- label unverified hypotheses clearly;
- warn against using final placement from an old episode as a future initial
  location.

### `service.py`

Owns the lifecycle.

```python
class SemMemoryService:
    def start_episode(self, payload: dict) -> dict: ...
    async def retrieve_long_term(self, payload: dict) -> dict: ...
    async def end_episode(self, payload: dict) -> dict: ...
```

`end_episode()` sequence:

```text
1. validate payload
2. save raw episode JSON
3. event_parser.parse_episode()
4. save raw_case and experience_graph
5. load existing schemas
6. consolidator.consolidate()
7. upsert schema records and update JSON indexes
8. exporter.case_to_epi_chunks() -> save episodic chunk JSONL
9. exporter.graph_to_sem_chunks() -> save semantic graph chunk JSONL
10. exporter.schemas_to_sem_chunks() -> save semantic schema chunk JSONL
11. insert_chunks_from_json(mem_epi, episodic_chunk_jsonl)
12. insert_chunks_from_json(mem_sem, semantic_graph_chunk_jsonl)
13. insert_chunks_from_json(mem_sem, semantic_schema_chunk_jsonl)
14. save report
```

Do not call:

```python
build_memory.process_memory(...)
handle_insert(...)
```

from the SemMemory path.

## 6. LightRAG Prompt Adjustment

Keep LightRAG as the final KG builder, but make its entity/relation extraction
prompt SemMemory-aware. The existing prompt already extracts:

```text
concept
affordances
constraints
temporal
spatial
relationships
```

These fields can carry SemMemory dimensions:

```text
entity_semantics -> concept + affordances
spatial_semantics -> spatial
temporal_semantics -> temporal
rule_semantics -> constraints
event-centric edges -> relationships
```

The implementation modifies:

```text
MemVerse/MemoryKB/Long_Term_Memory/Graph_Construction/lightrag/prompt.py
PROMPTS["entity_extraction"]
PROMPTS["entity_continue_extraction"]
```

The prompt changes instruct LightRAG to:

- preserve SemMemory node ids when present, such as `event:...`, `entity:...`,
  `schema:...`, `rule:...`, `phase:...`, `hypothesis:...`, `correction:...`,
  and `case:...`;
- treat `event`, `schema`, `hypothesis`, `correction`, `rule`, `phase`,
  `episodic_case`, `semantic_experience_graph`, `action`, `constraint`,
  `receptacle`, and `object` as valid entity types;
- map explicit `entity_semantics`, `spatial_semantics`, `temporal_semantics`,
  and `rule_semantics` lines into the existing `concept`, `spatial`, `temporal`,
  and `constraints` fields;
- preserve relationships named in the text, such as `acts_on`, `requires`,
  `causes`, `violates`, `supports`, `before`, `located_at`,
  `has_precondition`, `has_effect`, `fails_when`, `supported_by`, and
  `applies_to`;
- avoid replacing specific SemMemory node ids with broad abstract labels.

Do not modify LightRAG storage schemas in the first implementation.

## 7. API Contracts

### `POST /alfred/start_episode`

Request:

```json
{
  "episode_id": "alfred_base_12",
  "instruction": "put a clean mug in the microwave",
  "eval_set": "base",
  "available_actions": ["go to sink", "pick up mug"],
  "metadata": {
    "scene_id": "FloorPlan1",
    "task_type": "pick_clean_then_place",
    "trial_id": "..."
  }
}
```

Side effects:

- writes `active_episodes/<episode_id>.json`;
- does not write long-term memory.

### `POST /alfred/retrieve_long_term`

Request:

```json
{
  "episode_id": "alfred_base_12",
  "instruction": "put a clean mug in the microwave",
  "step": 3,
  "phase": "acquire_object",
  "current_state_summary": {
    "latest_perception": "mug and sink are visible",
    "visible_objects": ["mug", "sink", "counter"],
    "holding": null,
    "current_status": {},
    "active_entities": {},
    "active_relationships": []
  },
  "recent_feedback": [
    {"action": "go to counter", "success": true}
  ],
  "top_k": 8
}
```

Side effects:

- no long-term memory write;
- optional access counters only.

### `POST /alfred/end_episode`

Request:

```json
{
  "episode_id": "alfred_base_12",
  "instruction": "put a clean mug in the microwave",
  "result": {
    "task_success": 1,
    "task_progress": 1.0,
    "num_steps": 8,
    "num_invalid_actions": 1
  },
  "history": [
    {"step": 1, "action_desc": "go to counter", "success": true}
  ],
  "step_records": [
    {
      "step": 1,
      "image_path": "path/to/step_1.png",
      "perception": {
        "scene_description": "The agent sees a counter.",
        "visible_objects": ["counter"]
      },
      "action": {"action_id": 12, "action_desc": "go to counter"},
      "feedback": {"last_action_success": 1, "env_feedback": "success"}
    }
  ],
  "final_working_memory": {
    "latest_perception": "...",
    "current_status": {},
    "active_entities": {},
    "active_relationships": [],
    "action_history": [],
    "environment_feedback": []
  }
}
```

Response:

```json
{
  "status": "ok",
  "episode_id": "alfred_base_12",
  "num_graph_nodes": 12,
  "num_graph_edges": 18,
  "schema_updates": [],
  "exported_to_lightrag": {
    "mem_epi_chunks": 1,
    "mem_sem_chunks": 2
  },
  "debug": {
    "semmemory_llm_calls": 2,
    "used_generic_insert": false,
    "used_build_memory": false,
    "used_insert_chunks_from_json": true,
    "used_lightrag_ainsert": true
  }
}
```

## 8. Data Models

### Raw Case

Stored as JSON and exported to `mem_epi`.

```json
{
  "id": "case:alfred_base_12",
  "type": "episodic_case",
  "episode_id": "alfred_base_12",
  "instruction": "put a clean mug in the microwave",
  "task_type": "pick_clean_then_place",
  "result": {"task_success": 1, "task_progress": 1.0},
  "ordered_steps": [
    {"step": 1, "action": "go to counter", "success": true}
  ],
  "key_lessons": [
    "clean object before final placement"
  ],
  "failure_modes": [],
  "summary_for_retrieval": "Successful clean-and-place task: locate mug, clean at sink, place in microwave."
}
```

### Event-Centric Semantic Experience Graph

Stored as JSON and exported to `mem_sem`.

```json
{
  "id": "graph:alfred_base_12",
  "type": "semantic_experience_graph",
  "episode_id": "alfred_base_12",
  "nodes": [
    {
      "id": "event:alfred_base_12:3",
      "node_type": "event",
      "label": "pick up mug",
      "semantic_dimensions": {
        "entity_semantics": {
          "participants": ["entity:mug"],
          "agent_state": {"holding_before": null, "holding_after": "mug"}
        },
        "spatial_semantics": {
          "relevant_relations": ["entity:mug on receptacle:counter"],
          "reachability": "reachable_or_assumed_reachable"
        },
        "temporal_semantics": {
          "step": 3,
          "phase": "acquire_object",
          "previous_event_id": "event:alfred_base_12:2"
        },
        "rule_semantics": {
          "preconditions": ["object_visible", "hand_empty"],
          "effects": ["agent_holding_object"],
          "constraints": ["target must be visible and reachable"],
          "failure_conditions": ["object_not_visible", "hand_not_empty"]
        }
      },
      "provenance": {
        "step": 3,
        "image_path": "...",
        "confidence": 0.8
      }
    }
  ],
  "edges": [
    {
      "source": "event:alfred_base_12:3",
      "relation": "acts_on",
      "target": "entity:mug",
      "evidence": ["step:3"],
      "confidence": 0.8
    }
  ],
  "metadata": {
    "parser": "llm_v1",
    "semmemory_layer": "experience_graph"
  }
}
```

### Schema Record

Stored as JSON and exported to `mem_sem`.

```json
{
  "id": "schema:pickup_visible_reachable_object",
  "type": "schema",
  "name": "pickup visible reachable object",
  "schema_kind": "action_rule",
  "action_type": "pickup",
  "entity_roles": ["target_object"],
  "preconditions": ["object_visible", "object_reachable", "hand_empty"],
  "effects": ["agent_holding_object"],
  "failure_conditions": ["object_not_visible", "object_in_closed_receptacle"],
  "applicable_phases": ["acquire_object"],
  "positive_support": [
    {"episode_id": "alfred_base_12", "event_id": "event:alfred_base_12:3"}
  ],
  "negative_support": [],
  "support_count": 1,
  "failure_count": 0,
  "confidence": 0.7,
  "status": "active",
  "source_operators": ["abstraction", "lifting"]
}
```

## 9. Prompt Design

### `event_extraction.txt`

Purpose:

```text
full episode -> raw_case + event-centric semantic experience graph
```

Must require:

- strict JSON only;
- no markdown;
- no invented observations;
- event-centric graph;
- every KG node has all four semantic dimensions;
- failed actions become event/rule/constraint nodes when semantically relevant;
- final placement is provenance, not future initial object location.

### `schema_consolidation.txt`

Purpose:

```text
experience_graph + existing schemas -> schema ops and records
```

Must require:

- strict JSON only;
- separate `source_operator` values:
  - `decoupling`;
  - `abstraction`;
  - `lifting`;
  - `extension`;
- hypotheses are explicitly unverified;
- invalidated schemas remain traceable;
- support events are preserved.

## 10. MemVerse Integration

### `MemVerse/app.py`

Add:

```python
@app.post("/alfred/start_episode")
async def alfred_start_episode(payload: dict):
    return await handle_alfred_start_episode(payload)

@app.post("/alfred/retrieve_long_term")
async def alfred_retrieve_long_term(payload: dict):
    return await handle_alfred_retrieve_long_term(payload)

@app.post("/alfred/end_episode")
async def alfred_end_episode(payload: dict):
    return await handle_alfred_end_episode(payload)
```

Keep existing `/insert` and `/query` unchanged.

### `MemVerse/orchestrator.py`

Add global service:

```python
sem_memory = None
```

Initialize it after `mem_epi` and `mem_sem`:

```python
sem_memory = SemMemoryService(
    storage_dir=os.path.join("MemoryKB", "Long_Term_Memory", "semmemory"),
    episodic_rag=mem_epi,
    semantic_rag=mem_sem,
    llm_client=client,
    insert_chunks_from_json=insert_chunks_from_json,
)
```

Add handlers:

```python
async def handle_alfred_start_episode(payload):
    return sem_memory.start_episode(payload)

async def handle_alfred_retrieve_long_term(payload):
    return await sem_memory.retrieve_long_term(payload)

async def handle_alfred_end_episode(payload):
    return await sem_memory.end_episode(payload)
```

Do not call `update_long_term_memory()` from these handlers.

## 11. `memverse_alfred_agent.py` Changes

### Client

Add:

```python
class MemVerseClient:
    def start_episode(self, payload: dict) -> dict: ...
    def retrieve_long_term(self, payload: dict) -> dict: ...
    def end_episode(self, payload: dict) -> dict: ...
```

Keep `query()` and `insert()` only as legacy fallback.

### Episode Loop

At start:

```text
episode_id = stable id from env metadata if available
memverse.start_episode(...)
```

Before each planner call:

```text
ltm = memverse.retrieve_long_term(...)
augmented_instruction =
  user_instruction
  + ltm.long_term_context
  + local working memory context
```

After each environment step:

```text
update local working_memory
append one step_record
```

At end:

```text
memverse.end_episode({
  episode_id,
  instruction,
  result,
  history,
  step_records,
  final_working_memory
})
```

Remove:

```python
action = [35,129,79,133,155,156,129,39,133,34,127,35,159,79,133,39,35,129,38,143,133,144]
```

Do not call `reflect_and_summarize()` for long-term memory. The SemMemory
endpoint performs episode-end semantic parsing.

Do not call `self.memverse.insert(...)` at episode end in the SemMemory path.

## 12. Retrieval Contract

ALFRED SemMemory retrieval should not use generic `/query`, because generic
`/query` calls an LLM to generate `final_answer`.

Instead:

```text
/alfred/retrieve_long_term
  -> retriever builds semantic-state query
  -> mem_sem.aquery(... only_need_context=True)
  -> mem_epi.aquery(... only_need_context=True)
  -> deterministic filtering and formatting
  -> returns planner-ready long_term_context
```

Planner block:

```text
### Long-Term SemMemory
Current phase: acquire_object

Applicable schemas:
- pickup visible reachable object: requires object visible, reachable, hand empty.

Relevant prior cases:
- In a previous clean-and-place task, the agent succeeded by cleaning before final placement.

Warnings:
- Do not use final placement from prior episodes as current initial object location.

Missing checks:
- Confirm whether the mug is already clean.
```

Do not include raw JSON or long unfiltered LightRAG text in the planner prompt.

## 13. Implementation Milestones

### M0: Agent Cleanup

Files:

- `embodiedbench/evaluator/memverse_alfred_agent.py`

Tasks:

- remove hardcoded action override;
- add stable `episode_id`;
- collect `step_records`;
- add `start_episode`, `retrieve_long_term`, `end_episode` client calls;
- stop calling generic `insert()` at episode end in SemMemory mode.

Acceptance:

- planner action is no longer overwritten;
- no long-term memory writes happen during the episode;
- one complete episode payload can be sent at episode end.

### M1: SemMemory Skeleton

Files:

- `MemVerse/MemoryKB/SemMemory/__init__.py`
- `contracts.py`
- `store.py`
- `service.py`
- `MemVerse/app.py`
- `MemVerse/orchestrator.py`

Tasks:

- add three `/alfred/*` endpoints;
- persist active episode and raw episode JSON;
- return valid empty retrieval context when no memory exists.

Acceptance:

- `/alfred/start_episode` and `/alfred/end_episode` work without generic
  `/insert`;
- `build_memory.process_memory()` is not called.

### M2: LLM Event Parser

Files:

- `event_parser.py`
- `prompts/event_extraction.txt`

Tasks:

- call parser LLM on complete episode;
- output `raw_case` and `experience_graph`;
- validate all graph nodes have four semantic dimensions;
- store prompt and raw output.

Acceptance:

- failed actions become event/rule/constraint nodes when relevant;
- graph JSON is saved under `semmemory/graphs/`.

### M3: LLM Consolidator

Files:

- `consolidator.py`
- `prompts/schema_consolidation.txt`

Tasks:

- call consolidator LLM on graph + existing schemas;
- output Decoupling / Abstraction / Lifting / Extension ops;
- upsert schema, hypothesis, correction, and invalidation records.

Acceptance:

- repeated pickup successes can update one pickup schema;
- multi-role objects can create role-separated schema candidates;
- Extension creates unverified hypotheses, not confirmed facts.

### M4: SemMemory Chunk Export

Files:

- `exporter.py`
- `service.py`

Tasks:

- map raw case to `output_text` chunks for `mem_epi`;
- map experience graph to `output_text` chunks for `mem_sem`;
- map schema records to `output_text` chunks for `mem_sem`;
- call existing `insert_chunks_from_json()`.

Acceptance:

- SemMemory export does not call `build_memory.process_memory()`;
- SemMemory export does call `insert_chunks_from_json()`;
- LightRAG receives structured SemMemory text and builds KG through its normal
  insertion pipeline.

### M5: LightRAG Prompt Tuning If Needed

Files:

- `MemVerse/MemoryKB/Long_Term_Memory/Graph_Construction/lightrag/prompt.py`

Tasks:

- inspect extracted KG quality after M4;
- if event/schema/rule nodes are lost, update LightRAG entity extraction prompt;
- add SemMemory examples to the prompt if needed.

Acceptance:

- extracted KG preserves event nodes;
- extracted KG preserves schema/rule nodes;
- extracted KG preserves entity/spatial/temporal/rule dimensions through
  concept/affordances/constraints/temporal/spatial fields.

### M6: Retrieval

Files:

- `retriever.py`
- `service.py`

Tasks:

- build semantic-state query;
- retrieve from `mem_sem` and `mem_epi` with `only_need_context=True`;
- filter and format planner context.

Acceptance:

- retrieval does not call generic `/query`;
- retrieval does not call `generate_final_answer()`;
- output is compact and phase-aware.

### M7: Smoke Test

Run:

```bash
cd MemVerse
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then run one ALFRED episode.

Expected artifacts:

```text
MemVerse/MemoryKB/Long_Term_Memory/semmemory/raw_episodes/<episode_id>.json
MemVerse/MemoryKB/Long_Term_Memory/semmemory/cases/<episode_id>_case.json
MemVerse/MemoryKB/Long_Term_Memory/semmemory/graphs/<episode_id>_graph.json
MemVerse/MemoryKB/Long_Term_Memory/semmemory/schemas/schemas.json
MemVerse/MemoryKB/Long_Term_Memory/semmemory/chunks/<episode_id>_episodic.jsonl
MemVerse/MemoryKB/Long_Term_Memory/semmemory/chunks/<episode_id>_semantic_graph.jsonl
MemVerse/MemoryKB/Long_Term_Memory/semmemory/chunks/<episode_id>_semantic_schema.jsonl
MemVerse/MemoryKB/Long_Term_Memory/semmemory/reports/<episode_id>_report.json
```

Expected report flags:

```json
{
  "used_generic_insert": false,
  "used_build_memory": false,
  "used_insert_chunks_from_json": true,
  "used_lightrag_ainsert": true,
  "used_lightrag_custom_kg": false,
  "semmemory_llm_calls": 2
}
```

## 14. Test Plan

Unit tests:

- endpoint payload validation;
- store writes raw episode, case, graph, schemas, chunks, prompts, outputs,
  reports;
- parser validation rejects nodes missing semantic dimension fields;
- consolidator validation accepts schema ops;
- exporter creates JSONL rows with `output_text`;
- SemMemory service does not call `build_memory.process_memory()`;
- SemMemory service calls `insert_chunks_from_json()`;
- retriever uses context-only query;
- retriever filters invalidated schema and contradicted preconditions;
- agent removes hardcoded action override.

Smoke tests:

- MemVerse starts with `/alfred/*` endpoints;
- one episode produces raw case, graph, schema report, and chunk JSONL;
- retrieval returns empty valid context before memory exists;
- retrieval returns schema/case context after one completed episode.

Metrics to log:

- task_success;
- task_progress;
- num_invalid_actions;
- num_steps;
- parser node/edge count;
- consolidation op count;
- schema count;
- `mem_epi` chunk export count;
- `mem_sem` chunk export count;
- LightRAG insert track ids;
- retrieval candidate count;
- filtered candidate count;
- SemMemory LLM call count.

## 15. First Coding Slice

Implement the smallest vertical path:

1. `contracts.py`
2. `store.py`
3. `event_parser.py`
4. `consolidator.py`
5. `exporter.py`
6. `service.py`
7. `/alfred/*` routes in `MemVerse/app.py`
8. `/alfred/*` handlers in `MemVerse/orchestrator.py`
9. `MemVerseClient` methods and episode payload collection in
   `memverse_alfred_agent.py`
10. remove the hardcoded action override

The first slice is complete when one ALFRED episode can be collected locally,
sent once at episode end, parsed into an event-centric semantic experience
graph, consolidated into schema memory, saved as JSON artifacts, exported as
SemMemory chunk JSONL, and inserted into `mem_epi` / `mem_sem` through the
existing LightRAG insertion path without calling generic MemVerse memory
builders.
