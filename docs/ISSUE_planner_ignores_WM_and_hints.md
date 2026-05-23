# 🚨 Critical Issue: VLM Planner Ignores Working Memory and Feedback Hints

**Date**: January 7, 2026  
**Severity**: CRITICAL  
**Status**: Root cause identified, solution needed

---

## Problem Summary

**Observation**: VLM Planner receives correct feedback hints and complete WM context but still generates wrong action IDs.

### Evidence from Logs (eb_log.txt, line ~1029-1190)

#### 1. Feedback Hints ARE Correct ✅

```json
{
  "prompt_hint": "The environment consists of multiple objects on the right counter, including a spatula, along with other tools like a screwdriver and some fruits. The robot has navigated successfully to the right counter but repeatedly attempts to pick up a ball that is not its goal. The robot must ensure it is trying to interact with the spatula instead.",
  "adjustment_suggestions": [
    "Navigate to the right counter in the kitchen again.",
    "Look around to confirm the spatula's position.",
    "Pick up the spatula instead of the ball.",  // ✅ EXPLICIT INSTRUCTION
    "Navigate to the left counter in the kitchen after picking up the spatula.",
    "Place the spatula at the right receptacle of the left counter."
  ]
}
```

#### 2. Working Memory IS Correct ✅

```
Semantic memory context:
[object] spatula | source=ltm_entity_hint, visible=True, spatial_position=on counter, 
                   last_seen_position=center on the right counter, color=['black'], 
                   size_hint=small, materials=['metal', 'silicone'], shape_hint=['rectangular', 'flat'], 
                   texture=smooth, top_features=['has handle', 'has a long handle'], 
                   last_seen_ts=1767772919.363211, category=cooking_utensil, 
                   affordances=['flip', 'spread'], temperature_tolerance=hot, colors=['black'], 
                   supports_hot_items=True
```

#### 3. Latest Observation IS Correct ✅

After step 13 navigation:
```json
{
  "detected_objects": [
    {"label": "screwdriver", "color": ["black", "silver"], "size": "small", "position": "left side"},
    {"label": "red ball", "color": ["red"], "size": "small", "position": "center"},
    {"label": "piece of cheese", "color": ["yellow"], "size": "small", "position": "near ball"},
    {"label": "spatula", "color": ["black"], "size": "small", "position": "right side"}  // ✅ SPATULA VISIBLE!
  ]
}
```

#### 4. BUT Planner Output IS WRONG ❌

```
[MetaFlat] Planned action ids: [10, 17, 11, 54]
```

**Action ID Mapping**:
- Action 10 = `navigate to the right counter in the kitchen` ✅ Correct
- **Action 17 = `pick up the clamp`** ❌ **WRONG! Should be "pick up the spatula"**
- Action 11 = `navigate to the left counter in the kitchen` ✅ Correct
- Action 54 = `place at the right receptacle of the left counter` ✅ Correct

---

## Root Cause Analysis

### Issue 1: VLM Cannot Map Object Names to Action IDs

**Problem**: The VLM Planner receives:
1. Feedback hint: "Pick up the spatula"
2. WM context: spatula entity with full attributes
3. Observation: spatula visible on right side

But it outputs **Action ID 17 (pick up the clamp)** instead of the correct action ID for "pick up the spatula".

**Why**: The planner needs to:
1. See the instruction "pick up the spatula"
2. Find action ID X where `language_skill_set[X] == "pick up the spatula"`
3. But the planner appears to be **hallucinating action IDs** or **reusing cached/previous IDs**

### Issue 2: Action Space Is Dynamic

From `EBHabEnv.py`:
```python
self.language_skill_set = transform_action_to_natural_language(self.skill_set)
```

The action space is episode-specific:
- Each episode generates different objects (spatula, clamp, ball, etc.)
- Each object gets a unique action ID for pick/place
- **Action ID 17** might be "pick up clamp" in one episode and "pick up spatula" in another

**Current Problem**:
- VLM doesn't see the FULL `language_skill_set` mapping
- VLM doesn't know that it needs action ID 24 for "pick up spatula"
- VLM guesses or reuses previously seen action IDs

---

## Evidence of Repeated Mistakes

### Execution History

```
- step 2: action 17 'pick up the clamp', invalid=True  ❌
- step 9: action 16 'pick up the ball', invalid=True   ❌
- step 11: action 16 'pick up the ball', invalid=True  ❌
- step 14: action 17 'pick up the clamp', invalid=True ❌
```

**Pattern**: Agent keeps trying to pick up **wrong objects** (clamp, ball) even though:
1. Feedback explicitly says "pick up the spatula"
2. WM shows spatula is visible
3. Observation confirms spatula on right side

---

## Why Semantic Memory Can't Fix This

**Semantic Memory's Job** ✅:
- Detect entity gaps ("spatula" missing from WM)
- Align observed objects with gaps (observed "spatula" → instruction "a spatula")
- Store spatial attributes (position, color, size, etc.)
- Provide rich context to planner

**Semantic Memory CANNOT** ❌:
- Force planner to choose correct action IDs
- Override VLM's action ID generation
- Map object names to dynamic action IDs

This is a **Planner Architecture Problem**, not a memory problem.

---

## Current Planner Input (Hypothesized)

We need to verify what the VLM Planner actually sees. Likely scenarios:

### Scenario A: Missing Action Space
```python
# Planner receives:
- Instruction: "Move a spatula from the right counter..."
- WM context: [spatula entity with attributes]
- Feedback: "Pick up the spatula instead of the ball"
- Recent observations: spatula visible

# Planner DOES NOT receive:
- Full language_skill_set mapping
- "Action 24 = pick up the spatula"
- "Action 17 = pick up the clamp"
```

**Result**: VLM hallucinates action IDs or reuses familiar ones.

### Scenario B: Action Space Provided But Not Used
```python
# Planner receives:
- All of the above
- PLUS: language_skill_set = ["navigate to X", "pick up Y", ...]

# But VLM still outputs wrong IDs because:
- Too many actions (~69 actions) in context
- VLM loses track of mapping
- VLM pattern-matches "pick up" → returns first similar action ID seen in examples
```

---

## Proposed Solutions

### Option 1: Add Action Space to Planner Prompt (QUICK FIX)

**Modify**: `meta_flat_habitat_agent.py` planning stage

**Add to planner context**:
```python
### Available Actions for This Episode
You must ONLY use action IDs from this list:
{
  10: "navigate to the right counter in the kitchen",
  11: "navigate to the left counter in the kitchen",
  16: "pick up the ball",
  17: "pick up the clamp",
  24: "pick up the spatula",  // ← THE CORRECT ONE!
  54: "place at the right receptacle of the left counter",
  ...
}

IMPORTANT: When feedback says "Pick up the spatula", you must find the action ID 
where description contains "pick up the spatula" (in this case, action 24).
```

**Pros**:
- Simple to implement
- Makes action mapping explicit
- VLM can do exact string matching

**Cons**:
- Increases context size (~69 actions)
- May exceed token limits for long episodes

### Option 2: Filter Actions by Context (SMART FIX)

**Modify**: Planner to only show **relevant actions**

```python
def filter_relevant_actions(instruction, wm_entities, language_skill_set):
    """Extract only actions relevant to current instruction and WM state."""
    relevant = []
    
    # Get entities from instruction and WM
    entities = extract_entities(instruction) + [node.label for node in wm_entities]
    
    # Filter actions that mention these entities
    for action_id, action_desc in enumerate(language_skill_set):
        for entity in entities:
            if entity.lower() in action_desc.lower():
                relevant.append((action_id, action_desc))
                break
    
    return relevant
```

**Example Output** (for spatula task):
```
Relevant actions for this step:
{
  10: "navigate to the right counter in the kitchen",
  11: "navigate to the left counter in the kitchen",
  24: "pick up the spatula",  // ← Only spatula-related action shown
  54: "place at the right receptacle of the left counter",
  55: "place at the left receptacle of the left counter"
}
```

**Pros**:
- Reduces context size
- Shows only task-relevant actions
- Easier for VLM to choose correctly

**Cons**:
- Requires entity extraction
- May filter out useful actions

### Option 3: Two-Stage Planning (ROBUST FIX)

**Stage 1**: VLM generates **natural language actions**
```python
plan = [
  "navigate to the right counter",
  "pick up the spatula",
  "navigate to the left counter",
  "place at right receptacle"
]
```

**Stage 2**: Deterministic mapper converts to action IDs
```python
def map_to_action_ids(nl_actions, language_skill_set):
    """Map natural language actions to IDs using fuzzy matching."""
    action_ids = []
    for nl_action in nl_actions:
        best_match = find_best_match(nl_action, language_skill_set)
        action_ids.append(best_match.id)
    return action_ids
```

**Pros**:
- VLM focuses on reasoning, not ID lookup
- Deterministic mapping eliminates hallucination
- Can use semantic similarity for matching

**Cons**:
- Requires robust fuzzy matching
- May introduce new mapping errors

---

## Immediate Action Required

### Step 1: Verify Current Planner Input

Check what `VLMPlanner` actually receives:

```python
# In meta_flat_habitat_agent.py, before calling planner.plan()
logger.info(f"[DEBUG] Action space size: {len(self.env.language_skill_set)}")
logger.info(f"[DEBUG] Sample actions: {self.env.language_skill_set[:10]}")
logger.info(f"[DEBUG] Spatula-related actions: {[a for a in self.env.language_skill_set if 'spatula' in a]}")
```

### Step 2: Check VLM Planner Prompt

Read `embodiedbench/planner/vlm_planner.py`:
- Does it receive `language_skill_set`?
- How does it format the action space in the prompt?
- Does the prompt explicitly instruct to map objects to action IDs?

### Step 3: Implement Quick Fix

Add filtered action list to planner context (Option 2).

---

## Summary

**Problem**: VLM Planner ignores correct WM context and feedback hints, outputs wrong action IDs

**Root Cause**: Planner cannot reliably map object names to dynamic action IDs in the episode-specific action space

**Impact**: Agent repeatedly picks wrong objects despite perfect perception and memory

**Solution**: Make action ID mapping explicit in planner prompt (Option 1 or 2)

**Status**: Needs immediate investigation and fix in `VLMPlanner`

---

**Next Steps**:
1. ✅ Identified problem: Planner action ID mapping failure
2. ⏳ Investigate VLMPlanner prompt structure
3. ⏳ Implement filtered action space in planner context
4. ⏳ Test with spatula episode to verify correct action selection
