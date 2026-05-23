# ✅ Fix: Enhanced VLM Planner Prompt Precision

**Date**: January 7, 2026  
**Issue**: VLM Planner outputs wrong action IDs despite correct WM and feedback hints  
**Root Cause**: Prompt guidance too weak, no explicit action selection rules  
**Solution**: Enhanced prompt with action selection rules and task-relevant highlighting

---

## Changes Made

### 1. Added Task-Relevant Action Highlighting

**File**: `embodiedbench/planner/vlm_planner.py`  
**New Method**: `get_task_relevant_actions()`

**Purpose**: Extract and highlight only actions relevant to the current task, reducing cognitive load on VLM.

**How It Works**:
```python
def get_task_relevant_actions(self, user_instruction, available_actions, wm_context=None):
    """Extract task-relevant actions to highlight in the prompt.
    
    Filters actions that contain entities mentioned in the instruction or WM context.
    """
    # Extract entities from instruction: "move a spatula from right counter to left counter"
    # → entities: ["spatula", "right counter", "left counter"]
    
    # Extract entities from WM context (visible objects)
    # → entities: ["spatula", "screwdriver", "apple", ...]
    
    # Filter actions containing these entities:
    # action 10: "navigate to the right counter" ✅ contains "right counter"
    # action 17: "pick up the clamp" ❌ not relevant
    # action 24: "pick up the spatula" ✅ contains "spatula"
    # action 54: "place at the right receptacle" ✅ contains "right" + "receptacle"
```

**Output Example**:
```
### TASK-RELEVANT ACTIONS (Focus on these):
  ► action id 10: navigate to the right counter in the kitchen
  ► action id 11: navigate to the left counter in the kitchen
  ► action id 24: pick up the spatula
  ► action id 54: place at the right receptacle of the left counter
  ► action id 55: place at the left receptacle of the left counter
```

**Benefits**:
- Reduces 69 actions → ~5-10 relevant actions
- Makes correct action ID obvious
- VLM can focus on task-relevant choices

---

### 2. Added Explicit Action Selection Rules

**File**: `embodiedbench/planner/vlm_planner.py`  
**Modified Method**: `process_prompt()`

**New Guidance Block**:
```python
action_guidance = '''
### ACTION SELECTION RULES (CRITICAL):
1. You MUST select action IDs ONLY from the available actions list above.

2. When you need to "pick up X", find the EXACT action where description contains "pick up X".
   Example: If you need "pick up the spatula", search for action ID where description = "pick up the spatula"

3. DO NOT guess or hallucinate action IDs. DO NOT reuse action IDs from previous steps without checking.

4. When feedback or hints say "Pick up the spatula instead of Y", you MUST:
   - Search the action list for "pick up the spatula"
   - Use that exact action ID
   - NOT use the action ID for "pick up Y"

5. Double-check: Does the action description match what you want to do?
'''
```

**Why This Helps**:
- **Explicit rule #2**: Tells VLM to do exact string matching
- **Explicit rule #4**: Addresses the exact error pattern (spatula vs. clamp)
- **Explicit rule #5**: Adds verification step

---

### 3. Integrated Feedback Hints into Prompt

**File**: `embodiedbench/planner/vlm_planner.py`  
**Modified Method**: `process_prompt()`

**New Parameter**: `feedback_hints=None`

**Added Block**:
```python
if feedback_hints:
    action_guidance += f'''
### FEEDBACK FROM PREVIOUS ATTEMPTS:
{feedback_hints}

IMPORTANT: The feedback above tells you what went wrong. Adjust your plan accordingly!
'''
```

**Impact**:
- Feedback hints now appear BEFORE the task, not just in instruction
- Highlighted as critical information
- Explicitly tells VLM to adjust based on feedback

---

### 4. Updated Planner Invocation

**File**: `embodiedbench/evaluator/meta_flat_habitat_agent.py`  
**Modified Method**: `_plan_with_vlm()`

**Changes**:
```python
# Prepare WM context string for planner's action filtering
wm_context_str = None
if semantic_result and semantic_result.nodes:
    wm_context_str = semantic_result.to_text_block()

# Prepare feedback hints string for planner's guidance
feedback_hints_str = None
if feedback_hint:
    feedback_hints_str = feedback_hint

# Call planner with enhanced context
try:
    action, reasoning = self.planner.act(
        img_path, 
        effective_instruction,
        wm_context=wm_context_str,        # NEW: for action filtering
        feedback_hints=feedback_hints_str  # NEW: for explicit guidance
    )
except TypeError:
    # Fallback: planner doesn't support new parameters yet
    action, reasoning = self.planner.act(img_path, effective_instruction)
```

**Benefits**:
- WM context used to filter relevant actions
- Feedback hints highlighted in prompt
- Backward compatible (try/except fallback)

---

### 5. Updated VLMPlanner.act() Signature

**File**: `embodiedbench/planner/vlm_planner.py`  
**Modified Method**: `act()`

**New Signature**:
```python
def act(self, observation, user_instruction, wm_context=None, feedback_hints=None):
    # ...
    prompt = self.process_prompt(
        user_instruction, 
        prev_act_feedback=self.episode_act_feedback,
        wm_context=wm_context,          # NEW
        feedback_hints=feedback_hints    # NEW
    )
```

---

## Example: Before vs. After

### BEFORE (Weak Prompt)

```
Available actions:
action id 0: navigate to the cabinet 7, action id 1: navigate to the chair 2, ...
action id 17: pick up the clamp, ... action id 24: pick up the spatula, ... [69 actions total]

## Now the human instruction is: Move a spatula from the right counter to the right receptacle of the left counter.

You are supposed to output in json. At the end, output the action id (0 ~ 68) from the available actions to excute.
```

**Problems**:
- ❌ 69 actions overwhelming
- ❌ No guidance on how to select correct action
- ❌ No emphasis on exact matching
- ❌ Feedback hints buried in instruction text

**Result**: VLM outputs action 17 (clamp) instead of 24 (spatula)

---

### AFTER (Enhanced Prompt)

```
Available actions:
action id 0: navigate to the cabinet 7, action id 1: navigate to the chair 2, ...
action id 17: pick up the clamp, ... action id 24: pick up the spatula, ... [69 actions total]

### TASK-RELEVANT ACTIONS (Focus on these):
  ► action id 10: navigate to the right counter in the kitchen
  ► action id 11: navigate to the left counter in the kitchen
  ► action id 24: pick up the spatula
  ► action id 54: place at the right receptacle of the left counter

### ACTION SELECTION RULES (CRITICAL):
1. You MUST select action IDs ONLY from the available actions list above.
2. When you need to "pick up X", find the EXACT action where description contains "pick up X".
   Example: If you need "pick up the spatula", search for action ID where description = "pick up the spatula"
3. DO NOT guess or hallucinate action IDs. DO NOT reuse action IDs from previous steps without checking.
4. When feedback or hints say "Pick up the spatula instead of Y", you MUST:
   - Search the action list for "pick up the spatula"
   - Use that exact action ID (action 24 in this case)
   - NOT use the action ID for "pick up Y"
5. Double-check: Does the action description match what you want to do?

### FEEDBACK FROM PREVIOUS ATTEMPTS:
The robot has navigated successfully to the right counter but repeatedly attempts to pick up a ball that is not its goal. 
The robot must ensure it is trying to interact with the spatula instead.
IMPORTANT: The feedback above tells you what went wrong. Adjust your plan accordingly!

## Now the human instruction is: Move a spatula from the right counter to the right receptacle of the left counter.

You are supposed to output in json. At the end, output the action id (0 ~ 68) from the available actions to excute.
```

**Improvements**:
- ✅ Highlights 4 relevant actions (from 69)
- ✅ Explicit rules for action selection
- ✅ Example: "pick up the spatula" → action 24
- ✅ Feedback prominently displayed with urgency
- ✅ Multiple layers of verification guidance

**Expected Result**: VLM outputs action 24 (spatula) correctly

---

## Technical Details

### Entity Extraction Patterns

```python
patterns = [
    # Extract from action verbs: "move a spatula" → "spatula"
    r'(?:move|pick up|place|put|navigate to|go to)\s+(?:a|an|the)?\s*([\w\s]+?)(?:\s+from|\s+to|\s+at|\s+in|$)',
    
    # Extract from prepositions: "from the right counter" → "right counter"
    r'(?:from|to|at|in)\s+(?:a|an|the)?\s*([\w\s]+?)(?:\s+to|\s+from|\s+and|,|\.|\s+in|$)',
]
```

### WM Context Extraction

```python
# Extract visible objects from WM context
# Pattern 1: [object] spatula | visible=True
wm_entities = re.findall(r'\[object\]\s+([\w\s]+?)\s+\|', wm_lower)

# Pattern 2: object:spatula (source=...)
wm_entities += re.findall(r'object:([\w\s]+?)\s+', wm_lower)
```

### Action Filtering Logic

```python
relevant_actions = []
for action_id, action_desc in enumerate(available_actions):
    action_lower = action_desc.lower()
    # Check if any entity appears in this action
    for entity in entities:
        if entity in action_lower:
            relevant_actions.append((action_id, action_desc))
            break
```

---

## Testing Verification

### Expected Behavior

1. **Task-relevant actions highlighted**:
   - Log should show: `### TASK-RELEVANT ACTIONS (Focus on these):`
   - Should list ~5-10 actions (not all 69)

2. **Action selection rules present**:
   - Log should show: `### ACTION SELECTION RULES (CRITICAL):`
   - Should include 5 explicit rules

3. **Feedback hints integrated**:
   - Log should show: `### FEEDBACK FROM PREVIOUS ATTEMPTS:`
   - Should include hints like "Pick up the spatula instead of the ball"

4. **Correct action selected**:
   - When instruction says "pick up the spatula"
   - And feedback says "Pick up the spatula instead of the ball"
   - VLM should output action 24 (or whichever ID = "pick up the spatula")
   - NOT action 17 (clamp) or action 16 (ball)

### Log Patterns to Check

```bash
# Check if relevant actions are highlighted
grep "TASK-RELEVANT ACTIONS" eb_log.txt

# Check if action rules are present
grep "ACTION SELECTION RULES" eb_log.txt

# Check if feedback is integrated
grep "FEEDBACK FROM PREVIOUS ATTEMPTS" eb_log.txt

# Verify correct action selected
grep "Planned action ids.*24" eb_log.txt  # Should see spatula action
```

---

## Impact Analysis

### Before Fix

**Failure Pattern** (from eb_log.txt):
```
Step 2: action 17 'pick up the clamp', invalid=True
Step 9: action 16 'pick up the ball', invalid=True
Step 11: action 16 'pick up the ball', invalid=True
Step 14: action 17 'pick up the clamp', invalid=True
```
- 4/14 steps wasted on wrong objects
- Despite correct WM and feedback hints
- Agent stuck in loop

### After Fix (Expected)

```
Step 2: action 24 'pick up the spatula', invalid=False, task_progress=0.667
Step 3: action 11 'navigate to the left counter', invalid=False
Step 4: action 54 'place at the right receptacle', invalid=False, task_success=1.0
```
- Direct path to goal
- No wasted actions
- Task completed efficiently

### Performance Improvement

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Invalid actions | 4/14 (28.6%) | 0/4 (0%) |
| Steps to complete | >14 (failed) | ~4 (success) |
| Success rate | 0% | 100% (on spatula task) |
| VLM precision | Low (wrong IDs) | High (exact match) |

---

## Files Modified

1. **`embodiedbench/planner/vlm_planner.py`**
   - ✅ Added `get_task_relevant_actions()` method
   - ✅ Enhanced `process_prompt()` with action selection rules
   - ✅ Updated `act()` signature to accept `wm_context` and `feedback_hints`

2. **`embodiedbench/evaluator/meta_flat_habitat_agent.py`**
   - ✅ Modified `_plan_with_vlm()` to prepare and pass WM context
   - ✅ Added try/except for backward compatibility

3. **`docs/FIX_enhanced_planner_prompt_precision.md`** (this file)
   - ✅ Comprehensive documentation of changes

---

## Summary

**Problem**: VLM Planner had access to all action IDs but chose wrong ones due to weak prompt guidance.

**Root Cause Analysis**: ✅ User was **100% correct** - the prompt lacked precision and strength:
- No explicit rules for action ID selection
- No task-relevant action highlighting  
- No emphasis on exact string matching
- Feedback hints not integrated prominently

**Solution**: Enhanced prompt with:
1. Task-relevant action highlighting (69 actions → 5-10 relevant)
2. Explicit action selection rules (5 critical rules)
3. Feedback hints prominently displayed
4. Verification guidance at multiple levels

**Expected Outcome**: VLM should now select correct action IDs by following explicit rules and focusing on relevant actions.

**Status**: ✅ Implemented and ready for testing
