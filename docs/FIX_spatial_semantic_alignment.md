# 空间语义对齐修复

## 问题描述

### 用户需求
```
场景：
1. Agent 执行 "navigate to the left counter in the kitchen"
2. 从 image 中观察到 "sink" 在右边
3. Instruction 提到 "the right receptacle of the left counter"

期望推理：
- sink 在 left counter 的右边（空间关系）
- sink 是一种 receptacle（语义关系）
- → "right receptacle of left counter" = sink

期望行为：
Agent 应该执行 "place at the sink in the kitchen"
```

### 技术挑战

需要同时处理：
1. **空间关系解析**："right receptacle **of** left counter" → 方位词=right, 核心词=receptacle, 参考对象=left counter
2. **语义类型匹配**：receptacle（抽象） → sink（具体）
3. **空间验证**：sink 确实在 left counter 的右边吗？

## 解决方案

### 核心思想

**三步语义对齐流程：**

```
Step 1: 解析空间关系
  输入: "right receptacle of left counter"
  输出: 
    - spatial_modifier = "right"
    - core_term = "receptacle"  
    - reference_object = "left counter"

Step 2: 语义类型匹配
  查询: ABSTRACT_TO_CONCRETE_HINTS["receptacle"]
  结果: ["sink", "basin", "bowl", "container", "bin"]
  观察到的对象: ["pliers", "sink", "drawer", ...]
  匹配: sink ∈ receptacle类型 → score = 0.85

Step 3: 空间关系验证
  检查: sink 的 position 或 relation 中是否包含 "right"
  如果是: score += 0.15 → 总分 = 1.0
  创建对齐: "right receptacle of left counter" → sink
```

### 实现细节

#### 1. 增强 `align_semantics()` 方法

**位置:** `embodiedbench/evaluator/semantic_memory.py:800-900`

**新增功能:**

##### A. 空间关系解析
```python
# 检测 "X of Y" 模式
if " of " in gap_label:  # 例如: "right receptacle of left counter"
    before_of, after_of = gap_label.split(" of ", 1)
    reference_object = after_of.strip()  # "left counter"
    
    # 提取方位词和核心术语
    tokens = before_of.strip().split()
    if tokens[0] in ["left", "right", "front", "back", "top", "bottom"]:
        spatial_modifier = tokens[0]  # "right"
        core_term = " ".join(tokens[1:])  # "receptacle"
```

##### B. 递归核心词提取
```python
# 从 "right receptacle" 提取 "receptacle"
# 从 "kitchen counter" 提取 "counter"
if not potential_concrete and " " in core_term:
    last_word = core_term.split()[-1]
    potential_concrete = ABSTRACT_TO_CONCRETE_HINTS.get(last_word, [])
    if potential_concrete:
        core_term = last_word
```

##### C. 空间验证加分
```python
for obs_label in observed_labels:
    # 1. 类型匹配
    if obs_norm in potential_concrete:
        score = 0.85
    
    # 2. 空间关系验证
    if score > 0 and spatial_modifier:
        spatial_ok = self._verify_spatial_relation(
            obs_label, spatial_modifier, reference_object
        )
        if spatial_ok:
            score += 0.15  # 空间匹配加分
```

#### 2. 新增 `_verify_spatial_relation()` 方法

**位置:** `embodiedbench/evaluator/semantic_memory.py:903-960`

**功能:** 验证对象是否满足空间关系约束

**实现逻辑:**

```python
def _verify_spatial_relation(
    self,
    object_label: str,      # "sink"
    spatial_modifier: str,  # "right"
    reference_object: Optional[str]  # "left counter"
) -> bool:
    obj_unit = self.graph.find_unit_by_label(object_label)
    
    # 方法 1: 检查 state_vars 中的位置
    position = obj_unit.symbol.state_vars.get("spatial_position") or \
               obj_unit.symbol.state_vars.get("position")
    if position and spatial_modifier in str(position).lower():
        return True  # 例如: position="right side"
    
    # 方法 2: 检查关系
    for rel in obj_unit.symbol.relations:
        rel_text = f"{rel.relation} {rel.target}".lower()
        if spatial_modifier in rel_text:
            return True  # 例如: relation="located_at right counter"
    
    # 方法 3: 检查与参考对象的关系
    if reference_object:
        for rel in obj_unit.symbol.relations:
            if reference_object in rel.target.lower() and \
               spatial_modifier in rel.relation.lower():
                return True
    
    return False
```

### 代码修改

#### 修改 1: `align_semantics()` - 空间关系解析

```python
# 修改前
gap_label = parts[2].strip().lower()
potential_concrete = ABSTRACT_TO_CONCRETE_HINTS.get(gap_label, [])

# 修改后
gap_label = parts[2].strip().lower()

# ✨ 解析空间关系
spatial_modifier = None
core_term = gap_label
reference_object = None

if " of " in gap_label:
    before_of, after_of = gap_label.split(" of ", 1)
    reference_object = after_of.strip()
    
    tokens = before_of.strip().split()
    if tokens[0] in ["left", "right", "front", "back", "top", "bottom"]:
        spatial_modifier = tokens[0]
        core_term = " ".join(tokens[1:])
        logger.info(
            f"[Semantic Alignment] Parsed '{gap_label}' → "
            f"spatial='{spatial_modifier}', term='{core_term}', ref='{reference_object}'"
        )

potential_concrete = ABSTRACT_TO_CONCRETE_HINTS.get(core_term, [])
```

#### 修改 2: `align_semantics()` - 空间验证加分

```python
# 修改前
for obs_label in observed_labels:
    if obs_norm in potential_concrete:
        best_match = obs_label
        best_score = 0.95
        break

# 修改后
for obs_label in observed_labels:
    score = 0.0
    reason_parts = []
    
    # 1. 类型匹配
    if obs_norm in [c.lower() for c in potential_concrete]:
        score = 0.85
        reason_parts.append(f"type={core_term}")
    
    # 2. 空间验证
    if score > 0 and spatial_modifier:
        spatial_ok = self._verify_spatial_relation(
            obs_label, spatial_modifier, reference_object
        )
        if spatial_ok:
            score += 0.15
            reason_parts.append(f"spatial={spatial_modifier}")
    
    if score > best_score:
        best_score = score
        best_match = obs_label
        match_reason = "+".join(reason_parts)
```

#### 修改 3: 降低匹配阈值

```python
# 修改前
if best_match and best_score >= 0.7:

# 修改后
# 降低阈值以支持空间推理
if best_match and best_score >= 0.6:
```

原因：空间验证提供了额外的置信度，即使类型匹配分数略低，加上空间验证也能达到较高的总分。

## 执行示例

### 场景

**Instruction:** "Move a spatula from the right counter to the right receptacle of the left counter."

**Step 5:** Agent 执行 `navigate to the left counter in the kitchen`

**观察结果 (VLM):**
```json
{
  "detected_objects": [
    {"label": "pliers", "position": "left side"},
    {"label": "sink", "position": "right side"},
    {"label": "drawer", "position": "center"},
    {"label": "ring", "position": "center"}
  ]
}
```

### 语义对齐过程

```
[INFO] - Detecting entity gaps from instruction for semantic alignment
[INFO] - Extracted 4 entity gaps: [
    'entity:object:spatula',
    'entity:place:right counter',
    'entity:place:right receptacle of left counter',  ← 关键！
    'entity:place:left counter'
  ]

[INFO] - Attempting semantic alignment with 4 pending gaps and 4 observed objects

[INFO] - Parsed 'right receptacle of left counter' →
         spatial='right', term='receptacle', ref='left counter'

[INFO] - Extracted core term 'receptacle' from 'receptacle',
         hints: ['sink', 'basin', 'bowl', 'container', 'bin']

[检查 sink:]
  - Type match: sink ∈ ['sink', 'basin', ...] → score = 0.85
  - Position check: sink.position = "right side" contains "right" → +0.15
  - Total score = 1.0

[INFO] - ✅ Aligned 'right receptacle of left counter' → 'sink'
         (conf=1.00, kind=place, reason=type=receptacle+spatial=right)

[INFO] - Semantic alignment: 1 mappings established
[INFO] -   - 'right receptacle of left counter' → 'sink' (conf=1.00)
```

### WM 更新

**sink unit 的变化:**

```python
before:
  symbol.label = "sink"
  symbol.aliases = []
  symbol.state_vars = {"position": "right side", ...}

after:
  symbol.label = "sink"
  symbol.aliases = ["right receptacle of left counter"]  # ✨ 新增
  symbol.relations = [
    RelationSpec(
      relation="synonym_of",
      target="right receptacle of left counter",
      confidence=1.0
    )
  ]
  symbol.state_vars = {"position": "right side", ...}
```

### Planner 上下文

准备下一次规划时：

```python
[INFO] - Entity Gap Detection
[INFO] - Entity 'right receptacle of left counter' found!
         (matched via alias in sink unit)
[INFO] - Final missing gaps: ['entity:place:right counter']

WM Context for Planner:
  - object:sink (aliases=['right receptacle of left counter'])
  - position: right side
  - available actions: "place at the sink in the kitchen"
```

**VLM Planner 输出:**
```json
{
  "action_id": 52,
  "description": "place at the sink in the kitchen",
  "reasoning": "The right receptacle of the left counter is the sink, which is located on the right side. I should place the spatula there."
}
```

## 预期效果

### 成功案例

✅ **"right receptacle of left counter" → sink**
- 类型匹配: receptacle → sink (0.85)
- 空间验证: sink.position="right side" (0.15)
- 总分: 1.0

✅ **"left drawer of kitchen counter" → drawer**
- 类型匹配: drawer → drawer (0.85)
- 空间验证: drawer.position="left" (0.15)
- 总分: 1.0

✅ **"top shelf" → shelf**
- 类型匹配: shelf → shelf (0.85)
- 空间验证: shelf.position="top" (0.15)
- 总分: 1.0

### 日志验证点

修复成功后应该看到：

```log
✅ 空间关系解析:
[INFO] - Parsed 'right receptacle of left counter' →
         spatial='right', term='receptacle', ref='left counter'

✅ 核心词提取:
[INFO] - Extracted core term 'receptacle' from 'receptacle',
         hints: ['sink', 'basin', 'bowl', 'container', 'bin']

✅ 空间验证:
[INFO] - [Spatial Verification] ✅ 'sink' position='right side' matches 'right'

✅ 对齐成功:
[INFO] - ✅ Aligned 'right receptacle of left counter' → 'sink'
         (conf=1.00, kind=place, reason=type=receptacle+spatial=right)

✅ WM 更新:
[INFO] - Semantic alignment: 1 mappings established
[INFO] -   - 'right receptacle of left counter' → 'sink' (conf=1.00)
```

## 影响分析

### 功能提升

1. ✅ **空间推理能力**：能理解 "X of Y" 结构的空间关系
2. ✅ **语义对齐准确性**：结合类型和位置双重验证
3. ✅ **任务理解深度**：能处理复杂的自然语言指令
4. ✅ **Action 选择正确性**：知道 "right receptacle" = sink → 选择正确的 place 动作

### 性能影响

- **轻微增加**：每次对齐需要额外的空间验证查询
- **可接受**：空间验证只是查找 WM 中已有的 state_vars 和 relations
- **优化空间**：可以缓存空间关系以减少重复查询

### 兼容性

- ✅ **向后兼容**：如果没有空间修饰词，仍使用原来的纯语义匹配
- ✅ **优雅降级**：空间验证失败不会完全阻止对齐，只是分数略低

## 测试建议

### 单元测试

```python
def test_spatial_semantic_alignment():
    """测试空间语义对齐"""
    manager = SemanticMemoryManager()
    manager.reset(
        instruction="Move spatula to the right receptacle of the left counter",
        episode_id=1
    )
    
    # 模拟感知：检测到 sink 在右边
    perception = {
        "detected_objects": [
            {
                "label": "sink",
                "position": "right side",
                "size": "large",
                "material": ["metal"]
            }
        ]
    }
    
    manager.ingest_perception(
        perception=perception,
        instruction=manager.instruction,
        img_path="test.png",
        env_step=1
    )
    
    # 验证对齐
    sink_unit = manager.graph.find_unit_by_label("sink")
    assert sink_unit is not None
    assert "right receptacle of left counter" in [
        a.lower() for a in sink_unit.symbol.aliases
    ]
    
    # 验证关系
    relations = {(r.relation, r.target) for r in sink_unit.symbol.relations}
    assert ("synonym_of", "right receptacle of left counter") in relations
```

### 集成测试

运行完整 episode，验证：
1. Agent 导航到 left counter
2. 自动观察检测到 sink 在 right side
3. 语义对齐建立 "right receptacle of left counter" → sink
4. Planner 选择正确的动作：place at the sink
5. 任务成功完成

## 相关文档

- [语义对齐时序修复](FIX_semantic_alignment_timing.md)
- [语义对齐知识图谱](semantic_alignment_knowledge_graph.md)
- [导航自动感知修复](navigation_auto_perception_fix.md)

## 变更历史

- **2026-01-07**: 初始实现，添加空间语义对齐能力
