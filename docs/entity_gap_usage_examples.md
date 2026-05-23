# Entity Gap 检测与 LTM 位置检索 - 使用示例

## 示例 1: 基础用法（系统自动运行）

### 场景
```
Instruction: "Move a spatula from the right counter to the left counter."
WM 初始状态: 只有 right_counter, left_counter
LTM 状态: 之前看到过 spatula 在 right counter 上
```

### 执行流程（完全自动）

```python
# 1. Agent 调用 prepare_planner_context（无需手动操作）
result = semantic_memory.prepare_planner_context(instruction)

# 系统内部自动执行：
#   a) 从 instruction 提取实体: ["spatula", "right counter", "left counter"]
#   b) WM read 发现 spatula 缺失 → gap: "entity:object:spatula"
#   c) LTM search 检索 spatula → hint: position="on right counter"
#   d) 应用 hint 到 WM → 创建 spatula 节点
#   e) 重新 read WM → 返回完整上下文

# 2. Planner 收到完整上下文
print(f"Nodes: {[node.primary_label() for node in result.nodes]}")
# 输出: ['spatula', 'right counter', 'left counter']

# 3. 检查 spatula 的位置信息
spatula = next(node for node in result.nodes if node.primary_label() == "spatula")
print(f"Position: {spatula.symbol.state_vars.get('spatial_position')}")
# 输出: 'on right counter'
```

### 日志输出

```
[SemanticMemoryManager] Preparing planner context for goal: Move a spatula from the right counter...
[SemanticMemoryManager] WM read result: nodes=2 gaps=3
[SemanticMemoryManager] Entity gaps detected: ['entity:object:spatula']
[SemanticMemoryManager] Attribute gaps detected: ['attribute:color:right counter']
[SemanticMemoryManager] LTM search returned 2 hints
[SemanticMemoryManager] Created WM node from LTM entity hint: spatula (object)
[SemanticMemoryManager] Applied 2 hints to WM (1 units updated)
[SemanticMemoryManager] WM updated with LTM hints, re-reading...
[SemanticMemoryManager] No gaps detected in WM read
[SemanticMemoryManager] Planner context prepared: nodes=3 gaps=0 hints=0
```

---

## 示例 2: 复杂场景 - 多个缺失对象

### 场景
```
Instruction: "Put the red knife in the drawer near the sink."
WM 初始状态: 只有 sink
LTM 状态: 之前看到过 knife 在 table 上，drawer 在 cabinet 中
```

### 执行流程

```python
result = semantic_memory.prepare_planner_context(
    "Put the red knife in the drawer near the sink."
)

# 系统检测到的 gaps:
# - entity:object:knife (包含属性提示 "red")
# - entity:place:drawer

# LTM 返回的 hints:
# Hint 1: {
#   "entity_kind": "object",
#   "entity_label": "knife",
#   "position": "on table",
#   "attributes": {"color": ["red", "silver"], "size": ["small"]}
# }
# Hint 2: {
#   "entity_kind": "place",
#   "entity_label": "drawer",
#   "position": "in cabinet",
#   "attributes": {}
# }

# WM 更新后:
for node in result.nodes:
    print(f"{node.primary_label()}: {node.symbol.state_vars}")

# 输出:
# knife: {'source': 'ltm_entity_hint', 'visible': False, 
#         'spatial_position': 'on table', 'colors': ['red', 'silver'], 
#         'size_hint': 'small'}
# drawer: {'source': 'ltm_entity_hint', 'visible': False, 
#          'spatial_position': 'in cabinet'}
# sink: {'visible': True}
```

---

## 示例 3: 属性匹配 + 位置检索组合

### 场景
```
Instruction: "Pick up a small red object with green top."
WM 初始状态: 空
LTM 状态: strawberry (red, small, green leaves on top, on brown table)
```

### 执行流程

```python
result = semantic_memory.prepare_planner_context(
    "Pick up a small red object with green top."
)

# 系统行为:
# 1. 从 instruction 提取实体（描述性）: 
#    "a small red object with green top"
# 2. 检测 entity gap: "entity:object:a small red object with green top"
# 3. LTM 尝试匹配:
#    - 提取属性: color=[red, green], size=[small], feature=[green top]
#    - 通过属性索引查询: 找到 strawberry
#    - 规范化: "a small red object with green top" → "strawberry"
# 4. 返回 hint:
hint = {
    "entity_label": "strawberry",  # ← 规范化后的名称
    "position": "on brown table",
    "attributes": {
        "color": ["red"],
        "size": ["small"],
        "feature": ["green leaves on top"]
    }
}

# 5. WM 创建 strawberry 节点（而不是 "a small red object with green top"）
strawberry = result.nodes[0]
print(strawberry.primary_label())  # "strawberry"
print(strawberry.symbol.state_vars['spatial_position'])  # "on brown table"
```

---

## 示例 4: 对象已在 WM 但缺少位置

### 场景
```
WM 中已有: spatula (visible=True, 但没有 spatial_position)
Instruction: "Put the spatula on the left counter."
LTM 状态: spatula 历史位置 "on right counter"
```

### 执行流程

```python
# WM 初始状态
spatula = wm.find_unit_by_label("spatula")
print(spatula.symbol.state_vars)
# 输出: {'visible': True, 'colors': ['silver']}

# Planner context 准备
result = semantic_memory.prepare_planner_context(
    "Put the spatula on the left counter."
)

# 系统行为:
# 1. WM read: spatula 存在，但缺少 spatial_position
# 2. Gap 检测: 不会生成 entity gap（因为 spatula 已存在）
# 3. 但会生成 attribute gap: "state_vars missing for spatula"
# 4. LTM search 可能返回位置信息作为 state_vars hint

# 注意：这种情况可能需要额外优化
# 当前实现主要针对"完全缺失的对象"
```

---

## 示例 5: 探索引导（未来优化）

### 场景
```
LTM: spatula 上次在 right counter 上（50 steps ago）
当前视野: 看不到 spatula
```

### 增强后的行为（建议实现）

```python
# 当 visible=False 且 position 已知时，生成探索提示
hint = {
    "entity_label": "spatula",
    "position": "on right counter",
    "visible": False,
    "exploration_suggestion": (
        "spatula was previously observed on right counter, "
        "but is not currently visible. Navigate to right counter to verify."
    )
}

# Planner 可以根据此提示优先生成导航动作
# plan = [navigate_to_right_counter, look_around, pick_up_spatula, ...]
```

---

## 示例 6: 位置冲突处理（未来优化）

### 场景
```
LTM: spatula 在 right counter（历史）
当前感知: spatula 在 table（新观察）
```

### 建议处理逻辑

```python
# 在 ingest_perception 时检测冲突
if wm_node.state_vars.get('spatial_position') != perception_position:
    logger.warning(
        f"Position updated for {label}: "
        f"{wm_node.state_vars['spatial_position']} → {perception_position}"
    )
    
    # 更新 WM
    wm_node.state_vars['spatial_position'] = perception_position
    wm_node.state_vars['last_position_update'] = current_step
    
    # 同时更新 LTM（在下次 sync 时）
    # LTM 会记录位置变化历史
```

---

## 代码集成示例

### 在 HabitatMetaFlatAgent 中使用

```python
class HabitatMetaFlatAgent:
    def run_single_episode(self):
        # ... 初始化 ...
        
        # 每次规划前，自动触发 entity gap 检测和 LTM 检索
        fusion_ctx = self.fusion.fuse(p_out, instruction)
        
        # fusion 内部会调用:
        # semantic_result = self.semantic_memory.prepare_planner_context(instruction)
        # ↑ 这里就会自动检测 entity gaps 并从 LTM 检索位置
        
        # Planner 收到的上下文已包含从 LTM 检索的对象和位置
        current_plan = self._plan_with_vlm(img_path, instruction, wm=wm)
        
        # ... 执行 ...
```

### 无需修改现有代码

**关键点：** 所有功能都在 `prepare_planner_context` 内部自动触发，不需要修改 Agent 的主循环或规划逻辑。

---

## 调试技巧

### 1. 检查提取的实体

```python
entities = semantic_memory.graph.extract_key_entities_from_goal(instruction)
print(f"Extracted entities: {entities}")
```

### 2. 检查检测到的 gaps

```python
result = semantic_memory.operator.read(query)
entity_gaps = [g for g in result.gaps if g.startswith("entity:")]
print(f"Entity gaps: {entity_gaps}")
```

### 3. 检查 LTM hints

```python
hints = semantic_memory.ltm.search(query, result.gaps)
for hint in hints:
    if hint.get("entity_kind"):
        print(f"Entity: {hint['entity_label']}")
        print(f"Position: {hint['position']}")
        print(f"Attributes: {hint['attributes']}")
```

### 4. 检查 WM 更新

```python
before_count = len(semantic_memory.graph.nodes)
updated, _ = semantic_memory._apply_attribute_hints_to_wm(hints, step=1)
after_count = len(semantic_memory.graph.nodes)
print(f"Nodes added: {after_count - before_count}")
print(f"Updated: {updated}")
```

---

## 常见问题

### Q1: 为什么 LTM 中有 spatula 但没有返回 hint？

**可能原因：**
1. Entity gap 未被正确生成（检查实体提取）
2. LTM 节点标签不匹配（检查 `_get_node_by_label`）
3. Position 信息缺失（检查 LTM 节点的 attributes 和 edges）

**调试：**
```python
# 检查 LTM 节点
node = semantic_memory.ltm._get_node_by_label("spatula")
if node:
    print(f"LTM node found: {node.label}")
    print(f"Attributes: {node.attributes}")
    # 检查空间关系边
    edges = [e for e in semantic_memory.ltm.edges.values() if e.source == node.id]
    print(f"Edges: {[(e.relation, semantic_memory.ltm.nodes.get(e.target).label) for e in edges]}")
else:
    print("LTM node NOT found")
```

### Q2: WM 节点被创建但没有位置信息？

**可能原因：**
1. LTM hint 中 position="unknown"
2. Hint 应用时映射错误

**调试：**
```python
# 检查 hint 内容
print(f"Hint position: {hint.get('position')}")
print(f"Hint attributes: {hint.get('attributes')}")

# 检查 WM 节点
unit = semantic_memory.graph.find_unit_by_label("spatula")
print(f"WM state_vars: {unit.symbol.state_vars}")
```

### Q3: 实体提取不准确？

**可能原因：**
介词短语正则过于宽松或过于严格

**解决方法：**
调整 `extract_key_entities_from_goal` 中的正则模式：
```python
prep_patterns = [
    r"from\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+to|\s+and|\s+in|\.)",  # 添加句号
    # ...
]
```

---

## 性能考虑

### 时间复杂度
- Entity 提取: O(n) - n 是 instruction 长度
- Gap 检测: O(m × k) - m 是提取的实体数，k 是 WM 节点数
- LTM 检索: O(g × l) - g 是 gaps 数，l 是 LTM 节点数
- WM 更新: O(h) - h 是 hints 数

### 空间复杂度
- 每个实体 hint ~1KB
- 通常 entity gaps < 5 个/次
- 总开销可忽略

### 优化建议
1. 缓存实体提取结果（相同 instruction）
2. LTM 节点索引优化（已实现 node_index）
3. Batch 应用 hints（已实现）

---

## 总结

✅ **使用简单**: 完全自动，无需额外代码
✅ **日志清晰**: 每步操作都有详细日志
✅ **向后兼容**: 不影响现有功能
✅ **可扩展**: 易于添加新的 gap 类型和检索策略
