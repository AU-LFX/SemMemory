# Entity Gap 检测与 LTM 位置检索 - 实现总结

## 实现概述

已成功实现**方案二：增强 Gap 检测 - 直接从 goal 生成 entity missing gap**，使系统能够：
1. 从 instruction 中提取关键实体
2. 检测 WM 中缺失的实体并生成 `entity:kind:label` gap
3. 从 LTM 检索缺失实体的完整信息（包括位置、属性）
4. 将 LTM 信息注入 WM，创建新节点或更新现有节点

---

## 核心修改清单

### 1. SemanticWMGraph - 实体提取 (semantic_memory.py, ~460行)

#### 新增方法：

```python
def extract_key_entities_from_goal(self, goal: str) -> List[str]:
    """从 goal 文本中提取关键对象/地点名称。
    
    使用策略：
    1. ACTION_ENTITY_PATTERNS 提取动作目标
    2. 介词短语模式提取 (from/to/on/in/at + 实体)
    3. 去重并返回
    
    示例:
        "Move a spatula from the right counter to the left counter"
        → ["spatula", "right counter", "left counter"]
    """
```

```python
def infer_kind_from_entity(self, entity: str) -> str:
    """根据实体名称推断 kind（object/place）。
    
    策略：
        - 包含 place_keywords (counter, table, shelf, etc.) → "place"
        - 其他 → "object"
    """
```

### 2. MemoryOperatorEngine - Gap 检测增强 (semantic_memory.py, ~616行)

#### 修改方法：

```python
def read(self, query: WMReadQuery) -> WMReadResult:
    """根据查询抓取子图，并标记缺失字段，供 planner 调整策略。
    
    新增功能：
        - 检测 goal 中提到但 WM 中不存在的实体
        - 生成 "entity:kind:label" gap
    """
```

#### 新增方法：

```python
def _detect_missing_entities_in_goal(
    self, 
    goal: str, 
    existing_nodes: List[SemanticUnit]
) -> List[str]:
    """检测 goal 中提到但 existing_nodes 中不存在的实体。
    
    流程：
    1. 提取 goal 中的关键实体
    2. 构建已存在节点的标签集合（normalized）
    3. 找出缺失的实体（完全匹配 + 模糊匹配 threshold=0.8）
    4. 生成 "entity:kind:label" gap
    
    返回：
        ["entity:object:spatula", "entity:place:drawer", ...]
    """
```

### 3. LTMService - 实体检索 (semantic_memory.py, ~1296行)

#### 修改方法：

```python
def search(self, query: WMReadQuery, gaps: Sequence[str]) -> List[Dict[str, Any]]:
    """Search 操作：结合节点属性与关系生成可操作提示。
    
    新增功能：
        - 解析 entity gaps
        - 从 LTM 检索完整实体信息（位置、属性）
        - 返回结构化 hint
    """
```

#### 新增方法：

```python
def _parse_entity_gap(self, gap: str) -> Optional[Tuple[str, str]]:
    """解析 entity gap: 'entity:object:spatula' → ('object', 'spatula')"""

def _extract_position_from_node(self, node: LTNode) -> str:
    """从 LTM 节点提取位置信息。
    
    策略：
    1. 查找空间关系边 (on/in/at/near/inside/above/below/beside)
    2. 从 attributes 提取 spatial_position/last_seen_position
    3. 从 state_vars 提取（如果保留了 WM state）
    4. 回退：返回 "unknown"
    """

def _summarize_node_attributes(self, node: LTNode) -> str:
    """汇总节点的关键属性为简短字符串。
    
    示例：
        "color=silver/gray, size=medium, material=metal"
    """

def _collect_all_node_attributes(self, node: LTNode) -> Dict[str, List[str]]:
    """收集节点的所有属性值（用于 hint 详细信息）。
    
    返回：
        {
            "color": ["silver", "gray"],
            "size": ["medium"],
            "material": ["metal"],
            "position": ["on right counter"]
        }
    """
```

### 4. SemanticMemoryManager - WM 更新 (semantic_memory.py, ~2048行)

#### 修改方法：

```python
def _apply_attribute_hints_to_wm(
    self,
    hints: Sequence[Dict[str, Any]],
    step: int,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """应用 LTM hints 到 WM，支持属性 hints 和实体 hints。
    
    新增功能：
        - 识别 entity hints (entity_kind, entity_label)
        - 创建新 WM 节点或更新现有节点
        - 保留位置和属性信息
    """
```

#### 新增方法：

```python
def _create_unit_from_entity_hint(
    self,
    hint: Dict[str, Any],
    step: int,
) -> Optional[SemanticUnit]:
    """从 LTM entity hint 创建新的 WM 节点。
    
    构建内容：
    - state_vars: {
        "source": "ltm_entity_hint",
        "visible": False,  # 标记为需要探索确认
        "spatial_position": "on right counter",
        "colors": ["silver", "gray"],
        "size_hint": "medium",
        ...
      }
    - 创建 clip 用于溯源
    - TTL=120, confidence=0.7
    """

def _update_unit_from_entity_hint(
    self,
    unit: SemanticUnit,
    hint: Dict[str, Any],
    step: int,
) -> None:
    """用 LTM entity hint 更新已存在的 WM 节点。
    
    更新策略：
    - 只填充缺失的字段（不覆盖已有值）
    - 更新 recency
    - 添加 clip 引用
    """
```

### 5. 增强日志 (semantic_memory.py, ~3062行)

```python
def prepare_planner_context(self, goal: str) -> WMReadResult:
    """组合 read + LTM 检索，为 planner 生成最终上下文。
    
    新增日志：
    - 显示提取的 goal
    - 分类显示 gaps (entity / attribute / other)
    - 显示 LTM hints 数量
    - 显示 WM 更新结果
    """
```

---

## 数据流示例

### 场景：Move a spatula from the right counter

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: prepare_planner_context(goal)                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ extract_key_entities_from_goal(goal)                        │
│   → ["spatula", "right counter", "left counter"]           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ operator.read(query)                                         │
│   - select_nodes() → [right_counter, left_counter]         │
│   - _detect_missing_entities_in_goal()                     │
│     → spatula not in WM                                     │
│     → gap: "entity:object:spatula"                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ltm.search(query, gaps)                                      │
│   - _parse_entity_gap("entity:object:spatula")             │
│     → ("object", "spatula")                                 │
│   - _get_node_by_label("spatula")                          │
│     → LTM node found                                        │
│   - _extract_position_from_node()                          │
│     → "on right counter" (from edge or attributes)         │
│   - _summarize_node_attributes()                           │
│     → "color=silver/gray, size=medium, material=metal"    │
│   - Return hint:                                            │
│     {                                                        │
│       "entity_kind": "object",                              │
│       "entity_label": "spatula",                            │
│       "position": "on right counter",                       │
│       "attributes": {                                        │
│         "color": ["silver", "gray"],                        │
│         "size": ["medium"],                                 │
│         "material": ["metal"]                               │
│       },                                                     │
│       "summary": "spatula is a object. Last seen at: ..."  │
│     }                                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ _apply_attribute_hints_to_wm(hints, step)                   │
│   - Detect entity_hint (has entity_kind)                   │
│   - spatula not in WM → _create_unit_from_entity_hint()   │
│   - Create SemanticUnit:                                    │
│     {                                                        │
│       "id": "object_a7c3d5e2...",                           │
│       "kind": "object",                                     │
│       "label": "spatula",                                   │
│       "state_vars": {                                        │
│         "source": "ltm_entity_hint",                        │
│         "visible": False,                                   │
│         "spatial_position": "on right counter",            │
│         "colors": ["silver", "gray"],                      │
│         "size_hint": "medium",                             │
│         "materials": ["metal"]                             │
│       }                                                      │
│     }                                                        │
│   - graph.add_unit(spatula_unit, step=2)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Re-read WM (if updated)                                      │
│   - select_nodes() → [spatula, right_counter, left_counter]│
│   - No entity gaps!                                         │
│   - Return complete context to Planner                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键数据结构

### Entity Gap 格式

```
"entity:{kind}:{label}"

示例:
    "entity:object:spatula"
    "entity:place:drawer"
    "entity:object:red ball"
```

### Entity Hint 格式

```python
{
    "summary": "spatula is a object. Last seen at: on right counter. Attributes: color=silver/gray, size=medium, material=metal",
    "applicability": "entity:object:spatula",
    "entity_kind": "object",
    "entity_label": "spatula",
    "position": "on right counter",
    "attributes": {
        "color": ["silver", "gray"],
        "size": ["medium"],
        "material": ["metal"],
        "shape": ["flat"],
        "position": ["on right counter"]
    },
    "graph_path": [
        {
            "source": "spatula",
            "relation": "on",
            "target": "right counter",
            "confidence": 0.9,
            "target_kind": "place"
        }
    ]
}
```

---

## 日志输出示例

```
[SemanticMemoryManager] Preparing planner context for goal: Move a spatula from the right counter...
[SemanticMemoryManager] WM read result: nodes=2 gaps=3
[SemanticMemoryManager] Entity gaps detected: ['entity:object:spatula']
[SemanticMemoryManager] LTM search returned 3 hints
[SemanticMemoryManager] Created WM node from LTM entity hint: spatula (object)
[SemanticMemoryManager] Applied 3 hints to WM (1 units updated)
[SemanticMemoryManager] WM updated with LTM hints, re-reading...
[SemanticMemoryManager] Planner context prepared: nodes=3 gaps=0 hints=2
```

---

## 测试验证

### 测试脚本
文件：`test_entity_gap_detection.py`

### 测试用例

1. **Entity Gap Detection**
   - 输入：WM 中只有 right_counter, left_counter
   - Instruction: "Move a spatula from..."
   - 验证：检测到 `entity:object:spatula`

2. **LTM Position Retrieval**
   - LTM 中存在 spatula 节点，带位置 "on right counter"
   - 验证：LTM 返回包含位置的 hint

3. **WM Update**
   - 应用 hints 到 WM
   - 验证：WM 中创建了 spatula 节点
   - 验证：position 信息被正确设置
   - 验证：重新读取 WM 不再有 entity gap

---

## 后续优化建议

### 1. 位置置信度衰减
```python
position_age = current_step - node.attributes.get("last_seen_step", 0)
if position_age > 50:
    hint["summary"] += " (WARNING: position info may be outdated)"
    hint["confidence"] = max(0.3, 0.9 - position_age / 100)
```

### 2. 智能探索引导
```python
if ltm_hint["position"] and not wm_node.state_vars.get("visible"):
    exploration_hint = (
        f"Note: {ltm_hint['entity_label']} was previously observed "
        f"{ltm_hint['position']}, but is not currently visible. "
        f"Consider navigating to that location to verify."
    )
```

### 3. 多候选位置支持
```python
positions = [
    ("on right counter", 0.8, step=45),
    ("on table 2", 0.5, step=12),
]
hint["summary"] = (
    f"{label} has been observed at multiple locations: "
    f"most recently {positions[0][0]} (confidence {positions[0][1]})"
)
```

### 4. 冲突检测
```python
# 如果 WM 和 LTM 的位置信息冲突
if wm_position != ltm_position:
    logger.warning(
        f"Position conflict for {label}: WM={wm_position}, LTM={ltm_position}"
    )
    # 优先使用更新的信息
```

---

## 与现有系统的集成

### 无缝集成点

1. **prepare_planner_context**: 自动触发，无需额外调用
2. **ingest_perception**: 持续更新 WM/LTM，位置信息自动保存
3. **on_episode_end**: LTM consolidate 会保留位置信息

### 向后兼容

- 保留所有原有功能
- 只增加新的 gap 类型和 hint 类型
- 不影响现有的 attribute gap 处理

---

## 总结

✅ **已完成**：
- Entity 提取（从 instruction）
- Entity gap 检测（WM 中缺失的对象）
- LTM 位置检索（从历史观察）
- WM 节点创建/更新（带位置和属性）
- 完整日志追踪

✅ **关键优势**：
- 无侵入性（只扩展，不修改核心逻辑）
- 语义清晰（entity:kind:label 格式一致）
- 可扩展（支持未来新的 gap 类型）
- 可追溯（完整的 clip 引用链）

✅ **实际效果**：
现在当 agent 接收到 "Move a spatula from the right counter" 时：
1. 自动检测 WM 中没有 spatula
2. 从 LTM 检索 spatula 的历史位置
3. 创建 WM 节点并注入位置信息
4. Planner 收到完整上下文，可以直接生成探索/拿取计划
