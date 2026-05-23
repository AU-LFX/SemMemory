# LTM 位置检索缺陷分析与解决方案

## 问题场景

**Instruction:** `"Move a spatula from the right counter to the right receptacle of the left counter."`

**当前状态:** WM 中没有记录 spatula 的任何信息

**期望行为:** 系统应该能从 LTM 检索到 spatula 的位置信息

**实际问题:** **当前实现无法直接从 LTM 检索 spatula 的位置！**

---

## 根本原因分析

### 1. WM Read 的查询逻辑缺陷

#### 当前实现 (`select_nodes`)

```python
def select_nodes(self, query: WMReadQuery) -> List[SemanticUnit]:
    """根据查询关注的 kind 与 utility 提取一组节点。"""
    focus = query.focus_kinds or list(self.by_kind.keys())
    selected: List[SemanticUnit] = []
    
    # ❌ 问题：只按 kind 筛选，不解析 goal 中的关键实体！
    for kind in focus:
        for unit_id in self.by_kind.get(kind, []):
            selected.append(self.nodes[unit_id])
    
    selected.sort(key=lambda u: (self.utility.get(u.id, 0), u.metadata.recency), reverse=True)
    return selected[: query.max_nodes]
```

#### 问题所在

```
query.goal = "Move a spatula from the right counter..."
query.focus_kinds = ["object", "place", "link", "rule"]

select_nodes 的行为:
    → 返回所有 kind="object" 的节点（按 utility 排序）
    → 返回所有 kind="place" 的节点
    → ...

结果：
    如果 WM 中没有 "spatula" 节点，select_nodes 返回的 nodes 列表中就不包含它
    → read() 检测不到 "spatula missing" 的 gap
    → LTM.search() 永远不会被触发去检索 spatula 的位置
```

**核心问题：`select_nodes` 不解析 `goal` 文本，不提取关键实体（spatula, right counter, left counter），因此无法发现"instruction 中提到但 WM 中不存在"的对象！**

---

### 2. Gap 检测的局限性

#### 当前 Gap 检测逻辑

```python
def read(self, query: WMReadQuery) -> WMReadResult:
    nodes = self.graph.select_nodes(query)  # ← 只返回已存在的节点
    
    required = query.required_fields or ["state_vars"]
    gaps: List[str] = []
    
    # Gap 检测 1: 缺失字段
    for req in required:
        missing = [node.primary_label() for node in nodes if req not in node.symbol.state_vars]
        if missing:
            gaps.append(f"{req} missing for {', '.join(missing[:4])}")
    
    # Gap 检测 2: 缺失属性
    gaps.extend(self._collect_attribute_gaps(nodes))
    
    return WMReadResult(nodes=nodes, edges=edges, gaps=gaps)
```

**问题：Gap 检测只针对 `nodes` 列表中已存在的节点，无法检测"指令中提到但 WM 中完全不存在"的对象！**

**示例：**

```
Instruction: "Move a spatula from..."
WM 中存在: [right_counter, left_counter, apple, bowl]
WM 中不存在: spatula

select_nodes 返回: [right_counter, left_counter, apple, bowl]

Gap 检测结果:
    ✓ "state_vars missing for right_counter" (如果 right_counter 缺少 state_vars)
    ✓ "attribute:color:apple" (如果 apple 缺少 color)
    ✗ spatula 完全不在 nodes 列表中，所以不会生成任何 gap！
```

---

### 3. LTM Search 的触发条件

```python
def prepare_planner_context(self, goal: str) -> WMReadResult:
    result = self.operator.read(query)
    
    # ❌ 只有当 result.gaps 存在时才触发 LTM 检索
    if result.gaps:
        hints = self.ltm.search(query, result.gaps)
        ...
    
    return result
```

**问题链：**

```
WM 中没有 spatula 
    ↓
select_nodes 不返回 spatula 
    ↓
read() 检测不到 spatula 相关的 gap 
    ↓
result.gaps = [] 
    ↓
ltm.search() 不被调用 
    ↓
LTM 中的 spatula 位置信息无法被检索
```

---

## 解决方案设计

### 方案 1: 增强 `select_nodes` - 提取指令中的关键实体

#### 实现逻辑

```python
def select_nodes(self, query: WMReadQuery) -> List[SemanticUnit]:
    """根据查询关注的 kind 与 utility 提取一组节点，并尝试匹配 goal 中的关键实体。"""
    
    # 1. 提取 goal 中的关键实体
    key_entities = self._extract_key_entities_from_goal(query.goal)
    # 例如: ["spatula", "right counter", "left counter"]
    
    # 2. 按 kind 筛选已存在的节点
    focus = query.focus_kinds or list(self.by_kind.keys())
    selected: List[SemanticUnit] = []
    matched_labels: Set[str] = set()
    
    for kind in focus:
        for unit_id in self.by_kind.get(kind, []):
            unit = self.nodes[unit_id]
            selected.append(unit)
            matched_labels.add(unit.primary_label().lower())
    
    # 3. 为 goal 中提到但 WM 中不存在的实体创建 placeholder 节点
    for entity in key_entities:
        normalized = entity.lower().strip()
        if normalized not in matched_labels:
            # 创建 placeholder 节点（标记为 "missing_in_wm"）
            placeholder = self._create_placeholder_unit(
                label=entity,
                kind=self._infer_kind_from_entity(entity),
                provenance="goal_extraction"
            )
            selected.append(placeholder)
    
    selected.sort(key=lambda u: (self.utility.get(u.id, 0), u.metadata.recency), reverse=True)
    return selected[: query.max_nodes]

def _extract_key_entities_from_goal(self, goal: str) -> List[str]:
    """从 goal 文本中提取关键对象/地点名称。"""
    entities: List[str] = []
    
    # 1. 使用正则模式提取动作目标
    for pattern, kind in ACTION_ENTITY_PATTERNS:
        # e.g., "move a spatula from..." → "spatula"
        match = pattern.search(goal)
        if match:
            entities.append(match.group("label"))
    
    # 2. 提取常见介词短语中的实体
    # "from the right counter" → "right counter"
    # "to the left counter" → "left counter"
    prep_patterns = [
        r"from (?:the\s+)?(?P<label>[\w\s]+?)(?:\s+to|\s+and|$)",
        r"to (?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+from|$)",
        r"on (?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|$)",
        r"in (?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|$)",
    ]
    for pattern_str in prep_patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for match in pattern.finditer(goal):
            entities.append(match.group("label").strip())
    
    # 3. 去重并返回
    return list(dict.fromkeys(entities))

def _create_placeholder_unit(self, label: str, kind: str, provenance: str) -> SemanticUnit:
    """为 goal 中提到但 WM 中不存在的实体创建占位符节点。"""
    unit_id = f"placeholder_{hashlib.sha1(label.encode()).hexdigest()[:8]}"
    
    return SemanticUnit(
        id=unit_id,
        kind=kind,
        symbol=SemanticSymbol(
            type_candidates=[TypeCandidate(label=label, confidence=0.4)],
            state_vars={"missing_in_wm": True},  # ← 标记为缺失
        ),
        metadata=SemanticMetadata(
            confidence=0.4,
            recency=0,
            ttl=60,
            provenance=provenance,
        ),
        clip_refs=[],
    )

def _infer_kind_from_entity(self, entity: str) -> str:
    """根据实体名称推断 kind。"""
    place_keywords = ["counter", "table", "shelf", "cabinet", "room", "kitchen", "bedroom"]
    entity_lower = entity.lower()
    
    for keyword in place_keywords:
        if keyword in entity_lower:
            return "place"
    
    return "object"
```

#### 效果

```
Instruction: "Move a spatula from the right counter..."

select_nodes 执行过程:
    1. 提取关键实体: ["spatula", "right counter", "left counter"]
    2. 从 WM 匹配现有节点:
       - ✓ right_counter (已存在)
       - ✓ left_counter (已存在)
       - ✗ spatula (不存在)
    3. 为 spatula 创建 placeholder 节点:
       {
         "id": "placeholder_a7c3d5e2",
         "kind": "object",
         "label": "spatula",
         "state_vars": {"missing_in_wm": True}
       }
    4. 返回: [right_counter, left_counter, spatula_placeholder]

read() 检测 gaps:
    - spatula 节点存在，但 state_vars 只有 "missing_in_wm"
    - ✓ Gap: "state_vars missing for spatula"
    - ✓ Gap: "attribute:color:spatula"
    - ✓ Gap: "attribute:size:spatula"

prepare_planner_context:
    result.gaps = ["state_vars missing for spatula", ...]
    ↓
    触发 ltm.search(query, gaps)
    ↓
    LTM 返回 spatula 的先验知识和历史位置
```

---

### 方案 2: 增强 Gap 检测 - 直接从 goal 生成 "entity missing" gap

#### 实现逻辑

```python
def read(self, query: WMReadQuery) -> WMReadResult:
    nodes = self.graph.select_nodes(query)
    
    required = query.required_fields or ["state_vars"]
    gaps: List[str] = []
    
    # 原有逻辑：检测已存在节点的缺失字段
    for req in required:
        missing = [node.primary_label() for node in nodes if req not in node.symbol.state_vars]
        if missing:
            gaps.append(f"{req} missing for {', '.join(missing[:4])}")
    
    gaps.extend(self._collect_attribute_gaps(nodes))
    
    # ✨ 新增：检测 goal 中提到但 WM 中不存在的实体
    goal_entity_gaps = self._detect_missing_entities_in_goal(query.goal, nodes)
    gaps.extend(goal_entity_gaps)
    
    edges = self.graph.get_edges([node.id for node in nodes])
    return WMReadResult(nodes=nodes, edges=edges, gaps=gaps)

def _detect_missing_entities_in_goal(
    self, 
    goal: str, 
    existing_nodes: List[SemanticUnit]
) -> List[str]:
    """检测 goal 中提到但 existing_nodes 中不存在的实体。"""
    
    # 1. 提取 goal 中的关键实体
    key_entities = self._extract_key_entities_from_goal(goal)
    
    # 2. 构建已存在节点的标签集合
    existing_labels = {
        node.primary_label().lower().strip()
        for node in existing_nodes
    }
    
    # 3. 找出缺失的实体
    missing_gaps: List[str] = []
    for entity in key_entities:
        normalized = entity.lower().strip()
        if normalized not in existing_labels:
            # 生成 gap: "entity:object:spatula"
            kind = self._infer_kind_from_entity(entity)
            missing_gaps.append(f"entity:{kind}:{entity}")
    
    return missing_gaps

# _extract_key_entities_from_goal 同方案 1
```

#### LTM Search 增强

```python
def search(self, query: WMReadQuery, gaps: Sequence[str]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    
    attribute_gaps: List[Tuple[str, str]] = []
    entity_gaps: List[Tuple[str, str]] = []  # ← 新增
    residual_gaps: List[str] = []
    
    for gap in gaps:
        # 解析属性缺口
        attr_parsed = self._parse_attribute_gap(gap)  # "attribute:color:spatula"
        if attr_parsed:
            attribute_gaps.append(attr_parsed)
            continue
        
        # ✨ 解析实体缺口
        entity_parsed = self._parse_entity_gap(gap)  # "entity:object:spatula"
        if entity_parsed:
            entity_gaps.append(entity_parsed)
            continue
        
        residual_gaps.append(gap)
    
    # 处理实体缺口：从 LTM 检索完整节点信息
    for kind, label in entity_gaps:
        node = self._get_node_by_label(label)
        if not node:
            continue
        
        # 构建完整提示（包括位置信息）
        position_info = self._extract_position_from_node(node)
        attribute_summary = self._summarize_node_attributes(node)
        
        hints.append({
            "summary": (
                f"{label} is a {kind}. "
                f"Last seen at: {position_info}. "
                f"Attributes: {attribute_summary}"
            ),
            "applicability": f"entity:{kind}:{label}",
            "entity_kind": kind,
            "entity_label": label,
            "position": position_info,
            "attributes": self._collect_all_attributes(node),
            "graph_path": self._build_graph_path(node.id),
        })
    
    # 原有的属性缺口处理...
    for attr_kind, label in attribute_gaps:
        ...
    
    return hints

def _parse_entity_gap(self, gap: str) -> Optional[Tuple[str, str]]:
    """解析 entity gap: 'entity:object:spatula' → ('object', 'spatula')"""
    if not isinstance(gap, str) or not gap.startswith("entity:"):
        return None
    parts = gap.split(":", 2)
    if len(parts) != 3:
        return None
    return (parts[1].strip(), parts[2].strip())

def _extract_position_from_node(self, node: LTNode) -> str:
    """从 LTM 节点提取位置信息。"""
    # 查找 "on", "in", "at" 等空间关系边
    for edge in self.edges.values():
        if edge.source != node.id:
            continue
        if edge.relation in {"on", "in", "at", "near", "inside"}:
            target_node = self.nodes.get(edge.target)
            if target_node:
                return f"{edge.relation} {target_node.label}"
    
    # 从 state_vars 提取
    position = node.attributes.get("spatial_position") or node.attributes.get("last_seen_position")
    if position:
        return str(position)
    
    return "unknown"

def _summarize_node_attributes(self, node: LTNode) -> str:
    """汇总节点的关键属性。"""
    parts: List[str] = []
    
    # 颜色
    colors = self._collect_attribute_values(node, "color")
    if colors:
        parts.append(f"color={'/'.join(colors[:2])}")
    
    # 尺寸
    sizes = self._collect_attribute_values(node, "size")
    if sizes:
        parts.append(f"size={sizes[0]}")
    
    # 材质
    materials = self._collect_attribute_values(node, "material")
    if materials:
        parts.append(f"material={'/'.join(materials[:2])}")
    
    return ", ".join(parts) if parts else "no detailed attributes"
```

#### 效果

```
Instruction: "Move a spatula from the right counter..."

WM.read() 执行:
    1. select_nodes 返回: [right_counter, left_counter, apple]
    2. 检测缺失实体:
       - 提取 goal 实体: ["spatula", "right counter", "left counter"]
       - 已存在: {right_counter, left_counter}
       - 缺失: spatula
    3. 生成 gap: "entity:object:spatula"
    4. result.gaps = ["entity:object:spatula"]

LTM.search(gaps=["entity:object:spatula"]):
    1. 解析 entity gap: ("object", "spatula")
    2. 从 LTM 查找 spatula 节点
    3. 提取位置: "on right counter" (来自历史观察)
    4. 提取属性: "color=silver/gray, size=medium, material=metal"
    5. 返回 hint:
       {
         "summary": "spatula is a object. Last seen at: on right counter. Attributes: color=silver/gray, size=medium, material=metal",
         "applicability": "entity:object:spatula",
         "position": "on right counter",
         "attributes": {...}
       }

prepare_planner_context:
    1. 应用 hint 到 WM:
       - 创建 spatula 节点
       - 设置 state_vars: {spatial_position: "on right counter", colors: ["silver", "gray"], size_hint: "medium"}
    2. 重新 read()
    3. 返回完整上下文（包含 spatula 位置）
```

---

## 推荐方案

**推荐使用方案 2（增强 Gap 检测）**

### 理由

1. **侵入性更小**：不需要修改 `select_nodes` 的核心逻辑，只在 `read()` 和 `ltm.search()` 中增加新的 gap 类型处理
2. **语义更清晰**：`"entity:object:spatula"` 明确表示"指令中提到的实体在 WM 中缺失"，与现有的 `"attribute:color:spatula"` 形式一致
3. **扩展性更好**：未来可以支持更多 gap 类型（`"relation:inside:box"`, `"state:opened:door"` 等）
4. **不污染 WM**：方案 1 需要创建 placeholder 节点，可能影响 WM 的统计和剪枝逻辑

### 实现步骤

1. ✅ 在 `SemanticWMGraph` 中添加 `_extract_key_entities_from_goal()` 和 `_infer_kind_from_entity()`
2. ✅ 在 `MemoryOperatorEngine.read()` 中添加 `_detect_missing_entities_in_goal()` 调用
3. ✅ 在 `LTMService.search()` 中添加 `_parse_entity_gap()` 和实体检索逻辑
4. ✅ 在 `LTMService` 中添加 `_extract_position_from_node()` 和 `_summarize_node_attributes()`

---

## 测试案例

### Case 1: 基础对象缺失

```
Instruction: "Move a spatula from the right counter..."
WM 初始状态: {right_counter, left_counter, apple}

期望行为:
    1. read() 检测到 gap: "entity:object:spatula"
    2. LTM 返回: "spatula last seen at: on right counter"
    3. WM 创建 spatula 节点，position="on right counter"
    4. Planner 收到完整上下文，生成精确计划
```

### Case 2: 对象已在 WM，但缺少属性

```
Instruction: "Pick up the red ball"
WM 初始状态: {ball (no color info)}

期望行为:
    1. read() 检测到 gap: "attribute:color:ball"
    2. LTM 返回: "ball typically has color: red"
    3. WM 更新 ball 节点，color=["red"]
```

### Case 3: 对象和位置都缺失

```
Instruction: "Put the knife in the drawer"
WM 初始状态: {table, chair}

期望行为:
    1. read() 检测到 gaps: ["entity:object:knife", "entity:place:drawer"]
    2. LTM 返回:
       - "knife last seen at: on kitchen counter"
       - "drawer is part of: left cabinet"
    3. WM 创建 knife 和 drawer 节点
```

---

## 后续优化

### 1. 智能探索引导

当 LTM 提示 "spatula last seen at: on right counter" 但当前视野看不到时：

```python
# 在 planning prompt 中注入探索提示
if ltm_hint["position"] and not wm_node.state_vars.get("visible"):
    exploration_hint = (
        f"Note: {ltm_hint['entity_label']} was previously observed "
        f"{ltm_hint['position']}, but is not currently visible. "
        f"Consider navigating to that location to verify."
    )
```

### 2. 位置置信度衰减

```python
# LTM 节点携带时间戳
position_age = current_step - node.attributes.get("last_seen_step", 0)
if position_age > 50:
    hint["summary"] += " (WARNING: position info may be outdated)"
    hint["confidence"] = max(0.3, 0.9 - position_age / 100)
```

### 3. 多候选位置

```python
# 如果 LTM 中记录了 spatula 在多个位置出现过
positions = [
    ("on right counter", 0.8, step=45),
    ("on table 2", 0.5, step=12),
]

hint["summary"] = (
    f"spatula has been observed at multiple locations: "
    f"most recently {positions[0][0]} (confidence {positions[0][1]}), "
    f"also seen at {positions[1][0]}"
)
```

---

## 总结

当前系统的主要缺陷是 **WM Read 操作无法检测"指令中提到但 WM 中不存在"的实体**，导致 LTM 检索永远不会被触发。

解决方案是在 `read()` 中增加 **entity gap 检测**，并在 `ltm.search()` 中增加 **实体检索和位置提取逻辑**。

这样，即使 WM 中没有 spatula，系统也能：
1. 从 instruction 中提取 "spatula"
2. 检测到 "entity:object:spatula" 缺口
3. 从 LTM 检索 spatula 的历史位置和属性
4. 将信息注入 WM
5. 生成精确的探索和操作计划
