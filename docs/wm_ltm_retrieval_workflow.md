# WM/LTM 检索工作流程文档

## 概述

当前系统实现了一个**自动的 WM→LTM 级联检索机制**，确保 agent 能够通过持续探索和记忆积累最终找到目标对象。

## 核心检索流程

### 1. Agent 查询 WM（Working Memory）

**触发点：** 每次规划前（`_plan_with_vlm` / `fusion.fuse`）

**入口函数：** `SemanticMemoryManager.prepare_planner_context(goal: str)`

**查询逻辑：**
```python
def prepare_planner_context(self, goal: str) -> WMReadResult:
    """组合 read + LTM 检索，为 planner 生成最终上下文。"""
    
    # 1. 构造 WM 读取查询
    query = WMReadQuery(
        goal=goal,  # e.g., "Move a spatula from the right counter to..."
        focus_kinds=["object", "place", "link", "rule"],
        risk_tags=["safety"],
        required_fields=["state_vars", "constraints"],
        max_nodes=12,
    )
    
    # 2. 从 WM 读取相关节点
    result = self.operator.read(query)
    
    # 3. 如果 WM 有缺口（gaps），触发 LTM 检索
    if result.gaps:
        hints = self.ltm.search(query, result.gaps)
        # 尝试用 LTM hints 补充 WM
        updated, passthrough = self._apply_attribute_hints_to_wm(hints, step=...)
        
        # 4. 如果 WM 更新了，重新读取
        if updated:
            result = self.operator.read(query)
            extra_hints = self.ltm.search(query, result.gaps)
            result.ltm_hints = passthrough + extra_hints
        else:
            result.ltm_hints = hints
    
    return result  # nodes + gaps + ltm_hints
```

### 2. WM Read 操作（检测缺口）

**函数：** `MemoryOperatorEngine.read(query: WMReadQuery)`

**检测缺口的策略：**

```python
# semantic_memory.py Line ~1670
def read(self, query: WMReadQuery) -> WMReadResult:
    # 1. 根据 goal 提取关键词，匹配 WM 节点
    tokens = self._tokenize_goal(query.goal)
    
    # 2. 按相关性排序节点（基于 label、perspective、state_vars 匹配）
    ranked_nodes = sorted(
        relevant_nodes,
        key=lambda node: self._score_node(node, tokens),
        reverse=True
    )
    
    # 3. 检查每个节点是否缺少 required_fields
    for node in selected_nodes:
        for field in query.required_fields:  # ["state_vars", "constraints"]
            if not node.state_vars:
                gaps.append(f"state_vars missing for {node.label}")
            if not node.constraints:
                gaps.append(f"constraints missing for {node.label}")
        
        # 4. 检查属性缺口
        for attr_kind in ATTRIBUTE_GAP_KINDS:  # color, feature, size, material, texture
            if not self._has_attribute(node, attr_kind):
                gaps.append(f"attribute:{attr_kind}:{node.label}")
    
    return WMReadResult(nodes=selected_nodes, gaps=gaps)
```

**缺口类型示例：**
- `"state_vars missing for spatula"` - 对象存在但缺少状态信息
- `"attribute:color:spatula"` - 对象存在但不知道颜色
- `"attribute:size:ball"` - 对象存在但不知道尺寸

### 3. LTM Search（知识检索）

**函数：** `LongTermMemory.search(query: WMReadQuery, gaps: Sequence[str])`

**检索策略：**

```python
def search(self, query: WMReadQuery, gaps: Sequence[str]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    
    # 1. 解析属性缺口 (e.g., "attribute:color:spatula")
    attribute_gaps: List[Tuple[str, str]] = []
    for gap in gaps:
        parsed = self._parse_attribute_gap(gap)  # -> ("color", "spatula")
        if parsed:
            attribute_gaps.append(parsed)
    
    # 2. 对每个属性缺口，从 LTM 检索
    for attr_kind, label in attribute_gaps:
        node = self._get_node_by_label(label)  # 在 LTM KG 中查找
        if not node:
            continue
        
        # 收集该对象的属性值
        values = self._collect_attribute_values(node, attr_kind)
        # 例如：spatula -> color: ["silver", "gray"]
        
        if values:
            hints.append({
                "summary": f"{label} typically has {attr_kind}: {', '.join(values[:3])}",
                "applicability": f"{attr_kind}:{label}",
                "attribute_values": values,
                "attribute_kind": attr_kind,
                "attribute_target": label,
            })
    
    # 3. 对其他缺口，基于图结构检索相关知识
    for gap in residual_gaps:
        # 排序节点，找最相关的
        ranked = sorted(
            self.nodes.values(),
            key=lambda node: self._score_node(node, perspective_tokens, gap),
            reverse=True
        )
        top_node = ranked[0]
        path = self._build_graph_path(top_node.id)
        hints.append({
            "summary": self._compose_summary(top_node, gap, path),
            "graph_path": path,
        })
    
    return hints
```

### 4. 应用 LTM Hints 到 WM

**函数：** `SemanticMemoryManager._apply_attribute_hints_to_wm(hints, step)`

```python
def _apply_attribute_hints_to_wm(self, hints, step):
    updated = 0
    passthrough = []
    
    for hint in hints:
        attr_kind = hint.get("attribute_kind")  # e.g., "color"
        target = hint.get("attribute_target")    # e.g., "spatula"
        values = hint.get("attribute_values")    # e.g., ["silver", "gray"]
        
        if not (attr_kind and target and values):
            passthrough.append(hint)
            continue
        
        # 1. 在 WM 中查找目标对象
        wm_node = self.graph._get_node_by_label(target)
        if not wm_node:
            passthrough.append(hint)
            continue
        
        # 2. 创建属性节点并链接到对象
        for value in values:
            attr_node_id = self.graph.write_node(
                kind=attr_kind,
                label=value,
                state_vars={"source": "ltm_hint"},
                perspective=["ltm", "attribute"],
                clip_id=...,
            )
            
            relation = ATTRIBUTE_HINT_RELATION.get(attr_kind, "has_property")
            self.graph.write_edge(
                source_id=wm_node.id,
                target_id=attr_node_id,
                relation=relation,  # e.g., "has_color"
                clip_id=...,
            )
            updated += 1
    
    return updated, passthrough
```

## 探索-记忆-检索闭环

### 场景：寻找 spatula

**Instruction:** `"Move a spatula from the right counter to the left counter."`

#### Step 1: 初始规划

```
Agent → WM: query("Move a spatula...")
WM Read: 找到 "right counter", "left counter"，但没有 "spatula"
WM Gaps: ["state_vars missing for spatula", "attribute:color:spatula", ...]

WM → LTM: search(gaps=["attribute:color:spatula", ...])
LTM Hints: [
    {"summary": "spatula typically has color: silver, gray", 
     "attribute_values": ["silver", "gray"]},
    {"summary": "spatula typically has size: medium",
     "attribute_values": ["medium"]},
]

WM: 将 LTM hints 应用到 WM（创建属性节点）
     现在 WM 知道：spatula 可能是银色/灰色，中等尺寸

Agent: 收到 planner context，包含：
    - WM nodes: [right counter, left counter, spatula(属性来自LTM)]
    - LTM hints: "spatula typically has color: silver, gray"

Planner: 生成计划 [navigate_kitchen, look_around, ...]
```

#### Step 2-5: 探索房间

```
每个 step 后：
    1. env 返回 observation image
    2. PerceptionModule.perceive(image) → 提取对象
       detected_objects: [
           {"label": "knife", "color": ["silver"], "size": "small"},
           {"label": "bowl", "color": ["white"], "size": "medium"},
       ]
    
    3. SemanticMemoryManager.ingest_perception() 
       → 为每个对象创建/更新 WM 节点
       → 存储视觉属性（color, size, material, etc.）
       
    4. 同步到 LTM (后台)
```

#### Step 6: 发现 spatula

```
Perception: detected_objects: [
    {"label": "spatula", "color": ["gray", "silver"], 
     "size": "medium", "position": "on right counter"}
]

SemanticMemoryManager.ingest_perception():
    1. 规范化 label: "spatula" → "spatula"
    2. 创建/更新 WM 节点:
       - kind: "object"
       - label: "spatula"
       - state_vars: {
           "colors": ["gray", "silver"],
           "size_hint": "medium",
           "visible": True,
           "spatial_position": "on right counter"
         }
    3. 创建属性边:
       spatula --has_color--> gray
       spatula --has_color--> silver
       spatula --has_size--> medium
    4. 更新索引:
       attribute_label_index["gray"] += {spatula_attr_node_id}
       attribute_object_index[(attr_id, "has_color")] += {spatula_id}
```

#### Step 7: 重新规划

```
Agent → WM: query("Move a spatula...")
WM Read: 
    - 找到 "spatula" 节点（刚刚创建）
    - state_vars 完整：colors=["gray","silver"], size_hint="medium", visible=True
    - 没有 gaps！

Agent: 收到 planner context，包含：
    - spatula (完整状态)
    - right counter (位置已知)
    - left counter (目标位置)

Planner: 生成精确计划 [
    navigate_to_right_counter,
    pick_up_spatula,
    navigate_to_left_counter,
    place_spatula
]
```

## 属性匹配机制

当指令包含描述性语言时（e.g., "a small red object with green top"），系统使用属性匹配：

### 1. 提取属性提示

**函数：** `_extract_attribute_hints(text: str)`

```python
def _extract_attribute_hints(self, text: str) -> Dict[str, List[str]]:
    hints: Dict[str, List[str]] = defaultdict(list)
    
    # 1. 提取颜色
    for color in COLOR_KEYWORDS:  # red, green, blue, ...
        if color in text.lower():
            hints["color"].append(color)
    
    # 2. 提取尺寸
    for size in SIZE_KEYWORDS:  # small, medium, large, ...
        if size in text.lower():
            hints["size"].append(size)
    
    # 3. 提取特征短语
    feature_phrases = self._extract_feature_phrases(text)
    # e.g., "with green top" → ["green top"]
    hints["feature"].extend(feature_phrases)
    
    return hints
```

### 2. 基于属性查询 WM

**函数：** `WorkingMemoryGraph.query_by_attribute_hints(attribute_hints)`

```python
def query_by_attribute_hints(
    self, 
    attribute_hints: Dict[str, List[str]],  # {"color": ["red"], "feature": ["green top"]}
    limit: int = 5
) -> List[str]:
    
    candidate_sets: List[Set[str]] = []
    
    for hint_type, values in attribute_hints.items():
        object_ids: Set[str] = set()
        
        for value in values:  # e.g., "red"
            norm = self._normalize_label(value)
            
            # 1. 直接索引匹配
            attr_node_ids = self.attribute_label_index.get(norm, set())
            for attr_id in attr_node_ids:
                relation = ATTRIBUTE_HINT_RELATION[hint_type]  # "has_color"
                key = (attr_id, relation)
                object_ids.update(self.attribute_object_index[key])
                # 找到所有 has_color --> "red" 的对象
            
            # 2. 同义词扩展（feature 类型）
            if hint_type == "feature" and not object_ids:
                for word in norm.split():
                    synonyms = FEATURE_SYNONYMS.get(word, set())
                    # e.g., "top" → {"leaves", "cap", "crown"}
                    for syn in synonyms:
                        # 重复索引查询...
            
            # 3. 模糊匹配
            if not object_ids:
                for attr_label in self.attribute_label_index.keys():
                    if norm in attr_label or attr_label in norm:
                        # 部分匹配也算
        
        if object_ids:
            candidate_sets.append(object_ids)
    
    # 4. 组合逻辑：优先 AND（交集），回退 OR（并集）
    if len(candidate_sets) == 1:
        combined = candidate_sets[0]
    else:
        intersection = set.intersection(*candidate_sets)
        combined = intersection if intersection else set.union(*candidate_sets)
    
    # 5. 按中心度排序，返回 top N
    ranked = sorted(combined, key=lambda id: self.nodes[id].centrality, reverse=True)
    return [self.nodes[id].label for id in ranked[:limit]]
```

### 3. 规范化对象标签

**函数：** `_canonicalize_label(label: str)`

```python
def _canonicalize_label(self, label: str) -> str:
    # 1. 尝试从 LTM 解析
    canonical = resolve_canonical_label(label)
    if canonical:
        return canonical
    
    # 2. 基于属性匹配
    hints = self._extract_attribute_hints(label)
    if hints:
        matches = self.graph.query_by_attribute_hints(hints, limit=1)
        if matches:
            return matches[0]
    
    # 3. 回退：返回原标签
    return label.lower().strip()
```

**示例：**

```
输入: "a small red object with green top"

Step 1: 提取属性
    hints = {
        "color": ["red", "green"],
        "size": ["small"],
        "feature": ["green top"]
    }

Step 2: 查询 WM
    - color="red" → 找到对象 {A, B, C}
    - color="green" → 找到对象 {C, D}
    - size="small" → 找到对象 {C, E}
    - feature="green top" → 找到对象 {C, F}（通过同义词 "top"↔"leaves"）
    
    交集: {C}
    
Step 3: 检查对象 C 的 label
    → "strawberry"

输出: "strawberry"
```

## 关键数据结构

### WM 索引

```python
# semantic_memory.py Line ~820
class WorkingMemoryGraph:
    # 节点存储
    nodes: Dict[str, MemoryNode]  # {node_id: MemoryNode}
    edges: Dict[str, MemoryEdge]  # {edge_id: MemoryEdge}
    
    # 标签索引（快速按 label 查找）
    label_index: Dict[str, Set[str]]  # {normalized_label: {node_id, ...}}
    
    # 属性索引（快速按属性查找）
    attribute_label_index: Dict[str, Set[str]]  
    # {normalized_attr_label: {attr_node_id, ...}}
    # 例如: {"red": {"color_node_123", "color_node_456"}}
    
    attribute_object_index: Dict[Tuple[str, str], Set[str]]
    # {(attr_node_id, relation): {object_node_id, ...}}
    # 例如: {("color_node_123", "has_color"): {"obj_789", "obj_012"}}
```

### MemoryNode 结构

```python
@dataclass
class MemoryNode:
    id: str
    kind: str  # "object", "place", "link", "rule", "color", "size", ...
    label: str  # "spatula", "red", "small", ...
    state_vars: Dict[str, Any]  # 动态状态
    constraints: List[str]
    attributes: Dict[str, Any]  # 静态属性
    perspective: List[str]  # ["perception", "execution", "ltm", ...]
    clip_ids: List[str]  # 证据溯源
    recency: float  # 最近访问时间
    utility: float  # 重要性得分
    centrality: float  # 图中心度
```

## 总结

### 检索链路

```
Agent Planning
    ↓
prepare_planner_context(instruction)
    ↓
WM.read(query)
    ↓
检测 gaps（缺失的 state_vars / attributes）
    ↓
LTM.search(query, gaps)
    ↓
返回 hints（属性值、关系路径）
    ↓
_apply_attribute_hints_to_wm(hints)
    ↓
更新 WM（创建属性节点/边）
    ↓
重新 WM.read(query)
    ↓
返回完整 context 给 Agent
```

### 记忆更新链路

```
每个 Step
    ↓
env.step() → observation image
    ↓
PerceptionModule.perceive(image)
    ↓
提取 detected_objects（带视觉属性）
    ↓
ingest_perception(detected_objects)
    ↓
创建/更新 WM 节点（对象 + 属性）
    ↓
更新索引（label_index, attribute_label_index, attribute_object_index）
    ↓
同步到 LTM（后台，on_episode_end 时整合）
    ↓
下次查询时可用
```

### 为什么一定能找到 spatula

1. **持续探索**：Agent 规划会生成 `look_around`, `navigate_to_*` 等探索动作
2. **完整记录**：每个 step 的 image 都会被感知模块处理，提取所有可见对象
3. **累积记忆**：所有对象都会进入 WM，带有完整的视觉属性（颜色、尺寸、位置）
4. **智能检索**：
   - 如果对象未见过，LTM 提供先验知识（"spatula 通常是银色"）
   - 如果对象已见过，WM 直接返回位置和属性
5. **属性匹配**：即使指令用描述性语言（"银色的扁平工具"），也能通过属性索引匹配到 "spatula"
6. **反馈循环**：无效动作会触发 `reperceive`/`replan`，强制 agent 重新探索

## 下一步优化建议

1. **主动探索策略**：当 WM gaps 显示缺少某对象时，优先规划探索动作
2. **空间记忆**：增强 `place` 节点的拓扑关系（"kitchen adjacent to living_room"）
3. **时间衰减**：对长时间未见的对象降低可见性权重
4. **置信度传播**：LTM hints 应携带置信度，避免误导
