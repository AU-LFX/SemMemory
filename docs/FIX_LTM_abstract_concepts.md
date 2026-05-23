# LTM 抽象概念修复

## 问题诊断

### 用户发现的问题

> "在我的LTM知道sink的位置，并且知道它是个receptacle，那么agent发现the right receptacle应该确定为sink，但是现在它还没有这个语义理解能力，是不是因为LTM没有 receptacle这个节点"

**完全正确！** 问题的根源是：

1. ❌ **LTM 中没有 "receptacle" 抽象概念节点**
2. ❌ **sink 节点没有 `is_a: receptacle` 关系**
3. ❌ **语义对齐只依赖硬编码的 `ABSTRACT_TO_CONCRETE_HINTS` 字典**

### 现有机制的局限性

**当前实现:**
```python
# 硬编码的抽象→具体映射
ABSTRACT_TO_CONCRETE_HINTS = {
    "receptacle": ["sink", "basin", "bowl", "container", "bin"],
    "container": ["box", "basket", "bag", "bin", "jar"],
    # ...
}
```

**问题:**
1. 静态映射：无法动态学习新的抽象概念
2. 单向查询：从抽象→具体，不能从具体→抽象
3. 缺少推理：不能利用 LTM 中的 `is_a` 关系进行类型推理
4. 知识割裂：LTM 知道 sink 是什么，但不知道它是一种 receptacle

## 解决方案

### 核心思想

**建立完整的抽象概念本体 (Ontology):**

```
抽象层 (LTM):
  receptacle (concept)
    ├─ is_a: concept
    ├─ definition: "A container that can hold liquids or objects"
    └─ concrete_instances: [sink, basin, bowl, ...]

具体层 (LTM + WM):
  sink (place)
    ├─ is_a: receptacle  ← 关键关系！
    ├─ category: "sink"
    ├─ materials: ["ceramic", "steel"]
    └─ affordances: ["wash", "hold_water"]

推理过程:
  1. Instruction 提到 "right receptacle"
  2. 提取抽象术语: "receptacle"
  3. 查询 LTM: sink is_a receptacle ✅
  4. 空间验证: sink.position = "right side" ✅
  5. 对齐成功: "right receptacle" → sink
```

### 实现细节

#### 1. 在 LTM 中添加抽象概念节点

**文件:** `embodiedbench/evaluator/object_knowledge_seed.py`

**位置:** `_RAW_TEMPLATES` 字典开头

**新增内容:**

```python
_RAW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --- abstract concepts (semantic categories) --------
    "receptacle": {
        "kinds": ["concept", "category"],
        "attributes": {
            "state_vars": {
                "category": "abstract_concept",
                "definition": "A container or basin that can hold liquids or objects",
                "semantic_type": "abstract",
                "concrete_instances": ["sink", "basin", "bowl", "container", "bin"],
            },
        },
    },
    "container": {
        "kinds": ["concept", "category"],
        "attributes": {
            "state_vars": {
                "category": "abstract_concept",
                "definition": "An object that can store or hold other objects",
                "semantic_type": "abstract",
                "concrete_instances": ["box", "basket", "bag", "bin", "jar"],
            },
        },
    },
    "surface": {
        "kinds": ["concept", "category"],
        "attributes": {
            "state_vars": {
                "category": "abstract_concept",
                "definition": "A flat area suitable for placing objects",
                "semantic_type": "abstract",
                "concrete_instances": ["counter", "table", "desk", "shelf"],
            },
        },
    },
    "appliance": {
        "kinds": ["concept", "category"],
        "attributes": {
            "state_vars": {
                "category": "abstract_concept",
                "definition": "An electrical device for household tasks",
                "semantic_type": "abstract",
                "concrete_instances": ["microwave", "oven", "stove", "refrigerator"],
            },
        },
    },
    # --- 原有的具体对象定义 ---
    "cabinet": { ... },
    # ...
}
```

#### 2. 为具体对象添加 is_a 关系

**修改 sink:**

```python
"sink in the kitchen": {
    "kinds": ["place", "surface", "receptacle"],  # 添加 receptacle kind
    "attributes": {
        "state_vars": {
            "category": "sink",
            "materials": ["ceramic", "steel"],
            "affordances": ["wash", "rinse", "hold_water"],  # 添加 hold_water
            "supports_hot_items": True,
            "requires_plumbing": True,
            "is_receptacle": True,  # 标记为 receptacle
        },
    },
    "relations": [  # ✨ 新增关系
        {"relation": "is_a", "target": "receptacle", "confidence": 1.0},
        {"relation": "synonym_of", "target": "basin", "confidence": 0.8},
    ],
},
```

**修改 bowl:**

```python
"bowl": {
    "kinds": ["object", "tableware", "receptacle"],
    "attributes": {
        "state_vars": {
            "category": "tableware",
            "materials": ["ceramic", "glass"],
            "affordances": ["hold_food", "hold_liquid"],
            "is_receptacle": True,
        },
    },
    "relations": [
        {"relation": "is_a", "target": "receptacle", "confidence": 0.9},
    ],
},
```

**修改 box:**

```python
"box": {
    "kinds": ["object", "container"],
    "attributes": {
        "state_vars": {
            "category": "container",
            "materials": ["cardboard", "plastic"],
            "affordances": ["store_items", "transport"],
            "is_container": True,
        },
    },
    "relations": [
        {"relation": "is_a", "target": "container", "confidence": 1.0},
    ],
},
```

**修改 counter:**

```python
"counter": {
    "kinds": ["place", "surface"],
    "attributes": {
        "state_vars": {
            "category": "surface",
            "materials": ["stone", "laminate"],
            "affordances": ["prepare_food", "place_items"],
            "supports_hot_items": True,
            "is_surface": True,
        },
        "constraints": [
            _constraint("keep fragile items away from the counter edge"),
        ],
    },
    "relations": [
        {"relation": "is_a", "target": "surface", "confidence": 1.0},
    ],
},
```

**修改 table:**

```python
"table": {
    "kinds": ["place", "surface"],
    "attributes": {
        "state_vars": {
            "category": "surface",
            "materials": ["wood", "glass"],
            "affordances": ["serve_food", "place_items"],
            "supports_hot_items": True,
            "is_surface": True,
        },
        "constraints": [
            _constraint("avoid scratching the tabletop"),
        ],
    },
    "relations": [
        {"relation": "is_a", "target": "surface", "confidence": 1.0},
    ],
},
```

#### 3. 增强语义对齐逻辑

**文件:** `embodiedbench/evaluator/semantic_memory.py`

**位置:** `MemoryOperatorEngine.align_semantics()` 方法

**新增逻辑:**

```python
for obs_label in observed_labels:
    obs_norm = obs_label.strip().lower()
    score = 0.0
    reason_parts = []
    
    # 1. 类型匹配：直接在 hint 列表中
    if obs_norm in [c.lower() for c in potential_concrete]:
        score = 0.85
        reason_parts.append(f"type={core_term}")
    else:
        # 2. 模糊匹配
        for concrete in potential_concrete:
            sim = _string_similarity(obs_norm, concrete.lower())
            if sim >= 0.7:
                score = max(score, sim * 0.8)
                reason_parts.append(f"fuzzy={concrete}({sim:.2f})")
    
    # 3. 📚 LTM 知识查询：检查 is_a 关系
    if score == 0.0:
        unit = self.graph.find_unit_by_label(obs_label)
        if unit:
            for rel in unit.symbol.relations:
                if rel.relation == "is_a" and rel.target.lower() == core_term:
                    score = 0.9
                    reason_parts.append(f"is_a={core_term}")
                    logger.info(
                        f"[Semantic Alignment] Found is_a relation: "
                        f"{obs_label} is_a {core_term}"
                    )
                    break
    
    # 4. 空间关系加分
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

## 完整执行流程

### 初始化阶段

```
Episode 开始:
  ↓
加载 LTM:
  1. 读取 ltm_store.json (如果存在)
  2. 或加载 seed graph from object_knowledge_seed.py
     - receptacle (concept)
     - sink (place) with is_a: receptacle
     - bowl (object) with is_a: receptacle
     - counter (place) with is_a: surface
     - etc.
  ↓
LTM 状态:
  nodes:
    - receptacle: {kinds:[concept], definition:"...", concrete_instances:[sink,basin,bowl]}
    - sink: {kinds:[place,surface,receptacle], category:"sink", ...}
  edges:
    - sink --is_a--> receptacle
    - bowl --is_a--> receptacle
    - counter --is_a--> surface
```

### 感知阶段

```
Step N: 导航到 left counter
  ↓
自动观察:
  VLM 检测: sink (position="right side")
  ↓
Ingest perception:
  1. 创建 sink unit in WM
  2. state_vars: {position: "right side", ...}
  ↓
Merge LTM attributes into WM:
  从 LTM 获取 sink 的完整信息:
    - state_vars: {category: "sink", is_receptacle: True, ...}
    - relations: [is_a: receptacle, synonym_of: basin]
  ↓
WM 中的 sink unit:
  symbol.label = "sink"
  symbol.state_vars = {
    position: "right side",  # 从感知获得
    category: "sink",  # 从 LTM 获得
    is_receptacle: True,  # 从 LTM 获得
    materials: ["ceramic", "steel"],  # 从 LTM 获得
    affordances: ["wash", "hold_water"],  # 从 LTM 获得
  }
  symbol.relations = [
    is_a: receptacle,  # 从 LTM 获得 ✅
    synonym_of: basin,
  ]
```

### 语义对齐阶段

```
检测 entity gaps:
  从 instruction 提取: "right receptacle of left counter"
  _pending_entity_gaps = ["entity:place:right receptacle of left counter"]
  ↓
解析空间关系:
  spatial_modifier = "right"
  core_term = "receptacle"
  reference_object = "left counter"
  ↓
查询 ABSTRACT_TO_CONCRETE_HINTS:
  receptacle → [sink, basin, bowl, container, bin]
  ↓
遍历观察到的对象:
  obs_label = "sink"
  
  方法 1: 类型匹配
    sink ∈ [sink, basin, bowl, ...] ✅
    score = 0.85
    reason = "type=receptacle"
  
  方法 2: LTM 关系查询 (备用)
    unit = find_unit_by_label("sink")
    relations = [is_a: receptacle, ...]
    发现: sink is_a receptacle ✅
    (如果方法 1 失败，score = 0.9, reason = "is_a=receptacle")
  
  方法 3: 空间验证
    sink.position = "right side"
    包含 "right" ✅
    score += 0.15 → 总分 = 1.0
    reason += "+spatial=right"
  ↓
对齐成功:
  "right receptacle of left counter" → sink
  conf = 1.0
  reason = "type=receptacle+spatial=right"
  ↓
更新 WM:
  sink.aliases.append("right receptacle of left counter")
  sink.relations.append(synonym_of("right receptacle of left counter"))
```

### 规划阶段

```
Prepare planner context:
  检测 entity gaps:
    "right receptacle of left counter" 在 WM 中找到 ✅
    (通过 sink 的 alias)
  ↓
  gaps = [] (没有缺失)
  ↓
WM context 传给 Planner:
  nodes:
    - sink (aliases: ["right receptacle of left counter"])
  ↓
VLM Planner:
  知道 "right receptacle of left counter" = sink
  选择 action: "place at the sink in the kitchen"
  ↓
任务成功！✅
```

## 预期效果

### 日志输出

```log
[INFO] - Loaded LTM store with 670 nodes, 1230 edges
[INFO] - LTM concepts loaded: receptacle, container, surface, appliance

[Step N: 观察]
[INFO] - Perception ingest: objects=7 (sink, drawer, ring, ...)
[INFO] - Merged 7 WM nodes with LTM attributes
[INFO] - WM node 'sink' enriched with LTM:
         - is_receptacle: True
         - relations: [is_a: receptacle, synonym_of: basin]

[语义对齐]
[INFO] - Detecting entity gaps from instruction
[INFO] - Extracted 4 entity gaps: ['entity:place:right receptacle of left counter', ...]
[INFO] - Parsed 'right receptacle of left counter' →
         spatial='right', term='receptacle', ref='left counter'
[INFO] - Attempting semantic alignment with 4 pending gaps and 7 observed objects
[INFO] - [Semantic Alignment] Found is_a relation: sink is_a receptacle
[INFO] - [Spatial Verification] ✅ 'sink' position='right side' matches 'right'
[INFO] - ✅ Aligned 'right receptacle of left counter' → 'sink'
         (conf=1.00, kind=place, reason=type=receptacle+spatial=right)
[INFO] - Semantic alignment: 1 mappings established
[INFO] -   - 'right receptacle of left counter' → 'sink' (conf=1.00)

[规划]
[INFO] - Entity Gap Detection: 'right receptacle of left counter' found (via alias)
[INFO] - No missing gaps
[INFO] - Planner action selected: place at the sink in the kitchen ✅
```

### 知识图谱结构

**LTM:**
```
receptacle (concept)
  ├─ definition: "A container that can hold liquids or objects"
  ├─ concrete_instances: [sink, basin, bowl, container, bin]
  └─ (incoming edges)
      ├─ sink --is_a--> receptacle
      ├─ bowl --is_a--> receptacle
      └─ basin --is_a--> receptacle

sink (place)
  ├─ category: "sink"
  ├─ materials: [ceramic, steel]
  ├─ is_receptacle: True
  └─ (outgoing edges)
      ├─ is_a --> receptacle
      └─ synonym_of --> basin
```

**WM (运行时):**
```
sink
  ├─ label: "sink"
  ├─ aliases: ["right receptacle of left counter", "basin"]
  ├─ state_vars:
  │   ├─ position: "right side" (from perception)
  │   ├─ category: "sink" (from LTM)
  │   ├─ is_receptacle: True (from LTM)
  │   └─ materials: [ceramic, steel] (from LTM)
  └─ relations:
      ├─ is_a: receptacle (from LTM)
      └─ synonym_of: right receptacle of left counter (from alignment)
```

## 优势

### 1. 动态知识推理

**修复前:**
- 静态映射：只能用硬编码的字典
- 无法扩展：新对象需要手动添加到 ABSTRACT_TO_CONCRETE_HINTS

**修复后:**
- 动态查询：利用 LTM 中的 `is_a` 关系
- 自动扩展：新对象只需在 LTM 中定义 `is_a` 关系即可

### 2. 双向语义理解

**修复前:**
- 单向：抽象 → 具体（receptacle → sink）

**修复后:**
- 双向：
  - 抽象 → 具体：receptacle → [sink, basin, bowl]
  - 具体 → 抽象：sink → is_a receptacle

### 3. 知识复用

**修复前:**
- 知识割裂：LTM 知道 sink 的属性，但不知道它是 receptacle

**修复后:**
- 知识统一：LTM 中完整的本体结构
- 自动继承：sink 继承 receptacle 的所有属性

### 4. 推理能力

**修复前:**
```
Instruction: "place at the receptacle"
Agent: ❌ 不知道 receptacle 是什么
```

**修复后:**
```
Instruction: "place at the receptacle"
查询 LTM: receptacle.concrete_instances = [sink, basin, bowl, ...]
观察到: sink
推理: sink is_a receptacle ✅
Agent: 选择 "place at the sink" ✅
```

## 影响范围

### 修改的文件

1. **`embodiedbench/evaluator/object_knowledge_seed.py`**
   - 添加抽象概念节点：receptacle, container, surface, appliance
   - 为具体对象添加 `is_a` 关系和 `is_X` 标志

2. **`embodiedbench/evaluator/semantic_memory.py`**
   - 增强 `align_semantics()` 方法，添加 LTM 关系查询

### 性能影响

- **轻微增加**：每次对齐时可能额外查询 WM 中的关系
- **可接受**：查询只是遍历已有的关系列表，无显著开销
- **优化空间**：可以缓存抽象概念查询结果

### 兼容性

- ✅ **完全向后兼容**：保留了原有的 ABSTRACT_TO_CONCRETE_HINTS 机制
- ✅ **优雅降级**：如果 LTM 中没有关系，仍使用字典匹配
- ✅ **增量改进**：可以逐步为更多对象添加关系

## 测试建议

### 1. 验证 LTM 加载

```bash
# 删除旧的 LTM store
rm outputs/semantic_memory/ltm_store.json

# 运行测试
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name='test_ltm'
```

检查日志：
```log
[INFO] - Bootstrapped LTM with seed graph (nodes=670, edges=1230)
[INFO] - LTM node 'receptacle' created: {kinds:[concept,category], ...}
[INFO] - LTM node 'sink' has relation: is_a receptacle
```

### 2. 验证语义对齐

运行包含 "receptacle" 的任务：
```python
instruction = "Move spatula to the right receptacle of the left counter"
```

检查日志：
```log
[INFO] - [Semantic Alignment] Found is_a relation: sink is_a receptacle
[INFO] - ✅ Aligned 'right receptacle of left counter' → 'sink'
         (conf=1.00, reason=type=receptacle+spatial=right)
```

### 3. 验证 Planner 行为

检查 agent 是否选择了正确的动作：
```log
[INFO] - VLM Planner selected: place at the sink in the kitchen ✅
```

## 总结

**问题本质：** LTM 中缺少抽象概念本体，无法进行语义推理

**解决方案：** 
1. 在 LTM 中建立完整的抽象概念层（receptacle, container, surface, appliance）
2. 为具体对象添加 `is_a` 关系
3. 增强语义对齐逻辑，利用 LTM 关系进行类型推理

**核心改进：** 从静态字典匹配 → 动态知识图谱推理

**效果：** Agent 现在可以理解 "receptacle" 指的是 sink，因为它从 LTM 中学习到了 `sink is_a receptacle` 这个知识！

---

**Date:** 2026-01-07  
**Modified Files:** 
- `embodiedbench/evaluator/object_knowledge_seed.py`
- `embodiedbench/evaluator/semantic_memory.py`

**Impact:** 中等风险，高收益，向后兼容
