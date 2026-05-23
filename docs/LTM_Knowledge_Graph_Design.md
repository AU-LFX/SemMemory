# LTM Knowledge Graph Design: Complete Ontology

## 1. Overview

LTM（Long-Term Memory）不应该仅仅是对象的集合，而应该是一个**完整的知识图谱（Knowledge Graph）**，包含：
- 具体对象（concrete objects）
- 抽象概念（abstract concepts）
- 属性与特征（properties & attributes）
- 状态（states）
- 动作（actions）
- 空间关系（spatial relations）
- 约束与规则（constraints & rules）

这样的设计使得 agent 可以进行**语义推理、类型推理、关系推理、约束推理**，而不仅仅是对象识别。

---

## 2. Node Type Taxonomy

### 2.1 Spatial Concepts（空间概念）

定义空间结构和容纳关系的抽象概念。

| Concept | Definition | Concrete Instances | Abstract Properties |
|---------|------------|-------------------|---------------------|
| **receptacle** | 能容纳液体或物体的物理空间或容器 | sink, basin, bowl, container, bin, drawer, cabinet | containment, volume, accessibility |
| **container** | 设计用于存储、持有或运输其他物体的对象或空间 | box, basket, bag, bin, jar, cabinet, drawer | enclosure, portability, storage_capacity |
| **surface** | 适合放置、休息或工作的平面或水平区域 | counter, table, desk, shelf, floor, tray | flatness, support, accessibility |

**用途示例：**
- "place the apple on a surface" → agent 可以选择 counter, table, desk, shelf 任意一个
- "put the dishes in the receptacle" → agent 可以选择 sink, basin, bowl

---

### 2.2 Functional Concepts（功能概念）

定义对象功能和用途的抽象概念。

| Concept | Definition | Concrete Instances | Abstract Properties |
|---------|------------|-------------------|---------------------|
| **appliance** | 执行特定家务任务的电气或机械设备 | microwave, oven, stove, refrigerator, dishwasher, toaster | powered, task_specific, automated |
| **tool** | 设计用于执行特定任务或操作的手持对象 | spatula, knife, screwdriver, hammer, wrench, pliers | graspable, purpose_built, manipulable |
| **utensil** | 用于饮食、烹饪或食品准备的工具或器具 | spatula, spoon, fork, knife, ladle, tongs | food_related, handheld, cleanable |

**用途示例：**
- "heat the food with an appliance" → agent 可以选择 microwave 或 oven
- "cut with a utensil" → agent 可以选择 knife

---

### 2.3 Material Properties（材料属性）

定义对象材料特性的属性节点。

| Property | Definition | Applies To | Constraints/Affordances |
|----------|------------|------------|------------------------|
| **fragile** | 易碎、易损坏或被毁；需要小心处理 | glass, ceramic, porcelain, crystal | handle_with_care, avoid_dropping, gentle_placement |
| **heat_resistant** | 能承受高温而不损坏或变形 | metal, ceramic, glass, stone | hot_object_placement, cooking, heating |
| **waterproof** | 不透水；不受潮湿或液体暴露损坏 | plastic, metal, rubber, sealed_wood | washing, outdoor_use, liquid_storage |

**用途示例：**
- "place the hot pot somewhere" → agent 选择 heat_resistant 的 surface
- "this object is fragile" → agent 应用 handle_with_care 约束

---

### 2.4 State Concepts（状态概念）

定义对象可能处于的状态。

| State | Definition | Opposite | Applies To | Effects/Preconditions |
|-------|------------|----------|------------|----------------------|
| **open** | 访问不受阻；门、盖子或屏障未关闭 | closed | door, drawer, cabinet, container, window | accessible_interior, visible_contents |
| **closed** | 访问受阻；门、盖子或屏障关闭 | open | door, drawer, cabinet, container, window | contents_protected, access_blocked |
| **clean** | 没有污垢、污染物或不需要的物质 | dirty | dish, utensil, surface, floor, appliance | wash, wipe, sanitize |
| **dirty** | 污染、弄脏或覆盖有不需要的物质 | clean | dish, utensil, surface, floor, appliance | clean, wash |

**用途示例：**
- "open the drawer" → 改变 drawer 的状态从 closed → open
- "the dish is dirty" → 触发 required_action: clean, wash

---

### 2.5 Action Concepts（动作概念）

定义 agent 可执行的动作及其前提条件和效果。

| Action | Definition | Preconditions | Effects | Constraints |
|--------|------------|---------------|---------|-------------|
| **pick_up** | 抓取并从表面或位置提起对象 | object_graspable, object_reachable, hands_free | object_held, object_removed_from_surface | object_weight_manageable, object_not_fixed |
| **place** | 将对象放置或定位在特定位置 | object_held, target_surface_accessible | object_at_location, hands_free | surface_supports_object, sufficient_space |
| **navigate** | 从当前位置移动到目标位置 | path_exists, target_reachable | position_changed, at_target_location | no_obstacles, sufficient_clearance |

**用途示例：**
- "pick up the apple" → 检查 preconditions: object_graspable, hands_free
- "place on the table" → 检查 constraints: surface_supports_object

---

### 2.6 Spatial Relations（空间关系）

定义对象之间的空间关系。

| Relation | Definition | Inverse/Opposite | Examples | Constraints |
|----------|------------|------------------|----------|-------------|
| **on** | 由另一物体的上表面支撑并与之接触 | under | book on table, plate on counter | surface_contact, gravity_support |
| **in** | 包含在另一对象或空间的内部或边界内 | contains | food in bowl, utensil in drawer | spatial_enclosure, boundary_defined |
| **near** | 距离很近但没有直接接触 | far | chair near table, sink near counter | distance < threshold |

**用途示例：**
- "place the apple on the table" → 建立关系: apple.on = table
- "find the spatula in the drawer" → 查询关系: spatula.in = drawer

---

### 2.7 Constraints & Rules（约束与规则）

定义行为约束和安全规则。

| Constraint/Rule | Category | Definition | Applies To | Recommendations/Violations |
|-----------------|----------|------------|------------|---------------------------|
| **handle_with_care** | safety_rule | 需要温和、小心的操作以避免损坏 | fragile, valuable, delicate | slow_movement, secure_grip, padded_surface / dropping, rough_handling |
| **temperature_safety** | safety_constraint | 处理热或冷物体时需要的预防措施 | hot_object, cold_object | use_protection, wait_for_cooling / burns, frostbite |
| **food_hygiene** | hygiene_rule | 食品处理中的清洁和安全标准 | food, utensil, cooking_surface | clean_before_use, sanitize_after_raw_food / contamination |

**用途示例：**
- object has property "fragile" → 应用 handle_with_care 约束
- "cooking with raw meat" → 应用 food_hygiene 规则

---

## 3. Relation Types

### 3.1 is_a Relation（类型继承）

定义具体对象与抽象概念之间的类型关系。

**示例：**
```python
sink.relations = [
    {relation: "is_a", target: "receptacle", confidence: 1.0},
    {relation: "is_a", target: "surface", confidence: 0.8},
]
bowl.relations = [
    {relation: "is_a", target: "receptacle", confidence: 0.9},
]
spatula.relations = [
    {relation: "is_a", target: "utensil", confidence: 1.0},
    {relation: "is_a", target: "tool", confidence: 0.7},
]
```

**推理逻辑：**
- Instruction: "place in a receptacle"
- WM observation: "sink"
- LTM query: sink.is_a.receptacle? → YES
- Alignment: "receptacle" → "sink" ✅

---

### 3.2 synonym_of Relation（同义词）

定义对象之间的同义关系。

**示例：**
```python
sink.relations = [{relation: "synonym_of", target: "basin"}]
couch.relations = [{relation: "synonym_of", target: "sofa"}]
```

---

### 3.3 part_of Relation（部分关系）

定义对象之间的组成关系。

**示例：**
```python
handle.relations = [{relation: "part_of", target: "drawer"}]
burner.relations = [{relation: "part_of", target: "stove"}]
```

---

### 3.4 has_property Relation（属性关系）

连接对象和材料属性。

**示例：**
```python
glass_bowl.relations = [{relation: "has_property", target: "fragile"}]
metal_pot.relations = [{relation: "has_property", target: "heat_resistant"}]
```

---

### 3.5 requires_constraint Relation（约束关系）

连接对象/动作和必须遵守的约束。

**示例：**
```python
pick_up_glass.relations = [{relation: "requires_constraint", target: "handle_with_care"}]
cook_meat.relations = [{relation: "requires_constraint", target: "food_hygiene"}]
```

---

## 4. Implementation in object_knowledge_seed.py

### 4.1 Abstract Concept Nodes

```python
_RAW_TEMPLATES: Dict[str, Dict[str, Any]] = {
	# ==================================================================================
	# ABSTRACT CONCEPTS & SEMANTIC CATEGORIES
	# ==================================================================================
	
	# --- Spatial Concepts ---
	"receptacle": {
		"kinds": ["concept", "spatial_category"],
		"attributes": {
			"state_vars": {
				"category": "spatial_concept",
				"definition": "A physical space or container that can hold objects or liquids",
				"semantic_type": "abstract",
				"concrete_instances": ["sink", "basin", "bowl", "container", "bin"],
				"abstract_properties": ["containment", "volume", "accessibility"],
			},
		},
	},
	
	# --- Functional Concepts ---
	"appliance": {...},
	"tool": {...},
	"utensil": {...},
	
	# --- Material Properties ---
	"fragile": {
		"kinds": ["property", "material_attribute"],
		"attributes": {
			"state_vars": {
				"category": "material_property",
				"applies_to": ["glass", "ceramic", "porcelain"],
				"constraints": ["handle_with_care", "avoid_dropping"],
			},
		},
	},
	
	# --- State Concepts ---
	"open": {...},
	"closed": {...},
	"clean": {...},
	"dirty": {...},
	
	# --- Action Concepts ---
	"pick_up": {
		"kinds": ["action", "manipulation_action"],
		"attributes": {
			"state_vars": {
				"category": "action_concept",
				"preconditions": ["object_graspable", "object_reachable"],
				"effects": ["object_held", "object_removed_from_surface"],
				"constraints": ["object_weight_manageable"],
			},
		},
	},
	
	# --- Spatial Relations ---
	"on": {...},
	"in": {...},
	"near": {...},
	
	# --- Constraints & Rules ---
	"handle_with_care": {...},
	"temperature_safety": {...},
	"food_hygiene": {...},
}
```

---

### 4.2 Concrete Objects with is_a Relations

```python
"sink in the kitchen": {
	"kinds": ["place", "surface", "receptacle"],  # 多重类型
	"attributes": {
		"state_vars": {
			"category": "navigation_target",
			"is_receptacle": True,
			"is_surface": True,
		},
	},
	"relations": [
		{
			"relation": "is_a",
			"target": "receptacle",
			"confidence": 1.0,
		},
		{
			"relation": "is_a",
			"target": "surface",
			"confidence": 0.8,
		},
		{
			"relation": "synonym_of",
			"target": "basin",
			"confidence": 0.9,
		},
	],
},
```

---

## 5. Semantic Alignment with LTM Relations

### 5.1 Enhanced align_semantics() Logic

```python
def align_semantics(self, instruction: str, wm_entity: str) -> float:
	"""
	三层匹配策略：
	1. 字典匹配（ABSTRACT_TO_CONCRETE_HINTS）
	2. LTM 关系查询（is_a, synonym_of, has_property）
	3. 空间关系验证（position, spatial relations）
	"""
	
	# Layer 1: Dictionary lookup
	core_term = self._extract_core_term(instruction)
	if core_term in ABSTRACT_TO_CONCRETE_HINTS:
		if wm_entity in ABSTRACT_TO_CONCRETE_HINTS[core_term]:
			return 0.85  # 字典匹配
	
	# Layer 2: LTM relation query
	wm_node = self.wm.get_entity(wm_entity)
	if wm_node and hasattr(wm_node, 'relations'):
		for rel in wm_node.relations:
			if rel.relation == "is_a" and rel.target == core_term:
				logger.info(f"[LTM Relation] {wm_entity} is_a {core_term} (confidence={rel.confidence})")
				return 0.9  # LTM 关系匹配
			elif rel.relation == "synonym_of" and rel.target in instruction:
				return 0.85  # 同义词匹配
	
	# Layer 3: Spatial verification
	spatial_modifier, reference_obj = self._parse_spatial_relation(instruction)
	if spatial_modifier and self._verify_spatial_relation(wm_entity, spatial_modifier, reference_obj):
		return 0.95  # 空间+语义双重验证
	
	return 0.0  # 无匹配
```

---

## 6. Usage Examples

### Example 1: Type Reasoning
```
Instruction: "place the apple in a receptacle"
WM Observation: ["sink", "counter", "box"]

Alignment Process:
1. Extract core_term: "receptacle"
2. Query LTM relations:
   - sink.is_a.receptacle? → YES (confidence=1.0) ✅
   - counter.is_a.receptacle? → NO
   - box.is_a.container? → YES, but not receptacle
3. Result: align "receptacle" → "sink"
4. Action: place(apple, sink)
```

---

### Example 2: Spatial + Semantic Reasoning
```
Instruction: "place at the right receptacle of left counter"
WM Observation: 
  - "sink" (position="right", near="counter")
  - "left counter" (position="left")

Alignment Process:
1. Parse spatial relation:
   - core_term: "receptacle"
   - spatial_modifier: "right"
   - reference_object: "left counter"
2. Check LTM relation:
   - sink.is_a.receptacle? → YES ✅
3. Verify spatial constraint:
   - sink.position == "right"? → YES ✅
   - sink.near == "counter"? → YES ✅
4. Result: align "right receptacle" → "sink" (score=0.95)
5. Action: place(object, sink)
```

---

### Example 3: Constraint Application
```
Instruction: "pick up the glass bowl"
WM Observation: "bowl_1" (material="glass")

Reasoning Process:
1. Query LTM:
   - bowl.is_a.receptacle? → YES
   - bowl.has_property.fragile? → YES (material=glass)
2. Apply constraint:
   - fragile.requires_constraint.handle_with_care
3. Execution:
   - pick_up(bowl_1, force="gentle", speed="slow")
4. Safety check:
   - Avoid violations: [dropping, rough_handling, impact]
```

---

### Example 4: Action Precondition Check
```
Instruction: "place the heavy box on the shelf"

Precondition Verification:
1. Query action concept:
   - place.preconditions: [object_held, target_surface_accessible]
   - place.constraints: [surface_supports_object, sufficient_space]
2. Check object state:
   - box.is_held? → YES ✅
   - box.weight: "heavy"
3. Check target:
   - shelf.is_surface? → YES ✅
   - shelf.load_capacity > box.weight? → Need to verify ⚠️
4. Decision:
   - If load_capacity OK: execute place(box, shelf)
   - Else: suggest alternative surface (e.g., "floor", "table")
```

---

## 7. Benefits of Complete Knowledge Graph

### 7.1 Type Reasoning
- Instruction 使用抽象词（receptacle, appliance, surface）
- Agent 能映射到具体对象（sink, microwave, counter）
- 避免硬编码字典，支持动态扩展

### 7.2 Constraint Reasoning
- 对象属性自动关联约束（fragile → handle_with_care）
- 动作自动验证前提条件（pick_up → object_graspable）
- 提高安全性和鲁棒性

### 7.3 Spatial Reasoning
- 组合空间关系和类型推理（"right receptacle of left counter"）
- 支持模糊空间描述（"near", "left", "right"）
- 提高导航和操作准确性

### 7.4 Hierarchical Reasoning
- 多层次类型继承（sink is_a receptacle, receptacle is_a container）
- 属性传播（receptacle.abstract_properties → sink）
- 支持复杂推理链

### 7.5 Explainability
- 所有推理过程可追溯（log 显示匹配路径）
- 约束违规可检测（fragile object dropped → violation）
- 提高调试能力

---

## 8. Future Extensions

### 8.1 Temporal Concepts
- before/after: 时间顺序关系
- duration: 动作持续时间
- sequence: 动作序列约束

### 8.2 Affordance Reasoning
- object.affordances: [graspable, openable, heatable]
- action.requires_affordance: [pick_up → graspable]
- 自动筛选可操作对象

### 8.3 Probabilistic Reasoning
- relation.confidence: 0.0-1.0
- 多候选对象排序（按 confidence 降序）
- 不确定性传播

### 8.4 Learning from Experience
- 新观察 → 更新 LTM 节点属性
- 失败经验 → 添加新约束
- 成功模式 → 强化关系权重

---

## 9. Summary

LTM Knowledge Graph 应该是一个**多层次、多类型、多关系**的完整本体：

**节点类型：**
- 具体对象 (sink, bowl, spatula)
- 抽象概念 (receptacle, appliance, surface)
- 属性 (fragile, heat_resistant)
- 状态 (open, clean)
- 动作 (pick_up, place, navigate)
- 空间关系 (on, in, near)
- 约束与规则 (handle_with_care, food_hygiene)

**关系类型：**
- is_a: 类型继承
- synonym_of: 同义词
- part_of: 组成关系
- has_property: 属性关系
- requires_constraint: 约束关系

**推理能力：**
- 类型推理 (sink is_a receptacle)
- 空间推理 (sink at right of counter)
- 约束推理 (fragile → handle_with_care)
- 前提条件验证 (pick_up → object_graspable)

这样的设计使得 agent 不仅能理解"sink 是 receptacle"，还能理解"为什么是"、"如何使用"、"需要注意什么"，从而实现真正的语义理解和推理能力。
