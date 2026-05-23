# Bug修复总结

## 问题分析

### 1. 警告: `'dict' object has no attribute 'id'`

**位置**: `meta_flat_habitat_agent.py` 第168行 `print_ltm_retrieval()`

**原因**: 
- `prepare_planner_context()` 返回的nodes是dict列表 (通过`n.to_dict()`转换)
- `print_ltm_retrieval()` 尝试访问 `n.id` 和 `edge.source/target`
- dict对象没有`.id`属性

**修复**:
```python
# 修复前:
node_map = {n.id: n.label for n in nodes}

# 修复后:
node_map = {}
for n in nodes:
    if isinstance(n, dict):
        node_map[n.get('id', '')] = n.get('label', 'N/A')
    else:
        node_map[n.id] = n.label
```

同样处理edge对象的dict/Edge兼容性。

### 2. 日志显示大量"unknown"节点类型

**原因**:
- `print_wm_state()` 使用 `node.properties.get('kind')` 获取节点类型
- 但代码中创建节点时使用的是 `properties={'type': 'object'}` 而不是 `kind`
- 导致所有节点都被归类为"unknown"

**修复**:
```python
# 修复前:
kind = node.properties.get('kind', 'unknown')

# 修复后:
kind = node.properties.get('type') or node.properties.get('kind', 'unknown')
```

同时在两处应用:
1. `print_wm_state()` (行79)
2. `print_ltm_retrieval()` (行149, 153)

### 3. LTM检索范围不足

**问题**: 日志显示 "Retrieved 2 hints from LTM",但WM有17个节点

**原因**:
- `integrate_from_ltm()` 只检索与goal关键词匹配的WM节点
- 第一个episode的关键词是 ["sofa", "apple", "plum", "sink"]
- 很多WM节点(如"chair", "ball", "table")不匹配任何关键词
- `instruction_entity`节点和`object`节点没有被优先检索

**错误**: 使用了 `properties=` 参数,但API是 `property_filters=`
```python
# ❌ 错误:
self.wm.find_nodes(properties={"type": "instruction_entity"}, limit=50)

# ✅ 正确:
self.wm.find_nodes(property_filters={"type": "instruction_entity"}, limit=50)
```

**修复**:
```python
def integrate_from_ltm(self, goal: str, step: int) -> List[Dict[str, Any]]:
    anchor_ids: Set[str] = set()
    
    # 🔥 优先添加所有instruction_entity节点（修正API参数名）
    instruction_entities = self.wm.find_nodes(property_filters={"type": "instruction_entity"}, limit=50)
    for entity_node in instruction_entities:
        anchor_ids.add(entity_node.id)
    
    # 🔥 添加所有object节点
    object_nodes = self.wm.find_nodes(property_filters={"type": "object"}, limit=50)
    for obj_node in object_nodes:
        anchor_ids.add(obj_node.id)
    
    # 基于关键词的检索
    for kw in keywords[:8]:
        ranked = self.wm.semantic_find_nodes(kw, exact=False, limit=6)
        for n, _sc in ranked:
            anchor_ids.add(n.id)
```

同样在 `read()` 方法中应用:
```python
def read(self, goal: str, step: int, max_nodes: int = 30):
    # 索引召回...
    
    # 🔥 优先检索instruction_entity节点
    instruction_entities = self.wm.find_nodes(property_filters={"type": "instruction_entity"}, limit=20)
    for entity_node in instruction_entities:
        seed_ids[entity_node.id] = 0.9  # 高分数
        # 加入属性邻居
        for edge in self.wm.edges.values():
            if edge.source == entity_node.id and edge.relation in ["has_attribute", "has_color", "has_size"]:
                seed_ids[attr_node.id] = 0.85
    
    # 关键词召回时也扩展属性
    for kw in extract_goal_keywords(goal):
        for n, sc in ranked:
            if n.properties.get("type") == "object":
                # 加入对象的属性邻居
                ...
```

**效果**:
- 现在会检索所有object节点和instruction_entity节点
- 日志会显示 "Checked 17 WM nodes for LTM retrieval"
- 更多的节点会被赋予LTM先验知识

## 修改文件

### 1. `embodiedbench/evaluator/meta_flat_habitat_agent.py`

**修改1**: 行79 - `print_wm_state()` 节点类型获取
```python
# 优先使用'type'字段，fallback到'kind'字段
kind = node.properties.get('type') or node.properties.get('kind', 'unknown')
```

**修改2**: 行149, 153 - `print_ltm_retrieval()` 节点类型获取
```python
# dict对象
node_kind = node.get('properties', {}).get('type') or node.get('properties', {}).get('kind', 'unknown')

# Node对象
node_kind = node.properties.get('type') or node.properties.get('kind', 'unknown')
```

**修改3**: 行165-185 - `print_ltm_retrieval()` 处理dict/Node兼容性
```python
# 处理nodes的dict/Node兼容性
node_map = {}
for n in nodes:
    if isinstance(n, dict):
        node_map[n.get('id', '')] = n.get('label', 'N/A')
    else:
        node_map[n.id] = n.label

# 处理edges的dict/Edge兼容性
for i, edge in enumerate(edges[:10], 1):
    if isinstance(edge, dict):
        edge_source = edge.get('source', '')
        edge_target = edge.get('target', '')
        edge_relation = edge.get('relation', 'unknown')
        edge_conf = edge.get('metadata', {}).get('confidence', 0.0)
    else:
        edge_source = edge.source
        edge_target = edge.target
        edge_relation = edge.relation
        edge_conf = edge.metadata.confidence
```

### 2. `embodiedbench/evaluator/semantic_memory.py`

**修改1**: 行804-850 - `integrate_from_ltm()` 增强LTM检索范围
```python
# 优先检索instruction_entity和object节点
instruction_entities = self.wm.find_nodes(properties={"type": "instruction_entity"}, limit=50)
for entity_node in instruction_entities:
    anchor_ids.add(entity_node.id)

object_nodes = self.wm.find_nodes(properties={"type": "object"}, limit=50)
for obj_node in object_nodes:
    anchor_ids.add(obj_node.id)
```

**修改2**: 行1546 - `prepare_planner_context()` 日志增强
```python
logger.info(f"  - Checked {len(self.wm.nodes)} WM nodes for LTM retrieval")
```

## 预期效果

### 修复前的日志
```
📊 WM STATE [After Initial Perception] - Step 0
📈 Statistics: 17 nodes, 13 edges

🔷 Nodes by type:
  unknown: 17 nodes    # ❌ 所有节点都是unknown

[SemanticMemory] 📚 Step 1: Integrating relevant knowledge from LTM...
  - Retrieved 2 hints from LTM    # ❌ 只检索了2个节点

🔍 LTM RETRIEVAL [For Planning]
📦 Retrieved Nodes (top 10):
[WARNING] [MetaFlat] Semantic memory read failed: 'dict' object has no attribute 'id'    # ❌ 报错
```

### 修复后的日志
```
📊 WM STATE [After Initial Perception] - Step 0
📈 Statistics: 17 nodes, 13 edges

🔷 Nodes by type:
  object: 4 nodes    # ✅ 正确分类
    - shelf (conf=0.85, props={'type': 'object', ...})
    - blue object (conf=0.85, props={'type': 'object', ...})
  instruction_entity: 4 nodes    # ✅ 正确分类
    - small red object (conf=0.90, props={'type': 'instruction_entity', ...})
  attribute: 5 nodes    # ✅ 正确分类
  goal: 1 nodes    # ✅ 正确分类

[SemanticMemory] 📚 Step 1: Integrating relevant knowledge from LTM...
  - Checked 17 WM nodes for LTM retrieval    # ✅ 检查了所有节点
  - Retrieved 8 hints from LTM    # ✅ 更多节点被检索

🔍 LTM RETRIEVAL [For Planning]
📦 Retrieved Nodes (top 10):
  1. [attribute] with (conf=0.80, props={'type': 'attribute'})    # ✅ 正确显示类型
  2. [goal] instruction_requirements (conf=0.95, props={'type': 'goal'})
  3. [instruction_entity] large gray piece of furniture (conf=0.90, ...)
  4. [object] blue object (conf=0.85, ...)

🔗 Retrieved Edges (top 10):    # ✅ 正确显示边
  1. 'instruction_requirements' --[requires]--> 'small red object' (conf=0.95)
  2. 'small red object' --[has_attribute]--> 'small' (conf=0.85)
```

## 语义理解改进

### 之前的问题
1. 节点类型全部显示为"unknown" → 无法理解WM中存储了什么类型的信息
2. LTM检索范围窄 → 很多对象没有先验知识支持
3. dict兼容性问题 → 日志输出不完整,有警告

### 修复后的优势
1. **类型清晰**: 可以看到object、instruction_entity、attribute、goal等不同类型的节点
2. **检索完整**: 所有object和instruction_entity都会检索LTM,获得先验知识
3. **日志完整**: 无警告,可以看到完整的节点和边信息
4. **调试友好**: 开发者可以清楚看到WM的结构和LTM检索过程

## 测试建议

运行相同的测试任务,检查日志中:
- [ ] 无 `'dict' object has no attribute` 警告
- [ ] WM节点按类型正确分组(不是全部unknown)
- [ ] LTM检索显示 "Checked N WM nodes"
- [ ] Retrieved Nodes 显示正确的节点类型
- [ ] Retrieved Edges 正确显示边关系

```bash
python main.py --config configs/eb-hab.yaml --max_episodes 1 2>&1 | tee test_log.txt
grep -A 20 "WM STATE" test_log.txt  # 查看WM状态
grep -A 20 "LTM RETRIEVAL" test_log.txt  # 查看LTM检索
grep "WARNING" test_log.txt  # 检查是否还有警告
```
