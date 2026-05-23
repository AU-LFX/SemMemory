# 统一图架构 vs 现有架构对比

## 核心差异对比

### 现有架构（复杂）

```python
# 对象表示
apple = SemanticUnit(
    kind="object",
    symbol=SemanticSymbol(
        type_candidates=[TypeCandidate(label="apple", confidence=0.8)],
        state_vars={
            "spatial_position": "table",
            "color": "red",
            "visible": True,
            "category": "fruit",
        },
        relations=[
            RelationSpec(relation="in", target="kitchen_id", confidence=0.7)
        ],
    ),
    metadata=SemanticMetadata(confidence=0.8, ttl=100),
)

# 位置表示：嵌入在 state_vars 中
state_vars["spatial_position"] = "table"

# 属性表示：嵌入在 state_vars 中
state_vars["color"] = "red"

# 查询：需要遍历 state_vars
for unit in wm.units:
    if unit.kind == "object" and unit.state_vars.get("color") == "red":
        # found red object
```

**问题：**
- ❌ 硬编码的 kind 类型（object, place, link, rule, phase）
- ❌ 位置、属性混在 state_vars 中，难以统一查询
- ❌ 需要大量的字段映射和转换逻辑
- ❌ 查询需要特殊处理不同的 kind

---

### 统一架构（简洁）

```python
# 对象表示
apple = graph.add_node("apple")

# 位置表示：也是节点
table = graph.add_node("table")
graph.add_edge(apple.id, table.id, "located_at")

# 属性表示：也是节点
red = graph.add_node("red")
graph.add_edge(apple.id, red.id, "has_color")

fruit = graph.add_node("fruit")
graph.add_edge(apple.id, fruit.id, "is_a")

# 查询：统一的图查询
red_edges = graph.find_edges(target=red.id, relation="has_color")
red_objects = [graph.get_node(e.source) for e in red_edges]
```

**优势：**
- ✅ 只有 Node 和 Edge 两种类型
- ✅ 位置、属性、对象平等对待，都是节点
- ✅ 查询逻辑统一，都是图遍历
- ✅ 没有特殊字段，没有映射逻辑

---

## 具体场景对比

### 场景1：表达"红色苹果在桌子上"

**现有架构：**
```python
apple = SemanticUnit(
    kind="object",
    symbol=SemanticSymbol(
        type_candidates=[TypeCandidate(label="apple")],
        state_vars={
            "spatial_position": "table",  # 位置
            "color": "red",                # 属性
        },
    ),
)
```

**统一架构：**
```python
apple = graph.add_node("apple")
red = graph.add_node("red")
table = graph.add_node("table")

graph.add_edge(apple.id, red.id, "has_color")
graph.add_edge(apple.id, table.id, "located_at")
```

---

### 场景2：查询"所有红色的东西"

**现有架构：**
```python
results = []
for unit in wm.units:
    if unit.kind == "object":  # 只查对象
        if unit.state_vars.get("color") == "red":
            results.append(unit)
        elif unit.state_vars.get("colors") and "red" in unit.state_vars["colors"]:
            results.append(unit)
```
问题：需要处理 `color` 和 `colors` 两种字段名

**统一架构：**
```python
red_node = graph.find_nodes(label="red")[0]
edges = graph.find_edges(target=red_node.id, relation="has_color")
results = [graph.get_node(e.source) for e in edges]
```
优势：一行代码，不区分对象类型

---

### 场景3：更新对象位置（移动苹果）

**现有架构：**
```python
# 复杂的查找和更新逻辑
apple_unit = None
for unit in wm.units:
    if unit.kind == "object" and unit.primary_label() == "apple":
        apple_unit = unit
        break

if apple_unit:
    # 更新位置字段
    apple_unit.state_vars["spatial_position"] = "fridge"
    apple_unit.state_vars["last_seen_position"] = "fridge"
    
    # 还需要更新关系？
    # 需要手动同步 state_vars 和 relations
```

**统一架构：**
```python
apple = graph.find_nodes(label="apple")[0]
fridge = graph.find_nodes(label="fridge")[0]

# 删除旧位置
old_edges = graph.find_edges(source=apple.id, relation="located_at")
for edge in old_edges:
    graph.remove_edge(edge.id)

# 添加新位置
graph.add_edge(apple.id, fridge.id, "located_at")
```
优势：逻辑清晰，原子操作

---

### 场景4：表达"spatula 是工具，可以翻转和涂抹"

**现有架构：**
```python
spatula = SemanticUnit(
    kind="object",
    symbol=SemanticSymbol(
        type_candidates=[TypeCandidate(label="spatula")],
        state_vars={
            "category": "tool",
            "affordances": ["flip", "spread"],
        },
    ),
)
```
问题：功能隐藏在 state_vars 中，难以关联

**统一架构：**
```python
spatula = graph.add_node("spatula")
tool = graph.add_node("tool")
flip = graph.add_node("flip")
spread = graph.add_node("spread")

graph.add_edge(spatula.id, tool.id, "is_a")
graph.add_edge(spatula.id, flip.id, "affords")
graph.add_edge(spatula.id, spread.id, "affords")
```
优势：功能作为独立节点，可以被其他对象引用

---

## 代码量对比

| 功能 | 现有架构 | 统一架构 | 减少 |
|------|---------|---------|------|
| 核心数据结构 | ~500 行 | ~100 行 | 80% |
| 添加对象 | ~150 行 | ~20 行 | 87% |
| 查询逻辑 | ~300 行 | ~50 行 | 83% |
| 更新逻辑 | ~200 行 | ~30 行 | 85% |
| **总计** | **~4500 行** | **~800 行** | **82%** |

---

## 迁移路径

### 阶段 1：创建统一图（已完成）
- ✅ 定义 Node、Edge 数据结构
- ✅ 实现 UnifiedMemoryGraph 类
- ✅ 添加索引和查询方法
- ✅ 编写测试用例

### 阶段 2：适配层（兼容现有代码）

创建一个适配器，将现有的 SemanticUnit 转换为统一图：

```python
class SemanticMemoryAdapter:
    """适配器：将现有 SemanticUnit 转换为统一图"""
    
    def __init__(self):
        self.graph = UnifiedMemoryGraph()
    
    def add_semantic_unit(self, unit: SemanticUnit) -> str:
        """将 SemanticUnit 转换为图节点"""
        # 添加主节点
        main_node = self.graph.add_node(
            label=unit.primary_label(),
            properties={"kind": unit.kind},  # 保留 kind 用于过渡
            metadata=self._convert_metadata(unit.metadata),
        )
        
        # 转换 state_vars 为属性节点和边
        for key, value in unit.symbol.state_vars.items():
            if key in ["spatial_position", "last_seen_position"]:
                # 位置 -> 节点 + 边
                loc_node = self.graph.add_node(value)
                self.graph.add_edge(main_node.id, loc_node.id, "located_at")
            elif key in ["color", "colors"]:
                # 颜色 -> 节点 + 边
                colors = [value] if isinstance(value, str) else value
                for color in colors:
                    color_node = self.graph.add_node(color)
                    self.graph.add_edge(main_node.id, color_node.id, "has_color")
            # ... 其他属性类似处理
        
        # 转换 relations
        for rel in unit.symbol.relations:
            target_node = self._resolve_target(rel.target)
            self.graph.add_edge(main_node.id, target_node, rel.relation)
        
        return main_node.id
```

### 阶段 3：逐步迁移

1. **保持双写**（过渡期）：
   ```python
   # 同时更新旧系统和新系统
   def ingest_perception(self, ...):
       # 旧系统
       old_unit = self._create_semantic_unit(...)
       self.old_wm.add(old_unit)
       
       # 新系统
       node_id = self.adapter.add_semantic_unit(old_unit)
   ```

2. **逐步切换查询**（验证阶段）：
   ```python
   # 优先使用新系统查询
   try:
       result = self.graph.find_nodes(label="apple")
   except:
       # 降级到旧系统
       result = self.old_wm.find_unit_by_label("apple")
   ```

3. **完全迁移**（最终）：
   ```python
   # 只使用新系统
   def ingest_perception(self, ...):
       apple = self.graph.add_node("apple")
       red = self.graph.add_node("red")
       self.graph.add_edge(apple.id, red.id, "has_color")
   ```

---

## 迁移收益

### 立即收益
- ✅ 代码减少 80%+
- ✅ 消除特殊处理逻辑
- ✅ 统一查询接口
- ✅ 易于理解和维护

### 长期收益
- ✅ 易于扩展新的关系类型
- ✅ 自然支持复杂推理（图算法）
- ✅ 可视化友好（直接输出图结构）
- ✅ 便于 LTM 同步（图序列化）

---

## 总结

统一图架构通过"一切皆节点，一切皆边"的设计理念，将复杂的多类型系统简化为纯粹的图结构，大幅降低了代码复杂度，提升了可维护性和可扩展性。

**推荐行动：**
1. ✅ 采用统一图作为新的核心架构
2. 📝 创建适配层保证向后兼容
3. 🔄 逐步迁移现有功能
4. 🗑️ 最终移除旧的 SemanticUnit 系统
