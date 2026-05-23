# 统一内存图 - 快速开始

## 🚀 5分钟上手指南

### 1. 基本用法

```python
from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph

# 创建图
graph = UnifiedMemoryGraph()

# 添加节点
apple = graph.add_node("apple")
red = graph.add_node("red")
table = graph.add_node("table")

# 添加边
graph.add_edge(apple.id, red.id, "has_color")
graph.add_edge(apple.id, table.id, "located_at")

# 查询
red_things = graph.find_edges(target=red.id, relation="has_color")
print(f"红色的东西: {[graph.get_node(e.source).label for e in red_things]}")
```

### 2. 运行测试

```bash
cd /home/dministrator/EmbodiedBench-problemsolving
python embodiedbench/tests/test_unified_graph.py
```

### 3. 运行演示

```bash
python embodiedbench/examples/unified_graph_demo.py
```

---

## 📖 核心概念

### Node（节点）- 一切皆节点

```python
# 对象
apple = graph.add_node("apple", properties={"type": "fruit"})

# 属性
red = graph.add_node("red", properties={"type": "color"})

# 位置
table = graph.add_node("table", properties={"type": "location"})

# 功能
flip = graph.add_node("flip", properties={"type": "action"})
```

### Edge（边）- 一切皆边

```python
# 属性关系
graph.add_edge(apple.id, red.id, "has_color")

# 位置关系
graph.add_edge(apple.id, table.id, "located_at")

# 类型关系
graph.add_edge(apple.id, fruit.id, "is_a")

# 功能关系
graph.add_edge(spatula.id, flip.id, "affords")
```

---

## 🔍 常用查询

### 查找节点

```python
# 精确匹配
nodes = graph.find_nodes(label="apple", exact=True)

# 模糊匹配
nodes = graph.find_nodes(label="app")  # 匹配 "apple", "application" 等

# 属性过滤
nodes = graph.find_nodes(property_filters={"type": "fruit"})
```

### 查找边

```python
# 按关系类型
edges = graph.find_edges(relation="has_color")

# 按源节点
edges = graph.find_edges(source=apple.id)

# 按目标节点
edges = graph.find_edges(target=red.id)

# 组合查询
edges = graph.find_edges(source=apple.id, relation="located_at")
```

### 图遍历

```python
# 从apple开始，沿着 located_at 和 in 关系遍历
nodes = graph.traverse(
    start_node=apple.id,
    relations=["located_at", "in"],
    max_depth=3
)

# 获取邻居
neighbors = graph.get_neighbors(apple.id, relation="has_color", direction="out")
```

---

## 💡 常见模式

### 模式1：表达对象属性

```python
# 红色的大苹果
apple = graph.add_node("apple")
red = graph.add_node("red")
large = graph.add_node("large")

graph.add_edge(apple.id, red.id, "has_color")
graph.add_edge(apple.id, large.id, "has_size")
```

### 模式2：表达对象位置

```python
# 苹果在桌子上，桌子在厨房里
apple = graph.add_node("apple")
table = graph.add_node("table")
kitchen = graph.add_node("kitchen")

graph.add_edge(apple.id, table.id, "on")
graph.add_edge(table.id, kitchen.id, "in")
```

### 模式3：更新对象位置

```python
# 移动苹果从桌子到冰箱
fridge = graph.add_node("fridge")

# 删除旧位置
old_edges = graph.find_edges(source=apple.id, relation="on")
for edge in old_edges:
    graph.remove_edge(edge.id)

# 添加新位置
graph.add_edge(apple.id, fridge.id, "in")
```

### 模式4：查询对象位置

```python
# 苹果在哪里？
loc_edges = graph.find_edges(source=apple.id, relation="located_at")
if loc_edges:
    location = graph.get_node(loc_edges[0].target)
    print(f"苹果在 {location.label}")
```

### 模式5：查询属性

```python
# 找到所有红色的东西
red_node = graph.find_nodes(label="red")[0]
color_edges = graph.find_edges(target=red_node.id, relation="has_color")
red_objects = [graph.get_node(e.source).label for e in color_edges]
print(f"红色的东西: {red_objects}")
```

---

## 🔧 高级功能

### 元数据（Metadata）

```python
from embodiedbench.evaluator.unified_graph import NodeMetadata, EdgeMetadata

# 带元数据的节点
apple = graph.add_node(
    "apple",
    metadata=NodeMetadata(
        confidence=0.9,
        timestamp=100,
        ttl=200,
        provenance="perception",
        ephemeral=False  # 长期知识
    )
)

# 带元数据的边
graph.add_edge(
    apple.id, table.id, "located_at",
    metadata=EdgeMetadata(
        ephemeral=True  # 临时状态
    )
)
```

### 清理过期节点

```python
# 删除超过TTL的节点
removed_nodes, removed_edges = graph.prune(current_step=150)
print(f"清理了 {removed_nodes} 个节点, {removed_edges} 条边")
```

### 导入/导出

```python
# 导出
data = graph.to_dict()

# 导入
new_graph = UnifiedMemoryGraph()
new_graph.from_dict(data)
```

### 统计信息

```python
stats = graph.stats()
print(f"节点数: {stats['nodes']}")
print(f"边数: {stats['edges']}")
print(f"平均出度: {stats['avg_out_degree']:.2f}")
```

---

## 🎯 实际应用

### 完整示例：处理观察和动作

```python
from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph, NodeMetadata, EdgeMetadata

class MemoryManager:
    def __init__(self):
        self.graph = UnifiedMemoryGraph()
        self.step = 0
    
    def observe(self, objects):
        """处理观察"""
        for obj_data in objects:
            # 添加对象
            obj = self.graph.add_node(
                obj_data["label"],
                metadata=NodeMetadata(timestamp=self.step, provenance="perception")
            )
            
            # 添加属性
            if "color" in obj_data:
                color = self.graph.add_node(obj_data["color"])
                self.graph.add_edge(obj.id, color.id, "has_color")
            
            # 添加位置（临时状态）
            if "position" in obj_data:
                pos = self.graph.add_node(obj_data["position"])
                self.graph.add_edge(
                    obj.id, pos.id, "located_at",
                    metadata=EdgeMetadata(ephemeral=True)
                )
        
        self.step += 1
    
    def move_object(self, obj_label, to_location):
        """移动对象"""
        obj = self.graph.find_nodes(label=obj_label)[0]
        new_loc = self.graph.add_node(to_location)
        
        # 删除旧位置
        for edge in self.graph.find_edges(source=obj.id, relation="located_at"):
            self.graph.remove_edge(edge.id)
        
        # 添加新位置
        self.graph.add_edge(obj.id, new_loc.id, "located_at")
        self.step += 1
    
    def where_is(self, obj_label):
        """查询对象位置"""
        obj = self.graph.find_nodes(label=obj_label)[0]
        edges = self.graph.find_edges(source=obj.id, relation="located_at")
        if edges:
            loc = self.graph.get_node(edges[0].target)
            return loc.label
        return "未知"

# 使用
manager = MemoryManager()

# 观察
manager.observe([
    {"label": "apple", "color": "red", "position": "table"}
])

# 查询
print(manager.where_is("apple"))  # "table"

# 移动
manager.move_object("apple", "fridge")

# 再次查询
print(manager.where_is("apple"))  # "fridge"
```

---

## 📚 更多资源

- **设计文档**: `docs/unified_memory_graph_design.md`
- **对比分析**: `docs/architecture_comparison.md`
- **实现总结**: `docs/unified_graph_summary.md`
- **测试代码**: `embodiedbench/tests/test_unified_graph.py`
- **演示示例**: `embodiedbench/examples/unified_graph_demo.py`

---

## ❓ FAQ

### Q: 与旧的 SemanticUnit 有什么区别？

A: 旧系统区分对象、属性、位置等不同类型，新系统全部用节点和边表示，更简洁统一。

### Q: 如何迁移现有代码？

A: 可以创建适配层，将 SemanticUnit 转换为 Node/Edge，逐步迁移。

### Q: 性能如何？

A: 通过索引优化，查询性能优于旧系统。大规模图可以进一步优化或使用图数据库。

### Q: 支持什么类型的查询？

A: 支持标签查询、属性过滤、关系查询、图遍历、邻居查询等，覆盖所有常见场景。

---

## 🎉 开始使用吧！

```bash
# 运行测试验证
python embodiedbench/tests/test_unified_graph.py

# 运行演示了解更多
python embodiedbench/examples/unified_graph_demo.py
```
