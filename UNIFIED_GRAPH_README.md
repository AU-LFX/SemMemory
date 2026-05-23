# 统一内存图架构 (Unified Memory Graph)

## 概述

一个极简、通用的图结构内存系统，实现"一切皆节点，一切皆边"的设计理念。

**核心优势：**
- ✅ 对象、属性、位置平等对待
- ✅ 统一的查询接口
- ✅ 代码减少 80%+
- ✅ 易于理解和扩展

---

## 创建的文件

### 1. 核心实现

**`embodiedbench/evaluator/unified_graph.py`** (~600行)
- UnifiedMemoryGraph 类
- Node, Edge 数据结构
- 完整的图操作API

### 2. 测试和示例

**`embodiedbench/tests/test_unified_graph.py`** (~350行)
- 6个测试场景
- 验证所有核心功能

**`embodiedbench/examples/unified_graph_demo.py`** (~350行)
- EmbodiedMemoryManager 实际应用示例
- 3个演示场景

### 3. 文档

**`docs/unified_memory_graph_design.md`**
- 完整的架构设计
- 核心理念和优势
- 实现路径

**`docs/architecture_comparison.md`**
- 新旧架构详细对比
- 迁移路径和收益分析

**`docs/unified_graph_summary.md`**
- 实现总结
- 如何解决原有问题

**`docs/quick_start.md`**
- 5分钟快速上手
- 常用模式和示例

---

## 快速开始

### 基本用法

```python
from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph

# 创建图
graph = UnifiedMemoryGraph()

# 添加节点（对象、属性、位置都是节点）
apple = graph.add_node("apple")
red = graph.add_node("red")
table = graph.add_node("table")

# 添加边（所有关系都是边）
graph.add_edge(apple.id, red.id, "has_color")
graph.add_edge(apple.id, table.id, "located_at")

# 查询
red_things = graph.find_edges(target=red.id, relation="has_color")
print([graph.get_node(e.source).label for e in red_things])
# 输出: ['apple']
```

### 运行测试

```bash
cd /home/dministrator/EmbodiedBench-problemsolving
python embodiedbench/tests/test_unified_graph.py
```

### 运行演示

```bash
python embodiedbench/examples/unified_graph_demo.py
```

---

## 核心设计

### 一切皆节点

```python
# 不再区分对象、属性、位置
apple = Node(label="apple")       # 对象 → 节点
red = Node(label="red")           # 属性 → 节点  
table = Node(label="table")       # 位置 → 节点
fruit = Node(label="fruit")       # 类型 → 节点
```

### 一切皆边

```python
# 所有关系统一表示
Edge(apple, red, "has_color")     # 属性关系 → 边
Edge(apple, table, "located_at")  # 位置关系 → 边
Edge(apple, fruit, "is_a")        # 类型关系 → 边
```

---

## 与现有系统对比

| 维度 | 现有系统 | 统一图系统 |
|------|---------|-----------|
| 节点类型 | 5种 (object, place, link, rule, phase) | 1种 (Node) |
| 数据存储 | 3处 (state_vars, relations, constraints) | 1处 (properties) |
| 位置表示 | state_vars["spatial_position"] (特殊) | located_at edge (通用) |
| 属性表示 | state_vars["color"] (字段) | has_color edge (关系) |
| 代码复杂度 | 4500+ 行 | ~800 行 |

---

## 主要功能

### 节点操作
- `add_node()` - 添加或更新节点
- `find_nodes()` - 查找节点（支持模糊匹配、属性过滤）
- `get_node()` - 获取节点
- `remove_node()` - 删除节点

### 边操作
- `add_edge()` - 添加边
- `find_edges()` - 查找边（支持源/目标/关系过滤）
- `get_edge()` - 获取边
- `remove_edge()` - 删除边

### 图遍历
- `traverse()` - BFS遍历
- `get_neighbors()` - 获取邻居节点

### 维护
- `prune()` - 清理过期节点
- `to_dict()` / `from_dict()` - 导入导出
- `stats()` - 统计信息

---

## 使用场景

### 场景1：表达"红色苹果在桌子上"

```python
apple = graph.add_node("apple")
red = graph.add_node("red")
table = graph.add_node("table")

graph.add_edge(apple.id, red.id, "has_color")
graph.add_edge(apple.id, table.id, "located_at")
```

### 场景2：查询"所有红色的东西"

```python
red = graph.find_nodes(label="red")[0]
edges = graph.find_edges(target=red.id, relation="has_color")
results = [graph.get_node(e.source).label for e in edges]
```

### 场景3：移动对象

```python
# 移动 apple 从 table 到 fridge
fridge = graph.add_node("fridge")

# 删除旧位置
old_edges = graph.find_edges(source=apple.id, relation="located_at")
for edge in old_edges:
    graph.remove_edge(edge.id)

# 添加新位置
graph.add_edge(apple.id, fridge.id, "located_at")
```

---

## 文档索引

1. **快速开始** → `docs/quick_start.md`
2. **设计文档** → `docs/unified_memory_graph_design.md`
3. **架构对比** → `docs/architecture_comparison.md`
4. **实现总结** → `docs/unified_graph_summary.md`

---

## API 文档

### UnifiedMemoryGraph 类

```python
class UnifiedMemoryGraph:
    """统一内存图"""
    
    def add_node(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[NodeMetadata] = None,
        clip_refs: Optional[List[str]] = None,
        node_id: Optional[str] = None,
    ) -> Node:
        """添加或更新节点"""
    
    def find_nodes(
        self,
        label: Optional[str] = None,
        exact: bool = False,
        property_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Node]:
        """查找节点"""
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[EdgeMetadata] = None,
        clip_refs: Optional[List[str]] = None,
        edge_id: Optional[str] = None,
    ) -> Edge:
        """添加边"""
    
    def find_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relation: Optional[str] = None,
        limit: int = 100,
    ) -> List[Edge]:
        """查找边"""
    
    def traverse(
        self,
        start_node: str,
        relations: List[str],
        max_depth: int = 3,
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> List[Node]:
        """图遍历（BFS）"""
    
    def prune(self, current_step: int) -> Tuple[int, int]:
        """清理过期节点和边"""
```

---

## 常用关系类型

### 空间关系
- `located_at` - 位置
- `in`, `on`, `near`, `above`, `below` - 相对位置

### 语义关系
- `is_a` - 类型
- `has_property` - 属性
- `has_color`, `has_size`, `has_shape` - 视觉属性
- `made_of` - 材质

### 功能关系
- `affords`, `can` - 功能
- `used_for` - 用途

---

## 性能

### 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| add_node | O(1) |
| find_nodes (by label) | O(k), k=结果数 |
| add_edge | O(1) |
| find_edges | O(k), k=结果数 |
| traverse | O(V+E), V=节点数, E=边数 |
| prune | O(V+E) |

### 空间复杂度

- 节点存储: O(V)
- 边存储: O(E)
- 索引: O(V+E)

---

## 下一步

### 立即可做
1. ✅ 阅读文档了解设计理念
2. ✅ 运行测试验证功能
3. ✅ 运行演示查看实际应用

### 集成到现有系统
1. 创建适配层
2. 双写验证
3. 逐步迁移
4. 移除旧系统

---

## 总结

统一内存图通过"一切皆节点，一切皆边"的设计，将复杂的多类型系统简化为纯粹的图结构：

- ✅ **简洁** - 只有 Node 和 Edge
- ✅ **通用** - 可表达任意知识
- ✅ **灵活** - 易于扩展
- ✅ **高效** - 代码减少 80%+

这是一个更加清晰、易维护的内存架构！
