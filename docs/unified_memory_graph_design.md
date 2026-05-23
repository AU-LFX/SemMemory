# 统一内存图架构设计 (Unified Memory Graph Architecture)

## 核心理念

**一切皆节点，一切皆边** - 不区分对象、属性、位置等，统一表示为图中的节点和边。

## 当前问题

现有代码的复杂性来源：
- ❌ 硬编码的 `kind` 分类：object, place, link, rule, phase
- ❌ 特殊处理逻辑：对象位置单独处理、属性字段映射
- ❌ 不同类型的数据结构：state_vars vs relations vs constraints
- ❌ 固定的字段语义：spatial_position, visible, last_seen_ts 等

## 新设计：统一图结构

### 1. 唯一的节点类型 (Node)

```python
@dataclass
class Node:
    """统一的节点表示"""
    id: str                          # 唯一标识
    label: str                       # 节点标签（如 "apple", "red", "on_table"）
    properties: Dict[str, Any]       # 任意属性键值对
    metadata: NodeMetadata           # 元数据（置信度、时间戳等）
    clip_refs: List[str]            # 证据引用
```

**所有东西都是节点：**
- `apple` → Node(id="n1", label="apple")
- `red` → Node(id="n2", label="red")  
- `kitchen` → Node(id="n3", label="kitchen")
- `(0.5, 1.2, 0.3)` → Node(id="n4", label="position", properties={"x": 0.5, "y": 1.2, "z": 0.3})
- `edible` → Node(id="n5", label="edible")

### 2. 唯一的边类型 (Edge)

```python
@dataclass
class Edge:
    """统一的边表示"""
    id: str                          # 边唯一标识
    source: str                      # 源节点 id
    target: str                      # 目标节点 id
    relation: str                    # 关系类型（如 "has_color", "located_at", "is_a"）
    properties: Dict[str, Any]       # 边的属性（如权重、置信度）
    metadata: EdgeMetadata           # 元数据
    clip_refs: List[str]            # 证据引用
```

**所有关系都是边：**
- `apple --has_color--> red`
- `apple --located_at--> position(0.5, 1.2, 0.3)`
- `apple --is_a--> fruit`
- `apple --in--> kitchen`
- `spatula --can--> flip`

### 3. 示例图结构

```
场景：厨房里的红苹果在桌子上

节点：
n1: Node(label="apple")
n2: Node(label="red")  
n3: Node(label="kitchen")
n4: Node(label="table")
n5: Node(label="position", properties={"x": 0.5, "y": 1.2, "z": 0.3})
n6: Node(label="fruit")
n7: Node(label="edible")

边：
e1: Edge(source=n1, target=n2, relation="has_color")
e2: Edge(source=n1, target=n5, relation="located_at")
e3: Edge(source=n5, target=n4, relation="on")
e4: Edge(source=n4, target=n3, relation="in")
e5: Edge(source=n1, target=n6, relation="is_a")
e6: Edge(source=n1, target=n7, relation="has_property")
```

### 4. 查询示例

**查询1：苹果在哪里？**
```python
# 图遍历
apple = find_node(label="apple")
position_edge = find_outgoing_edge(apple, relation="located_at")
position_node = get_target_node(position_edge)
container_edge = find_outgoing_edge(position_node, relation="on")
table = get_target_node(container_edge)
# 结果：apple is on table
```

**查询2：红色的可食用物体**
```python
# 图遍历
red_nodes = find_nodes_with_incoming_edge(relation="has_color", source_label="*", target_label="red")
edible_nodes = find_nodes_with_outgoing_edge(relation="has_property", target_label="edible")
result = set(red_nodes) & set(edible_nodes)
# 结果：apple
```

## 优势

### 1. 极简性
- ✅ 只有 2 种数据结构：Node + Edge
- ✅ 没有硬编码的类型系统
- ✅ 没有特殊字段处理逻辑

### 2. 通用性
- ✅ 可以表达任意复杂的知识
- ✅ 易于扩展新的关系类型
- ✅ 自然支持多跳查询

### 3. 一致性
- ✅ 对象、属性、位置平等对待
- ✅ 所有更新操作统一为"添加节点+添加边"
- ✅ 所有查询操作统一为图遍历

## 与现有系统对比

| 维度 | 现有系统 | 统一图系统 |
|------|---------|-----------|
| 节点类型 | object, place, link, rule, phase (5种) | Node (1种) |
| 数据存储 | state_vars, relations, constraints (3处) | properties (1处) |
| 位置表示 | state_vars["spatial_position"] (特殊) | located_at edge (通用) |
| 属性表示 | state_vars["color"] (字段) | has_color edge (关系) |
| 代码复杂度 | 4500+ 行，大量特殊逻辑 | ~1000 行，纯图操作 |

## 实现路径

### Phase 1: 定义核心结构
```python
@dataclass
class Node:
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    clip_refs: List[str] = field(default_factory=list)

@dataclass
class Edge:
    id: str
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: EdgeMetadata = field(default_factory=EdgeMetadata)
    clip_refs: List[str] = field(default_factory=list)

@dataclass
class UnifiedGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    # 索引加速查询
    node_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))  # label -> node_ids
    edge_index_out: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # source -> edge_ids
    edge_index_in: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))   # target -> edge_ids
```

### Phase 2: 图操作 API
```python
class UnifiedGraph:
    def add_node(self, label: str, properties: Dict = None, ...) -> Node
    def add_edge(self, source_id: str, target_id: str, relation: str, ...) -> Edge
    def find_nodes(self, label: str = None, **property_filters) -> List[Node]
    def find_edges(self, source: str = None, target: str = None, relation: str = None) -> List[Edge]
    def traverse(self, start_node: str, relations: List[str], max_depth: int = 3) -> List[Node]
    def shortest_path(self, start: str, end: str) -> List[Edge]
```

### Phase 3: 迁移现有逻辑
```python
# 旧代码：
obj = SemanticUnit(kind="object", state_vars={"spatial_position": "table", "color": "red"})

# 新代码：
apple = graph.add_node(label="apple")
red = graph.add_node(label="red")
table = graph.add_node(label="table")
graph.add_edge(apple.id, red.id, relation="has_color")
graph.add_edge(apple.id, table.id, relation="located_at")
```

## 关系类型设计

常用关系类型（非硬编码，只是约定）：

### 空间关系
- `located_at`: 对象位置
- `in`, `on`, `near`, `above`, `below`: 相对空间关系

### 语义关系  
- `is_a`: 类型关系 (apple is_a fruit)
- `has_property`: 属性关系 (apple has_property edible)
- `has_color`, `has_size`, `has_shape`: 视觉属性
- `made_of`: 材质 (spatula made_of metal)

### 功能关系
- `affords`, `can`: 功能 (spatula can flip)
- `used_for`: 用途

### 时间关系
- `observed_at`: 观察时间
- `before`, `after`: 时序关系

## 临时状态 vs 长期知识

**临时状态** (Episode-scoped，不入LTM):
- 通过 `metadata.ephemeral = True` 标记
- 包括：observed_at, visible, current_position

**长期知识** (Cross-episode，入LTM):
- `metadata.ephemeral = False`
- 包括：is_a, has_color, made_of, affords

## 总结

这个统一图架构：
- 🎯 **简单**：只有节点和边两种概念
- 🎯 **通用**：可表达任意知识
- 🎯 **灵活**：易于扩展和修改
- 🎯 **清晰**：没有特殊逻辑，纯图操作

消除了现有代码的核心复杂性来源，为后续开发提供了坚实的基础。
