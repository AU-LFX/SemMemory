# Unified Graph 认知属性系统迁移指南

## 概述

`unified_graph.py` 已升级为支持认知语言学理论的语义内存系统，`semantic_memory.py` 已完成适配。

## 🎯 核心升级

### 1. **节点认知属性** (NodeMetadata)

新增 6 个认知属性用于概念表示：

```python
@dataclass
class NodeMetadata:
    # 基础属性
    confidence: float = 0.8
    timestamp: float
    ttl: int = 100
    provenance: str = "perception"
    
    # 🆕 认知属性
    centrality: float = 0.5          # 中心性（概念重要程度）
    prototypicality: float = 0.5     # 原型性（典型代表程度）
    family_resemblance: float = 0.5  # 家族相似性（与同类概念相似度）
    perspective: Optional[str] = None # 视角（空间/功能/文化等）
    vagueness: float = 0.0            # 模糊性（边界不清晰程度）
    gradience: float = 0.0            # 渐进性（程度连续变化）
```

**理论基础**：
- Rosch 的原型理论
- Wittgenstein 的家族相似性
- Zadeh 的模糊集合论

### 2. **边语义关系属性** (EdgeMetadata)

新增 6 个语义属性用于知识推理：

```python
@dataclass
class EdgeMetadata:
    # 基础属性
    confidence: float = 0.8
    timestamp: float
    ttl: int = 100
    provenance: str = "perception"
    
    # 🆕 语义关系属性
    relation_type: str = "associative"    # 关系类型分类
    is_symmetric: bool = False            # 对称性
    is_transitive: bool = False           # 传递性
    strength: float = 0.5                 # 关系强度
    directionality: str = "directed"      # 方向性
    semantic_role: Optional[str] = None   # 语义角色
```

**关系类型**（`relation_type`）：
- `taxonomic`: 分类关系（is_a, instance_of）
- `partitive`: 部分-整体（part_of, contains）
- `synonymy`: 同义关系（synonym_of）
- `antonymy`: 反义关系（antonym_of）
- `metaphorical`: 隐喻关系（metaphor_of）
- `causal`: 因果关系（causes, enables）
- `temporal`: 时序关系（before, after）
- `spatial`: 空间关系（on, in, near）
- `functional`: 功能关系（used_for, affords）
- `associative`: 一般关联（has_color, has_size）

## 📝 semantic_memory.py 适配内容

### 1. **新增关系语义配置表**

在 `semantic_memory.py` 中添加了 `RELATION_SEMANTICS` 字典，为每种关系类型预定义语义属性：

```python
RELATION_SEMANTICS = {
    # 分类关系 - 可传递
    "is_a": {
        "relation_type": "taxonomic",
        "is_transitive": True,
        "strength": 1.0,
        "directionality": "directed"
    },
    
    # 同义关系 - 对称
    "synonym_of": {
        "relation_type": "synonymy",
        "is_symmetric": True,
        "strength": 0.95,
        "directionality": "undirected"
    },
    
    # 空间关系
    "located_at": {
        "relation_type": "spatial",
        "is_symmetric": False,
        "strength": 0.9,
        "directionality": "directed",
        "semantic_role": "location"
    },
    
    # ... 更多配置
}
```

### 2. **新增辅助方法**

添加了 `_create_edge_with_semantics()` 方法，自动为边设置适当的语义属性：

```python
def _create_edge_with_semantics(
    self,
    graph: UnifiedMemoryGraph,
    source: str,
    target: str,
    relation: str,
    properties: Optional[Dict[str, Any]] = None,
    metadata: Optional[EdgeMetadata] = None,
    clip_refs: Optional[List[str]] = None,
) -> Edge:
    """创建带有适当语义属性的边"""
    # 自动从 RELATION_SEMANTICS 配置中提取语义属性
    if relation in RELATION_SEMANTICS and metadata is None:
        semantics = RELATION_SEMANTICS[relation]
        metadata = EdgeMetadata(
            relation_type=semantics.get("relation_type", "associative"),
            is_symmetric=semantics.get("is_symmetric", False),
            is_transitive=semantics.get("is_transitive", False),
            strength=semantics.get("strength", 0.5),
            directionality=semantics.get("directionality", "directed"),
            semantic_role=semantics.get("semantic_role"),
        )
    
    return graph.add_edge(source, target, relation, properties, metadata, clip_refs)
```

### 3. **更新边创建调用**

已更新的关键位置：

#### ✅ 同义关系（synonym_of）
```python
# 旧代码
self.wm_graph.add_edge(node.id, node.id, "synonym_of", ...)

# 新代码 - 自动设置 is_symmetric=True, strength=0.95
self._create_edge_with_semantics(
    graph=self.wm_graph,
    source=node.id,
    target=node.id,
    relation="synonym_of",
    ...
)
```

#### ✅ 属性关系（has_color, has_size, made_of）
```python
# 自动设置 relation_type="associative", strength=0.9
self._create_edge_with_semantics(
    graph=self.wm_graph,
    source=obj_node.id,
    target=color_node.id,
    relation="has_color",
    ...
)
```

#### ✅ 空间关系（located_at）
```python
# 自动设置 relation_type="spatial", semantic_role="location"
self._create_edge_with_semantics(
    graph=self.wm_graph,
    source=obj_node.id,
    target=loc_node.id,
    relation="located_at",
    ...
)
```

## 🔍 新增高级查询方法

### 节点查询（基于认知属性）

```python
# 查找高中心性的核心概念
central = graph.find_central_concepts(threshold=0.7)

# 查找原型成员
prototypes = graph.find_prototypes("bird", threshold=0.8)

# 查找模糊概念
vague = graph.find_vague_concepts(threshold=0.6)

# 查找渐进性属性
gradient = graph.find_gradient_attributes(threshold=0.5)
```

### 边查询（基于语义关系）

```python
# 查找对称关系
symmetric = graph.find_symmetric_relations()

# 查找可传递关系
transitive = graph.find_transitive_relations(relation_type="taxonomic")

# 查找高强度关系
strong = graph.find_strong_relations(threshold=0.8, relation_type="spatial")
```

### 推理方法

```python
# 传递闭包推理
# robin is_a bird, bird is_a animal => robin is_a animal
closure = graph.infer_transitive_closure("robin", "is_a")
```

## 🎓 使用最佳实践

### 1. **创建分类层次**（is_a 关系）

```python
# 在 LTM 中构建分类层次
apple = ltm_graph.add_node("apple")
fruit = ltm_graph.add_node("fruit")
food = ltm_graph.add_node("food")

# 使用传递关系
ltm_graph.add_edge(
    apple.id, fruit.id, "is_a",
    metadata=EdgeMetadata(
        relation_type="taxonomic",
        is_transitive=True,
        strength=1.0,
        provenance="ltm"
    )
)

# 自动推理：apple is_a food
closure = ltm_graph.infer_transitive_closure(apple.id, "is_a")
```

### 2. **表示同义词**（synonym_of 关系）

```python
# 对称关系
sofa = wm_graph.add_node("sofa")
couch = wm_graph.add_node("couch")

wm_graph.add_edge(
    sofa.id, couch.id, "synonym_of",
    metadata=EdgeMetadata(
        relation_type="synonymy",
        is_symmetric=True,
        strength=0.95
    )
)

# 查询所有同义关系
synonyms = wm_graph.find_symmetric_relations()
```

### 3. **处理模糊概念**

```python
# 标记模糊概念
nearby = wm_graph.add_node(
    "nearby",
    metadata=NodeMetadata(
        vagueness=0.8,      # 高模糊性
        gradience=0.7,      # 有渐进性
        perspective="spatial"
    )
)

# 查找所有模糊概念
vague_concepts = wm_graph.find_vague_concepts(threshold=0.5)
```

## 🔧 迁移清单

- [x] 添加 `RELATION_SEMANTICS` 配置表
- [x] 实现 `_create_edge_with_semantics()` 方法
- [x] 更新 `synonym_of` 关系创建
- [x] 更新 `has_color` 关系创建
- [x] 更新 `has_size` 关系创建
- [x] 更新 `made_of` 关系创建
- [x] 更新 `located_at` 关系创建
- [x] 添加高级查询方法到 `unified_graph.py`
- [x] 添加元数据过滤器支持
- [x] 添加传递闭包推理方法
- [ ] 🔄 优化 LTM 中的 is_a 关系查询（使用传递闭包）
- [ ] 🔄 为 LTM 节点添加认知属性（centrality, prototypicality）
- [ ] 🔄 实现基于家族相似性的概念聚类

## 📊 性能优化

### 元数据过滤器

新的查询方法支持高效的元数据过滤：

```python
# 精确匹配
nodes = graph.find_nodes(metadata_filters={"provenance": "ltm"})

# 范围查询
nodes = graph.find_nodes(metadata_filters={
    "centrality": (">", 0.7),
    "vagueness": ("<", 0.3)
})

# 组合查询
edges = graph.find_edges(metadata_filters={
    "relation_type": "taxonomic",
    "strength": (">", 0.8),
    "is_transitive": True
})
```

### 索引优化

`unified_graph.py` 维护了多个索引：
- `label_index`: 快速标签查询
- `out_edges` / `in_edges`: 快速邻居查询
- `relation_index`: 快速关系查询

元数据过滤在索引结果上执行，保持高效。

## 🧪 测试

运行认知属性测试：

```bash
python embodiedbench/tests/test_unified_graph_cognitive.py
```

测试覆盖：
- ✅ 认知属性查询
- ✅ 语义关系查询
- ✅ 元数据过滤器
- ✅ LTM 节点持久化
- ✅ 传递闭包推理

## 📚 相关文档

- `unified_graph.py`: 核心图结构和认知属性定义
- `semantic_memory.py`: 语义内存管理器
- `docs/semantic_memory_framework.md`: 架构设计文档

## 🎯 下一步

1. **扩展 LTM 知识库**：为常见对象类别添加 is_a 关系
2. **实现概念聚类**：基于 family_resemblance 进行无监督聚类
3. **优化查询性能**：利用传递闭包缓存
4. **增强推理能力**：实现隐喻映射和类比推理
