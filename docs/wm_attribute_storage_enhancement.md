# WM属性存储增强功能文档

## 概述
本次修改解决了日志中的警告,并实现了将`detected_objects`、`detected_object_attributes`、`instruction_entities_and_attributes`存储到工作记忆(WM)中,并在检索时利用这些信息的功能。

## 修改内容

### 1. 修复Node属性访问警告

#### 问题
代码中多处错误访问了`node.kind`,但Node对象没有直接的`kind`属性,应该通过`node.properties.get('kind')`访问。

#### 解决方案
**文件**: `embodiedbench/evaluator/meta_flat_habitat_agent.py`

- **行83**: `print_wm_state()`函数
  ```python
  # 修改前:
  logger.info(f"  {i}. [{node.kind}] {node.label}")
  
  # 修改后:
  node_kind = node.properties.get('kind', 'unknown')
  logger.info(f"  {i}. [{node_kind}] {node.label}")
  ```

- **行150+**: `print_ltm_retrieval()`函数
  ```python
  # 修改前:
  logger.info(f"  {i}. [{node.kind}] {node.label}")
  
  # 修改后:
  # 处理node可能是dict或Node对象的情况
  if isinstance(node, dict):
      node_kind = node.get('properties', {}).get('kind', 'unknown')
      node_label = node.get('label', 'N/A')
      # ...
  else:
      node_kind = node.properties.get('kind', 'unknown')
      node_label = node.label
      # ...
  ```

### 2. 增强感知输出数据结构

#### PerceptionOutput dataclass
**文件**: `embodiedbench/evaluator/meta_flat_habitat_agent.py` (行60-65)

新增字段用于存储指令实体信息:
```python
@dataclass
class PerceptionOutput:
    scene_summary: str
    detected_objects: List[str]
    detected_object_attributes: List[str]  # 已存在
    objects_relations: List[str]
    state_changes: List[str]
    environment_snapshot: Dict[str, Any]
    instruction_entities_and_attributes: List[str] = field(default_factory=list)  # 新增
```

### 3. 实现对象属性存储到WM

#### 新增方法: `_write_object_attributes`
**文件**: `embodiedbench/evaluator/semantic_memory.py`

功能:
- 将感知模块输出的详细对象属性写入WM
- 支持字符串格式("red apple")和字典格式({"object": "apple", "color": "red"})
- 创建属性节点并建立`has_attribute`/`has_color`/`has_size`等边

```python
def _write_object_attributes(self, obj_attrs: List[Any], step: int) -> None:
    """写入detected_object_attributes"""
    # 支持多种格式解析
    # 创建属性节点并关联到对象节点
    # 使用persistent=True确保属性信息稳定
```

### 4. 实现指令实体存储到WM

#### 新增方法: `_write_instruction_entities`
**文件**: `embodiedbench/evaluator/semantic_memory.py`

功能:
- 在第一步感知时,将指令中的实体和属性存入WM
- 创建特殊的`instruction_requirements`节点作为目标锚点
- 创建`instruction_entity`类型的节点,并通过`requires`边链接到目标节点
- 解析实体的属性并创建`has_attribute`边

核心逻辑:
```python
def _write_instruction_entities(self, instruction_entities: List[Any], step: int) -> None:
    # 创建goal节点
    goal_node = self._get_or_create_node("instruction_requirements", ntype="goal", ...)
    
    # 为每个实体创建节点
    for entity_item in instruction_entities:
        entity_node = self._get_or_create_node(entity_label, ntype="instruction_entity", ...)
        self._add_edge(goal_node.id, entity_node.id, "requires", ...)
        
        # 解析并创建属性节点
        # 建立 entity -> has_attribute -> attribute 边
```

### 5. 修改write_perception调用流程

**文件**: `embodiedbench/evaluator/semantic_memory.py` (write_perception方法)

```python
def write_perception(self, perception: Dict[str, Any], step: int, clip_id: Optional[str] = None) -> None:
    # 原有对象节点写入逻辑...
    
    # 🔥 新增: 写入对象属性
    obj_attrs = perception.get("detected_object_attributes", []) or []
    self._write_object_attributes(obj_attrs, step=step)
    
    # 🔥 新增: 在第一步写入指令实体
    instruction_entities = perception.get("instruction_entities_and_attributes", []) or []
    if step <= 1:
        self._write_instruction_entities(instruction_entities, step=step)
    
    # 原有关系写入逻辑...
```

### 6. 增强WM检索逻辑

#### 修改方法: `read`
**文件**: `embodiedbench/evaluator/semantic_memory.py`

新增功能:
1. **优先检索指令实体节点**: 查找所有`instruction_entity`类型节点,给予高初始分数(0.9)
2. **扩展属性邻居**: 当检索到实体或对象节点时,自动包含其属性节点(通过`has_*`关系)
3. **增强语义匹配**: 利用属性信息提高相关性判断

```python
def read(self, goal: str, step: int, max_nodes: int = 30) -> ...:
    # 原有索引召回...
    
    # 🔥 优先检索instruction_entity节点
    instruction_entities = self.wm.find_nodes(properties={"type": "instruction_entity"}, limit=20)
    for entity_node in instruction_entities:
        seed_ids[entity_node.id] = 0.9  # 高分数
        # 加入属性邻居
        for edge in self.wm.edges.values():
            if edge.source == entity_node.id and edge.relation in ["has_attribute", "has_color", ...]:
                seed_ids[attr_node.id] = 0.85
    
    # 关键词召回时也扩展属性
    for kw in extract_goal_keywords(goal):
        for n, sc in ranked:
            if n.properties.get("type") == "object":
                # 加入对象的属性邻居
                ...
```

### 7. 增强日志输出

#### ingest_perception日志
**文件**: `embodiedbench/evaluator/semantic_memory.py`

```python
def ingest_perception(self, perception: Dict[str, Any], step: int, clip_id: Optional[str] = None) -> None:
    logger.info(f"[SemanticMemory] 📝 Ingesting perception at step {step}...")
    logger.info(f"  - Detected objects: {perception.get('detected_objects', [])}")
    logger.info(f"  - Object attributes: {perception.get('detected_object_attributes', [])}")  # 新增
    logger.info(f"  - Instruction entities: {perception.get('instruction_entities_and_attributes', [])}")  # 新增
    logger.info(f"  - Objects relations: ...")
    # ...
```

## 数据流

```
感知模块 (PerceptionModule)
    ↓
    |- detected_objects: ["apple", "table"]
    |- detected_object_attributes: ["red apple", "wooden table"]
    |- instruction_entities_and_attributes: ["red apple", "kitchen table"]
    ↓
SemanticMemory.ingest_perception()
    ↓
WMOperators.write_perception()
    ↓
    |- _write_object_attributes() → 创建属性节点和has_*边
    |- _write_instruction_entities() → 创建instruction_entity节点和requires/has_attribute边
    ↓
WM图结构:
    [instruction_requirements] --requires--> [red apple (instruction_entity)]
                                                 ↓ has_attribute
                                            [red (attribute)]
    [apple (object)] --has_color--> [red (attribute)]
                     --has_size--> [small (attribute)]
```

## 检索增强效果

### 原有检索流程
```
goal: "找到红苹果"
  → 关键词: ["找到", "红", "苹果"]
  → 索引召回: [apple节点]
  → PPR扩展: 附近节点
```

### 增强后检索流程
```
goal: "找到红苹果"
  → 优先召回instruction_entity节点: [red apple (instruction_entity)]
  → 自动包含属性节点: [red (attribute)]
  → 关键词召回: [apple (object)] + 其属性邻居 [red, small, ...]
  → PPR扩展: 更精确的语义相关子图
  → 结果: 不仅知道要找apple,还知道要匹配red属性
```

## 语义理解提升

### 问题场景(修改前)
日志显示:"LLM检测到plum但WM中无plum,转而对ball执行pick"

原因:
- WM只存储对象标签,不存储详细属性
- 检索时无法区分"红苹果"和"绿苹果"
- 指令实体信息丢失,导致目标不明确

### 改进后
1. **初始感知**: 存储"red apple"为instruction_entity + attribute节点
2. **检索阶段**: 优先匹配instruction_entity,包含属性约束
3. **规划阶段**: LLM看到完整的"red apple"实体和属性关系
4. **执行阶段**: 更准确的目标匹配,减少幻觉

## 测试要点

1. **属性存储测试**:
   - 验证detected_object_attributes正确写入WM
   - 检查属性节点和边的创建

2. **指令实体测试**:
   - 验证第一步是否创建instruction_requirements节点
   - 检查instruction_entity节点和requires边

3. **检索增强测试**:
   - 验证read()优先返回instruction_entity节点
   - 检查属性邻居是否包含在检索结果中

4. **日志验证**:
   - 确认ingest_perception打印新增字段
   - 确认print_wm_state显示新的节点类型

## 注意事项

1. **节点类型**: 新增了`instruction_entity`和`goal`类型节点
2. **边关系**: 新增了`requires`和`has_attribute`关系
3. **持久性**: 所有属性和指令实体节点设置为`persistent=True`
4. **时间戳**: 仅在`step<=1`时存储指令实体(初始感知)

## 相关文件

- `embodiedbench/evaluator/meta_flat_habitat_agent.py`: 感知模块和日志输出
- `embodiedbench/evaluator/semantic_memory.py`: WM存储和检索逻辑
- `embodiedbench/evaluator/unified_graph.py`: Node/Edge数据结构

## 下一步优化建议

1. **动态属性更新**: 当检测到同一对象的新属性时,更新而非重复创建
2. **属性权重**: 不同属性(color vs size)可能需要不同的重要性权重
3. **属性冲突**: 处理"red apple"和"green apple"的属性冲突
4. **LLM总结**: 在episode_end时总结属性学习的模式
