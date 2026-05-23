# WM属性存储功能 - 快速参考

## 修改摘要

### ✅ 已完成

#### 1. 修复警告 (meta_flat_habitat_agent.py)
```python
# 行83: print_wm_state()
node_kind = node.properties.get('kind', 'unknown')  # 修复 node.kind 访问错误

# 行150+: print_ltm_retrieval()  
node_kind = node.properties.get('kind', 'unknown')  # 同样修复,支持dict和Node对象
```

#### 2. 新增字段 (meta_flat_habitat_agent.py, 行65)
```python
@dataclass
class PerceptionOutput:
    instruction_entities_and_attributes: List[str] = field(default_factory=list)  # 新增
```

#### 3. 新增WM存储方法 (semantic_memory.py)

**_write_object_attributes()**
- 位置: 行480+
- 功能: 存储detected_object_attributes
- 支持格式: 字符串("red apple") / 字典({"object": "apple", "color": "red"})
- 创建: 属性节点 + has_attribute/has_color/has_size边

**_write_instruction_entities()**
- 位置: 行520+
- 功能: 存储instruction_entities_and_attributes
- 仅在step<=1时调用
- 创建结构:
  ```
  [instruction_requirements] --requires--> [red apple (instruction_entity)]
                                               ↓ has_attribute
                                          [red (attribute)]
  ```

#### 4. 修改write_perception (semantic_memory.py, 行393+)
```python
# 获取新数据
obj_attrs = perception.get("detected_object_attributes", []) or []
instruction_entities = perception.get("instruction_entities_and_attributes", []) or []

# 写入WM
self._write_object_attributes(obj_attrs, step=step)
if step <= 1:
    self._write_instruction_entities(instruction_entities, step=step)
```

#### 5. 增强检索逻辑 (semantic_memory.py, read方法)
```python
# 优先检索instruction_entity节点
instruction_entities = self.wm.find_nodes(properties={"type": "instruction_entity"}, limit=20)
for entity_node in instruction_entities:
    seed_ids[entity_node.id] = 0.9  # 高分数
    # 包含属性邻居
    for edge with relation in ["has_attribute", "has_color", "has_size"]:
        seed_ids[attr_node.id] = 0.85
```

#### 6. 增强日志 (semantic_memory.py, ingest_perception)
```python
logger.info(f"  - Object attributes: {perception.get('detected_object_attributes', [])}")
logger.info(f"  - Instruction entities: {perception.get('instruction_entities_and_attributes', [])}")
```

## 数据结构变化

### 新节点类型
1. `instruction_entity`: 指令中的目标实体
2. `goal`: instruction_requirements节点,作为所有实体的锚点
3. `attribute`: 对象/实体的属性(颜色、大小等)

### 新边关系
1. `requires`: goal节点到instruction_entity的边
2. `has_attribute`: 实体/对象到属性的通用边
3. `has_color`, `has_size`, `has_shape`: 具体属性类型边

## 语义理解提升

### 之前
```
Instruction: "找到红苹果"
WM存储: [apple (object)]
检索: 找到任意apple → 可能拿错绿苹果
```

### 之后
```
Instruction: "找到红苹果"
WM存储: 
  - [red apple (instruction_entity)] --has_attribute--> [red (attribute)]
  - [apple (object)] --has_color--> [red (attribute)]
检索: 
  - 优先匹配instruction_entity
  - 包含属性约束
  - 减少目标混淆
```

## 测试检查点

- [ ] 无语法错误 (get_errors通过)
- [ ] 日志显示detected_object_attributes
- [ ] 日志显示instruction_entities_and_attributes
- [ ] WM中创建instruction_requirements节点
- [ ] WM中创建instruction_entity节点
- [ ] WM中创建attribute节点和has_*边
- [ ] 检索结果包含instruction_entity
- [ ] 检索结果包含属性邻居

## 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| Node.kind修复1 | meta_flat_habitat_agent.py | 83 |
| Node.kind修复2 | meta_flat_habitat_agent.py | 150+ |
| PerceptionOutput新字段 | meta_flat_habitat_agent.py | 65 |
| _write_object_attributes | semantic_memory.py | 480+ |
| _write_instruction_entities | semantic_memory.py | 520+ |
| write_perception修改 | semantic_memory.py | 393+ |
| read方法增强 | semantic_memory.py | 848+ |
| ingest_perception日志 | semantic_memory.py | 1520+ |

## 运行验证

```bash
# 检查语法
python -m py_compile embodiedbench/evaluator/meta_flat_habitat_agent.py
python -m py_compile embodiedbench/evaluator/semantic_memory.py

# 运行测试 (如果有)
cd embodiedbench
python -m pytest tests/test_semantic_memory.py -v

# 运行实际任务查看日志
python main.py --config configs/eb-hab.yaml --max_episodes 1
```

查看日志关键输出:
```
[SemanticMemory] 📝 Ingesting perception at step 0...
  - Detected objects: [...]
  - Object attributes: [...]        # ← 新增
  - Instruction entities: [...]     # ← 新增
  - Objects relations: [...]

📊 Working Memory State:
  instruction_entity nodes: [...]   # ← 新增
  attribute nodes: [...]            # ← 新增
```
