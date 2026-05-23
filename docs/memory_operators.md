# 语义内存算子文档

## 概述

新的统一图架构中实现了 **6 大核心语义理解算子**，这些算子是实现高级语义推理的关键。

---

## 算子列表

### 1. **align_semantics** - 语义对齐

**功能**: 将抽象概念映射到具体实例

**用途**: 当 planner 请求一个抽象概念（如 "receptacle"）时，自动查找并对齐到具体对象（如 "sink", "counter"）

**调用时机**: `prepare_planner_context()` 检测到 knowledge gap 时

**工作流程**:
```
1. 解析 gap: "entity:object:receptacle" → core_term="receptacle"
2. 查询预定义映射: receptacle → [sink, basin, bowl, counter, bin]
3. 从 LTM 查询 is_a 关系: sink is_a receptacle
4. 匹配观察到的对象: 当前看到 "sink"
5. 建立对齐: receptacle ← synonym_of → sink (confidence=0.9)
6. 在 WM 中添加/更新 sink 节点，附加 "receptacle" 别名
```

**示例**:
```python
# Goal: "Place the apple in the receptacle"
# WM 中只有: [apple, sink, counter]

gaps = ["entity:object:receptacle"]
observed = ["apple", "sink", "counter"]

alignments = operators.align_semantics(gaps, observed, step=10)
# Result: [
#   {
#     "abstract_term": "receptacle",
#     "concrete_term": "sink",
#     "confidence": 0.9,
#     "node_id": "node_abc123"
#   }
# ]

# 现在 planner 知道 "receptacle" 就是 "sink"
```

**关键参数**:
- `pending_gaps`: 待解析的实体列表（格式：`entity:kind:label`）
- `observed_labels`: 当前观察到的对象标签
- `step`: 当前步数

---

### 2. **consolidate** - 知识巩固

**功能**: 合并 LTM 中相似的节点，减少冗余

**用途**: Episode 结束后整理 LTM，避免 "apple" 和 "red_apple" 分别存储

**调用时机**: `on_episode_end()` → Step 6

**工作流程**:
```
1. 遍历 LTM 中所有节点对
2. 计算标签相似度（Jaccard）
3. 如果 similarity >= 0.82:
   a. 合并属性: node1.properties.update(node2.properties)
   b. 转移边: 将指向 node2 的边重定向到 node1
   c. 删除 node2
4. 返回统计: {"merged_pairs": 5}
```

**示例**:
```python
# LTM Before:
# - "apple" (red, edible)
# - "red_apple" (shiny, sweet)

consolidate_stats = operators.consolidate(threshold=0.82)
# Result: {"merged_pairs": 1}

# LTM After:
# - "apple" (red, edible, shiny, sweet)  ← 合并后的节点
```

**关键参数**:
- `threshold`: 相似度阈值（默认 0.82）

---

### 3. **decompose** - 概念分解

**功能**: 将高 fuzziness（模糊性）节点拆分为情境化子节点

**用途**: 当一个概念在不同情境下有不同含义时，拆分成多个子概念

**调用时机**: `on_episode_end()` → Step 7

**工作流程**:
```
1. 查找 fuzziness >= 0.6 的节点
2. 检查是否有多个 perspectives（视角）
3. 为每个 perspective 创建子节点:
   - 标签: "原标签:perspective"
   - fuzziness: 降低 0.2
   - 属性: 继承自父节点
4. 降低父节点的 fuzziness
```

**示例**:
```python
# LTM Before:
# - "counter" (fuzziness=0.7, perspectives=["kitchen", "bathroom"])

decompose_stats = operators.decompose(fuzziness_threshold=0.6)
# Result: {"decomposed_count": 2}

# LTM After:
# - "counter" (fuzziness=0.4)  ← 父节点
# - "counter:kitchen" (fuzziness=0.5, context-specific)
# - "counter:bathroom" (fuzziness=0.5, context-specific)
```

**关键参数**:
- `fuzziness_threshold`: Fuzziness 阈值（默认 0.6）

---

### 4. **abstract** - 抽象提取

**功能**: 将 WM 中稳定的布尔模式转化为规则

**用途**: 提取重复出现的条件，形成通用规则

**调用时机**: `on_episode_end()` → Step 1

**工作流程**:
```
1. 遍历 WM 中的节点
2. 查找稳定的布尔属性（值为 0/1 或 True/False）
3. 为每个稳定属性创建规则节点:
   - 标签: "rule_原节点标签"
   - 属性: {"type": "rule", "conditions": {...}}
4. 添加边: rule_node --applies_to--> 原节点
```

**示例**:
```python
# WM Before:
# - "microwave" (powered=True, door_closed=True)

operators.abstract(step=100)

# WM After:
# - "microwave" (powered=True, door_closed=True)
# - "rule_microwave" (conditions={powered:True, door_closed:True})
#   └─ applies_to → microwave
```

---

### 5. **correctness** - 正确性校验

**功能**: 根据执行反馈调整节点的风险等级

**用途**: 记录哪些对象/位置可能导致失败，避免重复错误

**调用时机**: `ingest_execution_feedback()` 收到负面反馈时

**工作流程**:
```
1. 从 feedback 提取 target_label
2. 在 WM 中查找匹配的节点
3. 调整 risk_level:
   - 失败: risk_level += 0.2
   - 成功: risk_level -= 0.1
4. 限制在 [0.0, 1.0] 范围内
```

**示例**:
```python
feedback = {
    "target": "fragile_vase",
    "invalid": True,  # 失败
    "reason": "object_broken"
}

operators.correctness(feedback, step=50)

# "fragile_vase".risk_level: 0.0 → 0.2
# planner 下次会避开这个对象
```

---

### 6. **simplify** - 图简化

**功能**: 剪枝 WM 中低价值的节点，保持图精简

**用途**: 防止 WM 过大影响性能

**调用时机**: `on_episode_end()` → Step 2

**工作流程**:
```
1. 检查 WM 节点数是否超过 max_nodes
2. 计算每个节点的价值分数:
   score = edge_count * 10 - recency * 0.1
3. 按分数排序，删除最低分的节点
4. 返回删除数量
```

**示例**:
```python
# WM: 300 个节点，max_nodes=256

removed = operators.simplify(step=100, max_nodes=256)
# Result: 44

# WM After: 256 个节点（保留高连接数和新近的节点）
```

**评分公式**:
```python
score = (入边数 + 出边数) * 10 - (当前步数 - 节点时间戳) * 0.1
```

---

## 算子调用顺序

### Episode 运行期间
```
step 1: 观察 → ingest_perception()
step 2: 动作 → ingest_execution_feedback()
        └─ 如果失败 → correctness()
step 3: planner 请求上下文 → prepare_planner_context()
        └─ 检测 gap → align_semantics()
```

### Episode 结束时（`on_episode_end()`）
```
Step 1: abstract()         # 提取规则
Step 2: simplify()         # 剪枝 WM
Step 3: 过滤临时节点
Step 4: WM → LTM 节点合并
Step 5: WM → LTM 边合并
Step 6: consolidate()      # 巩固 LTM
Step 7: decompose()        # 分解模糊节点
Step 8: 保存 LTM 到磁盘
```

---

## 算子之间的协作

### 语义对齐 + LTM 查询
```python
# align_semantics 依赖 LTM 中的 is_a 关系
# 如果 LTM 知道 "sink is_a receptacle"
# 就能自动解析 "place in receptacle" → "place in sink"
```

### 巩固 + 分解
```python
# consolidate: 合并过于相似的节点（去重）
# decompose: 拆分过于模糊的节点（细化）
# 两者互补，保持 LTM 的合理粒度
```

### 抽象 + 正确性
```python
# abstract: 提取稳定模式 → 规则
# correctness: 标记风险 → 避免
# 结合后可以形成 "if risk_level > 0.5 then avoid" 规则
```

---

## 实现方式

所有算子都支持**两种实现方式**：

### 1. 算法实现（当前）
- 基于图算法和启发式规则
- 速度快，确定性强
- 适合实时运行

### 2. LLM 实现（可扩展）
```python
def align_semantics_with_llm(self, gaps, observed):
    prompt = f"""
    Abstract concepts: {gaps}
    Observed objects: {observed}
    
    Map each abstract concept to the most likely object.
    Return JSON: {{"receptacle": "sink", ...}}
    """
    
    result = llm_client.complete(prompt)
    return parse_json(result)
```

**优势**: LLM 能处理更复杂的语义推理
**劣势**: 速度慢，需要 API 调用

---

## 统计与日志

所有算子运行后会更新 `self.stats`：

```python
self.stats = {
    "episodes": 10,
    "perceptions": 200,
    "actions": 150,
    "ltm_syncs": 10,
    "alignments": 35,        # align_semantics 次数
    "consolidations": 8,      # consolidate 合并对数
}
```

**日志示例**:
```
[Semantic Alignment] ✅ 'receptacle' → 'sink' (conf=0.90)
[Consolidate] Merged 'red_apple' into 'apple'
[Decompose] Decomposed 2 nodes
[Simplify] Removed 44 low-value nodes
```

---

## 总结

这 6 大算子构成了语义理解的核心能力：

1. **align_semantics**: 桥接抽象与具体
2. **consolidate**: 去除冗余知识
3. **decompose**: 细化模糊概念
4. **abstract**: 提取通用规则
5. **correctness**: 学习错误教训
6. **simplify**: 保持图精简

它们共同实现了从 **原始感知 → 结构化知识 → 高级推理** 的完整流程！
