# Initial Location Memory - 初始位置记忆功能

## 🎯 设计目标

既然环境是固定的（每个任务开始时重置到初始状态），LTM 应该记住**对象的初始位置**，这样下次遇到相同任务时，机器人可以直接知道去哪里找对象，而不是盲目搜索。

## 📊 核心设计

### 1. 区分两种空间信息

| 类型 | 属性/关系 | 是否存入LTM | 用途 |
|------|----------|------------|------|
| **临时位置** | `current_position` 属性<br>`located_at` 边 | ❌ 不存 | 对象被移动后的当前位置 |
| **初始位置** | `initial_location` 属性<br>`usually_found_at` 边 | ✅ 存入 | 对象在环境重置时的位置 |

### 2. 属性定义更新

```python
# 临时属性（不同步到LTM）
EPHEMERAL_PROPERTIES = {
    "spatial_position", "last_seen_position", "visible", "last_seen_ts",
    "step", "recency", "source", "timestamp", "last_access",
    "current_position",  # 🔥 当前位置 - 临时
    "holding",  # 是否被持有 - 临时状态
}

# 长期属性（同步到LTM）
PERSISTENT_PROPERTIES = {
    "category", "materials", "affordances", "colors", "size_hint",
    "shape_hint", "texture", "temperature_tolerance", "edibility",
    "features", "usage_notes",
    "initial_location",  # 🔥 初始位置 - 长期
    "spawn_point",  # 对象的出生点
}
```

### 3. 关系类型扩展

```python
RELATION_TYPES = {
    # 空间关系（临时 - 当前位置）
    "located_at", "on", "in", "near", "above", "below", "left_of", "right_of",
    
    # 🔥 空间关系（长期 - 初始布局知识）
    "usually_found_at",  # 通常在哪里找到
    "initially_at",       # 初始位置
    "spawns_at",          # 生成点
    
    # 语义关系（长期）
    "is_a", "has_property", "has_color", ...
}
```

## 🔄 工作流程

### Episode 开始时（step 1）

```
1. Perception 检测到对象: strawberry @ kitchen_counter
   
2. _add_object_from_perception() 处理:
   ├─ 创建对象节点: strawberry
   ├─ 创建位置节点: kitchen_counter
   ├─ 添加临时边: strawberry --located_at--> kitchen_counter (ephemeral=True)
   └─ 🔥 检查 env_step <= 1:
      ├─ 设置属性: strawberry.initial_location = "kitchen_counter"
      └─ 添加长期边: strawberry --usually_found_at--> kitchen_counter (ephemeral=False)
```

### Episode 进行中（step > 1）

```
3. 机器人执行: pick up strawberry, place at sofa
   
4. 位置更新:
   ├─ 删除旧的 located_at 边: strawberry --located_at--> kitchen_counter
   ├─ 添加新的 located_at 边: strawberry --located_at--> sofa (ephemeral=True)
   └─ 保留 usually_found_at 边: strawberry --usually_found_at--> kitchen_counter (不变)
```

### Episode 结束时（LTM 巩固）

```
5. on_episode_end() 调用 _llm_consolidate_wm_to_ltm():
   
6. LLM 分析 WM 数据:
   WM Knowledge:
   - strawberry {initial_location: "kitchen_counter"}
   - strawberry --usually_found_at--> kitchen_counter (ephemeral=False)
   - strawberry --located_at--> sofa (ephemeral=True) ❌ 被过滤掉
   
7. LLM 决策:
   {
     "action": "add_ltm_edge",
     "source": "strawberry",
     "target": "kitchen_counter",
     "relation": "usually_found_at"
   }
   
8. 保存到 LTM:
   strawberry --usually_found_at--> kitchen_counter ✅
```

### 下一个 Episode 开始时

```
9. prepare_planner_context() 查询 LTM:
   Goal: "pick up strawberry"
   
10. LTM 返回:
    ltm_hints: [
      {
        "label": "strawberry",
        "location": "kitchen_counter",
        "relation": "usually_found_at",
        "confidence": 0.9
      }
    ]
    
11. Planner 得到提示:
    "Based on LTM knowledge, strawberry is usually found at kitchen_counter.
     Suggest navigating there first."
```

## 🎨 示例场景

### 场景 1: 学习初始布局

**Episode 1:**
```
Step 1: Perception detects:
  - strawberry on kitchen_counter
  - apple on dining_table
  - banana on refrigerator_shelf

WM after step 1:
  strawberry {initial_location: "kitchen_counter"}
    --usually_found_at--> kitchen_counter (ephemeral=False)
  apple {initial_location: "dining_table"}
    --usually_found_at--> dining_table (ephemeral=False)
  banana {initial_location: "refrigerator_shelf"}
    --usually_found_at--> refrigerator_shelf (ephemeral=False)

LTM after episode end:
  ✅ strawberry --usually_found_at--> kitchen_counter
  ✅ apple --usually_found_at--> dining_table
  ✅ banana --usually_found_at--> refrigerator_shelf
```

**Episode 2 (same environment):**
```
Goal: "pick up strawberry"

Planner receives LTM hint:
  "strawberry is usually found at kitchen_counter"

Plan:
  1. navigate to kitchen_counter  ← 直接去正确位置！
  2. pick up strawberry
```

### 场景 2: 区分临时和初始位置

**Episode 1:**
```
Step 1: strawberry @ kitchen_counter (initial)
  → strawberry.initial_location = "kitchen_counter"
  → strawberry --usually_found_at--> kitchen_counter

Step 5: pick up strawberry

Step 6: place at sofa
  → strawberry --located_at--> sofa (ephemeral=True, 不存LTM)
  → strawberry.initial_location 依然是 "kitchen_counter"

Episode end:
  LTM 保存: strawberry --usually_found_at--> kitchen_counter ✅
  LTM 不保存: strawberry --located_at--> sofa ❌ (被过滤)
```

**Episode 2:**
```
Environment reset → strawberry 回到 kitchen_counter

Planner query LTM:
  ✅ "strawberry usually_found_at kitchen_counter" (正确！)
  ❌ 不会误导到 "sofa" (因为临时位置没存)
```

## 📝 LLM Prompt 更新

在 `_llm_consolidate_wm_to_ltm` 的 prompt 中：

```python
**Consolidation Strategy**:
- **🔥 Learn initial locations**: 
  * Use WM's "initial_location" property and "usually_found_at" edges
  * Example: If WM has {"strawberry": {"initial_location": "kitchen counter"}},
    add to LTM: strawberry --usually_found_at--> kitchen_counter
  * This helps the robot know where to look first in future episodes
  
- **Extract patterns**: If robot frequently found objects at certain locations,
  strengthen "usually_found_at" relations
  
- **Validate with outcomes**: Successful actions validate location knowledge
```

## ✅ 优势

1. **零样本导航**: 下次遇到相同环境，机器人知道去哪里找对象
2. **减少搜索时间**: 不需要盲目探索，直接去初始位置
3. **知识累积**: 多个 episode 后，LTM 包含完整的环境布局
4. **避免混淆**: 不会把临时位置当成初始位置

## 🚀 未来扩展

1. **统计频率**: 如果对象多次在同一位置找到，增加置信度
2. **多位置支持**: 对象可能在多个位置出现，记录概率分布
3. **时间衰减**: 长时间未见的位置信息逐渐降低权重
4. **主动验证**: 如果 LTM 说的位置找不到，尝试其他位置并更新 LTM

## 🔧 实现细节

### 代码位置

- **常量定义**: `semantic_memory.py` Line 54-78
- **感知处理**: `_add_object_from_perception()` Line 915-948
- **关系处理**: `_parse_and_add_relation()` Line 1020-1070
- **LTM 巩固**: `_llm_consolidate_wm_to_ltm()` Line 3253-3265

### 关键判断条件

```python
# 只在初始观察时记录 usually_found_at
if env_step <= 1 and "initial_location" not in obj_node.properties:
    obj_node.properties["initial_location"] = pos_label
    self.wm_graph.add_edge(
        obj_node.id, pos_node.id, "usually_found_at",
        metadata=EdgeMetadata(ephemeral=False)  # 🔥 长期边
    )
```

## 📊 测试验证

运行任务后，检查日志：

```
[SemanticMemory] 📍 Recorded initial location: 'strawberry' usually_found_at 'kitchen_counter'
...
[SemanticMemory] Episode End: Syncing to LTM
[SemanticMemory] LTM consolidation: added edge 'strawberry --usually_found_at--> kitchen_counter'
```

下次运行相同任务：

```
[SemanticMemory] LTM retrieval: strawberry usually_found_at kitchen_counter (confidence=0.9)
[Planner] Using LTM hint: navigate to kitchen_counter first
```

---

**总结**: 通过区分临时位置和初始位置，LTM 可以记住环境的固定布局，显著提升机器人在重复任务中的效率。
