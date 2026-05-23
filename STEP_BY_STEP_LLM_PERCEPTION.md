# ✅ 每步执行后调用LLM感知 - 实现完成

## 📋 需求

在每个step执行后调用LLM感知新的image，分析前面的执行动作和环境反馈，然后更新WM。

## 🎯 实现方案

### 1️⃣ 新增方法：`PerceptionModule.perceive_with_action_history()`

**位置**: `embodiedbench/evaluator/meta_flat_habitat_agent.py` 第438-591行

**功能**: 专门用于每步执行后的LLM感知，包含动作执行上下文

**方法签名**:
```python
def perceive_with_action_history(
    self,
    env: EBHabEnv,                      # 当前环境
    img_path: str,                      # 执行action后的新图像路径
    instruction: str,                   # 原始任务指令
    last_action: str,                   # 刚执行的动作描述
    recent_actions: List[Dict[str, Any]], # 最近的动作历史
    env_feedback: Optional[str] = None, # 环境反馈信息
) -> PerceptionOutput
```

**与原 `perceive()` 的区别**:

| 特性 | `perceive()` (初始感知) | `perceive_with_action_history()` (执行后感知) |
|------|------------------------|---------------------------------------------|
| **调用时机** | Episode开始时 | 每步执行action后 |
| **输入信息** | 仅当前图像 + 指令 | 图像 + 指令 + 动作历史 + 环境反馈 |
| **分析重点** | 场景布局、对象位置 | 动作执行结果、状态变化 |
| **Prompt** | "描述当前场景" | "分析动作执行效果和环境变化" |
| **上下文** | 无历史 | 最近5步动作历史 |

### 2️⃣ 修改执行循环调用逻辑

**位置**: `embodiedbench/evaluator/meta_flat_habitat_agent.py` 第1267-1333行

**原来的逻辑** ❌:
```python
# 保存图像
img_path = self.env.save_image(obs)

# 只做规则化correctness更新（不调用LLM）
self.semantic_memory.ingest_action_feedback(
    action_desc=action_desc,
    env_info=info,
    reward=float(reward),
    ...
)
```

**新的逻辑** ✅:
```python
# 1. 保存执行后的图像
img_path = self.env.save_image(obs)

# 2. 构建最近动作历史
recent_actions = [
    {
        "action_desc": step.action_desc,
        "reward": step.reward,
        "invalid": step.invalid,
        "env_feedback": step.env_feedback,
    }
    for step in wm.execution_trace[-5:]  # 最近5步
]

# 3. 🔥 调用LLM感知（分析执行结果）
p_out_post_action = self.perception.perceive_with_action_history(
    env=self.env,
    img_path=img_path,
    instruction=instruction,
    last_action=action_desc,
    recent_actions=recent_actions,
    env_feedback=info.get("env_feedback"),
)

# 4. 添加到感知历史
wm.perception_history.append(p_out_post_action)

# 5. 🔥 将LLM感知结果写入WM图
self.semantic_memory.ingest_perception(
    asdict(p_out_post_action),
    step=self.env._current_step,
    clip_id=img_path,
)

# 6. 更新WM摘要
wm.semantic_memory_digest = self.semantic_memory.export_digest(limit=32)
```

### 3️⃣ LLM Prompt设计

#### System Prompt（第466-490行）

```
You are a perception module for a household robot analyzing execution results.
You receive:
1) The CURRENT RGB image after executing an action
2) The action that was just executed
3) Recent action history and environment feedback
4) The overall task instruction

Your job is to analyze:
- What changed after executing the action?
- Did the action succeed or fail? Why?
- What's the current state of relevant objects?
- What should the robot know for next steps?

Output a JSON with these exact fields:
{
  "scene_summary": string,
  "detected_objects": [string],
  "objects_relations": [string],
  "state_changes": [string],
  "environment_snapshot": string
}
```

#### User Prompt（第492-528行）

```
[Task instruction]: {instruction}

[Just executed action]: {last_action}

[Environment feedback]: {env_feedback}

[Recent action history (last 5 steps)]:
  1. Action: pick up spoon, Reward: 0.5, Invalid: False, Feedback: success
  2. Action: navigate to table, Reward: 0.3, Invalid: False
  3. Action: place on table, Reward: 1.0, Invalid: False, Feedback: task completed
  ...

[Current robot state]:
- Episode: 5
- Step: 12
- Holding object: True

Analyze the current image and tell me:
1. What is the result of the last action?
2. What changed in the environment?
3. What objects are visible and how are they arranged?
4. What state information is relevant for next steps?
```

## 🔄 完整工作流程

```
Episode 开始
│
├─ 【初始感知】perceive()
│   └─ LLM分析初始场景
│       └─ ingest_perception() → 写入WM
│
├─ 【规划】plan()
│   └─ 从WM读取上下文
│       └─ 生成action序列
│
├─ 【执行循环】
│   ├─ 执行 action_1
│   │   ├─ env.step() → obs, reward, info
│   │   ├─ 保存图像 img_path_1
│   │   │
│   │   ├─ 🔥🔥🔥 【新增】每步LLM感知
│   │   │   ├─ perceive_with_action_history(
│   │   │   │     img_path=img_path_1,
│   │   │   │     last_action="pick up spoon",
│   │   │   │     recent_actions=[...],
│   │   │   │     env_feedback="success"
│   │   │   │   )
│   │   │   ├─ → PerceptionOutput(
│   │   │   │      scene_summary="Robot successfully picked up spoon...",
│   │   │   │      detected_objects=["spoon", "table", ...],
│   │   │   │      state_changes=["spoon moved from table to gripper"],
│   │   │   │      ...
│   │   │   │    )
│   │   │   │
│   │   │   └─ ingest_perception() → 🔥 更新WM图
│   │   │       ├─ 添加新检测到的对象节点
│   │   │       ├─ 更新对象关系边
│   │   │       ├─ 记录状态变化
│   │   │       └─ 更新置信度
│   │   │
│   │   └─ export_digest() → 更新WM摘要
│   │
│   ├─ 执行 action_2
│   │   ├─ env.step()
│   │   ├─ 🔥 perceive_with_action_history()
│   │   └─ 🔥 ingest_perception() → 更新WM
│   │
│   └─ ... (每步都调用LLM感知)
│
└─ Episode 结束
    └─ on_episode_end() → WM巩固到LTM
```

## 📊 LLM调用频率对比

### 修改前 ❌

| 阶段 | LLM调用 | 频率 |
|------|---------|------|
| Episode开始 | ✅ | 1次 |
| 每步执行后 | ❌ | 0次（规则更新） |
| 反馈触发重感知 | ✅ | 0-3次（按需） |
| **总计** | - | **1-4次/episode** |

### 修改后 ✅

| 阶段 | LLM调用 | 频率 |
|------|---------|------|
| Episode开始 | ✅ | 1次 |
| **每步执行后** | **✅** | **N次（N=步数）** |
| 反馈触发重感知 | ✅ | 0-3次（按需） |
| **总计** | - | **N+1 ~ N+4次/episode** |

**典型50步episode**: 从 1-4次 → **51-54次** LLM调用 🚀

## 🎯 关键改进点

### 1. 上下文感知
- **修改前**: 每步更新只基于reward和validity（规则）
- **修改后**: 每步调用LLM分析图像+动作+反馈（语义理解）

### 2. 动作历史整合
```python
recent_actions = [
    {
        "action_desc": "pick up spoon",
        "reward": 0.5,
        "invalid": False,
        "env_feedback": "success"
    },
    # ... 最近5步
]
```

### 3. 环境反馈分析
```python
env_feedback = info.get("env_feedback")
# 例如: "Object successfully picked up" 或 "Action failed: object not reachable"
```

### 4. 状态变化追踪
```json
{
  "state_changes": [
    "spoon moved from table to robot gripper",
    "table surface is now empty",
    "robot is now holding an object"
  ]
}
```

## 💡 错误处理

### 容错机制（第1305-1320行）

```python
try:
    # 尝试LLM感知
    p_out_post_action = self.perception.perceive_with_action_history(...)
    self.semantic_memory.ingest_perception(asdict(p_out_post_action), ...)
    
except Exception as e:
    logger.warning(f"Post-action LLM perception failed: {e}")
    
    # 🔥 失败时回退到规则更新
    try:
        self.semantic_memory.ingest_action_feedback(
            action_desc=action_desc,
            env_info=info,
            reward=float(reward),
            step=step,
        )
    except Exception as e2:
        logger.warning(f"Fallback correctness update also failed: {e2}")
```

**三层保护**:
1. 首选：LLM感知 + WM更新
2. 回退：规则化correctness更新
3. 最终：记录错误但不中断执行

## 📈 性能考虑

### 成本分析

假设一个episode平均50步：

| 项目 | 修改前 | 修改后 | 增加 |
|------|--------|--------|------|
| LLM调用次数 | 1-4次 | 51-54次 | **~50次** |
| 每次Token（估算） | ~1000 | ~1500（含历史） | +50% |
| 总Token/episode | ~2000 | ~76,500 | **38倍** |
| 延迟（估算） | ~2秒 | ~100秒 | **+98秒** |

### 优化建议

1. **选择性调用**: 只在关键步骤调用（如每5步或检测到状态变化时）
2. **批量处理**: 累积多步后一次性分析
3. **缓存机制**: 相似场景复用之前的感知结果
4. **模型选择**: 使用更快的模型（如gpt-4o-mini而非gpt-4o）

### 可配置开关（建议添加）

```python
# 在config中添加
enable_step_by_step_perception: bool = True  # 是否启用每步LLM感知
perception_interval: int = 1  # 每N步调用一次（1=每步）
```

## ✅ 验证结果

```
======================================================================
验证修改后的代码
======================================================================
✅ Python语法检查通过
✅ 找到新方法: perceive_with_action_history
✅ 执行循环中调用了 perceive_with_action_history
✅ 包含post-action感知日志
✅ perceive_with_action_history 参数完整
======================================================================
验证完成！
======================================================================
```

## 🚀 下一步建议

1. **运行时测试**: 
   ```bash
   python embodiedbench/main.py --config embodiedbench/configs/eb-hab.yaml
   ```

2. **监控LLM调用**:
   - 查看日志中的 `[PerceptionModule] Post-action LLM perception output:`
   - 统计每个episode的LLM调用次数和成本

3. **评估效果**:
   - 比较修改前后的任务成功率
   - 分析WM图的丰富程度（节点/边数量）
   - 检查规划质量的提升

4. **性能优化**:
   - 如果成本过高，添加调用频率控制
   - 考虑使用异步调用减少延迟
   - 实现结果缓存机制

## 📝 代码变更摘要

### 新增代码
- **PerceptionModule.perceive_with_action_history()**: 第438-591行（154行）
  - 构建包含动作历史的prompt
  - 调用LLM分析执行结果
  - 返回结构化PerceptionOutput

### 修改代码
- **HabitatMetaFlatAgent.run_single_episode()**: 第1267-1333行（67行）
  - 构建recent_actions列表
  - 调用perceive_with_action_history()
  - 将结果写入WM图
  - 添加完整的错误处理

### 总代码增量
- **新增**: ~154行
- **修改**: ~67行
- **总计**: ~221行

## 🎉 实现完成！

现在系统在**每个step执行后**都会：
1. ✅ 调用LLM感知新图像
2. ✅ 分析执行动作和环境反馈
3. ✅ 生成结构化的PerceptionOutput
4. ✅ 更新WM图（添加节点、边、状态变化）
5. ✅ 为下一步规划提供最新的语义理解

**核心优势**: 从基于规则的置信度更新 → 基于LLM的语义理解和状态追踪！
