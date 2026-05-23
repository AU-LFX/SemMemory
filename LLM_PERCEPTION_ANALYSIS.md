# LLM感知调用分析报告

## 问题1：调用LLM感知完生成的结构是什么？

### ✅ 答案：生成 `PerceptionOutput` 结构化数据类

```python
@dataclass
class PerceptionOutput:
    """对当前 Habitat 场景的抽象感知"""
    scene_summary: str                              # 场景总结（房间类型、主要家具等）
    detected_objects: List[str]                     # 检测到的对象列表
    objects_relations: List[str]                    # 对象之间的空间关系
    state_changes: List[str]                        # 状态变化记录
    environment_snapshot: Dict[str, Any]            # 环境快照
```

### 📋 具体字段示例

```json
{
  "scene_summary": "厨房场景，有餐桌、水槽、冰箱，机器人手中拿着勺子",
  
  "detected_objects": [
    "spoon",
    "dining_table", 
    "sink",
    "fridge",
    "counter"
  ],
  
  "objects_relations": [
    "the spoon is in robot's hand",
    "the fridge is left of the sink",
    "the counter is next to the dining table"
  ],
  
  "state_changes": [
    "drawer is now open",
    "cup moved from table to counter"
  ],
  
  "environment_snapshot": {
    "episode": 5,
    "step": 12,
    "is_holding": true,
    "max_steps": 500,
    "env_comment_from_model": "Kitchen with central dining table, sink on north wall, fridge to the left..."
  }
}
```

### 🔄 LLM生成流程

#### 1. PerceptionModule 构建 messages (第282-324行)

```python
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text", 
                "text": """You are a perception module for a household robot...
                Your JSON MUST have the following fields:
                {
                  "scene_summary": string,
                  "detected_objects": [string],
                  "objects_relations": [string],
                  "state_changes": [string],
                  "environment_snapshot": string
                }"""
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},  # 🖼️ 第一人称图像
            {
                "type": "text", 
                "text": f"""Current observation:
                - Episode: {env._current_episode_num}
                - Step: {env._current_step}
                - Is holding: {env.is_holding}
                - Instruction: "{instruction}"
                """
            }
        ]
    }
]
```

#### 2. 调用 VLM (第374行)

```python
raw_output = self.vision_model.respond_freeform(messages)
# 返回原始JSON字符串
```

#### 3. 解析并规范化 (第377-398行)

```python
# 修复JSON格式
fixed = fix_json(raw_output)
parsed = json.loads(fixed)

# 提取字段
scene_summary = parsed.get("scene_summary", "").strip()
detected_objects = parsed.get("detected_objects", []) or []
objects_relations = parsed.get("objects_relations", []) or []
state_changes = parsed.get("state_changes", []) or []
env_comment = parsed.get("environment_snapshot", "").strip()

# 构建environment_snapshot字典
snapshot = {
    "episode": int(env._current_episode_num),
    "step": int(env._current_step),
    "is_holding": bool(env.is_holding),
    "max_steps": int(env._max_episode_steps),
    "env_comment_from_model": env_comment,  # LLM生成的环境描述
}
```

#### 4. 返回结构化对象 (第414-420行)

```python
return PerceptionOutput(
    scene_summary=scene_summary,
    detected_objects=detected_objects,
    objects_relations=objects_relations,
    state_changes=state_changes,
    environment_snapshot=snapshot,
)
```

---

## 问题2：每个step后有没有调用LLM更新WM？

### ⚠️ 答案：**没有**！每步只做规则化的correctness更新

### 📊 当前实现逻辑

```
Episode开始
│
├─ 【第983-995行】初始感知（调用LLM）
│   └─ perception.perceive() → PerceptionOutput
│        └─ semantic_memory.ingest_perception() → 写入WM
│
├─ 【执行循环】
│   ├─ 规划 → 执行action
│   ├─ 保存新图像 (第1102行)
│   │
│   ├─ 【第1107-1116行】每步调用 ingest_action_feedback
│   │   ├─ ❌ 不调用LLM！
│   │   └─ ✅ 只做规则化correctness更新
│   │
│   └─ 【第1158-1180行】反馈触发时（可选）
│       ├─ 如果 feedback_decision.action == "reperceive"
│       │   └─ ✅ 重新调用 perception.perceive()（调用LLM）
│       └─ 否则：不调用LLM
│
└─ Episode结束
    └─ semantic_memory.on_episode_end()
        └─ ❌ 不调用LLM，只做算法巩固
```

### 🔍 详细分析

#### A. 初始感知（调用LLM）✅

**位置**: 第983-995行

```python
# 🟢 调用LLM进行感知
p_out = self.perception.perceive(
    self.env, img_path, instruction,
    feedback_hint=feedback_hint_for_perception,
)

# 存储到WorkingMemory
wm.perception_history.append(p_out)

# 写入SemanticMemory的WM图
self.semantic_memory.ingest_perception(
    asdict(p_out),              # 转为dict
    step=self.env._current_step, 
    clip_id=img_path
)
```

**调用**: `PerceptionModule.perceive()` → **调用VLM生成结构化感知**

---

#### B. 每步执行后（不调用LLM）❌

**位置**: 第1107-1116行

```python
# 保存action执行后的新图像
img_path = self.env.save_image(obs)

# 🔴 虽然传入了img_path，但实际不调用LLM！
try:
    self.semantic_memory.ingest_action_feedback(
        action_id=action_id,
        action_desc=action_desc,
        env_info=info,
        reward=float(reward),
        env_step=int(info.get("env_step", self.env._current_step)),
        img_path=img_path,  # ⚠️ 传入了图像，但方法内部不使用
    )
except Exception as e:
    logger.warning(f"[MetaFlat] Semantic memory executor ingest failed: {e}")
```

**调用**: `semantic_memory.ingest_action_feedback()` (第1326-1330行)

```python
def ingest_action_feedback(self, action_desc: str, env_info: Dict[str, Any], reward: float, step: int) -> None:
    """
    每步 action 执行后：做 correctness check（更新置信度）
    """
    self.current_step = int(step)
    # 🔴 只调用correctness，不调用LLM！
    self.wm_ops.correctness(action_desc, env_info, reward, step=int(step))
```

**注意**: 
- ⚠️ Agent传入了 `img_path`，但方法签名不接受此参数
- ⚠️ 实际调用的 `correctness()` 是**纯规则化**的置信度更新，**不涉及LLM**

---

#### C. 反馈触发重感知（调用LLM）✅

**位置**: 第1158-1180行

```python
if feedback_decision.action in ("reperceive", "reperceive_and_replan"):
    # 🟢 重新调用LLM感知
    p_out = self.perception.perceive(
        self.env, img_path, instruction,
        feedback_hint=feedback_hint_for_perception,
    )
    wm.perception_history.append(p_out)
    
    # 写入WM
    self.semantic_memory.ingest_perception(
        asdict(p_out), 
        step=self.env._current_step, 
        clip_id=img_path
    )
```

**触发条件**:
- 反馈控制器判断需要重新感知
- 通常在连续invalid actions或进度停滞时触发
- **不是每步都触发**

---

#### D. Episode结束（不调用LLM）❌

**位置**: 第1254行

```python
# Episode结束时的巩固
semantic_memory_report = self.semantic_memory.on_episode_end()
```

**调用**: `semantic_memory.on_episode_end()` (第1349-1374行)

```python
def on_episode_end(self) -> Dict[str, Any]:
    """
    Episode 结束：巩固 WM → LTM
    """
    step = int(self.current_step)
    
    # 🔴 纯算法操作，不调用LLM
    created_rules = self.wm_ops.abstract(step=step, min_support=3)
    removed_wm = self.wm_ops.simplify(step=step)
    writeback = self.ltm_ops.add_from_wm(self.wm, step=step)
    consolidated = self.ltm_ops.consolidate()
    self.save_ltm()
    
    return {
        "wm_rules_created": created_rules,
        "wm_nodes_removed": removed_wm,
        "ltm_nodes_added": writeback,
        "ltm_consolidated": consolidated,
    }
```

---

### 📈 LLM调用频率统计

| 阶段 | 是否调用LLM | 调用频率 |
|------|-------------|----------|
| **初始感知** | ✅ 是 | 每个episode **1次** |
| **每步执行后** | ❌ 否 | 0次（纯规则更新） |
| **反馈触发重感知** | ✅ 是 | **按需**（通常0-3次/episode） |
| **Episode结束** | ❌ 否 | 0次（纯算法巩固） |

**典型Episode的LLM调用次数**: 1-4次

---

### 🔧 每步执行后的实际更新机制

虽然不调用LLM，但每步会更新WM：

```python
# semantic_memory.py 第560-620行
def correctness(self, action_desc: str, env_info: Dict[str, Any], reward: float, step: int) -> None:
    """
    基于action执行结果更新WM节点的置信度
    - 成功 → 提升置信度
    - 失败 → 降低置信度
    - 持续失败 → 移除节点
    """
    # 1. 解析action涉及的对象
    objects = self._extract_objects_from_action(action_desc)
    
    # 2. 根据reward和env_info更新置信度
    if reward > 0 or not env_info.get("was_prev_action_invalid"):
        # 成功：提升置信度
        for obj in objects:
            node = self.wm.get_node_by_label(obj)
            if node:
                node.metadata.confidence = min(1.0, node.metadata.confidence + 0.1)
    else:
        # 失败：降低置信度
        for obj in objects:
            node = self.wm.get_node_by_label(obj)
            if node:
                node.metadata.confidence = max(0.0, node.metadata.confidence - 0.15)
                if node.metadata.confidence < 0.3:
                    self.wm.remove_node(node.id)  # 移除低置信度节点
```

---

## 💡 总结

### 当前设计特点

1. **LLM感知输出**: 结构化的 `PerceptionOutput` dataclass，包含5个字段
2. **LLM调用频率**: 主要在episode开始时，**不在每步执行后**调用
3. **每步更新方式**: 使用规则化的correctness机制（基于reward和validity）
4. **优点**: 
   - 减少LLM调用成本
   - 提高执行效率
   - 避免重复感知开销
5. **缺点**: 
   - 每步执行后无法获取新的视觉理解
   - 依赖初始感知的准确性
   - 环境变化可能无法及时捕获

### 如需实现"每步调用LLM"

参考之前提供的 `FINAL_ADAPTATION_REPORT.md` 中的**选项A**或**选项B**，添加:

```python
# 选项：每步调用LLM感知
def ingest_execution_step_with_llm(
    self, 
    action_desc: str,
    env_info: Dict[str, Any],
    reward: float,
    step: int,
    img_path: str,
    recent_actions: List[Dict]
) -> None:
    """每步执行后调用LLM进行感知+更新WM"""
    # 1. 调用LLM感知（包含动作历史）
    perception = self._llm_perceive_with_history(img_path, action_desc, recent_actions)
    self.wm_ops.write_perception(perception, step=step, clip_id=img_path)
    
    # 2. 同时做correctness更新
    self.wm_ops.correctness(action_desc, env_info, reward, step=step)
```

**但需要考虑**: LLM调用成本、延迟、token消耗等因素
