# Meta Flat Habitat Agent 适配新 Semantic Memory 的修改指南

## 主要修改点

### 1. 删除 FusionContext 相关代码
- 删除 `FusionContext` dataclass 定义
- 删除 `MemoryWebFusionEngine` 类
- 从 `WorkingMemory` 删除 `fusion_history` 字段和 `latest_fusion()` 方法
- 从 `build_task_model` 删除 `fusion` 参数

### 2. 修改感知流程
修改 `PerceptionModule.perceive()` 返回的 `PerceptionOutput`,使其结构化能直接更新WM:
- `detected_objects`: List[Dict] (包含 label, type, color, size等属性)
- `objects_relations`: List[Dict] (包含 source, target, relation)

### 3. 每个 execution step 后的感知更新
在 `run_single_episode()` 的执行循环中,每执行一个action后:
```python
# 保存图像
img_path = self.env.save_image(obs)

# 调用 semantic_memory 更新WM (包含action历史)
self.semantic_memory.ingest_execution_feedback(
    action_id=action_id,
    action_desc=action_desc,
    env_info=info,
    reward=reward,
    env_step=step,
    img_path=img_path,  # 执行后的新视图
)
```

### 4. Episode 结束时的 LLM 调用
在 `run_single_episode()` 结束前:
```python
# 传入完整动作历史
episode_log = getattr(self.env, "episode_log", [])
wm.semantic_memory_report = self.semantic_memory.on_episode_end(
    action_history=episode_log
)
```

### 5. 删除 fusion 相关调用
- 删除 `self.fusion = MemoryWebFusionEngine()`
- 删除所有 `fusion_ctx = self.fusion.fuse(...)` 调用
- 删除所有 `wm.fusion_history.append(...)` 调用
- 修改 `build_task_model(instruction, env)` 签名,删除 fusion 参数

## 新的工作流程

```
感知 (LLM) → 更新 WM
    ↓
检索 LTM → 融合到 WM  
    ↓
规划 (LLM + WM context)
    ↓
执行 action
    ↓
感知更新 (LLM + action history) → 更新 WM
    ↓
(循环直到 episode 结束)
    ↓
Episode 总结 (LLM + full action history) → 更新 WM → 巩固到 LTM
```

## 关键点

1. **初始感知**: 使用 `PerceptionModule.perceive()`,返回结构化数据
2. **步骤感知**: 使用 `semantic_memory.ingest_execution_feedback()`,包含动作历史
3. **Episode 总结**: 使用 `semantic_memory.on_episode_end()`,必须传入完整 action_history
4. **直接更新WM**: 不再需要 fusion 中间层,感知结果直接写入 semantic memory 的 WM
