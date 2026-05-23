# 初始LTM知识图谱构建工具

## 概述

`build_initial_ltm.py` 是一个用于通过环境探索构建初始长期记忆(LTM)知识图谱的工具。

### 核心思路

1. **自动探索**: Agent在环境中执行一系列动作序列
2. **感知记录**: 每步执行后保存图像并调用LLM感知
3. **直接写LTM**: 将感知结果直接写入LTM（跳过WM）
4. **构建知识图谱**: 积累对象、位置、属性、关系等先验知识

### 与正常agent的区别

| 维度 | 正常Agent | LTM构建工具 |
|------|-----------|-------------|
| 目标 | 完成特定任务 | 探索并记录环境 |
| 记忆流程 | 感知→WM→LTM | 感知→直接LTM |
| 动作选择 | 基于规划 | 遍历预定义列表 |
| 节点类型 | ephemeral+persistent | 全部persistent |
| 置信度 | 动态更新 | 固定高置信度 |
| 用途 | 执行任务 | 构建先验知识 |

## 文件结构

```
embodiedbench/evaluator/build_initial_ltm.py
    ├── ActionExecutionRecord      # 动作执行记录
    ├── InitialLTMBuilder           # 主构建器类
    │   ├── get_exploration_actions()       # 获取探索动作列表
    │   ├── execute_action_and_perceive()   # 执行动作并感知
    │   ├── write_perception_to_ltm()       # 写入LTM
    │   ├── build_ltm_from_exploration()    # 主流程
    │   ├── save_ltm()                      # 保存LTM
    │   └── save_summary()                  # 保存摘要
    └── main()                      # 入口函数
```

## 使用方法

### 1. 基本使用

```bash
# 直接运行
python -m embodiedbench.evaluator.build_initial_ltm
```

### 2. 自定义参数

修改 `main()` 函数中的参数：

```python
builder = InitialLTMBuilder(
    env=env,
    output_dir="outputs/ltm_construction",      # 输出目录
    model_name="gpt-4o-mini",                   # 感知LLM
    max_steps_per_location=50,                  # 每位置最大步数
    ltm_save_path="outputs/initial_ltm.json",   # LTM保存路径
)

builder.build_ltm_from_exploration(
    max_total_steps=100,    # 总步数限制
    save_interval=10,       # 保存间隔
)
```

### 3. 在代码中使用

```python
from embodiedbench.evaluator.build_initial_ltm import InitialLTMBuilder
from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv

# 初始化环境
env = EBHabEnv(eval_set="complex_instruction", num_episodes=1)
env.reset()

# 创建构建器
builder = InitialLTMBuilder(
    env=env,
    ltm_save_path="my_custom_ltm.json"
)

# 执行构建
builder.build_ltm_from_exploration(max_total_steps=200)
```

## 工作流程

### 1. 动作选择策略

优先级排序：
1. **导航动作** (priority=1): 探索主要位置
   - navigate to sofa, table, counter, sink, fridge, etc.
2. **开关动作** (priority=2): 观察容器内部
   - open drawer, close cabinet, etc.
3. **pick/place动作** (priority=3): 观察对象-容器关系
   - pick up apple, place at sink, etc.

### 2. 执行与感知流程

```
For each action:
    ├── 1. Execute action in environment
    ├── 2. Save observation image
    ├── 3. Call LLM perception
    │   ├── Detect objects
    │   ├── Extract attributes (color, size, material, etc.)
    │   ├── Identify relations (on, in, near, etc.)
    │   └── Document state changes
    ├── 4. Write to LTM
    │   ├── Create object nodes (persistent)
    │   ├── Create attribute nodes
    │   ├── Create location nodes
    │   ├── Add has_color/has_size edges
    │   └── Add usually_found_at edges
    └── 5. Save checkpoint (every N steps)
```

### 3. LTM写入策略

**节点创建**:
- 所有对象: `type="object", ephemeral=False, ttl=9999`
- 所有属性: `type="attribute", kinds=["attribute", "color"]`
- 所有位置: `type="location"`

**边创建**:
- 属性关系: `has_color`, `has_size`, `has_material`, `has_shape`
- 位置关系: `usually_found_at`, `located_at`
- 置信度: 0.85-0.90 (高置信度，因为是观察得到)
- 所有边都是持久化 (`ephemeral=False`)

## 输出文件

### 1. LTM图谱文件

路径: `outputs/initial_ltm_habitat.json`

内容:
```json
{
  "nodes": [...],  // 所有对象、属性、位置节点
  "edges": [...],  // 所有关系边
  "metadata": {
    "created_at": "...",
    "total_nodes": 150,
    "total_edges": 280
  }
}
```

### 2. 图像文件

路径: `outputs/ltm_construction/images/`

格式: `step_0001_action_12.png`

### 3. 构建摘要

路径: `outputs/ltm_construction/construction_summary.json`

内容:
```json
{
  "total_steps": 100,
  "total_actions_executed": 100,
  "ltm_stats": {
    "nodes": 150,
    "edges": 280
  },
  "execution_records": [
    {
      "step": 0,
      "action_id": 12,
      "action_name": "navigate to the sofa",
      "reward": 0.0,
      "success": false,
      "invalid": false,
      "detected_objects": ["sofa", "apple", "cup"]
    },
    ...
  ]
}
```

## 关键特性

### 1. 智能动作选择

- 自动从环境的 `language_skill_set` 中筛选有价值的探索动作
- 优先导航到关键位置（家具、容器）
- 跳过重复或低价值动作

### 2. 鲁棒的感知

- 使用与agent相同的 `PerceptionModule`
- 完整的对象属性提取（颜色、大小、材质、形状、状态）
- 支持不确定性标注（如 "red(?)"）

### 3. 持久化LTM

- 所有节点和边都标记为持久化
- 高置信度（0.85-0.90）
- 无TTL限制（ttl=9999）
- 适合作为先验知识

### 4. 增量保存

- 每N步自动保存LTM
- 防止意外中断导致数据丢失
- 支持断点续传（TODO）

## 使用场景

### 1. 初始化LTM

在首次使用系统前，运行此工具构建包含常见对象和位置的LTM：

```bash
python -m embodiedbench.evaluator.build_initial_ltm
```

### 2. 扩充LTM

在新场景中运行，补充LTM中缺失的知识：

```python
# 加载已有LTM
builder = InitialLTMBuilder(
    env=env,
    ltm_save_path="outputs/existing_ltm.json"  # 会先加载已有数据
)
builder.build_ltm_from_exploration(max_total_steps=50)
```

### 3. 场景特定LTM

为特定场景构建专用LTM：

```python
env = EBHabEnv(eval_set="rearrange_pick")  # 特定任务
builder = InitialLTMBuilder(
    env=env,
    ltm_save_path="outputs/ltm_rearrange_pick.json"
)
builder.build_ltm_from_exploration(max_total_steps=100)
```

## 与Agent集成

构建好LTM后，在agent中使用：

```python
from embodiedbench.evaluator.meta_flat_habitat_agent import MetaFlatHabitatAgent

agent = MetaFlatHabitatAgent(
    planner=planner,
    model_name="gpt-4o-mini",
    ltm_save_path="outputs/initial_ltm_habitat.json"  # 使用预构建的LTM
)

# Agent会自动加载LTM作为先验知识
episode_summary = agent.run_episode(...)
```

## 注意事项

### 1. 成本控制

- LLM调用次数 = 执行步数
- 建议从小规模开始（max_total_steps=50）
- 评估覆盖率后再增加

### 2. 环境兼容性

- 需要环境支持 `language_skill_set`
- 需要环境支持图像保存 (`save_obs_image`)
- 测试环境：Habitat-lab v0.3.0

### 3. LTM质量

- 依赖LLM感知质量
- 建议使用高质量模型（gpt-4o, gpt-4o-mini）
- 可通过 `construction_summary.json` 检查质量

### 4. 存储空间

- 图像文件较大（每步~500KB）
- 100步约50MB
- 可在构建完成后删除图像

## 未来优化

- [ ] 支持断点续传
- [ ] 智能去重（相似场景不重复探索）
- [ ] 多episode并行构建
- [ ] LTM质量评估指标
- [ ] 自动调参（根据LTM覆盖率）
- [ ] 支持视频输入（连续帧）

## 调试

### 查看日志

日志会输出详细的执行过程：

```
[InitialLTMBuilder] Initialized with model=gpt-4o-mini
[InitialLTMBuilder] LTM will be saved to: outputs/initial_ltm.json
[Step 0] Executing action 12: navigate to the sofa
[Step 0] Perception complete:
  - Detected 5 objects: ['sofa', 'apple', 'cup', 'table', 'rug']
  - Found 4 relations
[Step 0] 💾 Writing perception to LTM...
  ✓ Object 'sofa' added to LTM
  ✓ Added edge: sofa --[has_color]--> brown
  ✓ Added relation: apple --[usually_found_at]--> sofa
[Step 0] ✅ LTM updated: 12 nodes, 8 edges
```

### 检查LTM内容

```python
from embodiedbench.evaluator.semantic_memory import SemanticMemoryManager

sm = SemanticMemoryManager(ltm_save_path="outputs/initial_ltm.json")
print(f"Nodes: {len(sm.ltm.nodes)}")
print(f"Edges: {len(sm.ltm.edges)}")

# 查看特定对象
nodes = sm.ltm.find_nodes(label="apple", exact=True)
for node in nodes:
    print(node.to_dict())
```

## 联系

如有问题或建议，请提issue或联系开发者。
