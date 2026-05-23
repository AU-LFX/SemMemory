# LTM 空间信息补充工具使用说明

## 功能

在已有的 `output/semantic_ltm.json` 基础上，补充**位置信息**和**空间关系**。

### 已有 LTM 的问题
- ✅ 包含对象和基本属性（颜色、类别等）
- ❌ 缺少位置信息（对象在哪里）
- ❌ 缺少空间关系（对象之间的空间关系）

### 此工具的解决方案
使用**自定义空间感知 prompt**，专注于：
1. **对象位置**: apple is on table, cup is in sink
2. **空间关系**: table is near sofa, shelf is above counter
3. **对象属性补充**: 补充颜色、大小、材质、状态

## 与原版的区别

| 特性 | 原版 (PerceptionModule) | 新版 (自定义空间感知) |
|------|------------------------|---------------------|
| Prompt | 通用场景理解 | **专注空间位置** |
| 输出格式 | 通用对象检测 | **结构化位置数据** |
| 空间关系 | 简单文本描述 | **详细空间关系** |
| 位置精度 | 模糊 | **精确容器/表面** |
| LTM 更新 | 新建 LTM | **增量更新已有 LTM** |

## 使用方法

### 1. 运行工具

```bash
python -m embodiedbench.evaluator.build_initial_ltm
```

### 2. 输出

#### 更新的 LTM
- **文件**: `output/semantic_ltm.json` (原地更新)
- **新增节点**: 位置节点 (table, counter, sink, etc.)
- **新增边**: 
  - `usually_found_at`: apple --[usually_found_at]--> table
  - `spatial_near`: table --[spatial_near]--> sofa
  - `spatial_above`: shelf --[spatial_above]--> counter
  - `has_color`: apple --[has_color]--> red
  - `has_material`: table --[has_material]--> wood

#### 日志和图像
- **目录**: `output/ltm_spatial_construction/`
- **图像**: `images/step_XXXX.png`
- **摘要**: `construction_summary.json`

## 自定义 Prompt 示例

```python
spatial_prompt = """
Please analyze this image and provide:

1. **Detected Objects**: List ALL visible objects

2. **Object Locations**: For EACH object, describe:
   - WHERE it is located (on/in/near which furniture)
   - Spatial relationships with other objects
   - Format: "object_name is [relation] location_name"

3. **Object Attributes**: Color, size, material, state

4. **Scene Layout**: Overall spatial structure

**Output Format (JSON):**
{
  "detected_objects": ["apple", "table", "cup"],
  "object_locations": [
    {"object": "apple", "location": "on table", "container": "table", "spatial_relation": "on"},
    {"object": "cup", "location": "in sink", "container": "sink", "spatial_relation": "in"}
  ],
  "spatial_relations": [
    "apple is on table",
    "table is near sofa"
  ],
  "object_attributes": [
    {"label": "apple", "color": "red", "size": "small"}
  ],
  "scene_layout": "Kitchen with central counter and dining table"
}
```

## 代码架构

### 核心类

```python
class InitialLTMBuilder:
    def __init__(self, ltm_save_path="output/semantic_ltm.json"):
        # 加载已有 LTM
        self.semantic_memory = SemanticMemoryManager(ltm_save_path=ltm_save_path)
        # 使用 RemoteModel 直接调用 LLM
        self.llm_client = RemoteModel(model_name="gpt-4o-mini")
    
    def perceive_spatial_information(self, image_path, action_name):
        """自定义空间感知 - 专注位置和空间关系"""
        # 使用自定义 prompt
        response = self.llm_client.query(prompt=spatial_prompt, images=[image_base64])
        # 返回结构化的空间信息
        return SpatialPerceptionOutput(...)
    
    def write_perception_to_ltm(self, perception, step):
        """增量更新 LTM - 补充位置和空间信息"""
        # 1. 检查对象是否已存在（不重复创建）
        # 2. 补充对象属性（如果缺失）
        # 3. 创建位置节点和边（重点！）
        # 4. 创建空间关系边（重点！）
```

### 数据流

```
环境执行动作
    ↓
保存图像
    ↓
自定义空间感知 (perceive_spatial_information)
    ↓ 输出: SpatialPerceptionOutput
    ├── detected_objects: ["apple", "table"]
    ├── object_locations: [{"object": "apple", "container": "table", "spatial_relation": "on"}]
    ├── spatial_relations: ["apple is on table", "table is near sofa"]
    └── object_attributes: [{"label": "apple", "color": "red"}]
    ↓
增量更新 LTM (write_perception_to_ltm)
    ├── 查找已有对象节点
    ├── 创建/更新位置节点
    ├── 创建 usually_found_at 边
    ├── 创建 spatial_near/above/below 边
    └── 补充属性边 (has_color, has_material)
    ↓
保存到 output/semantic_ltm.json
```

## 参数调整

### 步数控制

```python
builder.build_ltm_from_exploration(
    max_total_steps=100,    # 总步数（越多覆盖越全，成本越高）
    save_interval=10,       # 每N步保存一次
)
```

### 动作优先级

工具自动从环境的 `language_skill_set` 中筛选：

```python
优先级 1: navigate to [location]  # 探索主要位置
优先级 2: open/close [container]  # 观察容器内部
优先级 3: pick/place [object]     # 观察对象-容器关系
```

### LLM 模型

```python
builder = InitialLTMBuilder(
    model_name="gpt-4o-mini",  # 或 "gpt-4o" (更准确但更贵)
)
```

## 验证 LTM 更新

### 检查新增节点和边

```python
from embodiedbench.evaluator.semantic_memory import SemanticMemoryManager

sm = SemanticMemoryManager(ltm_save_path="output/semantic_ltm.json")

# 查看总数
print(f"Total nodes: {len(sm.ltm.nodes)}")
print(f"Total edges: {len(sm.ltm.edges)}")

# 查找特定对象的位置信息
apple_nodes = sm.ltm.find_nodes(label="apple", exact=True)
if apple_nodes:
    apple_node = apple_nodes[0]
    # 查找位置边
    location_edges = sm.ltm.find_edges(source=apple_node.id, relation="usually_found_at")
    for edge in location_edges:
        target = sm.ltm.get_node(edge.target)
        print(f"apple is usually found at: {target.label}")
```

### 查看空间关系

```python
# 查找所有空间关系边
spatial_edges = sm.ltm.find_edges(relation="spatial_near")
for edge in spatial_edges:
    source = sm.ltm.get_node(edge.source)
    target = sm.ltm.get_node(edge.target)
    print(f"{source.label} is near {target.label}")
```

## 注意事项

### 1. 文件路径
- ⚠️ 确保 `output/semantic_ltm.json` 存在
- ⚠️ 工具会**原地更新**此文件（建议先备份）

### 2. 成本控制
- LLM 调用次数 = 执行步数
- 建议从 50 步开始测试
- 每步约 0.001-0.002 美元 (gpt-4o-mini)

### 3. 环境要求
- 需要 Habitat 环境正常运行
- 需要 OpenAI API key 配置
- 图像保存功能正常

### 4. 质量检查
- 查看 `construction_summary.json` 了解执行结果
- 检查图像以验证感知质量
- 对比 LTM 更新前后的节点/边数量

## 故障排除

### 问题：LTM 文件未找到
```bash
FileNotFoundError: output/semantic_ltm.json
```
**解决**: 确保文件存在，或修改 `ltm_save_path` 参数

### 问题：空间关系未创建
**检查**: 
1. 查看日志中的 "Processing spatial relations"
2. 确认 `perception.spatial_relations` 不为空
3. 检查正则表达式是否匹配关系文本

### 问题：重复节点
**原因**: 标签标准化可能失败
**解决**: 检查 `normalize_label()` 函数

## 下一步

使用增强后的 LTM 运行 agent：

```bash
python -m embodiedbench.main \
    env=eb-hab \
    model_name=gpt-4o-mini \
    exp_name='with_spatial_ltm'
```

Agent 将自动加载 `output/semantic_ltm.json`，利用位置信息进行更准确的规划！
