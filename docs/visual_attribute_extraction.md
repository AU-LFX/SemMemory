# 视觉属性提取与记忆更新增强

## 改进概要

现在系统会在每个step的perception阶段从image中提取**完整的视觉属性**并存入memory，包括：
- 颜色 (color)
- 尺寸 (size)
- 材质 (material)
- 形状 (shape)
- 纹理 (texture)
- 特征 (features)
- 位置 (position)

## 实现细节

### 1. Perception Prompt增强 (`meta_flat_habitat_agent.py`)

**之前**：只要求物体名称列表
```json
{
  "detected_objects": ["cup", "table", "book"]
}
```

**现在**：要求结构化对象描述
```json
{
  "detected_objects": [
    {
      "label": "cup",
      "color": ["red"],
      "size": "small",
      "material": ["ceramic"],
      "shape": "cylindrical",
      "features": ["has handle"],
      "position": "on the table in center"
    },
    {
      "label": "table",
      "color": ["brown"],
      "size": "large",
      "material": ["wood"],
      "shape": "rectangular",
      "texture": "smooth",
      "position": "center of room"
    }
  ]
}
```

### 2. 关系描述增强

**之前**：`"the cup is on the table"`

**现在**：`"the small red cup is on the wooden table"`

包含颜色/尺寸等描述词，让关系更明确，避免歧义。

### 3. Memory存储映射

Perception输出 → WM state_vars 映射：

| Perception字段 | WM字段 | 类型 | 用途 |
|---------------|--------|------|------|
| `color` | `colors` | list[str] | 属性匹配、去重 |
| `size` | `size_hint` | str | 属性匹配 |
| `material` | `materials` | list[str] | 属性匹配 |
| `shape` | `shape_hint` | str | 属性匹配 |
| `texture` | `texture` | str | 属性匹配 |
| `features` | `top_features` | list[str] | 属性匹配、特征推理 |
| `position` | `spatial_position` | str | 空间推理 |

### 4. 属性驱动的对象识别流程

```
Image → VLM Perception
  ↓
提取: "small red object with green top"
  ↓
Attribute hints: {color: [red, green], size: [small], feature: [green top, top]}
  ↓
LTM查询: 
  - has_color → red ✓
  - has_size → small ✓
  - has_feature → {top, leaves} ✓ (同义词扩展)
  ↓
匹配: strawberry (colors=[red], size_hint=small, top_features=[green leaves])
  ↓
WM更新: canonical label = "strawberry"
```

## 示例场景

### 输入Image内容
- 一张木桌子
- 桌上有一个红色杯子
- 杯子旁边有一本绿色的书

### VLM输出
```json
{
  "scene_summary": "Kitchen area with a wooden dining table in the center.",
  "detected_objects": [
    {
      "label": "table",
      "color": ["brown"],
      "size": "large",
      "material": ["wood"],
      "shape": "rectangular",
      "texture": "smooth",
      "position": "center of room"
    },
    {
      "label": "cup",
      "color": ["red"],
      "size": "small",
      "material": ["ceramic"],
      "shape": "cylindrical",
      "features": ["has handle"],
      "position": "on the table"
    },
    {
      "label": "book",
      "color": ["green"],
      "size": "medium",
      "material": ["paper"],
      "shape": "rectangular",
      "position": "on the table beside the cup"
    }
  ],
  "objects_relations": [
    "the small red cup is on the wooden table",
    "the green book is on the wooden table beside the red cup"
  ]
}
```

### Memory存储 (WM)
```
[object] table | visible=True, colors=[brown], size_hint=large, materials=[wood], 
                 shape_hint=rectangular, texture=smooth, spatial_position=center of room

[object] cup | visible=True, colors=[red], size_hint=small, materials=[ceramic],
               shape_hint=cylindrical, top_features=[has handle], spatial_position=on the table

[object] book | visible=True, colors=[green], size_hint=medium, materials=[paper],
                shape_hint=rectangular, spatial_position=on the table beside the cup

[link] cup on table | relations: on->table_<hash>
[link] book on table | relations: on->table_<hash>
[link] book beside cup | relations: adjacent->cup_<hash>
```

## 属性索引与查询

LTM会为每个属性值创建索引节点：

```
Attribute Nodes:
  - [attribute:color] red → 链接到 {cup, strawberry, apple, ...}
  - [attribute:color] green → 链接到 {book, pear, leaves, ...}
  - [attribute:size] small → 链接到 {cup, strawberry, ...}
  - [attribute:material] wood → 链接到 {table, shelf, ...}
```

查询时可通过组合属性快速定位：
```python
hints = {"color": ["red"], "size": ["small"], "feature": ["handle"]}
candidates = ltm.query_by_attribute_hints(hints)
# → ["cup", "mug", ...]
```

## 测试验证

运行系统后检查日志：
```
[PerceptionModule] LLM perception output:
{
  "detected_objects": [
    {"label": "cup", "color": ["red"], "size": "small", ...}
  ]
}

[SemanticMemoryManager] Perception ingest: objects=3 relations=2
[SemanticMemoryManager] Object profiles enriched with visual attributes

[MetaFlat] Semantic WM digest:
[object] cup | colors=[red], size_hint=small, materials=[ceramic], ...
```

## 优势

1. **精确匹配**：通过多属性组合避免歧义（红杯子 vs 蓝杯子）
2. **语义推理**：可识别 "small red thing" → strawberry（属性驱动）
3. **关系明确**：link中包含描述词（"red cup on wooden table"）
4. **可追溯性**：每个属性都关联到image clip_id
5. **增量更新**：每step的新观察更新已有对象的属性

## 下一步优化建议

1. **属性冲突检测**：如果同一对象在不同step看到不同颜色，记录变化或标记不确定性
2. **置信度传播**：VLM可能对某些属性不确定，可要求输出confidence并存储
3. **时序属性**：记录"之前是红色，现在是蓝色"这样的状态变化
4. **空间关系图**：利用position字段构建room layout的拓扑图
