# 修复：语义对齐时序问题

## 问题描述

### 现象
日志显示：
1. ✅ Entity gap 检测正常工作，检测到 `entity:place:right receptacle`
2. ✅ VLM 正确识别了 `sink` 对象（step 6 轻量级观察）
3. ❌ 但没有日志显示 `receptacle → sink` 的语义对齐
4. ❌ 没有创建 `synonym_of(sink, receptacle)` 关系

### 根本原因：时序不匹配

**原设计的时序问题：**
```
时间线：
┌─────────────────────────────────────────────────────────────┐
│ Step N: 执行动作 (navigate)                                  │
│   → 自动触发轻量级观察                                        │
│   → ingest_perception() 被调用                              │
│   → 此时 _pending_entity_gaps = [] (还未设置)              │
│   → align_semantics() 没有被调用或无法匹配                   │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Step N+1: 准备规划下一个动作                                 │
│   → prepare_planner_context() 被调用                        │
│   → 检测 entity gaps                                        │
│   → 设置 _pending_entity_gaps = ['entity:place:receptacle'] │
│   → 但此时 sink 已经在上一步被摄取进 WM 了！               │
└─────────────────────────────────────────────────────────────┘
```

**时序差导致的问题：**
- 感知摄取发生在**执行阶段**（动作后自动观察）
- Entity gaps 检测发生在**规划阶段**（准备下一次规划时）
- 两者之间存在时间差，导致对齐机会错失

### 具体案例分析

从 `eb_log.txt` 日志：

**Step 6 (16:22:15):**
```log
[INFO] - Auto-triggering lightweight observation after successful navigation
[INFO] - Perception ingest clip=clip_055c1a38acec objects=7 relations=0 rules=0
        sample_objects=['pliers', 'drawer', 'wire', 'ring']
        # 完整列表包含: pliers, drawer, wire, ring, small cube, sink, arm
```
- ✅ VLM 检测到 7 个对象，其中包括 `sink`
- ❌ 但 `_pending_entity_gaps` 此时为空
- ❌ `align_semantics()` 没有被调用

**Step 7 规划阶段 (16:22:45):**
```log
[INFO] - Entity Gap Detection] Key entities extracted: 
        ['spatula', 'right counter', 'right receptacle', 'left counter']
[INFO] - Entity 'right receptacle' NOT found -> gap: entity:place:right receptacle
[INFO] - Final missing gaps: ['entity:place:right counter', 
        'entity:place:right receptacle', 'entity:place:left counter']
```
- ✅ 正确检测到 `right receptacle` 缺失
- ✅ 设置 `_pending_entity_gaps`
- ❌ 但为时已晚，`sink` 已经在上一步被摄入 WM，没有机会对齐

## 解决方案

### 核心思想
让 `_pending_entity_gaps` 在整个 episode 生命周期中**持续维护**，而不是只在规划阶段临时设置。

### 实现策略

1. **感知摄取时主动检测 gaps**
   - 在 `ingest_perception()` 中，如果 `_pending_entity_gaps` 为空
   - 主动从 `instruction` 中提取 entity gaps
   - 确保每次感知摄取都有机会进行语义对齐

2. **对齐成功后移除已解决的 gaps**
   - 当 `align_semantics()` 成功建立映射时
   - 从 `_pending_entity_gaps` 中移除对应的 gap
   - 避免重复对齐

3. **Episode 重置时清空**
   - 在 `reset()` 方法中清空 `_pending_entity_gaps`
   - 确保每个 episode 独立追踪

## 代码修改

### 1. 修改 `ingest_perception()` 方法

**位置:** `embodiedbench/evaluator/semantic_memory.py:3278-3290`

**修改前:**
```python
# ✨ 语义对齐：检查待解析实体是否能与本次观察建立映射
if self._pending_entity_gaps and object_labels:
    alignments = self.operator.align_semantics(
        pending_gaps=self._pending_entity_gaps,
        observed_labels=object_labels,
        step=env_step
    )
    if alignments:
        logger.info(
            f"[SemanticMemoryManager] Semantic alignment: {len(alignments)} mappings established"
        )
```

**修改后:**
```python
# ✨ 语义对齐：检查待解析实体是否能与本次观察建立映射
# 如果 _pending_entity_gaps 为空，先尝试从当前 instruction 中提取
if not self._pending_entity_gaps and self.instruction:
    logger.info("[SemanticMemoryManager] Detecting entity gaps from instruction for semantic alignment")
    temp_query = WMReadQuery(goal=self.instruction)
    temp_result = self.operator.read(temp_query)
    entity_gaps = [g for g in temp_result.gaps if g.startswith("entity:")]
    if entity_gaps:
        self._pending_entity_gaps = entity_gaps
        logger.info(f"[SemanticMemoryManager] Extracted {len(entity_gaps)} entity gaps for alignment: {entity_gaps[:3]}")

if self._pending_entity_gaps and object_labels:
    logger.info(
        f"[SemanticMemoryManager] Attempting semantic alignment with {len(self._pending_entity_gaps)} pending gaps "
        f"and {len(object_labels)} observed objects"
    )
    alignments = self.operator.align_semantics(
        pending_gaps=self._pending_entity_gaps,
        observed_labels=object_labels,
        step=env_step
    )
    if alignments:
        logger.info(
            f"[SemanticMemoryManager] Semantic alignment: {len(alignments)} mappings established"
        )
        for alignment in alignments:
            logger.info(
                f"  - '{alignment['abstract_term']}' → '{alignment['concrete_term']}' "
                f"(conf={alignment['confidence']:.2f})"
            )
            # 从待解析列表中移除已对齐的实体
            aligned_gap = f"entity:{alignment.get('kind', 'place')}:{alignment['abstract_term']}"
            if aligned_gap in self._pending_entity_gaps:
                self._pending_entity_gaps.remove(aligned_gap)
        logger.info(f"[SemanticMemoryManager] Remaining pending gaps: {len(self._pending_entity_gaps)}")
```

**关键改进:**
- ✅ 主动从 instruction 提取 entity gaps（如果尚未设置）
- ✅ 对齐成功后移除已解决的 gap
- ✅ 增加详细日志输出，便于调试

### 2. 修改 `align_semantics()` 返回值

**位置:** `embodiedbench/evaluator/semantic_memory.py:840-849`

**修改:**
```python
alignments.append({
    "abstract_term": gap_label,
    "concrete_term": best_match,
    "confidence": best_score,
    "unit_id": unit.id,
    "kind": gap_kind,  # ✨ 添加 kind 字段用于追踪
})
```

**原因:** 需要知道对齐的实体类型（object/place），才能正确构造 gap 字符串进行移除。

### 3. 修改 `reset()` 方法

**位置:** `embodiedbench/evaluator/semantic_memory.py:1915-1920`

**修改:**
```python
def reset(self, instruction: str, episode_id: Optional[int] = None) -> None:
    """在 episode 开始时重置 WM、Clip 并记录任务指令。"""
    self.graph.reset()
    self.clips.clear()
    self.episode_id = episode_id
    self.instruction = instruction
    # ✨ 重置待解析实体列表
    self._pending_entity_gaps = []
```

**原因:** 确保每个 episode 独立追踪 entity gaps。

## 预期效果

修复后的执行流程：

```
时间线：
┌─────────────────────────────────────────────────────────────┐
│ Episode 开始: reset()                                        │
│   → _pending_entity_gaps = []                              │
│   → instruction = "Move spatula from right counter to       │
│                     right receptacle of left counter"       │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 执行 navigate 后自动观察                             │
│   → ingest_perception() 被调用                              │
│   → 检测到 _pending_entity_gaps 为空                        │
│   → 主动从 instruction 提取 entity gaps:                    │
│     ['entity:object:spatula',                               │
│      'entity:place:right counter',                          │
│      'entity:place:right receptacle',                       │
│      'entity:place:left counter']                           │
│   → VLM 检测到对象: ['pliers', 'drawer', 'wire', 'ring',    │
│                      'small cube', 'sink', 'arm']           │
│   → 调用 align_semantics():                                 │
│     - 检查 'right receptacle' 是否在 ABSTRACT_TO_CONCRETE   │
│     - 发现映射: receptacle → [sink, basin, bowl]            │
│     - 在 observed_labels 中找到 'sink'                      │
│     - 匹配成功！confidence=0.95                             │
│   → 更新 WM:                                                │
│     - unit(sink).aliases.append('right receptacle')        │
│     - unit(sink).relations.append(                         │
│         synonym_of('right receptacle', conf=0.95))         │
│   → 从 _pending_entity_gaps 移除:                           │
│     'entity:place:right receptacle'                        │
│   → 日志输出:                                               │
│     "[Semantic Alignment] Aligned 'right receptacle' ->     │
│      'sink' (confidence=0.95, kind=place)"                 │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Step 7: 规划阶段                                             │
│   → prepare_planner_context() 被调用                        │
│   → 检测 entity gaps                                        │
│   → 发现 'right receptacle' 已存在（通过 alias）            │
│   → gaps 列表不再包含 'entity:place:right receptacle'       │
│   → WM context 包含完整的 receptacle 信息                   │
└─────────────────────────────────────────────────────────────┘
```

## 验证点

修复后，日志应该包含：

```log
[INFO] - [SemanticMemoryManager] Detecting entity gaps from instruction for semantic alignment
[INFO] - [SemanticMemoryManager] Extracted 4 entity gaps for alignment: ['entity:object:spatula', 'entity:place:right counter', 'entity:place:right receptacle']
[INFO] - [SemanticMemoryManager] Attempting semantic alignment with 4 pending gaps and 7 observed objects
[INFO] - [Semantic Alignment] Aligned 'right receptacle' -> 'sink' (confidence=0.95, kind=place)
[INFO] - [SemanticMemoryManager] Semantic alignment: 1 mappings established
[INFO] -   - 'right receptacle' → 'sink' (conf=0.95)
[INFO] - [SemanticMemoryManager] Remaining pending gaps: 3
```

## 影响范围

### 功能改进
1. ✅ **语义对齐成功率提升**：不再依赖时序巧合
2. ✅ **抽象术语理解增强**：能正确映射 receptacle→sink, container→bowl 等
3. ✅ **WM 完整性提升**：synonym_of 关系被正确建立
4. ✅ **Planner 上下文质量提升**：包含更准确的实体信息

### 性能影响
- **轻微增加**: 每次感知摄取时可能额外调用一次 `operator.read()` 来提取 gaps
- **预期开销**: 可忽略（read 操作很轻量，只是遍历现有节点）

### 向后兼容性
- ✅ **完全兼容**: 不改变公共 API
- ✅ **优雅降级**: 如果 instruction 为空，行为与原来一致

## 测试建议

### 单元测试
```python
def test_semantic_alignment_timing():
    """测试语义对齐在感知摄取时正确触发"""
    manager = SemanticMemoryManager()
    manager.reset(
        instruction="Move spatula from right counter to right receptacle of left counter",
        episode_id=1
    )
    
    # 模拟第一次感知摄取（检测到 sink）
    perception = {
        "detected_objects": [
            {"label": "sink", "color": ["metal"], "size": "large"}
        ]
    }
    
    manager.ingest_perception(
        perception=perception,
        instruction=manager.instruction,
        img_path="test.png",
        env_step=1
    )
    
    # 验证 sink 节点被创建并包含 'right receptacle' 别名
    sink_unit = manager.graph.find_unit_by_label("sink", kind="place")
    assert sink_unit is not None
    assert "right receptacle" in [a.lower() for a in sink_unit.symbol.aliases]
    
    # 验证 synonym_of 关系被创建
    relations = {(r.relation, r.target) for r in sink_unit.symbol.relations}
    assert ("synonym_of", "right receptacle") in relations
```

### 集成测试
运行完整的 episode，验证：
1. 日志中出现语义对齐的记录
2. 后续规划阶段不再报告 receptacle 缺失
3. Agent 能正确执行涉及抽象术语的任务

## 相关文档

- [语义对齐知识图谱设计](semantic_alignment_knowledge_graph.md)
- [导航自动感知修复](navigation_auto_perception_fix.md)
- [轻量级观察优化](CHANGELOG_lightweight_observation.md)

## 变更历史

- **2026-01-07**: 初始修复，解决语义对齐时序问题
