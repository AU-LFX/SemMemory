# 语义对齐时序修复总结

## 问题发现

用户问："日志中有没有体现说agent把sink和receptacle关联了起来？"

通过日志分析发现：
- ✅ 系统检测到 `entity:place:right receptacle` 缺失
- ✅ VLM 在 step 6 检测到 `sink` 对象
- ❌ **但没有日志显示两者被关联**
- ❌ 没有 `synonym_of(sink, receptacle)` 关系被创建

## 根本原因

**时序不匹配问题：**

```
错误的时序：
Step 6 执行阶段:
  → navigate 动作执行完成
  → 自动触发轻量级观察
  → VLM 检测到 sink
  → ingest_perception() 被调用
  → 此时 _pending_entity_gaps = [] (未设置)
  → align_semantics() 无法匹配 ❌

Step 7 规划阶段:
  → prepare_planner_context() 被调用
  → 检测到 entity gap: "right receptacle"
  → 设置 _pending_entity_gaps
  → 但 sink 已经在上一步被摄入 WM 了 ❌
```

**核心问题：** `_pending_entity_gaps` 只在规划阶段临时设置，而感知摄取发生在执行阶段，两者错开导致对齐机会丢失。

## 修复方案

### 核心思想
让 `_pending_entity_gaps` 在整个 episode 生命周期中**持续维护**。

### 实现细节

1. **主动检测（Proactive Detection）**
   - 在 `ingest_perception()` 中，如果 `_pending_entity_gaps` 为空
   - 主动从 `instruction` 提取 entity gaps
   - 确保每次感知都有对齐机会

2. **动态更新（Dynamic Update）**
   - 对齐成功后，从 `_pending_entity_gaps` 移除已解决的 gap
   - 避免重复对齐，保持列表准确

3. **生命周期管理（Lifecycle Management）**
   - Episode 开始时在 `reset()` 中清空
   - 确保各 episode 独立追踪

## 代码修改

### 文件：`embodiedbench/evaluator/semantic_memory.py`

#### 修改 1: `ingest_perception()` - 主动检测 + 动态移除
```python
# 修改前（约第 3278 行）
if self._pending_entity_gaps and object_labels:
    alignments = self.operator.align_semantics(...)

# 修改后
if not self._pending_entity_gaps and self.instruction:
    # 主动从 instruction 提取 gaps
    temp_query = WMReadQuery(goal=self.instruction)
    temp_result = self.operator.read(temp_query)
    entity_gaps = [g for g in temp_result.gaps if g.startswith("entity:")]
    if entity_gaps:
        self._pending_entity_gaps = entity_gaps

if self._pending_entity_gaps and object_labels:
    alignments = self.operator.align_semantics(...)
    if alignments:
        for alignment in alignments:
            # 从列表中移除已对齐的 gap
            aligned_gap = f"entity:{alignment['kind']}:{alignment['abstract_term']}"
            if aligned_gap in self._pending_entity_gaps:
                self._pending_entity_gaps.remove(aligned_gap)
```

#### 修改 2: `align_semantics()` - 返回 kind 字段
```python
# 修改前（约第 850 行）
alignments.append({
    "abstract_term": gap_label,
    "concrete_term": best_match,
    "confidence": best_score,
    "unit_id": unit.id,
})

# 修改后
alignments.append({
    "abstract_term": gap_label,
    "concrete_term": best_match,
    "confidence": best_score,
    "unit_id": unit.id,
    "kind": gap_kind,  # 新增：用于构造 gap 字符串
})
```

#### 修改 3: `reset()` - 清空 pending gaps
```python
# 修改前（约第 1915 行）
def reset(self, instruction: str, episode_id: Optional[int] = None):
    self.graph.reset()
    self.clips.clear()
    self.episode_id = episode_id
    self.instruction = instruction

# 修改后
def reset(self, instruction: str, episode_id: Optional[int] = None):
    self.graph.reset()
    self.clips.clear()
    self.episode_id = episode_id
    self.instruction = instruction
    self._pending_entity_gaps = []  # 新增
```

## 预期效果

修复后的正确时序：

```
✅ 正确的时序：
Step 6 执行阶段:
  → navigate 动作执行完成
  → 自动触发轻量级观察
  → VLM 检测到 sink, drawer, ring, etc.
  → ingest_perception() 被调用
  → 检测到 _pending_entity_gaps 为空
  → 主动从 instruction 提取 gaps:
    ['entity:object:spatula',
     'entity:place:right counter',
     'entity:place:right receptacle',  ← 关键！
     'entity:place:left counter']
  → 调用 align_semantics()
  → 检查 'right receptacle' 在 ABSTRACT_TO_CONCRETE_HINTS
  → 发现映射: receptacle → [sink, basin, bowl]
  → 在 observed_labels 中找到 'sink'
  → ✅ 匹配成功！
  → 更新 WM:
    - sink.aliases.append('right receptacle')
    - sink.relations.append(synonym_of('right receptacle'))
  → 从 _pending_entity_gaps 移除 'entity:place:right receptacle'
  → 日志: "Aligned 'right receptacle' → 'sink' (conf=0.95)"

Step 7 规划阶段:
  → prepare_planner_context() 被调用
  → 检测 entity gaps
  → ✅ 'right receptacle' 已存在（通过 alias）
  → gap 列表不再包含它
  → Planner 获得完整的 receptacle 信息
```

## 验证要点

修复成功后，日志应该包含：

```log
✅ 期待看到的日志：
[INFO] - [SemanticMemoryManager] Detecting entity gaps from instruction for semantic alignment
[INFO] - [SemanticMemoryManager] Extracted 4 entity gaps for alignment: ['entity:object:spatula', ...]
[INFO] - [SemanticMemoryManager] Attempting semantic alignment with 4 pending gaps and 7 observed objects
[INFO] - [Semantic Alignment] Aligned 'right receptacle' -> 'sink' (confidence=0.95, kind=place)
[INFO] - [SemanticMemoryManager] Semantic alignment: 1 mappings established
[INFO] -   - 'right receptacle' → 'sink' (conf=0.95)
[INFO] - [SemanticMemoryManager] Remaining pending gaps: 3
```

## 影响分析

### 功能提升
1. ✅ **抽象术语理解增强**：receptacle→sink, container→bowl 等映射生效
2. ✅ **WM 完整性提升**：synonym_of 关系正确建立
3. ✅ **Planner 上下文准确**：不再报告已对齐的实体缺失
4. ✅ **任务成功率提升**：Agent 能正确识别抽象术语对应的具体对象

### 性能影响
- **轻微增加**：每次感知摄取可能额外调用一次 `operator.read()`
- **可接受**：read 操作轻量，只遍历现有节点，无显著开销

### 兼容性
- ✅ **完全向后兼容**：不改变公共 API
- ✅ **优雅降级**：instruction 为空时行为与原来一致

## 测试建议

### 快速验证
运行一个包含抽象术语的任务：
```bash
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name='test_alignment'
```

检查日志中是否出现：
1. ✅ "Detecting entity gaps from instruction"
2. ✅ "Semantic alignment: X mappings established"
3. ✅ "'receptacle' → 'sink'" 或类似的映射记录

### 完整测试
观察完整 episode 执行：
1. Agent 导航到目标位置
2. 自动观察触发
3. 语义对齐成功
4. 后续规划不再报告对应的 entity gap
5. Agent 正确执行涉及抽象术语的动作

## 相关文档

- [详细修复文档](FIX_semantic_alignment_timing.md)
- [语义对齐知识图谱](semantic_alignment_knowledge_graph.md)
- [导航自动感知修复](navigation_auto_perception_fix.md)

## 总结

**修复前：** 语义对齐依赖规划阶段设置 `_pending_entity_gaps`，而感知在执行阶段进行，两者时序错开导致对齐失败。

**修复后：** `_pending_entity_gaps` 在整个 episode 生命周期中持续维护，感知摄取时主动检测并动态更新，确保对齐机制正常工作。

**核心改进：** 从**被动等待**（规划阶段设置）变为**主动检测**（感知阶段提取），解决了时序不匹配问题。

---

**Date:** 2026-01-07  
**Modified Files:** `embodiedbench/evaluator/semantic_memory.py`  
**Impact:** 低风险，高收益，向后兼容
