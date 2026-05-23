# -*- coding: utf-8 -*-
"""Semantic memory framework for the EB Habitat agent.

This module implements:
    - Lightweight evidence clips ("Clip") referencing saved observations.
    - Unified semantic units ``U = Symbol + Clip`` that cover object / spatial / temporal / rule semantics.
    - A bounded working-memory graph with indexes, utility tracking, and simplified operator engine
      (Write, Read, Correctness, Abstract, Simplify, Integrate).
    - Stubs for long-term memory (KG + schema registry) search / consolidation so the agent can
      fetch structured hints when WM exposes knowledge gaps.

The implementation intentionally favours transparency and auditability over complexity:
    * All semantic updates emit structured logs via the global ``logger``.
    * Every semantic field mutation points back to at least one clip id.
    * TTL + utility based pruning keeps WM small while allowing frequently read nodes to persist.

The code is engineered so it can be imported without Habitat runtime dependencies, allowing us to
unit-test the memory logic in isolation.

WM ↔ LTM Interaction Strategy (Simplified):
    
    ✅ NEW DESIGN: Simple and Efficient
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Working Memory (WM):
        - Episode 内的完整信息暂存区
        - 每次感知、每次 action 后只更新 WM
        - 可以存储所有信息（空间、对象、位置、状态等）
        - 没有大小限制（可以很大）
        - Episode 结束时清空
    
    Long-Term Memory (LTM):
        - 跨 Episode 的知识库
        - 只在 Episode 结束时接收 WM 的知识
        - 只存储长期有效的知识（属性、关系、规则）
        - 自动过滤临时状态（位置、可见性、时间戳）
    
    Interaction Timeline:
        Episode Start:
            WM: {} (空白)
            LTM: {知识库} (保持不变)
        
        During Episode:
            Step 1: 感知 → 更新 WM ✓
            Step 2: 执行 → 更新 WM ✓
            Step 3: 感知 → 更新 WM ✓
            ...
            (不与 LTM 交互，只操作 WM)
        
        Episode End:
            WM → LTM: 统一同步 ✓
            - Consolidate: 融合知识
            - Decompose: 拆分模糊节点
            - Forget: 清理过期知识
            - Save: 持久化到磁盘
        
        Next Episode Start:
            WM: {} (重新清空)
            WM ← LTM: 按需查询长期知识 ✓
    
    Ephemeral States (filtered when WM → LTM):
        ❌ spatial_position: 对象位置 (每个 episode 都变化)
        ❌ last_seen_position: 最后看到的位置
        ❌ visible: 可见性标志
        ❌ last_seen_ts: 时间戳
        ❌ step, recency: 临时元数据
        ❌ source: 来源标记
    
    Long-Term Knowledge (persisted in LTM):
        ✅ category: 对象类别 (e.g., 'fruit', 'tool')
        ✅ materials: 材质 (e.g., ['metal', 'plastic'])
        ✅ affordances: 功能 (e.g., ['flip', 'spread'])
        ✅ colors, size_hint, shape_hint: 物理属性
        ✅ temperature_tolerance, edibility: 属性
        ✅ relations: 语义关系 (is_a, synonym_of, etc.)
    
    Benefits:
        ✓ 简单：WM 只在 episode 内工作，不用频繁同步
        ✓ 高效：减少 I/O 操作，提升性能
        ✓ 清晰：职责分离，WM = 暂存，LTM = 知识库
        ✓ 灵活：WM 可以很大，存储所有细节

API Methods:
    
    ingest_step(perception, instruction, img_path, action_id, action_desc, env_info, reward, env_step):
        ★ 推荐使用：原子化处理一个完整的 step
        - 同时处理观察(perception)和动作执行(action)
        - 一个 step = 观察 + 动作，作为一个原子操作
        - 只更新 WM，不同步到 LTM
        - 自动处理语义对齐、对象位置更新等
    
    ingest_perception(perception, instruction, img_path, env_step):
        ⚠️ 兼容方法：仅处理观察
        - 单独摄取感知结果
        - 如果使用 ingest_step，则不需要调用此方法
    
    ingest_execution_feedback(action_id, action_desc, env_info, reward, env_step):
        ⚠️ 兼容方法：仅处理动作反馈
        - 单独摄取执行反馈
        - 如果使用 ingest_step，则不需要调用此方法
    
    on_episode_end():
        ★ Episode 结束时调用
        - 将 WM 同步到 LTM（唯一的同步时机）
        - 运行 consolidate, decompose, forget
        - 持久化知识到磁盘
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from embodiedbench.main import logger
from embodiedbench.evaluator.object_knowledge_seed import (
    build_ltm_seed_graph,
    get_concept_definition,
    resolve_canonical_label,
)


ACTION_ENTITY_PATTERNS = [
    # "Move X from Y to Z" - extract X as object (skip articles)
    (re.compile(r"^(?:move)\s+(?:a|an|the)\s+(?P<label>[\w\s]+?)\s+from\s+", re.IGNORECASE), "object"),
    # "Navigate/Go/Move to X"
    (re.compile(r"^(?:navigate|go|move)\s+to\s+(?:the\s+)?(?P<label>.+)$", re.IGNORECASE), "object"),
    # "Pick up X"
    (re.compile(r"^(?:pick)\s+up\s+(?:the\s+)?(?P<label>.+)$", re.IGNORECASE), "object"),
    # "Open/Close X"
    (re.compile(r"^(?:open|close)\s+(?:the\s+)?(?P<label>.+)$", re.IGNORECASE), "object"),
    # "Place at/on X"
    (re.compile(r"^(?:place)\s+(?:at|on)\s+(?:the\s+)?(?P<label>.+)$", re.IGNORECASE), "object"),
]

# 语义关系类型定义
SEMANTIC_RELATIONS = {
    # Spatial relations
    "left_of": "spatial",
    "right_of": "spatial", 
    "on": "spatial",
    "in": "spatial",
    "above": "spatial",
    "below": "spatial",
    "near": "spatial",
    "inside": "spatial",
    
    # Semantic relations
    "synonym_of": "semantic",  # 同义词关系，如 receptacle <-> sink
    "is_a": "semantic",        # 类型关系，如 sink is_a receptacle
    "part_of": "semantic",     # 部分关系，如 faucet part_of sink
    "contains": "semantic",    # 包含关系
}

# 常见的抽象-具体映射（用于初始化 LTM 知识）
ABSTRACT_TO_CONCRETE_HINTS = {
    "receptacle": ["sink", "basin", "bowl", "container", "bin"],
    "container": ["box", "basket", "bag", "bin", "jar"],
    "surface": ["counter", "table", "desk", "shelf"],
    "appliance": ["microwave", "oven", "stove", "refrigerator", "dishwasher"],
}



ATTRIBUTE_FIELD_MAP: Dict[str, Tuple[str, str, bool]] = {
    "color": ("has_color", "color", False),
    "colors": ("has_color", "color", True),
    "size_hint": ("has_size", "size", False),
    "shape_hint": ("has_shape", "shape", False),
    "top_features": ("has_feature", "feature", True),
    "features": ("has_feature", "feature", True),
    "texture": ("has_texture", "texture", False),
    "materials": ("made_of", "material", True),
    "material": ("made_of", "material", False),
    "affordances": ("affords", "affordance", True),
    "usage_notes": ("has_usage", "usage", False),
}

ATTRIBUTE_RELATION_DEFAULT = "has_property"
ATTRIBUTE_HINT_RELATION = {
    "color": "has_color",
    "feature": "has_feature",
    "size": "has_size",
    "material": "made_of",
    "texture": "has_texture",
}

ATTRIBUTE_GAP_KINDS: Set[str] = set(ATTRIBUTE_HINT_RELATION.keys())

ATTRIBUTE_EXCLUDE_KEYS = {
    "visible",
    "last_seen_ts",
    "source",
    "passable",
    "step",
    "reward",
    "confidence",
    "risk_level",
    "state",
}

COLOR_KEYWORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "brown",
}

SIZE_KEYWORDS = {"tiny", "small", "medium", "large", "huge", "big"}

FEATURE_SYNONYMS: Dict[str, Set[str]] = {
    "top": {"leaves", "leaf", "stem", "crown", "cap"},
    "leaves": {"top", "leaf", "foliage"},
    "skin": {"surface", "peel", "rind", "coating"},
    "pattern": {"texture", "marking", "design"},
}


ATTRIBUTE_RELATION_TO_KIND: Dict[str, str] = {}
ATTRIBUTE_KIND_TO_RELATION: Dict[str, str] = {}
ATTRIBUTE_KIND_PRIMARY_KEY: Dict[str, str] = {}
ATTRIBUTE_KIND_ALLOW_LIST: Dict[str, bool] = {}
ATTRIBUTE_KIND_TO_KEYS: Dict[str, List[str]] = defaultdict(list)

for attr_field, (relation, attr_kind, allow_list) in ATTRIBUTE_FIELD_MAP.items():
    ATTRIBUTE_RELATION_TO_KIND[relation] = attr_kind
    ATTRIBUTE_KIND_TO_RELATION.setdefault(attr_kind, relation)
    ATTRIBUTE_KIND_PRIMARY_KEY.setdefault(attr_kind, attr_field)
    ATTRIBUTE_KIND_ALLOW_LIST[attr_kind] = ATTRIBUTE_KIND_ALLOW_LIST.get(attr_kind, False) or allow_list
    ATTRIBUTE_KIND_TO_KEYS[attr_kind].append(attr_field)


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------
# 这些轻量级结构描述了「语义单元＝Symbol+Clip」的骨架，是 WM/LTM 的最小组成部分。


def _generate_id(prefix: str) -> str:
    """生成带前缀的短 UUID，方便在日志中追踪语义实体。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _string_similarity(a: str, b: str) -> float:
    """基于标准化 token 的 Jaccard 相似度，帮助对齐候选语义单元。"""
    a_norm = (a or "").strip().lower()
    b_norm = (b or "").strip().lower()
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    # simple token-based Jaccard
    a_tokens = set(a_norm.replace("_", " ").split())
    b_tokens = set(b_norm.replace("_", " ").split())
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


@dataclass
class Clip:
    """Clip = 证据引用（轻量摘要 + 原始文件 URI），保证每个语义结论可追溯。"""
    id: str
    kind: str
    uri: str
    digest: str
    timestamps: Dict[str, float]
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TypeCandidate:
    """符号候选类型，对应目标 label 及其置信度。"""
    label: str
    confidence: float = 0.5
    concept_id: Optional[str] = None


@dataclass
class RelationSpec:
    """关系规格，描述 unit 之间的 directed edge 及辅助元数据。"""
    relation: str
    target: str  # target unit id or symbolic handle
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticSymbol:
    """Symbol 部分记录结构化语义字段：候选类型、状态变量、关系和约束。"""
    type_candidates: List[TypeCandidate] = field(default_factory=list)
    state_vars: Dict[str, Any] = field(default_factory=dict)
    relations: List[RelationSpec] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)  # 同义词/别名列表，用于语义对齐


@dataclass
class SemanticMetadata:
    """统一的元数据，用于调节置信度、TTL、来源等动态属性。"""
    confidence: float = 0.5
    recency: float = 0.0
    ttl: int = 50
    provenance: str = "perception"
    fuzziness: float = 0.2


@dataclass
class SemanticUnit:
    """完整语义单元 U：带唯一 id、kind（object/place/...）以及 clip 证据。"""
    id: str
    kind: str
    symbol: SemanticSymbol = field(default_factory=SemanticSymbol)
    metadata: SemanticMetadata = field(default_factory=SemanticMetadata)
    clip_refs: List[str] = field(default_factory=list)

    def primary_label(self) -> str:
        for cand in self.symbol.type_candidates:
            if cand.label:
                return cand.label
        return self.kind

    def to_digest(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.primary_label(),
            "aliases": list(self.symbol.aliases) if self.symbol.aliases else [],
            "state_vars": dict(self.symbol.state_vars),
            "conditions": list(self.symbol.conditions),
            "constraints": list(self.symbol.constraints),
            "relations": [asdict(rel) for rel in self.symbol.relations[:4]],
            "confidence": round(self.metadata.confidence, 3),
            "recency": self.metadata.recency,
            "ttl": self.metadata.ttl,
            "provenance": self.metadata.provenance,
            "clip_refs": list(self.clip_refs),
        }


@dataclass
class SemanticUnitCandidate:
    """待写入 WM 的临时候选，尚未占用 graph 节点 id。"""
    kind: str
    symbol: SemanticSymbol
    metadata: SemanticMetadata
    clip_refs: List[str]
    key: Optional[str] = None  # optional deterministic key used for matching

    def to_unit(self, unit_id: Optional[str] = None) -> SemanticUnit:
        return SemanticUnit(
            id=unit_id or _generate_id(self.kind or "unit"),
            kind=self.kind,
            symbol=self.symbol,
            metadata=self.metadata,
            clip_refs=list(self.clip_refs),
        )


@dataclass
class WMReadQuery:
    """Planner 的读取请求，指定目标、关注 kind 与必须的字段。"""
    goal: str
    focus_kinds: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    max_nodes: int = 12


@dataclass
class WMReadResult:
    """Planner/反馈读取 WM 时得到的子图、缺口列表和 LTM 提示。"""
    nodes: List[SemanticUnit]
    edges: List[Tuple[str, str, str]]
    gaps: List[str]
    ltm_hints: List[Dict[str, Any]] = field(default_factory=list)
    alignments: List[Dict[str, Any]] = field(default_factory=list)  # ✨ 新增：记录语义对齐映射

    def to_text_block(self) -> str:
        lines: List[str] = []
        
        # ✨ 添加当前状态摘要（关键对象的位置信息）
        key_objects_summary = []
        for node in self.nodes:
            if node.kind == "object":
                label = node.primary_label()
                spatial_pos = node.symbol.state_vars.get("spatial_position")
                last_seen_pos = node.symbol.state_vars.get("last_seen_position")
                if spatial_pos or last_seen_pos:
                    loc = spatial_pos or last_seen_pos
                    key_objects_summary.append(f"  • {label}: {loc}")
        
        if key_objects_summary:
            lines.append("### Current object locations:")
            lines.extend(key_objects_summary)
            lines.append("")  # 空行分隔
        
        for node in self.nodes:
            label = node.primary_label()
            
            # ✨ 显式添加别名信息（用于显示语义对齐结果）
            alias_desc = ""
            if node.symbol.aliases:
                alias_list = ", ".join(f"'{a}'" for a in sorted(node.symbol.aliases))
                alias_desc = f" (also known as: {alias_list})"
            
            # ✨ 为对象添加醒目的位置信息（优先显示）
            location_prefix = ""
            if node.kind == "object":
                spatial_pos = node.symbol.state_vars.get("spatial_position")
                last_seen_pos = node.symbol.state_vars.get("last_seen_position")
                if spatial_pos or last_seen_pos:
                    loc = spatial_pos or last_seen_pos
                    location_prefix = f" **[LOCATION: {loc}]**"
            
            state_desc = ", ".join(
                f"{k}={v}" for k, v in node.symbol.state_vars.items()
            ) or "state:unknown"
            rel_desc = "; ".join(
                f"{r.relation}->{r.target}" for r in node.symbol.relations[:3]
            )
            constraint_desc = "; ".join(
                [str(c) for c in node.symbol.constraints[:2]]
            )
            lines.append(
                f"[{node.kind}] {label}{location_prefix}{alias_desc} | {state_desc}" +
                (f" | relations: {rel_desc}" if rel_desc else "") +
                (f" | constraints: {constraint_desc}" if constraint_desc else "")
            )
        
        # ✨ 添加语义对齐摘要（在节点列表后）
        if self.alignments:
            lines.append("\n### Semantic alignments:")
            for align in self.alignments:
                lines.append(
                    f"- '{align['abstract_term']}' → '{align['concrete_term']}' "
                    f"(confidence={align['confidence']:.2f})"
                )
        
        if self.gaps:
            formatted = [self._format_gap(gap) for gap in self.gaps]
            lines.append("Missing fields:" + ", ".join(formatted))
        for hint in self.ltm_hints:
            lines.append(f"LTM hint: {hint.get('summary', '')}")
        return "\n".join(lines)

    @staticmethod
    def _format_gap(gap: str) -> str:
        if not isinstance(gap, str):
            return str(gap)
        if gap.startswith("attribute:"):
            parts = gap.split(":", 2)
            if len(parts) == 3:
                attr_kind = parts[1].strip()
                label = parts[2].strip()
                return f"missing {attr_kind} for {label}"
        return gap


@dataclass
class LTNode:
    """LTM 节点：表示概念/事件/对象，包含多维属性用于高级推理。"""
    id: str
    label: str
    kinds: List[str] = field(default_factory=list)
    centrality: float = 0.3
    graduality: float = 0.2
    fuzziness: float = 0.4
    family_resemblance: List[str] = field(default_factory=list)
    perspective: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    last_activated: float = field(default_factory=lambda: time.time())


@dataclass
class LTEdge:
    """LTM 边：记录节点之间的语义关系及其可信度/新近性。"""
    id: str
    source: str
    target: str
    relation: str
    recency: float
    confidence: float
    evidence_clips: List[str] = field(default_factory=list)
    usage_count: int = 0


# ---------------------------------------------------------------------------
# Working memory graph
# ---------------------------------------------------------------------------


class SemanticWMGraph:
    """工作记忆图：容量受限的小图结构，按 kind/utility 管理节点生命周期。"""
    def __init__(self, max_nodes: int = 256):
        self.max_nodes = max_nodes
        self.nodes: Dict[str, SemanticUnit] = {}
        self.by_kind: Dict[str, List[str]] = defaultdict(list)
        self.utility: Dict[str, float] = defaultdict(float)
        self.last_update_step: Dict[str, int] = {}

    # -- core operations -----------------------------------------------------

    def reset(self) -> None:
        """清空所有节点及索引，在 episode 重置或 debug 时使用。"""
        self.nodes.clear()
        self.by_kind.clear()
        self.utility.clear()
        self.last_update_step.clear()

    def add_unit(self, unit: SemanticUnit, step: int) -> None:
        """在 WM 中登记一个全新的语义单元并初始化 utility。"""
        self.nodes[unit.id] = unit
        self.by_kind[unit.kind].append(unit.id)
        self.utility[unit.id] = 1.0
        self.last_update_step[unit.id] = step
        logger.debug(f"[SemanticWMGraph] Added unit {unit.id} ({unit.kind})")

    def update_unit(self, unit_id: str, candidate: SemanticUnitCandidate, step: int) -> None:
        """多信号融合策略：逐字段更新，置信度/TTL/Clip 都同步刷新。"""
        unit = self.nodes[unit_id]
        # recency/置信度/TTL：决定节点能否长留 WM
        unit.metadata.recency = step
        unit.metadata.confidence = min(
            1.0,
            0.7 * unit.metadata.confidence + 0.3 * candidate.metadata.confidence,
        )
        unit.metadata.ttl = max(unit.metadata.ttl, candidate.metadata.ttl)
        unit.metadata.provenance = candidate.metadata.provenance or unit.metadata.provenance
        # 类型候选融合：根据 label 聚合，避免同义词反复写入
        combined = {tc.label: tc for tc in unit.symbol.type_candidates}
        for cand_tc in candidate.symbol.type_candidates:
            if cand_tc.label in combined:
                prev = combined[cand_tc.label]
                prev.confidence = min(1.0, (prev.confidence + cand_tc.confidence) / 2)
            else:
                combined[cand_tc.label] = cand_tc
        unit.symbol.type_candidates = sorted(
            combined.values(), key=lambda x: x.confidence, reverse=True
        )[:4]
        # 状态变量直接覆盖：后来的传感器通常更可信
        for key, value in candidate.symbol.state_vars.items():
            unit.symbol.state_vars[key] = value
        # 关系去重后追加，保持 WM 结构稀疏
        existing_relations = {(rel.relation, rel.target) for rel in unit.symbol.relations}
        for rel in candidate.symbol.relations:
            sig = (rel.relation, rel.target)
            if sig not in existing_relations:
                unit.symbol.relations.append(rel)
                existing_relations.add(sig)
        # 条件/约束直接附加——Planner 读取时再裁剪
        if candidate.symbol.conditions:
            unit.symbol.conditions.extend(candidate.symbol.conditions)
        if candidate.symbol.constraints:
            unit.symbol.constraints.extend(candidate.symbol.constraints)
        # Clip 引用用于可追溯性，最多保存 6 条最新轨迹
        for clip_id in candidate.clip_refs:
            if clip_id not in unit.clip_refs:
                unit.clip_refs.append(clip_id)
        unit.clip_refs = unit.clip_refs[-6:]
        # utility 随每次更新微增，帮助 read prioritization
        self.utility[unit_id] = min(self.utility[unit_id] + 0.5, 5.0)
        self.last_update_step[unit_id] = step
        logger.debug(f"[SemanticWMGraph] Updated unit {unit_id} ({unit.kind})")

    def match_candidate(self, candidate: SemanticUnitCandidate) -> Optional[str]:
        """匹配策略：优先用 kind + label 相似度，必要时也支持 key 精确命中，并检查别名。"""
        kind = candidate.kind
        label = candidate.symbol.type_candidates[0].label if candidate.symbol.type_candidates else None
        candidates = self.by_kind.get(kind, [])
        best_id: Optional[str] = None
        best_score = 0.0
        key = candidate.key
        
        for unit_id in candidates:
            unit = self.nodes[unit_id]
            
            # 1. Key 精确匹配
            if key and key == unit_id:
                return unit_id
            
            # 2. Primary label 相似度匹配
            score = _string_similarity(label or "", unit.primary_label())
            
            # 3. ✨ 检查别名匹配
            if label and unit.symbol.aliases:
                for alias in unit.symbol.aliases:
                    alias_score = _string_similarity(label, alias)
                    score = max(score, alias_score)
            
            if score > best_score:
                best_score = score
                best_id = unit_id
        
        if best_score >= 0.75:
            return best_id
        return None

    def find_unit_by_label(self, label: str, kind: Optional[str] = None) -> Optional[SemanticUnit]:
        """按 label 在 WM 中查找节点，供 LTM 属性回写使用。支持别名匹配。"""
        if not label:
            return None
        label_norm = label.strip().lower()
        best_unit: Optional[SemanticUnit] = None
        best_score = 0.0
        candidates = (
            self.by_kind.get(kind, [])
            if kind and kind in self.by_kind
            else list(self.nodes.keys())
        )
        for unit_id in candidates:
            unit = self.nodes.get(unit_id)
            if not unit:
                continue
            
            # 1. Primary label 匹配
            score = _string_similarity(label_norm, unit.primary_label().lower())
            
            # 2. ✨ 别名匹配
            if unit.symbol.aliases:
                for alias in unit.symbol.aliases:
                    alias_score = _string_similarity(label_norm, alias.lower())
                    score = max(score, alias_score)
            
            if score > best_score:
                best_score = score
                best_unit = unit
        
        return best_unit if best_score >= 0.6 else None

    def select_nodes(self, query: WMReadQuery) -> List[SemanticUnit]:
        """根据查询关注的 kind 与 utility 提取一组节点。"""
        focus = query.focus_kinds or list(self.by_kind.keys())
        selected: List[SemanticUnit] = []
        for kind in focus:
            for unit_id in self.by_kind.get(kind, []):
                selected.append(self.nodes[unit_id])
        selected.sort(key=lambda u: (self.utility.get(u.id, 0), u.metadata.recency), reverse=True)
        return selected[: query.max_nodes]

    def extract_key_entities_from_goal(self, goal: str) -> List[str]:
        """从 goal 文本中提取关键对象/地点名称。"""
        if not goal:
            return []
        entities: List[str] = []
        
        # 1. 使用 ACTION_ENTITY_PATTERNS 提取动作目标
        for pattern, _ in ACTION_ENTITY_PATTERNS:
            match = pattern.search(goal)
            if match:
                label = match.group("label").strip()
                if label:
                    entities.append(label)
                    logger.info(f"[Entity Extraction] Pattern matched: {pattern.pattern} -> '{label}'")
        
        # 2. 提取常见介词短语中的实体
        prep_patterns = [
            r"from\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+to|\s+and|\s+in|\s+of|$)",
            r"to\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+from|\s+in|\s+of|$)",
            r"on\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+to|\s+of|$)",
            r"in\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+to|\s+of|$)",
            r"at\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+to|\s+of|$)",
            r"of\s+(?:the\s+)?(?P<label>[\w\s]+?)(?:\s+and|\s+to|\s+in|[,.]|$)",
        ]
        for pattern_str in prep_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(goal):
                label = match.group("label").strip()
                # 过滤掉纯介词或太短的词
                if label and len(label) > 2 and label.lower() not in {"the", "and", "or"}:
                    entities.append(label)
                    logger.info(f"[Entity Extraction] Prep pattern matched: {pattern_str} -> '{label}'")
        
        # 3. 去重并返回（保持顺序）
        seen: Set[str] = set()
        unique_entities: List[str] = []
        for entity in entities:
            normalized = entity.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity)
        
        logger.info(f"[Entity Extraction] Final extracted entities from goal '{goal}': {unique_entities}")
        return unique_entities

    def infer_kind_from_entity(self, entity: str) -> str:
        """根据实体名称推断 kind（object/place）。"""
        place_keywords = [
            "counter", "table", "shelf", "cabinet", "drawer", "room",
            "kitchen", "bedroom", "bathroom", "living", "dining",
            "receptacle", "container", "box", "bin"
        ]
        entity_lower = entity.lower()
        
        for keyword in place_keywords:
            if keyword in entity_lower:
                return "place"
        
        return "object"

    def prune(self, step: int) -> List[str]:
        """按 TTL/utility 剪枝，返回被移除的节点 id。"""
        removed: List[str] = []
        # TTL / inactivity prune
        for unit_id, unit in list(self.nodes.items()):
            age = step - self.last_update_step.get(unit_id, 0)
            if age > unit.metadata.ttl or len(self.nodes) > self.max_nodes * 1.2:
                removed.append(unit_id)
        # utility-based pruning if still too big
        if len(self.nodes) - len(removed) > self.max_nodes:
            sorted_by_utility = sorted(
                (uid for uid in self.nodes if uid not in removed),
                key=lambda x: self.utility.get(x, 0.0),
            )
            overflow = len(self.nodes) - len(removed) - self.max_nodes
            removed.extend(sorted_by_utility[:overflow])
        for unit_id in removed:
            unit = self.nodes.pop(unit_id, None)
            if not unit:
                continue
            if unit_id in self.by_kind.get(unit.kind, []):
                self.by_kind[unit.kind].remove(unit_id)
            self.utility.pop(unit_id, None)
            self.last_update_step.pop(unit_id, None)
            logger.debug(f"[SemanticWMGraph] Pruned unit {unit_id} ({unit.kind})")
        return removed

    def snapshot(self, limit: int = 32) -> List[Dict[str, Any]]:
        """导出 WM digest，方便日志/调试/上报。"""
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda u: (u.metadata.recency, self.utility.get(u.id, 0.0)), reverse=True)
        return [node.to_digest() for node in nodes[:limit]]

    def ensure_unit(self, kind: str, label: str, step: int, provenance: str) -> SemanticUnit:
        """若引用的目标节点不存在，就以稳定 hash 创建一个 stub。"""
        # deterministic id by hashing kind + label for stable integration
        stable_id = hashlib.sha1(f"{kind}:{label}".encode("utf-8")).hexdigest()[:12]
        unit_id = f"{kind}_{stable_id}"
        if unit_id in self.nodes:
            return self.nodes[unit_id]
        unit = SemanticUnit(
            id=unit_id,
            kind=kind,
            symbol=SemanticSymbol(type_candidates=[TypeCandidate(label=label, confidence=0.6)]),
            metadata=SemanticMetadata(confidence=0.6, recency=step, ttl=80, provenance=provenance),
            clip_refs=[],
        )
        self.add_unit(unit, step)
        return unit

    def get_edges(self, node_ids: Sequence[str]) -> List[Tuple[str, str, str]]:
        """返回给定节点之间的 directed edges (source, target, relation)。"""
        node_set = set(node_ids)
        edges: List[Tuple[str, str, str]] = []
        for unit in self.nodes.values():
            if unit.id not in node_set:
                continue
            for rel in unit.symbol.relations:
                if rel.target in node_set:
                    edges.append((unit.id, rel.target, rel.relation))
        return edges


# ---------------------------------------------------------------------------
# Operator engine
# ---------------------------------------------------------------------------


class MemoryOperatorEngine:
    """WM 六大算子的工程实现（写/读/校验/抽象/简化/整合）。"""
    def __init__(self, graph: SemanticWMGraph, ltm: Optional['LTMService'] = None):
        self.graph = graph
        self.ltm = ltm  # ✨ 添加 LTM 引用，用于语义对齐

    def write(self, candidates: Sequence[SemanticUnitCandidate], step: int) -> List[Dict[str, Any]]:
        """将候选批量写入 WM，记录 add/update 日志。"""
        logs: List[Dict[str, Any]] = []
        for cand in candidates:
            if not cand:
                continue
            match_id = self.graph.match_candidate(cand)
            if match_id:
                self.graph.update_unit(match_id, cand, step)
                logs.append({"action": "update", "unit_id": match_id, "kind": cand.kind})
            else:
                unit = cand.to_unit()
                unit.metadata.recency = step
                self.graph.add_unit(unit, step)
                logs.append({"action": "add", "unit_id": unit.id, "kind": cand.kind})
        return logs

    def read(self, query: WMReadQuery) -> WMReadResult:
        """根据查询抓取子图，并标记缺失字段，供 planner 调整策略。"""
        nodes = self.graph.select_nodes(query)
        required = query.required_fields or ["state_vars"]
        gaps: List[str] = []
        
        # 检测已存在节点的缺失字段
        for req in required:
            missing = [node.primary_label() for node in nodes if req not in node.symbol.state_vars]
            if missing:
                gaps.append(f"{req} missing for {', '.join(missing[:4])}")
        
        # 检测已存在节点的属性缺口
        gaps.extend(self._collect_attribute_gaps(nodes))
        
        # ✨ 新增：检测 goal 中提到但 WM 中不存在的实体
        entity_gaps = self._detect_missing_entities_in_goal(query.goal, nodes)
        gaps.extend(entity_gaps)
        
        edges = self.graph.get_edges([node.id for node in nodes])
        return WMReadResult(nodes=nodes, edges=edges, gaps=gaps)

    def _detect_missing_entities_in_goal(
        self, 
        goal: str, 
        existing_nodes: List[SemanticUnit]
    ) -> List[str]:
        """检测 goal 中提到但 existing_nodes 中不存在的实体。"""
        if not goal:
            return []
        
        # 1. 提取 goal 中的关键实体
        key_entities = self.graph.extract_key_entities_from_goal(goal)
        logger.info(f"[Entity Gap Detection] Key entities extracted: {key_entities}")
        
        # 2. 构建已存在节点的标签集合（normalized），包含别名
        existing_labels: Set[str] = set()
        for node in existing_nodes:
            label = node.primary_label()
            if label:
                existing_labels.add(label.lower().strip())
            # ✨ 添加别名到可匹配标签集合
            if node.symbol.aliases:
                for alias in node.symbol.aliases:
                    existing_labels.add(alias.lower().strip())
        logger.info(f"[Entity Gap Detection] Existing labels in WM (with aliases): {sorted(existing_labels)}")
        
        # 3. 找出缺失的实体
        missing_gaps: List[str] = []
        for entity in key_entities:
            normalized = entity.lower().strip()
            
            # 检查是否已存在（完全匹配或高相似度）
            found = False
            if normalized in existing_labels:
                found = True
                logger.info(f"[Entity Gap Detection] Entity '{entity}' found (exact match)")
            else:
                # 模糊匹配：检查是否有高度相似的标签
                for existing_label in existing_labels:
                    similarity = _string_similarity(normalized, existing_label)
                    if similarity >= 0.8:
                        found = True
                        logger.info(f"[Entity Gap Detection] Entity '{entity}' found (fuzzy match with '{existing_label}', similarity={similarity:.2f})")
                        break
            
            if not found:
                # 生成 entity gap
                kind = self.graph.infer_kind_from_entity(entity)
                gap = f"entity:{kind}:{entity}"
                missing_gaps.append(gap)
                logger.info(f"[Entity Gap Detection] Entity '{entity}' NOT found -> gap: {gap}")
        
        logger.info(f"[Entity Gap Detection] Final missing gaps: {missing_gaps}")
        return missing_gaps

    def correctness(self, feedback: Dict[str, Any], step: int) -> None:
        """根据执行反馈调整 risk level，避免反复犯同一错误。"""
        # map invalid feedback to risk adjustments on relevant units (object/place or rule)
        target_label = str(feedback.get("target", "")).strip()
        if not target_label:
            return
        # pick best matching node regardless of kind
        best_id = None
        best_score = 0.0
        for unit in self.graph.nodes.values():
            score = _string_similarity(target_label, unit.primary_label())
            if score > best_score:
                best_id = unit.id
                best_score = score
        if not best_id:
            return
        unit = self.graph.nodes[best_id]
        risk_key = "risk_level"
        previous = float(unit.symbol.state_vars.get(risk_key, 0.0))
        delta = 0.2 if feedback.get("invalid", False) else -0.1
        unit.symbol.state_vars[risk_key] = max(0.0, min(1.0, previous + delta))
        unit.metadata.recency = step
        self.graph.utility[best_id] = min(self.graph.utility.get(best_id, 1.0) + 0.2, 5.0)

    def abstract(self, step: int) -> None:
        """抽象步骤：将稳定的 0/1 状态转写为条件，方便 schema 匹配。"""
        # simple smoothing: propagate stable boolean flags into conditions
        for unit in self.graph.nodes.values():
            stable_flags = {
                k: v for k, v in unit.symbol.state_vars.items()
                if isinstance(v, (int, float)) and v in (0, 1)
            }
            if not stable_flags:
                continue
            unit.symbol.conditions = [
                {"if": k, "equals": v} for k, v in stable_flags.items()
            ]
            unit.metadata.recency = step

    def simplify(self, step: int) -> None:
        """简化步骤：触发图剪枝，保持 WM 容量稳定。"""
        removed = self.graph.prune(step)
        if removed:
            logger.debug(f"[MemoryOperatorEngine] Simplify removed {len(removed)} units")

    def align_semantics(
        self, 
        pending_gaps: List[str], 
        observed_labels: List[str],
        step: int
    ) -> List[Dict[str, Any]]:
        """✨ 语义对齐：将 pending entity gaps 与新观察到的对象建立映射关系。
        
        Args:
            pending_gaps: 格式为 "entity:kind:label" 的待解析实体列表
            observed_labels: 当前 step 观察到的对象标签列表
            step: 当前步数
            
        Returns:
            对齐记录列表，每条记录包含 abstract_term, concrete_term, confidence
        """
        if not pending_gaps or not observed_labels:
            return []
        
        alignments: List[Dict[str, Any]] = []
        
        # 解析 pending gaps
        for gap in pending_gaps:
            parts = gap.split(":")
            if len(parts) < 3:
                continue
            gap_kind = parts[1]  # object/place
            gap_label = parts[2].strip().lower()
            
            # ✨ 解析空间关系："right receptacle of left counter"
            spatial_modifier = None
            core_term = gap_label
            reference_object = None
            
            # 检测 "X of Y" 模式
            if " of " in gap_label:
                before_of, after_of = gap_label.split(" of ", 1)
                reference_object = after_of.strip()
                
                # 提取方位词和核心术语
                tokens = before_of.strip().split()
                if len(tokens) >= 2 and tokens[0] in ["left", "right", "front", "back", "top", "bottom"]:
                    spatial_modifier = tokens[0]
                    core_term = " ".join(tokens[1:])
                    logger.info(
                        f"[Semantic Alignment] Parsed '{gap_label}' → "
                        f"spatial='{spatial_modifier}', term='{core_term}', ref='{reference_object}'"
                    )
                else:
                    core_term = before_of.strip()
            
            # ✨ 检查是否是抽象术语（从预定义字典中查找）
            # 支持带修饰词的情况，如 "right receptacle" 应该匹配 "receptacle"
            potential_concrete = ABSTRACT_TO_CONCRETE_HINTS.get(core_term, [])
            
            # 如果直接查找失败，尝试提取最后一个词（去除修饰词）
            if not potential_concrete and " " in core_term:
                # 提取核心词（最后一个词），如 "kitchen receptacle" -> "receptacle"
                last_word = core_term.split()[-1]
                potential_concrete = ABSTRACT_TO_CONCRETE_HINTS.get(last_word, [])
                if potential_concrete:
                    logger.info(
                        f"[Semantic Alignment] Extracted core term '{last_word}' from '{core_term}', "
                        f"hints: {potential_concrete}"
                    )
                    core_term = last_word
            
            # 🔍 候选对象池：observed_labels + LTM 查询结果
            candidate_labels = list(observed_labels)  # 从观察到的对象开始
            
            # ✨ 如果是抽象概念且有 potential_concrete hints，从 LTM 主动查询
            if potential_concrete and self.ltm:
                logger.info(
                    f"[Semantic Alignment] Querying LTM for concrete instances of '{core_term}': "
                    f"{potential_concrete}"
                )
                for concrete_hint in potential_concrete:
                    # 在 LTM 中查找该具体对象
                    ltm_node = self.ltm._get_node_by_label(concrete_hint)
                    if ltm_node:
                        # 检查是否有 is_a 关系指向抽象概念
                        outgoing_edges = [
                            edge for edge in self.ltm.edges.values()
                            if edge.source == ltm_node.id
                        ]
                        for edge in outgoing_edges:
                            if edge.relation == "is_a":
                                target_node = self.ltm.nodes.get(edge.target)
                                if target_node and target_node.label.lower() == core_term:
                                    # 找到匹配：concrete_hint is_a core_term
                                    if concrete_hint not in candidate_labels:
                                        candidate_labels.append(concrete_hint)
                                        logger.info(
                                            f"[LTM Query] Found '{concrete_hint}' is_a '{core_term}' "
                                            f"(confidence={edge.confidence:.2f})"
                                        )
                                    break
            
            # 如果候选池仍为空，尝试反向查询：找到所有指向 core_term 的 is_a 关系
            if not candidate_labels and core_term and self.ltm:
                ltm_abstract_node = self.ltm._get_node_by_label(core_term)
                if ltm_abstract_node:
                    # 查找所有指向该抽象概念的 is_a 边
                    incoming_edges = [
                        edge for edge in self.ltm.edges.values()
                        if edge.target == ltm_abstract_node.id and edge.relation == "is_a"
                    ]
                    logger.info(
                        f"[LTM Query] Reverse lookup: found {len(incoming_edges)} objects "
                        f"with 'is_a {core_term}' relation"
                    )
                    for edge in incoming_edges:
                        source_node = self.ltm.nodes.get(edge.source)
                        if source_node:
                            candidate_labels.append(source_node.label)
                            logger.info(
                                f"[LTM Query] Candidate: '{source_node.label}' is_a '{core_term}' "
                                f"(confidence={edge.confidence:.2f})"
                            )
            
            # 在候选对象池中寻找最佳匹配
            best_match = None
            best_score = 0.0
            match_reason = ""
            
            for obs_label in candidate_labels:
                obs_norm = obs_label.strip().lower()
                score = 0.0
                reason_parts = []
                
                # 1. 类型匹配：直接在 hint 列表中
                if obs_norm in [c.lower() for c in potential_concrete]:
                    score = 0.85
                    reason_parts.append(f"type={core_term}")
                else:
                    # 2. 模糊匹配
                    for concrete in potential_concrete:
                        sim = _string_similarity(obs_norm, concrete.lower())
                        if sim >= 0.7:
                            score = max(score, sim * 0.8)
                            reason_parts.append(f"fuzzy={concrete}({sim:.2f})")
                
                # 3. 📚 LTM 知识查询：检查对象是否有 is_a 关系指向抽象概念
                if score == 0.0 and self.ltm:
                    # 优先从 LTM 查询（更可靠）
                    ltm_node = self.ltm._get_node_by_label(obs_label)
                    if ltm_node:
                        outgoing_edges = [
                            edge for edge in self.ltm.edges.values()
                            if edge.source == ltm_node.id and edge.relation == "is_a"
                        ]
                        for edge in outgoing_edges:
                            target_node = self.ltm.nodes.get(edge.target)
                            if target_node and target_node.label.lower() == core_term:
                                score = 0.9
                                reason_parts.append(f"ltm_is_a={core_term}")
                                logger.info(
                                    f"[Semantic Alignment] Found LTM relation: "
                                    f"{obs_label} is_a {core_term} (confidence={edge.confidence:.2f})"
                                )
                                break
                    
                    # 后备：检查 WM 中的关系
                    if score == 0.0:
                        unit = self.graph.find_unit_by_label(obs_label)
                        if unit:
                            for rel in unit.symbol.relations:
                                if rel.relation == "is_a" and rel.target.lower() == core_term:
                                    score = 0.9
                                    reason_parts.append(f"wm_is_a={core_term}")
                                    logger.info(
                                        f"[Semantic Alignment] Found WM relation: "
                                        f"{obs_label} is_a {core_term}"
                                    )
                                    break
                
                # 4. 空间关系加分
                if score > 0 and spatial_modifier:
                    spatial_ok = self._verify_spatial_relation(
                        obs_label, spatial_modifier, reference_object
                    )
                    if spatial_ok:
                        score += 0.15  # 空间匹配显著加分
                        reason_parts.append(f"spatial={spatial_modifier}")
                    else:
                        reason_parts.append("spatial=?")
                
                if score > best_score:
                    best_score = score
                    best_match = obs_label
                    match_reason = "+".join(reason_parts)
            
            # 降低阈值以支持空间推理（0.6 而不是 0.7）
            if best_match and best_score >= 0.6:
                # 找到对齐关系，尝试从 WM 或 LTM 获取/创建节点
                unit = self.graph.find_unit_by_label(best_match, kind=gap_kind)
                
                # 如果 WM 中没有该节点，创建一个 stub（稍后会从 LTM 同步属性）
                if not unit:
                    logger.info(
                        f"[Semantic Alignment] Creating WM stub for '{best_match}' "
                        f"(kind={gap_kind}, source=ltm_alignment)"
                    )
                    unit = self.graph.ensure_unit(
                        kind=gap_kind,
                        label=best_match,
                        step=step,
                        provenance="ltm_alignment"
                    )
                    # 标记该节点需要从 LTM 同步（通过 state_vars 中的特殊标记）
                    unit.symbol.state_vars["_needs_ltm_sync"] = True
                
                if unit:
                    # 添加别名
                    if gap_label not in [a.lower() for a in unit.symbol.aliases]:
                        unit.symbol.aliases.append(gap_label)
                        logger.info(
                            f"[Semantic Alignment] ✅ Aligned '{gap_label}' → '{best_match}' "
                            f"(conf={best_score:.2f}, kind={gap_kind}, reason={match_reason})"
                        )
                    
                    # 添加 synonym_of 关系
                    existing_rels = {(r.relation, r.target) for r in unit.symbol.relations}
                    if ("synonym_of", gap_label) not in existing_rels:
                        unit.symbol.relations.append(
                            RelationSpec(
                                relation="synonym_of",
                                target=gap_label,
                                confidence=best_score,
                                metadata={"aligned_at_step": step}
                            )
                        )
                    
                    unit.metadata.recency = step
                    
                    alignments.append({
                        "abstract_term": gap_label,
                        "concrete_term": best_match,
                        "confidence": best_score,
                        "unit_id": unit.id,
                        "kind": gap_kind,  # ✨ 添加 kind 字段用于追踪
                    })
                else:
                    logger.warning(
                        f"[Semantic Alignment] ⚠️ Failed to create unit for '{best_match}'"
                    )
        
        return alignments

    def _verify_spatial_relation(
        self,
        object_label: str,
        spatial_modifier: str,
        reference_object: Optional[str]
    ) -> bool:
        """验证对象是否满足空间关系约束。
        
        Args:
            object_label: 待验证的对象标签（如 "sink"）
            spatial_modifier: 方位词（如 "right"）
            reference_object: 参考对象（如 "left counter"）
            
        Returns:
            如果满足空间关系返回 True
        """
        # 优先从 WM 查找
        obj_unit = self.graph.find_unit_by_label(object_label)
        
        # 如果 WM 没有，从 LTM 查找
        if not obj_unit and self.ltm:
            ltm_node = self.ltm._get_node_by_label(object_label)
            if ltm_node:
                logger.info(
                    f"[Spatial Verification] Object '{object_label}' found in LTM, checking attributes"
                )
                # 检查 LTM 节点的 state_vars
                state_vars = ltm_node.attributes.get("state_vars", {})
                position = state_vars.get("spatial_position") or state_vars.get("position")
                
                if position:
                    pos_lower = str(position).lower()
                    if spatial_modifier in pos_lower:
                        logger.info(
                            f"[Spatial Verification] ✅ LTM: '{object_label}' position='{position}' "
                            f"matches '{spatial_modifier}'"
                        )
                        return True
                
                # 检查 LTM 边的空间关系
                outgoing_edges = [
                    edge for edge in self.ltm.edges.values()
                    if edge.source == ltm_node.id
                ]
                for edge in outgoing_edges:
                    rel_text = f"{edge.relation} {self.ltm.nodes.get(edge.target, {}).get('label', '')}".lower()
                    if spatial_modifier in rel_text or (reference_object and reference_object in rel_text):
                        logger.info(
                            f"[Spatial Verification] ✅ LTM: '{object_label}' has relation '{rel_text}' "
                            f"matching '{spatial_modifier}'"
                        )
                        return True
                
                logger.info(
                    f"[Spatial Verification] ❌ LTM: '{object_label}' does not match spatial constraint"
                )
                return False
        
        # WM 中有该对象，检查其属性
        # 检查对象的 state_vars 中是否有位置信息
        position = obj_unit.symbol.state_vars.get("spatial_position") or obj_unit.symbol.state_vars.get("position")
        
        if position:
            pos_lower = str(position).lower()
            # 简单的位置匹配：检查位置描述中是否包含方位词
            if spatial_modifier in pos_lower:
                logger.info(
                    f"[Spatial Verification] ✅ WM: '{object_label}' position='{position}' "
                    f"matches '{spatial_modifier}'"
                )
                return True
        
        # 如果没有直接的位置信息，检查关系
        for rel in obj_unit.symbol.relations:
            rel_text = f"{rel.relation} {rel.target}".lower()
            if spatial_modifier in rel_text:
                logger.info(
                    f"[Spatial Verification] ✅ WM: '{object_label}' relation='{rel_text}' "
                    f"matches '{spatial_modifier}'"
                )
                return True
        
        # 检查参考对象
        if reference_object:
            ref_unit = self.graph.find_unit_by_label(reference_object)
            if ref_unit:
                # 查找两个对象之间的空间关系
                for rel in obj_unit.symbol.relations:
                    if reference_object in rel.target.lower() and spatial_modifier in rel.relation.lower():
                        logger.info(
                            f"[Spatial Verification] ✅ WM: Found relation: "
                            f"'{object_label}' {rel.relation} '{rel.target}'"
                        )
                        return True
        
        logger.info(
            f"[Spatial Verification] ❌ Could not verify '{spatial_modifier}' relation "
            f"for '{object_label}' (checked WM and LTM)"
        )
        return False

    def integrate(self, step: int) -> None:
        """整合步骤：确保关系引用的 target 节点存在，必要时补建 stub。"""
        # ensure relations reference existing node ids by promoting unknown targets
        for unit in list(self.graph.nodes.values()):
            for rel in unit.symbol.relations:
                if rel.target in self.graph.nodes:
                    continue
                if not rel.target:
                    continue
                target_unit = self.graph.ensure_unit(
                    kind="place" if "room" in rel.target.lower() else "object",
                    label=rel.target,
                    step=step,
                    provenance="integrate",
                )
                rel.target = target_unit.id

    def _collect_attribute_gaps(self, nodes: Sequence[SemanticUnit], max_gaps: int = 8) -> List[str]:
        gaps: List[str] = []
        for node in nodes:
            if node.kind not in {"object", "place"}:
                continue
            for attr_kind, keys in ATTRIBUTE_KIND_TO_KEYS.items():
                if not keys:
                    continue
                if any(self._has_attribute_value(node.symbol.state_vars.get(key)) for key in keys if key in node.symbol.state_vars):
                    continue
                label = node.primary_label()
                if not label:
                    continue
                token = f"attribute:{attr_kind}:{label}"
                if token in gaps:
                    continue
                gaps.append(token)
                if len(gaps) >= max_gaps:
                    return gaps
        return gaps

    @staticmethod
    def _has_attribute_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return any(MemoryOperatorEngine._has_attribute_value(item) for item in value)
        if isinstance(value, dict):
            return bool(value)
        return True


# ---------------------------------------------------------------------------
# LTM services (stubs)
# ---------------------------------------------------------------------------


class LTMService:
    """LTM = 知识图谱：节点表达语义单元，边记录多种关系。"""

    RELATION_FALLBACK = "association"
    RELATION_TYPES = {
        "synonymy": "synonymy",
        "synonym": "synonymy",
        "alias": "synonymy",
        "antonym": "antonymy",
        "opposite": "antonymy",
        "metaphor": "metaphor",
        "inheritance": "inheritance",
        "is_a": "inheritance",
        "inside": "inheritance",
        "in": "inheritance",
        "association": "association",
        "related-to": "association",
    }

    def __init__(self) -> None:
        self.nodes: Dict[str, LTNode] = {}
        self.edges: Dict[str, LTEdge] = {}
        self.node_index: Dict[str, str] = {}
        self.max_family_refs = 5
        self.attribute_label_index: Dict[str, Set[str]] = defaultdict(set)
        self.attribute_object_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    # -- core helpers -----------------------------------------------------

    def _normalize_label(self, label: str) -> str:
        return (label or "concept").strip().lower()

    def _get_node_by_label(self, label: str) -> Optional[LTNode]:
        norm = self._normalize_label(label)
        node_id = self.node_index.get(norm)
        return self.nodes.get(node_id) if node_id else None

    def _upsert_node(
        self,
        label: str,
        kinds: Sequence[str],
        perspective: str,
        clip_refs: Optional[Sequence[str]],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[LTNode, bool]:
        norm = self._normalize_label(label)
        node_id = self.node_index.get(norm)
        created = False
        if node_id is None:
            node_id = _generate_id("ltm_node")
            node = LTNode(
                id=node_id,
                label=label or "concept",
                kinds=list(dict.fromkeys(kinds)) if kinds else ["concept"],
                perspective=[perspective] if perspective else [],
            )
            node.attributes = attributes.copy() if attributes else {}
            node.evidence = list(clip_refs or [])
            self.nodes[node_id] = node
            self.node_index[norm] = node_id
            created = True
        else:
            node = self.nodes[node_id]
            node.kinds = list(dict.fromkeys(node.kinds + list(kinds or [])))
            if perspective and perspective not in node.perspective:
                node.perspective.append(perspective)
            if clip_refs:
                for clip_id in clip_refs:
                    if clip_id not in node.evidence:
                        node.evidence.append(clip_id)
            if attributes:
                node.attributes.setdefault("state_vars", {}).update(attributes.get("state_vars", {}))
                if attributes.get("constraints"):
                    node.attributes.setdefault("constraints", []).extend(attributes["constraints"])
                if attributes.get("conditions"):
                    node.attributes.setdefault("conditions", []).extend(attributes["conditions"])
        node.centrality = min(1.0, node.centrality + 0.05)
        node.graduality = min(1.0, node.graduality + 0.04)
        if attributes and attributes.get("constraints"):
            node.fuzziness = max(0.1, node.fuzziness - 0.05)
        else:
            node.fuzziness = min(1.0, node.fuzziness + 0.02)
        node.last_activated = time.time()
        self._update_family_resemblance(node)
        if "attribute" in node.kinds:
            self._track_attribute_node(node)
        return node, created

    def _update_family_resemblance(self, node: LTNode) -> None:
        scored: List[Tuple[float, str]] = []
        for other in self.nodes.values():
            if other.id == node.id:
                continue
            score = _string_similarity(node.label, other.label)
            if score > 0.45:
                scored.append((score, other.label))
        scored.sort(key=lambda x: x[0], reverse=True)
        node.family_resemblance = [label for _, label in scored[: self.max_family_refs]]

    def _track_attribute_node(self, node: LTNode, attr_kind: Optional[str] = None) -> None:
        if "attribute" not in node.kinds:
            return
        attr_type = attr_kind or node.attributes.get("attribute_type") or next(
            (kind for kind in node.kinds if kind != "attribute"),
            "attribute",
        )
        norm = self._normalize_label(node.label)
        self.attribute_label_index[norm].add(node.id)
        node.attributes.setdefault("attribute_type", attr_type)

    def _register_attribute_edge(self, source_id: str, target_id: str, relation: str) -> None:
        target_node = self.nodes.get(target_id)
        if not target_node or "attribute" not in target_node.kinds:
            return
        key = (target_id, relation)
        self.attribute_object_index[key].add(source_id)

    def _purge_attribute_indices_for_node(self, node_id: str) -> None:
        for key in list(self.attribute_object_index.keys()):
            attr_id, relation = key
            if attr_id == node_id:
                self.attribute_object_index.pop(key, None)
                continue
            targets = self.attribute_object_index[key]
            if node_id in targets:
                targets.discard(node_id)
                if not targets:
                    self.attribute_object_index.pop(key, None)

    def _canonical_relation(self, relation: Optional[str]) -> str:
        rel = (relation or self.RELATION_FALLBACK).lower()
        for key, value in self.RELATION_TYPES.items():
            if key in rel:
                return value
        if rel in ("above", "on", "near"):
            return "association"
        return self.RELATION_FALLBACK

    def _link_nodes(
        self,
        source_id: str,
        target_id: str,
        relation: Optional[str],
        confidence: float,
        clip_refs: Optional[Sequence[str]] = None,
    ) -> LTEdge:
        canonical = self._canonical_relation(relation)
        now = time.time()
        for edge in self.edges.values():
            if edge.source == source_id and edge.target == target_id and edge.relation == canonical:
                edge.confidence = min(1.0, 0.7 * edge.confidence + 0.3 * confidence)
                edge.recency = now
                edge.usage_count += 1
                if clip_refs:
                    for clip_id in clip_refs:
                        if clip_id not in edge.evidence_clips:
                            edge.evidence_clips.append(clip_id)
                self._register_attribute_edge(source_id, target_id, canonical)
                return edge
        edge_id = _generate_id("ltm_edge")
        edge = LTEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            relation=canonical,
            recency=now,
            confidence=min(1.0, confidence),
            evidence_clips=list(clip_refs or []),
            usage_count=1,
        )
        self.edges[edge_id] = edge
        self._register_attribute_edge(source_id, target_id, canonical)
        return edge

    def _ensure_attribute_node(
        self,
        label: str,
        attr_kind: str,
        perspective: str = "attribute",
    ) -> LTNode:
        node, _ = self._upsert_node(
            label=label,
            kinds=["attribute", attr_kind],
            perspective=perspective,
            clip_refs=None,
            attributes={"attribute_type": attr_kind},
        )
        self._track_attribute_node(node, attr_kind)
        return node

    def _ingest_state_attributes(self, node: LTNode, state_vars: Dict[str, Any]) -> None:
        if not state_vars:
            return
        for key, value in state_vars.items():
            if key in ATTRIBUTE_EXCLUDE_KEYS or value in (None, ""):
                continue
            relation, attr_kind, _ = ATTRIBUTE_FIELD_MAP.get(
                key,
                (ATTRIBUTE_RELATION_DEFAULT, "attribute", False),
            )
            if isinstance(value, (list, tuple, set)):
                values = value
            else:
                values = [value]
            for raw in values:
                if raw in (None, "") or isinstance(raw, bool):
                    continue
                label = str(raw).strip()
                if not label:
                    continue
                attr_node = self._ensure_attribute_node(label, attr_kind)
                self._link_nodes(node.id, attr_node.id, relation=relation, confidence=0.65, clip_refs=None)

    def _collect_attribute_relations(self, node_id: str) -> Dict[str, List[str]]:
        relations: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges.values():
            if edge.source != node_id:
                continue
            target = self.nodes.get(edge.target)
            if not target or "attribute" not in target.kinds:
                continue
            attr_kind = target.attributes.get("attribute_type") or ATTRIBUTE_RELATION_TO_KIND.get(edge.relation)
            if not attr_kind:
                attr_kind = next((kind for kind in target.kinds if kind != "attribute"), "attribute")
            label = target.label
            if not label:
                continue
            if label not in relations[attr_kind]:
                relations[attr_kind].append(label)
        return relations

    def _collect_attribute_values(self, node: LTNode, attr_kind: str) -> List[str]:
        values = list(self._collect_attribute_relations(node.id).get(attr_kind, []))
        if node.attributes:
            state_vars = node.attributes.get("state_vars", {}) or {}
        else:
            state_vars = {}
        for key in ATTRIBUTE_KIND_TO_KEYS.get(attr_kind, []):
            raw = state_vars.get(key)
            if raw in (None, ""):
                continue
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    if item in (None, ""):
                        continue
                    text = str(item).strip()
                    if text and text not in values:
                        values.append(text)
            else:
                text = str(raw).strip()
                if text and text not in values:
                    values.append(text)
        return values

    def _build_attribute_paths_for_kind(self, node_id: str, attr_kind: str, limit: int = 3) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        relation = ATTRIBUTE_KIND_TO_RELATION.get(attr_kind)
        if not relation:
            return paths
        for edge in self.edges.values():
            if edge.source != node_id or edge.relation != relation:
                continue
            target_node = self.nodes.get(edge.target)
            if not target_node or "attribute" not in target_node.kinds:
                continue
            paths.append(
                {
                    "source": self.nodes[node_id].label,
                    "relation": relation,
                    "target": target_node.label,
                    "confidence": round(edge.confidence, 3),
                    "target_kind": "attribute",
                }
            )
            if len(paths) >= limit:
                break
        return paths

    def _parse_attribute_gap(self, gap: str) -> Optional[Tuple[str, str]]:
        if not isinstance(gap, str) or not gap.startswith("attribute:"):
            return None
        parts = gap.split(":", 2)
        if len(parts) != 3:
            return None
        attr_kind = parts[1].strip()
        label = parts[2].strip()
        if not attr_kind or not label:
            return None
        return attr_kind, label

    def _parse_entity_gap(self, gap: str) -> Optional[Tuple[str, str]]:
        """解析 entity gap: 'entity:object:spatula' → ('object', 'spatula')"""
        if not isinstance(gap, str) or not gap.startswith("entity:"):
            return None
        parts = gap.split(":", 2)
        if len(parts) != 3:
            return None
        kind = parts[1].strip()
        label = parts[2].strip()
        if not kind or not label:
            return None
        return kind, label

    def _extract_position_from_node(self, node: LTNode) -> str:
        """从 LTM 节点提取位置信息。"""
        # 1. 查找空间关系边
        spatial_relations = {"on", "in", "at", "near", "inside", "above", "below", "beside"}
        for edge in self.edges.values():
            if edge.source != node.id:
                continue
            if edge.relation in spatial_relations:
                target_node = self.nodes.get(edge.target)
                if target_node:
                    return f"{edge.relation} {target_node.label}"
        
        # 2. 从 attributes 提取
        position = (
            node.attributes.get("spatial_position") 
            or node.attributes.get("last_seen_position")
            or node.attributes.get("position")
        )
        if position:
            return str(position)
        
        # 3. 从 state_vars 提取（如果 LTM 节点保留了 WM 的 state_vars）
        state_vars = node.attributes.get("state_vars", {})
        if isinstance(state_vars, dict):
            position = (
                state_vars.get("spatial_position")
                or state_vars.get("last_seen_position")
                or state_vars.get("position")
            )
            if position:
                return str(position)
        
        return "unknown"

    def _summarize_node_attributes(self, node: LTNode) -> str:
        """汇总节点的关键属性为简短字符串。"""
        parts: List[str] = []
        
        # 颜色
        colors = self._collect_attribute_values(node, "color")
        if colors:
            parts.append(f"color={'/'.join(colors[:2])}")
        
        # 尺寸
        sizes = self._collect_attribute_values(node, "size")
        if sizes:
            parts.append(f"size={sizes[0]}")
        
        # 材质
        materials = self._collect_attribute_values(node, "material")
        if materials:
            parts.append(f"material={'/'.join(materials[:2])}")
        
        # 形状
        shapes = self._collect_attribute_values(node, "shape")
        if shapes:
            parts.append(f"shape={shapes[0]}")
        
        return ", ".join(parts) if parts else "no detailed attributes"

    def _collect_all_node_attributes(self, node: LTNode) -> Dict[str, List[str]]:
        """收集节点的所有属性值（用于 hint 详细信息）。"""
        attributes: Dict[str, List[str]] = {}
        
        # 收集各种类型的属性
        for attr_kind in ["color", "size", "material", "shape", "texture", "feature"]:
            values = self._collect_attribute_values(node, attr_kind)
            if values:
                attributes[attr_kind] = values
        
        # 同时也包含 position
        position = self._extract_position_from_node(node)
        if position and position != "unknown":
            attributes["position"] = [position]
        
        return attributes

    def _score_node(self, node: LTNode, perspective_tokens: Sequence[str], gap: str) -> float:
        perspective_bonus = 0.15 if any(
            token and token.lower() in " ".join(node.perspective).lower()
            for token in perspective_tokens
        ) else 0.0
        gap_bonus = _string_similarity(gap, node.label) * 0.4
        return node.centrality * 0.4 + (1.0 - node.fuzziness) * 0.3 + node.graduality * 0.15 + perspective_bonus + gap_bonus

    def _build_graph_path(self, node_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        edges = [edge for edge in self.edges.values() if edge.source == node_id]
        edges.sort(key=lambda e: (e.confidence, e.recency), reverse=True)
        path: List[Dict[str, Any]] = []
        for edge in edges[:limit]:
            target_node = self.nodes.get(edge.target)
            target_label = target_node.label if target_node else edge.target
            target_kind = target_node.kinds[0] if target_node and target_node.kinds else ""
            path.append(
                {
                    "source": self.nodes[node_id].label,
                    "relation": edge.relation,
                    "target": target_label,
                    "confidence": round(edge.confidence, 3),
                    "target_kind": target_kind,
                }
            )
        return path

    def _compose_summary(self, node: LTNode, gap: str, path: List[Dict[str, Any]]) -> str:
        relation_desc = "; ".join(f"{step['relation']}->{step['target']}" for step in path) or "no related edges"
        return (
            f"Node '{node.label}' (centrality={node.centrality:.2f}, fuzziness={node.fuzziness:.2f})"
            f" may close gap '{gap}'. Relations: {relation_desc}."
        )

    def describe_node(self, label: str, path_limit: int = 4) -> Optional[Dict[str, Any]]:
        """返回指定 label 的节点属性/关系摘要，方便 WM 实时同步。"""
        node = self._get_node_by_label(label)
        if not node:
            return None
        connections = self._build_graph_path(node.id, limit=path_limit)
        attribute_relations = self._collect_attribute_relations(node.id)
        return {
            "label": node.label,
            "kinds": list(node.kinds),
            "attributes": dict(node.attributes),
            "perspective": list(node.perspective),
            "connections": connections,
            "attribute_relations": attribute_relations,
        }

    def query_relation_path(
        self, 
        start_label: str, 
        target_label: str,
        max_depth: int = 3
    ) -> Optional[Dict[str, Any]]:
        """✨ 查询两个实体之间的关系路径（用于语义推理）。
        
        例如：receptacle → (synonym_of) → sink → (left_of) → counter
        
        Args:
            start_label: 起始节点标签（如 "receptacle"）
            target_label: 目标节点标签（如 "left counter"）
            max_depth: 最大搜索深度
            
        Returns:
            包含路径、置信度的字典，或 None（未找到路径）
        """
        start_node = self._get_node_by_label(start_label)
        target_node = self._get_node_by_label(target_label)
        
        if not start_node or not target_node:
            # 检查别名
            if not start_node:
                start_node = self._find_node_by_alias(start_label)
            if not target_node:
                target_node = self._find_node_by_alias(target_label)
        
        if not start_node or not target_node:
            return None
        
        # BFS 查找路径
        from collections import deque
        
        queue = deque([(start_node.id, [])])  # (node_id, path)
        visited = {start_node.id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_node.id:
                # 找到路径
                confidence = 1.0
                for _, _, edge_conf in path:
                    confidence *= edge_conf
                
                return {
                    "found": True,
                    "path": path,
                    "confidence": confidence,
                    "steps": len(path),
                    "path_description": " → ".join([
                        f"{relation}({self.nodes[src].label if src in self.nodes else 'unknown'} → {self.nodes[tgt].label if tgt in self.nodes else 'unknown'})"
                        for src, tgt, _, relation in [(p[0], p[1], p[2], p[3]) for p in path]
                    ])
                }
            
            if len(path) >= max_depth:
                continue
            
            # 扩展邻居
            for edge in self.edges.values():
                if edge.source == current_id:
                    next_id = edge.target
                    if next_id not in visited:
                        visited.add(next_id)
                        new_path = path + [(edge.source, edge.target, edge.confidence, edge.relation)]
                        queue.append((next_id, new_path))
        
        return {"found": False, "reason": "No path found"}

    def _find_node_by_alias(self, alias: str) -> Optional[LTNode]:
        """通过别名查找节点。"""
        alias_norm = self._normalize_label(alias)
        for node in self.nodes.values():
            aliases = node.attributes.get("aliases", [])
            if any(self._normalize_label(a) == alias_norm for a in aliases):
                return node
        return None

    def _merge_similar_nodes(self, threshold: float = 0.82) -> int:
        merged = 0
        processed: Set[str] = set()
        for node in list(self.nodes.values()):
            if node.id in processed:
                continue
            for other in list(self.nodes.values()):
                if other.id == node.id or other.id in processed:
                    continue
                if _string_similarity(node.label, other.label) < threshold:
                    continue
                node.centrality = min(1.0, (node.centrality + other.centrality) / 2 + 0.1)
                node.graduality = min(1.0, max(node.graduality, other.graduality) + 0.05)
                node.fuzziness = max(0.1, (node.fuzziness + other.fuzziness) / 2 - 0.05)
                node.perspective = list(dict.fromkeys(node.perspective + other.perspective))
                node.evidence = list(dict.fromkeys(node.evidence + other.evidence))
                # redirect edges pointing to other -> node
                for edge in list(self.edges.values()):
                    if edge.source == other.id:
                        edge.source = node.id
                    if edge.target == other.id:
                        edge.target = node.id
                norm = self._normalize_label(other.label)
                self.node_index.pop(norm, None)
                self.nodes.pop(other.id, None)
                processed.add(other.id)
                merged += 1
        return merged

    def _delete_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if not node:
            return
        norm = self._normalize_label(node.label)
        self.node_index.pop(norm, None)
        if "attribute" in node.kinds:
            attr_norm = self._normalize_label(node.label)
            if attr_norm in self.attribute_label_index:
                self.attribute_label_index[attr_norm].discard(node.id)
                if not self.attribute_label_index[attr_norm]:
                    self.attribute_label_index.pop(attr_norm, None)
        self._purge_attribute_indices_for_node(node_id)
        for edge_id, edge in list(self.edges.items()):
            if edge.source == node_id or edge.target == node_id:
                self.edges.pop(edge_id, None)

    # -- public API -------------------------------------------------------

    def ingest_wm_digest(
        self,
        wm_snapshot: Sequence[Dict[str, Any]],
        perspective: str,
        clip_refs: Optional[Sequence[str]] = None,
        goal_context: str = "",
    ) -> Dict[str, int]:
        """Add 操作：将 WM digest 吸收到 LTM 图中，包括别名和语义/空间关系。"""
        stats = {"added": 0, "updated": 0, "edges": 0, "aliases": 0}
        for digest in wm_snapshot:
            label = digest.get("label") or digest.get("kind") or "concept"
            attributes = {
                "state_vars": digest.get("state_vars", {}),
                "conditions": digest.get("conditions", []),
                "constraints": digest.get("constraints", []),
                "goal_context": goal_context,
            }
            node, created = self._upsert_node(
                label=label,
                kinds=[digest.get("kind", "concept")],
                perspective=perspective,
                clip_refs=digest.get("clip_refs", []) or clip_refs,
                attributes=attributes,
            )
            self._ingest_state_attributes(node, attributes.get("state_vars", {}))
            stats["added" if created else "updated"] += 1
            
            # ✨ 同步别名到 LTM
            aliases = digest.get("aliases", [])
            if aliases:
                existing_aliases = node.attributes.get("aliases", [])
                for alias in aliases:
                    if alias and alias not in existing_aliases:
                        existing_aliases.append(alias)
                        stats["aliases"] += 1
                        # 为每个别名创建 synonym_of 关系
                        alias_node, _ = self._upsert_node(
                            label=alias,
                            kinds=[digest.get("kind", "concept")],
                            perspective="semantic_alignment",
                            clip_refs=clip_refs,
                            attributes={},
                        )
                        self._link_nodes(
                            source_id=node.id,
                            target_id=alias_node.id,
                            relation="synonym_of",
                            confidence=0.85,
                            clip_refs=clip_refs or [],
                        )
                        stats["edges"] += 1
                node.attributes["aliases"] = existing_aliases
            
            # 同步关系到 LTM（包括语义和空间关系）
            for rel in digest.get("relations", []) or []:
                target_label = rel.get("target") or rel.get("relation")
                if not target_label:
                    continue
                    
                relation_type = rel.get("relation", "association")
                
                # 判断关系类型
                if relation_type in SEMANTIC_RELATIONS:
                    # 语义关系：直接使用 target_label 作为标签
                    target_node, _ = self._upsert_node(
                        label=target_label,
                        kinds=[SEMANTIC_RELATIONS.get(relation_type, "concept")],
                        perspective=perspective,
                        clip_refs=digest.get("clip_refs", []),
                        attributes={"source": label},
                    )
                else:
                    # 普通关系
                    target_node, _ = self._upsert_node(
                        label=target_label,
                        kinds=[rel.get("relation", "association")],
                        perspective=perspective,
                        clip_refs=digest.get("clip_refs", []),
                        attributes={"source": label},
                    )
                
                self._link_nodes(
                    source_id=node.id,
                    target_id=target_node.id,
                    relation=relation_type,
                    confidence=rel.get("confidence", 0.5),
                    clip_refs=digest.get("clip_refs", []),
                )
                stats["edges"] += 1
        return stats

    def remove_stale(self, min_centrality: float = 0.25, max_idle: float = 900.0) -> Dict[str, int]:
        """Remove 操作：对长期未激活且重要性低的节点/边做遗忘。"""
        now = time.time()
        removed_nodes = 0
        for node_id, node in list(self.nodes.items()):
            idle = now - node.last_activated
            if node.centrality < min_centrality and idle > max_idle:
                self._delete_node(node_id)
                removed_nodes += 1
        removed_edges = 0
        for edge_id, edge in list(self.edges.items()):
            idle = now - edge.recency
            if edge.confidence < 0.2 and idle > max_idle / 2:
                self.edges.pop(edge_id, None)
                removed_edges += 1
        return {"removed_nodes": removed_nodes, "removed_edges": removed_edges}

    def query_by_attribute_hints(
        self,
        attribute_hints: Dict[str, Sequence[str]],
        limit: int = 5,
    ) -> List[str]:
        if not attribute_hints:
            return []
        candidate_sets: List[Set[str]] = []
        for hint_type, values in attribute_hints.items():
            if not values:
                continue
            relation = ATTRIBUTE_HINT_RELATION.get(hint_type, ATTRIBUTE_RELATION_DEFAULT)
            object_ids: Set[str] = set()
            for value in values:
                norm = self._normalize_label(value)
                # Direct match via attribute index
                attr_node_ids = self.attribute_label_index.get(norm, set())
                for attr_id in attr_node_ids:
                    key = (attr_id, relation)
                    object_ids.update(self.attribute_object_index.get(key, set()))
                # Synonym expansion for feature hints
                if hint_type == "feature" and not object_ids:
                    # Extract base words from multi-word features (e.g., "green top" -> "top")
                    words = norm.split()
                    for word in words:
                        synonyms = FEATURE_SYNONYMS.get(word, set())
                        for syn in synonyms:
                            syn_node_ids = self.attribute_label_index.get(syn, set())
                            for attr_id in syn_node_ids:
                                key = (attr_id, relation)
                                object_ids.update(self.attribute_object_index.get(key, set()))
                        # Also try the word itself
                        word_node_ids = self.attribute_label_index.get(word, set())
                        for attr_id in word_node_ids:
                            key = (attr_id, relation)
                            object_ids.update(self.attribute_object_index.get(key, set()))
                # Fuzzy match: check if value is contained in any attribute label
                if not object_ids:
                    for attr_label, node_ids in self.attribute_label_index.items():
                        if norm in attr_label or attr_label in norm:
                            for attr_id in node_ids:
                                key = (attr_id, relation)
                                object_ids.update(self.attribute_object_index.get(key, set()))
            if object_ids:
                candidate_sets.append(object_ids)
        if not candidate_sets:
            return []
        # Prefer intersection (AND logic) but fall back to union if no common results
        if len(candidate_sets) == 1:
            combined = candidate_sets[0]
        else:
            intersection = set.intersection(*candidate_sets)
            combined = intersection if intersection else set.union(*candidate_sets)
        if not combined:
            return []
        ranked = sorted(
            combined,
            key=lambda node_id: self.nodes.get(node_id).centrality if self.nodes.get(node_id) else 0.0,
            reverse=True,
        )
        return [self.nodes[node_id].label for node_id in ranked[:limit] if node_id in self.nodes]

    def search(self, query: WMReadQuery, gaps: Sequence[str]) -> List[Dict[str, Any]]:
        """Search 操作：结合节点属性与关系生成可操作提示。"""
        if not self.nodes or not gaps:
            return []
        hints: List[Dict[str, Any]] = []
        attribute_gaps: List[Tuple[str, str]] = []
        entity_gaps: List[Tuple[str, str]] = []  # ← 新增：实体缺口
        residual_gaps: List[str] = []
        
        # 解析不同类型的 gaps
        for gap in gaps:
            # 尝试解析属性缺口
            attr_parsed = self._parse_attribute_gap(gap)
            if attr_parsed:
                attribute_gaps.append(attr_parsed)
                continue
            
            # ✨ 尝试解析实体缺口
            entity_parsed = self._parse_entity_gap(gap)
            if entity_parsed:
                entity_gaps.append(entity_parsed)
                continue
            
            # 其他类型的缺口
            residual_gaps.append(gap)
        
        # ✨ 处理实体缺口：从 LTM 检索完整节点信息（包括位置）
        for kind, label in entity_gaps:
            node = self._get_node_by_label(label)
            if not node:
                continue
            
            # 提取位置信息
            position_info = self._extract_position_from_node(node)
            
            # 汇总属性
            attribute_summary = self._summarize_node_attributes(node)
            
            # 收集所有属性值
            all_attributes = self._collect_all_node_attributes(node)
            
            # 构建图路径
            path = self._build_graph_path(node.id)
            
            hints.append({
                "summary": (
                    f"{label} is a {kind}. "
                    f"Last seen at: {position_info}. "
                    f"Attributes: {attribute_summary}"
                ),
                "applicability": f"entity:{kind}:{label}",
                "entity_kind": kind,
                "entity_label": label,
                "position": position_info,
                "attributes": all_attributes,
                "graph_path": path,
            })
        
        # 处理属性缺口
        for attr_kind, label in attribute_gaps:
            node = self._get_node_by_label(label)
            if not node:
                continue
            values = self._collect_attribute_values(node, attr_kind)
            if not values:
                continue
            path = self._build_attribute_paths_for_kind(node.id, attr_kind)
            summary_values = ", ".join(values[:3])
            hints.append(
                {
                    "summary": f"{label} typically has {attr_kind}: {summary_values}",
                    "applicability": f"{attr_kind}:{label}",
                    "attribute_values": values,
                    "attribute_kind": attr_kind,
                    "attribute_target": label,
                    "graph_path": path,
                }
            )
        if not residual_gaps:
            return hints
        perspective_tokens = [query.goal, *query.focus_kinds, *query.risk_tags]
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda node: self._score_node(node, perspective_tokens, "baseline"),
            reverse=True,
        )
        for gap in residual_gaps:
            if not sorted_nodes:
                break
            ranked = sorted(
                sorted_nodes,
                key=lambda node: self._score_node(node, perspective_tokens, gap),
                reverse=True,
            )
            if not ranked:
                continue
            top_node = ranked[0]
            path = self._build_graph_path(top_node.id)
            hints.append(
                {
                    "summary": self._compose_summary(top_node, gap, path),
                    "applicability": ",".join(top_node.perspective[:2] or ["general"]),
                    "evidence_policy": top_node.attributes.get("evidence_policy", []),
                    "graph_path": path,
                }
            )
        return hints

    def consolidate(self, wm_snapshot: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate 操作：融合相似节点并返回统计信息。"""
        ingest_stats = self.ingest_wm_digest(wm_snapshot, perspective="episode")
        merged = self._merge_similar_nodes()
        return {
            "promoted_count": ingest_stats["added"] + ingest_stats["updated"],
            "merged_pairs": merged,
        }

    def decompose(self, wm_snapshot: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decompose 操作：将高 fuzziness 节点拆分为情境化子节点。"""
        decomposed = 0
        for digest in wm_snapshot:
            node = self._get_node_by_label(digest.get("label") or digest.get("kind") or "")
            if not node:
                continue
            if node.fuzziness < 0.6 or len(node.perspective) <= 1:
                continue
            for perspective in node.perspective:
                child_label = f"{node.label}:{perspective}"
                child_node, created = self._upsert_node(
                    label=child_label,
                    kinds=node.kinds,
                    perspective=perspective,
                    clip_refs=node.evidence[-2:],
                    attributes=node.attributes,
                )
                child_node.fuzziness = max(0.2, node.fuzziness - 0.2)
                child_node.graduality = min(1.0, node.graduality + 0.1)
                decomposed += 1 if created else 0
            node.fuzziness = max(0.2, node.fuzziness - 0.3)
        return {"decomposed_count": decomposed}

    def mutate(self, action: str, payload: Dict[str, Any]) -> None:
        """统一的 LTM 改写入口，方便外部触发 Add/Remove 等操作。"""
        if action == "add":
            self.ingest_wm_digest(
                payload.get("snapshot", []),
                perspective=payload.get("perspective", "general"),
                clip_refs=payload.get("clip_refs", []),
                goal_context=payload.get("goal", ""),
            )
        elif action == "remove":
            policy = payload.get("policy", {})
            self.remove_stale(
                min_centrality=policy.get("min_centrality", 0.25),
                max_idle=policy.get("max_idle", 900.0),
            )
        elif action == "consolidate":
            self.consolidate(payload.get("snapshot", []))
        elif action == "decompose":
            self.decompose(payload.get("snapshot", []))
        else:
            logger.debug(f"[LTMService] mutate action={action} payload={payload}")

    # -- persistence ------------------------------------------------------

    def dump_state(self) -> Dict[str, Any]:
        """导出当前 LTM 节点/边，方便持久化到磁盘。"""
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        """从持久化字典恢复 LTM 状态。"""
        if not payload:
            return
        self.nodes.clear()
        self.edges.clear()
        self.node_index.clear()
        self.attribute_label_index.clear()
        self.attribute_object_index.clear()
        for node_data in payload.get("nodes", []):
            try:
                node = LTNode(**node_data)
            except TypeError:
                continue
            self.nodes[node.id] = node
            self.node_index[self._normalize_label(node.label)] = node.id
        for edge_data in payload.get("edges", []):
            try:
                edge = LTEdge(**edge_data)
            except TypeError:
                continue
            self.edges[edge.id] = edge
        self._rebuild_attribute_indexes()

    def _rebuild_attribute_indexes(self) -> None:
        for node in self.nodes.values():
            if "attribute" in node.kinds:
                self._track_attribute_node(node)
        for edge in self.edges.values():
            self._register_attribute_edge(edge.source, edge.target, edge.relation)


# ---------------------------------------------------------------------------
# Semantic memory manager orchestrator
# ---------------------------------------------------------------------------


class SemanticMemoryManager:
    """语义内存总控：统一入口，负责 Clip 管理 + Operator 调度 + LTM 交互。

    可选的 ``llm_client`` 能够把自然语言感知转写为结构化槽位，方便 WM 算法继续处理。
    提供 ``ltm_store_path`` 将 LTM 永久化，跨 episode 复用知识。
    """
    def __init__(
        self,
        max_nodes: int = 256,
        llm_client: Optional[Any] = None,
        ltm_store_path: Optional[str] = None,
    ):
        self.graph = SemanticWMGraph(max_nodes=max_nodes)
        self.ltm = LTMService()
        self.operator = MemoryOperatorEngine(self.graph, self.ltm)  # ✨ 传递 ltm
        self.clips: Dict[str, Clip] = {}
        self.episode_id: Optional[int] = None
        self.instruction: str = ""
        self.llm_client = llm_client
        default_store = os.path.join("outputs", "semantic_memory", "ltm_store.json")
        self.ltm_store_path = os.path.abspath(ltm_store_path or default_store)
        self.action_vocabulary: List[str] = []
        self._action_target_lookup: Dict[str, str] = {}
        self._action_variants_by_base: Dict[str, Set[str]] = defaultdict(set)
        self._planner_query_step = 0
        # ✨ 语义对齐：跟踪当前 goal 中的未知实体（待与观察映射）
        self._pending_entity_gaps: List[str] = []  # 格式: "entity:kind:label"
        self._significant_alignment_occurred = False  # ✨ 标记是否发生了重要的语义对齐
        self._alignment_history: List[Dict[str, Any]] = []  # ✨ 持久化对齐记录
        self._currently_holding: Optional[str] = None  # ✨ 跟踪当前持有的对象
        self._load_ltm_store()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, instruction: str, episode_id: Optional[int] = None) -> None:
        """在 episode 开始时重置 WM、Clip 并记录任务指令。"""
        self.graph.reset()
        self.clips.clear()
        self.episode_id = episode_id
        self.instruction = instruction
        self._pending_entity_gaps = []
        self._significant_alignment_occurred = False  # ✨ 重置对齐标志
        self._alignment_history = []  # ✨ 重置对齐历史
        self._currently_holding = None  # ✨ 重置持有状态
        # ✨ 重置待解析实体列表
        self._pending_entity_gaps = []

    def register_action_vocabulary(self, action_names: Sequence[str]) -> None:
        """接收环境动作描述，提取带编号的实体，便于感知/执行阶段复用。"""
        self.action_vocabulary = [name for name in (action_names or []) if isinstance(name, str)]
        self._action_target_lookup.clear()
        self._action_variants_by_base.clear()
        if not self.action_vocabulary:
            return
        for phrase in self.action_vocabulary:
            target = self._extract_action_target_from_phrase(phrase)
            if not target:
                continue
            canonical = target.strip()
            lowered = canonical.lower()
            self._action_target_lookup[lowered] = canonical
            base = self._normalize_base_label(canonical)
            if base:
                self._action_variants_by_base[base].add(canonical)
        self._seed_action_concepts()

    def _extract_action_target_from_phrase(self, phrase: str) -> Optional[str]:
        if not phrase:
            return None
        cleaned = phrase.strip()
        for pattern, _ in ACTION_ENTITY_PATTERNS:
            match = pattern.match(cleaned)
            if match and match.group("label"):
                return self._strip_leading_article(match.group("label"))
        return cleaned if cleaned else None

    def _strip_leading_article(self, text: str) -> str:
        if not text:
            return ""
        trimmed = text.strip()
        if trimmed.lower().startswith("the "):
            return trimmed[4:].strip()
        return trimmed

    def _normalize_base_label(self, label: str) -> str:
        canonical = self._strip_leading_article(label).lower()
        tokens = canonical.split()
        if not tokens:
            return ""
        if tokens[-1].isdigit():
            tokens = tokens[:-1]
        return " ".join(tokens)

    def _canonicalize_label(self, label: str) -> str:
        if not label:
            return label
        cleaned = self._strip_leading_article(label)
        lowered = cleaned.lower()
        if lowered in self._action_target_lookup:
            return self._action_target_lookup[lowered]
        base = self._normalize_base_label(cleaned)
        variants = self._action_variants_by_base.get(base)
        if variants:
            if len(variants) == 1:
                return next(iter(variants))
            for variant in variants:
                variant_lower = variant.lower()
                if lowered in variant_lower or variant_lower in lowered:
                    return variant
        concept_label = resolve_canonical_label(cleaned)
        if concept_label:
            return concept_label
        attribute_label = self._canonicalize_via_attributes(cleaned)
        if attribute_label:
            return attribute_label
        return cleaned

    def _canonicalize_via_attributes(self, label: str) -> Optional[str]:
        hints = self._extract_attribute_hints(label)
        if not hints:
            return None
        candidates = self.ltm.query_by_attribute_hints(hints, limit=1)
        return candidates[0] if candidates else None

    def _extract_attribute_hints(self, text: str) -> Dict[str, List[str]]:
        hints: Dict[str, List[str]] = {}
        lowered = (text or "").lower()
        if not lowered:
            return hints
        tokens = re.findall(r"[a-z]+", lowered)
        # Extract colors
        color_hits = [tok for tok in tokens if tok in COLOR_KEYWORDS]
        if color_hits:
            hints["color"] = list(dict.fromkeys(color_hits))
        # Extract sizes
        size_hits = [tok for tok in tokens if tok in SIZE_KEYWORDS]
        if size_hits:
            hints["size"] = list(dict.fromkeys(size_hits))
        # Extract features from "with X" patterns and color+noun combinations
        feature_phrases = self._extract_feature_phrases(lowered)
        if feature_phrases:
            hints["feature"] = feature_phrases
        # Extract material hints
        material_keywords = {"wood", "wooden", "metal", "plastic", "glass", "ceramic", "fabric", "rubber", "steel"}
        material_hits = [tok for tok in tokens if tok in material_keywords]
        if material_hits:
            hints["material"] = list(dict.fromkeys(material_hits))
        return hints

    def _extract_feature_phrases(self, lowered_text: str) -> List[str]:
        phrases: List[str] = []
        # Pattern 1: "with X" or "having X"
        for marker in ("with", "having"):
            split_token = f" {marker} "
            if split_token not in lowered_text:
                continue
            segments = lowered_text.split(split_token)
            for segment in segments[1:]:
                stop_idx = len(segment)
                for stopper in (" to ", " on ", " at ", " and ", ",", ".", ";", " that "):
                    pos = segment.find(stopper)
                    if pos != -1:
                        stop_idx = min(stop_idx, pos)
                phrase = segment[:stop_idx].strip()
                phrase = self._strip_leading_article(phrase)
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        # Pattern 2: color + noun combinations (e.g., "green top", "green leaves")
        color_noun_pattern = re.compile(r'\b(' + '|'.join(COLOR_KEYWORDS) + r')\s+([a-z]+)\b')
        for match in color_noun_pattern.finditer(lowered_text):
            color = match.group(1)
            noun = match.group(2)
            # Combine color+noun as a feature hint
            combined = f"{color} {noun}"
            if combined not in phrases:
                phrases.append(combined)
            # Also add just the noun if it's a meaningful feature word
            feature_nouns = {"top", "leaves", "leaf", "stem", "skin", "surface", "pattern", "texture", "stripe", "spot", "dot"}
            if noun in feature_nouns and noun not in phrases:
                phrases.append(noun)
        return phrases

    def _parse_relation_entities(self, relation_text: str) -> Tuple[Optional[str], Optional[str], str]:
        """Extract source entity, target entity, and relation type from relation description.
        
        Returns:
            (source_label, target_label, relation_type)
        """
        if not relation_text:
            return None, None, "related-to"
        text_lower = relation_text.lower().strip()
        relation_type = "related-to"
        source_label: Optional[str] = None
        target_label: Optional[str] = None
        # pattern: "A is <relation> B" or "A <relation> B"
        for marker, rel_type in [
            (" is on ", "on"),
            (" on ", "on"),
            (" above ", "on"),
            (" is above ", "on"),
            (" is inside ", "inside"),
            (" inside ", "inside"),
            (" in ", "inside"),
            (" is in ", "inside"),
            (" to the left of ", "adjacent"),
            (" to the right of ", "adjacent"),
            (" left of ", "adjacent"),
            (" right of ", "adjacent"),
            (" near ", "adjacent"),
            (" beside ", "adjacent"),
        ]:
            if marker in text_lower:
                parts = text_lower.split(marker, 1)
                if len(parts) == 2:
                    source_label = self._strip_leading_article(parts[0].strip())
                    target_label = self._strip_leading_article(parts[1].strip())
                    relation_type = rel_type
                    break
        # fallback: first noun phrase as source, last as target
        if not source_label or not target_label:
            tokens = text_lower.split()
            if len(tokens) >= 2:
                source_label = tokens[0]
                target_label = tokens[-1]
        return source_label, target_label, relation_type

    def _extract_entities_from_action(self, action_desc: str) -> List[Dict[str, str]]:
        if not action_desc:
            return []
        entities: List[Dict[str, str]] = []
        text = action_desc.strip()
        for pattern, kind in ACTION_ENTITY_PATTERNS:
            match = pattern.match(text)
            if match and match.group("label"):
                label = self._strip_leading_article(match.group("label"))
                if label:
                    entities.append({"label": label, "kind": kind})
                break
        return entities

    def _update_holding_state_after_pick(
        self,
        action_desc: str,
        env_info: Dict[str, Any],
        env_step: int,
    ) -> None:
        """
        When a pick up action succeeds, track which object is being held.
        
        Example action_desc: "pick up the spatula"
        Extract: object_label = "spatula"
        """
        import re
        
        # Extract object label from action description
        # Pattern: "pick up [the] <object>"
        match = re.search(r'pick\s+up\s+(?:the\s+)?(.+)$', action_desc.lower())
        if not match:
            logger.debug(f"[SemanticMemoryManager] Could not extract object from pick action: {action_desc}")
            return
        
        object_label = match.group(1).strip()
        object_label = self._canonicalize_label(object_label)
        
        # Verify from env_info feedback that we're holding this object
        env_feedback = env_info.get("info", {}).get("feedback", "")
        if "you are holding" in env_feedback.lower() and object_label in env_feedback.lower():
            self._currently_holding = object_label
            logger.info(
                f"[SemanticMemoryManager] 🤖 Robot is now holding: {object_label}"
            )
        else:
            # Fallback: assume the object from action description
            self._currently_holding = object_label
            logger.debug(
                f"[SemanticMemoryManager] Assuming robot is holding: {object_label} (based on action)"
            )

    def _update_object_position_after_place(
        self,
        action_desc: str,
        env_info: Dict[str, Any],
        env_step: int,
    ) -> None:
        """
        When a place action succeeds, update the object's spatial_position in WM.
        
        Example action_desc: "place at the left counter in the kitchen"
        Extract: target_location = "left counter"
        """
        import re
        
        # Extract target location from action description
        # Pattern: "place at [the] <location>"
        match = re.search(r'place\s+at\s+(?:the\s+)?(.+?)(?:\s+in\s+the\s+\w+)?$', action_desc.lower())
        if not match:
            logger.debug(f"[SemanticMemoryManager] Could not extract placement location from: {action_desc}")
            return
        
        target_location = match.group(1).strip()
        target_location = self._canonicalize_label(target_location)
        
        # Get the object that was just placed (from _currently_holding)
        if not self._currently_holding:
            logger.debug(f"[SemanticMemoryManager] No object in _currently_holding during place action")
            return
        
        held_object_label = self._currently_holding
        
        # Update the object's spatial_position
        obj_unit = self.graph.find_unit_by_label(held_object_label, kind="object")
        if obj_unit:
            old_position = obj_unit.symbol.state_vars.get("spatial_position", "unknown")
            obj_unit.symbol.state_vars["spatial_position"] = f"on {target_location}"
            obj_unit.symbol.state_vars["last_seen_position"] = f"{target_location}"
            obj_unit.metadata.recency = env_step
            
            logger.info(
                f"[SemanticMemoryManager] ✅ Updated object position: {held_object_label} "
                f"[{old_position}] → [on {target_location}]"
            )
        else:
            logger.debug(
                f"[SemanticMemoryManager] Could not find object '{held_object_label}' in WM to update position"
            )
        
        # Clear the holding state after successful place
        self._currently_holding = None

    def _inject_action_references(self, action_desc: str, env_step: int) -> None:
        entities = self._extract_entities_from_action(action_desc)
        created = 0
        for entity in entities:
            label = self._canonicalize_label(entity.get("label", ""))
            created += self._ensure_action_reference(label, entity.get("kind", "object"), env_step)
        if created:
            logger.info(
                "[SemanticMemoryManager] Action vocab injected %d entities from action '%s'",
                created,
                action_desc,
            )

    def _ensure_action_reference(self, label: str, entity_kind: str, env_step: int) -> int:
        if not label:
            return 0
        unit = self.graph.find_unit_by_label(label, kind=entity_kind)
        if unit:
            unit.metadata.recency = env_step
            unit.symbol.state_vars.setdefault("visible", False)
            unit.symbol.state_vars.setdefault("source", "action_vocab")
            return 0
        profile = self._fallback_object_profile(label)
        state_vars = {
            "visible": False,
            "source": "action_vocab",
            "last_seen_ts": time.time(),
        }
        for key in (
            "category",
            "materials",
            "affordances",
            "temperature_tolerance",
            "usage_notes",
            "size_hint",
        ):
            value = profile.get(key)
            if value:
                state_vars[key] = value
        constraints: List[Dict[str, Any]] = []
        for constraint in profile.get("constraints", []) or []:
            if isinstance(constraint, dict):
                constraints.append(constraint)
            else:
                constraints.append({"text": constraint})
        candidate = SemanticUnitCandidate(
            kind=entity_kind or "object",
            symbol=SemanticSymbol(
                type_candidates=[TypeCandidate(label=label, confidence=0.7)],
                state_vars=state_vars,
                constraints=constraints,
            ),
            metadata=SemanticMetadata(confidence=0.65, recency=env_step, ttl=120, provenance="action_vocab"),
            clip_refs=[],
        )
        self.operator.write([candidate], env_step)
        return 1

    def _seed_action_concepts(self) -> None:
        if not self.action_vocabulary:
            return
        seeded = 0
        seen: Set[str] = set()
        for canonical in self._action_target_lookup.values():
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            seeded += self._ensure_action_reference(canonical, "object", env_step=0)
        if seeded:
            self._sync_ltm_from_wm(perspective="action_vocab_seed")

    # -- clip helpers ------------------------------------------------------

    def _register_clip(self, kind: str, uri: str, digest: str, annotations: Optional[Dict[str, Any]] = None) -> str:
        """把感知/交互证据封装成 Clip，返回用于溯源的 id。"""
        clip_id = _generate_id("clip")
        clip = Clip(
            id=clip_id,
            kind=kind,
            uri=uri,
            digest=digest,
            timestamps={"first_seen": time.time(), "last_seen": time.time()},
            annotations=annotations or {},
        )
        self.clips[clip_id] = clip
        logger.info(
            "[SemanticMemoryManager] Registered clip %s kind=%s uri=%s digest='%s'",
            clip_id,
            kind,
            uri,
            (digest or "")[:120],
        )
        return clip_id

    def _sync_ltm_from_wm(
        self,
        perspective: str,
        clip_refs: Optional[Sequence[str]] = None,
    ) -> None:
        """抽取当前 WM digest 并触发 LTM Add 算子。
        
        ⚠️ 只同步长期知识（属性、关系），过滤临时状态（位置、可见性）。
        """
        snapshot = self.graph.snapshot(limit=16)
        if not snapshot:
            return
        
        # ✨ 过滤掉临时状态字段，只保留长期知识
        filtered_snapshot = self._filter_ephemeral_states(snapshot)
        
        ingest_stats = self.ltm.ingest_wm_digest(
            filtered_snapshot,
            perspective=perspective,
            clip_refs=list(clip_refs or []),
            goal_context=self.instruction,
        )
        logger.info(
            "[SemanticMemoryManager] Synced %d WM nodes to LTM (perspective=%s, stats=%s)",
            len(filtered_snapshot),
            perspective,
            ingest_stats,
        )
        self._save_ltm_store()
    
    def _filter_ephemeral_states(self, snapshot: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤临时状态字段，只保留长期知识。
        
        临时状态（每个 episode 都会变化）：
        - spatial_position: 对象位置（每次环境重置会变）
        - last_seen_position: 最后看到的位置
        - visible: 是否可见
        - last_seen_ts: 最后看到的时间戳
        - step: 当前步数
        - recency: 最近访问时间
        
        长期知识（跨 episode 复用）：
        - category: 对象类别
        - materials: 材料
        - affordances: 功能
        - colors: 颜色
        - size_hint: 大小
        - temperature_tolerance: 温度容忍度
        - edibility: 可食用性
        - relations: 关系（is_a, synonym_of 等）
        """
        EPHEMERAL_FIELDS = {
            'spatial_position',
            'last_seen_position', 
            'visible',
            'last_seen_ts',
            'step',
            'recency',
            'source',  # 来源信息（action_vocab, perception 等）
            '_needs_ltm_sync',  # 内部标记
        }
        
        filtered = []
        for digest in snapshot:
            filtered_digest = dict(digest)
            
            # 过滤 state_vars 中的临时字段
            if 'state_vars' in filtered_digest:
                filtered_state_vars = {
                    k: v for k, v in filtered_digest['state_vars'].items()
                    if k not in EPHEMERAL_FIELDS
                }
                filtered_digest['state_vars'] = filtered_state_vars
            
            filtered.append(filtered_digest)
        
        return filtered

    def _merge_ltm_attributes_into_wm(
        self,
        labels: Sequence[str],
        env_step: int,
    ) -> None:
        """从 LTM 获取指定 label 的属性并回写 WM，保持双向同步。"""
        unique_labels = []
        for label in labels:
            if label and label not in unique_labels:
                unique_labels.append(label)
        
        # ✨ 添加所有标记为需要同步的节点
        for unit in self.graph.nodes.values():
            if unit.symbol.state_vars.get("_needs_ltm_sync"):
                label = unit.primary_label()
                if label and label not in unique_labels:
                    unique_labels.append(label)
                    logger.info(
                        f"[LTM→WM Sync] Found node '{label}' marked for LTM sync"
                    )
                # 清除标记
                del unit.symbol.state_vars["_needs_ltm_sync"]
        
        if not unique_labels:
            return
        merged = 0
        for label in unique_labels:
            snapshot = self.ltm.describe_node(label)
            if not snapshot:
                continue
            unit = self.graph.find_unit_by_label(label)
            if not unit:
                continue
            
            # 1. 合并 state_vars 属性（⚠️ 排除临时状态字段）
            attributes = snapshot.get("attributes", {}) or {}
            state_vars = attributes.get("state_vars", {})
            
            # ✨ 定义临时状态字段（不应从 LTM 恢复）
            EPHEMERAL_FIELDS = {
                'spatial_position',
                'last_seen_position', 
                'visible',
                'last_seen_ts',
                'step',
                'recency',
                'source',
                '_needs_ltm_sync',
            }
            
            for key, value in state_vars.items():
                # ✨ 跳过临时状态字段
                if key in EPHEMERAL_FIELDS:
                    continue
                # 只添加 WM 中不存在的长期属性
                if key not in unit.symbol.state_vars:
                    unit.symbol.state_vars[key] = value
            
            # 2. 应用属性关系（颜色、材料等）
            attribute_relations = snapshot.get("attribute_relations", {}) or {}
            self._apply_attribute_relations_to_unit(unit, attribute_relations)
            
            # 3. 应用约束
            constraints = attributes.get("constraints", [])
            for constraint in constraints:
                if constraint not in unit.symbol.constraints:
                    unit.symbol.constraints.append(constraint)
            
            # 4. 处理 connections（对象间的连接关系）
            suggested_relations = snapshot.get("connections", []) or []
            existing_relations = {(rel.relation, rel.target) for rel in unit.symbol.relations}
            for connection in suggested_relations:
                if connection.get("target_kind") == "attribute":
                    continue
                relation = connection.get("relation") or "association"
                target_label = connection.get("target")
                if not target_label:
                    continue
                inferred_kind = "place" if any(token in (target_label or "").lower() for token in ("room", "hall", "area")) else "object"
                target_unit = self.graph.ensure_unit(
                    kind=inferred_kind,
                    label=target_label,
                    step=env_step,
                    provenance="ltm_sync",
                )
                sig = (relation, target_unit.id)
                if sig in existing_relations:
                    continue
                unit.symbol.relations.append(
                    RelationSpec(
                        relation=relation,
                        target=target_unit.id,
                        confidence=connection.get("confidence", 0.5),
                        metadata={"source": "ltm_sync"},
                    )
                )
                existing_relations.add(sig)
            
            # 5. 📌 处理 LTM edges 中的语义关系（is_a, synonym_of, part_of 等）
            # 从 LTM 中查询所有从该节点出发的 edges
            ltm_node = self.ltm._get_node_by_label(label)
            if ltm_node:
                # 获取所有出边
                outgoing_edges = [
                    edge for edge in self.ltm.edges.values()
                    if edge.source == ltm_node.id
                ]
                
                for edge in outgoing_edges:
                    # 查找目标节点
                    target_node = self.ltm.nodes.get(edge.target)
                    if not target_node:
                        continue
                    
                    target_label = target_node.label
                    relation_type = edge.relation
                    
                    # 过滤属性类型的边（已在上面处理）
                    if "attribute" in target_node.kinds:
                        continue
                    
                    # 检查是否已存在该关系
                    sig = (relation_type, target_label)
                    if sig in {(rel.relation, rel.target) for rel in unit.symbol.relations if isinstance(rel.target, str)}:
                        continue
                    
                    # 添加语义关系到 WM（使用 target_label 而不是 target_id）
                    unit.symbol.relations.append(
                        RelationSpec(
                            relation=relation_type,
                            target=target_label,  # 直接使用 label，而不是 unit.id
                            confidence=edge.confidence,
                            metadata={"source": "ltm_semantic_relation"},
                        )
                    )
                    logger.info(
                        f"[LTM→WM] Synced semantic relation: {label} {relation_type} {target_label} "
                        f"(confidence={edge.confidence:.2f})"
                    )
            
            unit.metadata.recency = max(unit.metadata.recency, env_step)
            unit.metadata.provenance = unit.metadata.provenance or "ltm_sync"
            self.graph.utility[unit.id] = min(self.graph.utility.get(unit.id, 1.0) + 0.3, 5.0)
            merged += 1
        if merged:
            logger.info(
                "[SemanticMemoryManager] Refreshed %d WM nodes with LTM attributes",
                merged,
            )

    def _apply_attribute_relations_to_unit(
        self,
        unit: SemanticUnit,
        attribute_relations: Dict[str, Sequence[Any]],
    ) -> None:
        if not attribute_relations:
            return
        for attr_kind, values in attribute_relations.items():
            state_key = ATTRIBUTE_KIND_PRIMARY_KEY.get(attr_kind, attr_kind)
            if not state_key or state_key in unit.symbol.state_vars:
                continue
            normalized = self._normalize_attribute_values(values, attr_kind)
            if normalized is None:
                continue
            unit.symbol.state_vars[state_key] = normalized

    def _normalize_attribute_values(self, values: Sequence[Any], attr_kind: str) -> Optional[Any]:
        cleaned: List[str] = []
        for value in values or []:
            if value in (None, ""):
                continue
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    if item in (None, ""):
                        continue
                    text = str(item).strip()
                    if text and text not in cleaned:
                        cleaned.append(text)
                continue
            text = str(value).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            return None
        allow_list = ATTRIBUTE_KIND_ALLOW_LIST.get(attr_kind, False)
        if allow_list or len(cleaned) > 1:
            return cleaned
        return cleaned[0]

    def _apply_attribute_hints_to_wm(
        self,
        hints: Sequence[Dict[str, Any]],
        step: int,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """应用 LTM hints 到 WM，支持属性 hints 和实体 hints。"""
        if not hints:
            return False, []
        passthrough: List[Dict[str, Any]] = []
        updated_units: Set[str] = set()
        
        for hint in hints:
            # ✨ 处理实体 hint（entity:object:spatula）
            entity_kind = hint.get("entity_kind")
            entity_label = hint.get("entity_label")
            if entity_kind and entity_label:
                # 检查 WM 中是否已存在该实体
                existing_unit = self.graph.find_unit_by_label(entity_label, kind=entity_kind)
                if existing_unit:
                    # 已存在，更新其属性
                    self._update_unit_from_entity_hint(existing_unit, hint, step)
                    updated_units.add(existing_unit.id)
                else:
                    # 不存在，创建新节点
                    new_unit = self._create_unit_from_entity_hint(hint, step)
                    if new_unit:
                        self.graph.add_unit(new_unit, step)
                        updated_units.add(new_unit.id)
                        logger.info(
                            "[SemanticMemoryManager] Created WM node from LTM entity hint: %s (%s)",
                            entity_label,
                            entity_kind,
                        )
                continue
            
            # 处理属性 hint（attribute:color:spatula）
            attr_kind = hint.get("attribute_kind")
            label = hint.get("attribute_target")
            values = hint.get("attribute_values")
            if not attr_kind or not label or not values:
                passthrough.append(hint)
                continue
            
            unit = self.graph.find_unit_by_label(label)
            if not unit:
                passthrough.append(hint)
                continue
            
            state_key = ATTRIBUTE_KIND_PRIMARY_KEY.get(attr_kind, attr_kind)
            normalized = self._normalize_attribute_values(values, attr_kind)
            if normalized is None:
                passthrough.append(hint)
                continue
            
            if unit.symbol.state_vars.get(state_key) == normalized:
                continue
            
            unit.symbol.state_vars[state_key] = normalized
            unit.metadata.recency = max(unit.metadata.recency, step)
            updated_units.add(unit.id)
        
        if updated_units:
            logger.info(
                "[SemanticMemoryManager] Applied %d hints to WM (%d units updated)",
                len(hints),
                len(updated_units),
            )
        return bool(updated_units), passthrough

    def _create_unit_from_entity_hint(
        self,
        hint: Dict[str, Any],
        step: int,
    ) -> Optional[SemanticUnit]:
        """从 LTM entity hint 创建新的 WM 节点。"""
        entity_label = hint.get("entity_label")
        entity_kind = hint.get("entity_kind")
        if not entity_label or not entity_kind:
            return None
        
        # 构建 state_vars
        state_vars: Dict[str, Any] = {
            "source": "ltm_entity_hint",
            "visible": False,  # 标记为不可见（需要探索才能确认）
        }
        
        # 从 hint 中提取位置
        position = hint.get("position")
        if position and position != "unknown":
            state_vars["spatial_position"] = position
            state_vars["last_seen_position"] = position
        
        # 从 hint 中提取属性
        attributes = hint.get("attributes", {})
        if isinstance(attributes, dict):
            for attr_key, attr_values in attributes.items():
                if attr_key == "position":
                    continue  # 已处理
                state_key = ATTRIBUTE_KIND_PRIMARY_KEY.get(attr_key, attr_key)
                normalized = self._normalize_attribute_values(attr_values, attr_key)
                if normalized is not None:
                    state_vars[state_key] = normalized
        
        # 创建 SemanticUnit
        clip_id = self._register_clip(
            kind="ltm_hint",
            uri=f"ltm://entity/{entity_label}",
            digest=hint.get("summary", f"LTM hint for {entity_label}"),
        )
        
        unit = SemanticUnit(
            id=f"{entity_kind}_{hashlib.sha1(entity_label.encode()).hexdigest()[:12]}",
            kind=entity_kind,
            symbol=SemanticSymbol(
                type_candidates=[TypeCandidate(label=entity_label, confidence=0.7)],
                state_vars=state_vars,
            ),
            metadata=SemanticMetadata(
                confidence=0.7,
                recency=step,
                ttl=120,
                provenance="ltm_entity_hint",
            ),
            clip_refs=[clip_id],
        )
        
        return unit

    def _update_unit_from_entity_hint(
        self,
        unit: SemanticUnit,
        hint: Dict[str, Any],
        step: int,
    ) -> None:
        """用 LTM entity hint 更新已存在的 WM 节点。"""
        # 更新位置信息
        position = hint.get("position")
        if position and position != "unknown":
            if not unit.symbol.state_vars.get("spatial_position"):
                unit.symbol.state_vars["spatial_position"] = position
            if not unit.symbol.state_vars.get("last_seen_position"):
                unit.symbol.state_vars["last_seen_position"] = position
        
        # 更新属性
        attributes = hint.get("attributes", {})
        if isinstance(attributes, dict):
            for attr_key, attr_values in attributes.items():
                if attr_key == "position":
                    continue
                state_key = ATTRIBUTE_KIND_PRIMARY_KEY.get(attr_key, attr_key)
                if not unit.symbol.state_vars.get(state_key):
                    normalized = self._normalize_attribute_values(attr_values, attr_key)
                    if normalized is not None:
                        unit.symbol.state_vars[state_key] = normalized
        
        # 更新元数据
        unit.metadata.recency = max(unit.metadata.recency, step)
        if hint.get("summary"):
            clip_id = self._register_clip(
                kind="ltm_hint",
                uri=f"ltm://entity/{unit.primary_label()}",
                digest=hint["summary"],
            )
            if clip_id not in unit.clip_refs:
                unit.clip_refs.append(clip_id)

    def _ensure_store_dir(self) -> None:
        if not self.ltm_store_path:
            return
        store_dir = os.path.dirname(self.ltm_store_path)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)

    def _save_ltm_store(self) -> None:
        if not self.ltm_store_path:
            return
        try:
            self._ensure_store_dir()
            payload = self.ltm.dump_state()
            with open(self.ltm_store_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            logger.info(
                "[SemanticMemoryManager] Saved LTM store to %s (nodes=%d, edges=%d)",
                self.ltm_store_path,
                len(self.ltm.nodes),
                len(self.ltm.edges),
            )
        except OSError as exc:
            logger.warning(f"[SemanticMemoryManager] Failed to persist LTM: {exc}")

    def _load_ltm_store(self) -> None:
        payload: Optional[Dict[str, Any]] = None
        if self.ltm_store_path and os.path.exists(self.ltm_store_path):
            try:
                with open(self.ltm_store_path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"[SemanticMemoryManager] Failed to load LTM store: {exc}")
        if payload:
            self.ltm.load_state(payload)
            logger.info(
                "[SemanticMemoryManager] Loaded LTM store from %s (nodes=%d, edges=%d)",
                self.ltm_store_path,
                len(self.ltm.nodes),
                len(self.ltm.edges),
            )
            return
        seed_payload = build_ltm_seed_graph(perspective="seed")
        if not seed_payload.get("nodes"):
            logger.warning("[SemanticMemoryManager] No LTM seed graph available")
            return
        self.ltm.load_state(seed_payload)
        logger.info(
            "[SemanticMemoryManager] Bootstrapped LTM with seed graph (nodes=%d, edges=%d)",
            len(self.ltm.nodes),
            len(self.ltm.edges),
        )
        if self.ltm_store_path:
            self._save_ltm_store()

    # -- perception helpers ----------------------------------------------

    def _normalize_perception_input(
        self,
        perception: Any,
        instruction: str,
    ) -> Dict[str, Any]:
        """将自然语言/字典/原对象统一为结构化字段。"""
        raw_text = ""
        structured: Dict[str, Any]
        if isinstance(perception, str):
            raw_text = perception
            structured = self._extract_semantics_with_llm(perception, instruction)
        elif isinstance(perception, dict):
            structured = dict(perception)
            raw_text = str(
                structured.get("raw_text")
                or structured.get("scene_summary")
                or structured.get("description")
                or ""
            )
        else:
            structured = {
                "scene_summary": getattr(perception, "scene_summary", None),
                "detected_objects": getattr(perception, "detected_objects", None),
                "objects_relations": getattr(perception, "objects_relations", None),
                "state_changes": getattr(perception, "state_changes", None),
                "environment_snapshot": getattr(perception, "environment_snapshot", None),
                "raw_text": getattr(perception, "raw_text", None),
            }
            raw_text = str(
                structured.get("raw_text")
                or structured.get("scene_summary")
                or getattr(perception, "description", "")
            )
        normalized = {
            "scene_summary": structured.get("scene_summary") or raw_text.strip()[:200],
            "detected_objects": self._coerce_to_list(structured.get("detected_objects")),
            "objects_relations": self._coerce_to_list(structured.get("objects_relations")),
            "state_changes": self._coerce_to_list(structured.get("state_changes")),
            "environment_snapshot": structured.get("environment_snapshot") or {},
            "raw_text": raw_text or str(perception),
        }
        self._enrich_object_profiles(normalized, instruction)
        self._enrich_space_profiles(normalized, instruction)
        return normalized

    def _enrich_object_profiles(self, normalized: Dict[str, Any], instruction: str) -> None:
        detected = normalized.get("detected_objects") or []
        structured_objects: List[Dict[str, Any]] = []
        labels: List[str] = []
        for entry in detected:
            if isinstance(entry, dict):
                obj = dict(entry)
                label = obj.get("label") or obj.get("name") or obj.get("id") or obj.get("text")
            else:
                label = str(entry).strip()
                obj = {"label": label}
            if not label:
                continue
            canonical_label = self._canonicalize_label(label)
            obj["label"] = canonical_label
            structured_objects.append(obj)
            labels.append(canonical_label)
        if not structured_objects:
            normalized["detected_objects"] = []
            return
        profiles = self._query_object_profiles(labels, instruction, normalized.get("scene_summary", ""))
        for obj in structured_objects:
            label = obj["label"]
            profile = profiles.get(label) or profiles.get(label.lower())
            if not profile:
                profile = self._fallback_object_profile(label)
            obj.update({k: v for k, v in profile.items() if v not in (None, "")})
        normalized["detected_objects"] = structured_objects

    def _query_object_profiles(
        self,
        labels: Sequence[str],
        instruction: str,
        scene_summary: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not labels or not self.llm_client:
            return {}
        object_list = "\n".join(f"- {label}" for label in labels)
        prompt = (
            "You are an embodied-scene ontology builder. Given the task instruction,"
            " scene summary, and a list of salient objects, respond with JSON describing each object.\n"
            "For every object, include: label, category, typical_materials, affordances (list),"
            " size_hint, temperature_tolerance (hot/ambient/cold), usage_notes,"
            " placement_constraints (list), and suggested_actions (list of verbs)."
            " Use concise natural language phrases."
            f"\nInstruction: {instruction}\nScene summary: {scene_summary}\nObjects:\n{object_list}\n"
            "Return JSON of the form {\"objects\": [ ... ]}."
        )
        response = self._attempt_llm_completion(prompt)
        parsed = self._safe_json_parse(response)
        entries: List[Dict[str, Any]]
        if isinstance(parsed, dict):
            entries = parsed.get("objects") or parsed.get("items") or []
        elif isinstance(parsed, list):
            entries = parsed
        else:
            entries = []
        profiles: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or entry.get("name") or entry.get("id")
            if not label:
                continue
            key = label
            profiles[key] = {
                "category": entry.get("category"),
                "materials": entry.get("typical_materials") or entry.get("materials"),
                "affordances": entry.get("affordances"),
                "size_hint": entry.get("size_hint"),
                "temperature_tolerance": entry.get("temperature_tolerance"),
                "usage_notes": entry.get("usage_notes"),
                "constraints": entry.get("placement_constraints"),
                "suggested_actions": entry.get("suggested_actions"),
            }
            profiles[key.lower()] = profiles[key]
        return profiles

    def _fallback_object_profile(self, label: str) -> Dict[str, Any]:
        concept = get_concept_definition(label)
        if concept:
            attributes = concept.get("attributes", {}) or {}
            state_vars = attributes.get("state_vars", {}) or {}
            constraints = attributes.get("constraints", []) or []
            supports_hot = state_vars.get("supports_hot_items")
            temperature = state_vars.get("temperature_tolerance")
            if not temperature:
                if supports_hot:
                    temperature = "hot"
                elif state_vars.get("category") in {"appliance", "storage"}:
                    temperature = "ambient"
                else:
                    temperature = "ambient"
            return {
                "category": state_vars.get("category", "object"),
                "materials": state_vars.get("materials", []),
                "affordances": state_vars.get("affordances", []),
                "temperature_tolerance": temperature,
                "constraints": constraints,
                "usage_notes": state_vars.get("usage_notes", ""),
                "size_hint": state_vars.get("size_hint", "medium"),
                "suggested_actions": concept.get("suggested_actions", []),
            }

        lowered = (label or "object").lower()
        category = "object"
        materials = []
        affordances = []
        temperature = "ambient"
        constraints: List[str] = []
        usage_notes = ""
        size_hint = "medium"
        if any(token in lowered for token in ("shelf", "table", "desk", "counter")):
            category = "surface"
            materials = ["wood", "composite"]
            affordances = ["support_objects", "place_items"]
            temperature = "moderate"
            constraints.append("keep surface clear before placing fragile items")
            size_hint = "large"
        elif any(token in lowered for token in ("cup", "mug", "bowl", "container")):
            category = "container"
            materials = ["ceramic", "plastic"]
            affordances = ["hold_liquid", "carry_small_items"]
            temperature = "hot" if "mug" in lowered else "ambient"
            usage_notes = "Check if already filled before reusing."
        elif any(token in lowered for token in ("couch", "sofa", "chair", "bench")):
            category = "seating"
            materials = ["fabric", "wood"]
            affordances = ["sit", "place_soft_items"]
            constraints.append("avoid placing hot objects directly on fabric")
        elif "container" in lowered or "box" in lowered:
            category = "storage"
            affordances = ["store_items", "transport"]
            size_hint = "varies"
        return {
            "category": category,
            "materials": materials,
            "affordances": affordances,
            "temperature_tolerance": temperature,
            "constraints": constraints,
            "usage_notes": usage_notes,
            "size_hint": size_hint,
            "suggested_actions": [],
        }

    def _enrich_space_profiles(self, normalized: Dict[str, Any], instruction: str) -> None:
        summary = normalized.get("scene_summary") or normalized.get("raw_text") or ""
        relations = " ; ".join(str(rel) for rel in normalized.get("objects_relations", []))
        profile = self._query_space_profile(summary, relations, instruction)
        if not profile:
            profile = self._fallback_space_profile(summary)
        normalized["space_semantics"] = profile

    def _query_space_profile(self, summary: str, relations: str, instruction: str) -> Optional[Dict[str, Any]]:
        if not summary or not self.llm_client:
            return None
        prompt = (
            "You are a spatial reasoning assistant for a household robot."
            " Given the room summary, key relations, and task instruction, describe the room's"
            " navigability, reachable surfaces, connectivity to other spaces, potential obstacles,"
            " and surface/material properties. Respond with JSON: {\"layout\": str, \"connectivity\": [str],"
            " \"navigability\": str, \"surface_materials\": [str], \"reachable_surfaces\": [str],"
            " \"supports_hot_items\": bool, \"size_hint\": str, \"obstacles\": [str]}."
            f"\nInstruction: {instruction}\nRoom summary: {summary}\nRelations: {relations}\nJSON:"
        )
        response = self._attempt_llm_completion(prompt)
        parsed = self._safe_json_parse(response)
        return parsed if isinstance(parsed, dict) else None

    def _fallback_space_profile(self, summary: str) -> Dict[str, Any]:
        lowered = (summary or "room").lower()
        materials = ["wood"] if "floor" in lowered else ["mixed"]
        connectivity = []
        if "kitchen" in lowered:
            connectivity.append("dining room")
        if "living" in lowered or "couch" in lowered:
            connectivity.append("hallway")
        return {
            "layout": summary or "room",
            "connectivity": connectivity,
            "navigability": "open" if "open" in lowered or "spacious" in lowered else "mixed",
            "surface_materials": materials,
            "reachable_surfaces": ["tables", "shelves"] if "shelf" in lowered else ["general surfaces"],
            "supports_hot_items": "table" in lowered or "counter" in lowered,
            "size_hint": "large" if "large" in lowered else "medium",
            "obstacles": ["furniture cluster" if "crowded" in lowered else "none"],
        }

    def _extract_semantics_with_llm(self, text: str, instruction: str) -> Dict[str, Any]:
        """调用 LLM 将自然语言感知描述解析为结构化槽位。"""
        if not text:
            return self._fallback_perception_from_text(text)
        payload: Optional[Dict[str, Any]] = None
        if self.llm_client:
            prompt = (
                "You are a structured perception extractor for an embodied agent. "
                "Given the task instruction and the latest observation summary, "
                "return JSON with keys scene_summary (string), detected_objects (list of objects), "
                "objects_relations (list of relation descriptions or dicts), state_changes (list), "
                "environment_snapshot (JSON friendly dict)."
                " Focus on nouns and physical relations."
                f"\nInstruction: {instruction}\nPerception: {text}\nJSON:"
            )
            response = self._attempt_llm_completion(prompt)
            payload = self._safe_json_parse(response)
        if not payload:
            payload = self._fallback_perception_from_text(text)
        payload.setdefault("raw_text", text)
        return payload

    def _attempt_llm_completion(self, prompt: str) -> str:
        if not self.llm_client:
            return ""
        try:
            if hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt)
            if hasattr(self.llm_client, "complete"):
                return self.llm_client.complete(prompt)
            if callable(self.llm_client):
                return self.llm_client(prompt)
        except Exception as exc:  # pragma: no cover - 网络/权限异常无法稳定复现
            logger.warning(f"[SemanticMemoryManager] LLM extraction failed: {exc}")
        return ""

    def _safe_json_parse(self, payload: Optional[str]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        trimmed = payload.strip()
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            start = trimmed.find("{")
            end = trimmed.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(trimmed[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _fallback_perception_from_text(self, text: str) -> Dict[str, Any]:
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", text or "")
        dedup_objects = list(dict.fromkeys(tokens))[:8]
        sentences = [s.strip() for s in re.split(r"[.!?]", text or "") if s.strip()]
        relation_keywords = (" on ", " above ", " inside ", " next to ", " near ")
        relations = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in relation_keywords)
        ][:6]
        change_keywords = (" now ", " change", " became", " becomes")
        state_changes = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in change_keywords)
        ][:4]
        summary = sentences[0] if sentences else (text.strip()[:200] if text else "")
        return {
            "scene_summary": summary,
            "detected_objects": dedup_objects,
            "objects_relations": relations,
            "state_changes": state_changes,
            "environment_snapshot": {},
            "raw_text": text,
        }

    def _coerce_to_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if item]
        if isinstance(value, str):
            return [
                token.strip()
                for token in re.split(r"[,\n;]", value)
                if token and token.strip()
            ]
        return []

    def _normalize_feedback_input(
        self,
        action_desc: str,
        env_info: Any,
        instruction: str,
        reward: float,
    ) -> Dict[str, Any]:
        """统一执行反馈：先尝试调用 LLM 抽取，再融合原始结构化字段。"""
        raw_text = ""
        structured: Dict[str, Any]
        llm_payload: Dict[str, Any] = {}
        if isinstance(env_info, str):
            raw_text = env_info
            llm_payload = self._extract_feedback_with_llm(env_info, action_desc, instruction, reward)
            structured = dict(llm_payload)
        elif isinstance(env_info, dict):
            structured = dict(env_info)
            raw_text = str(
                structured.get("raw_text")
                or structured.get("env_feedback")
                or structured.get("description")
                or ""
            )
            if raw_text:
                llm_payload = self._extract_feedback_with_llm(raw_text, action_desc, instruction, reward)
        else:
            structured = {
                "env_feedback": getattr(env_info, "env_feedback", None),
                "target_object": getattr(env_info, "target_object", None),
                "was_prev_action_invalid": getattr(env_info, "was_prev_action_invalid", None),
                "invalid_ratio": getattr(env_info, "invalid_ratio", None),
                "raw_text": getattr(env_info, "raw_text", None),
            }
            raw_text = str(
                structured.get("raw_text")
                or structured.get("env_feedback")
                or getattr(env_info, "description", "")
            )
            if raw_text:
                llm_payload = self._extract_feedback_with_llm(raw_text, action_desc, instruction, reward)
        merged: Dict[str, Any] = {}
        if llm_payload:
            merged.update(llm_payload)
        merged.update(structured)
        invalid = self._is_truthy(
            merged.get("invalid")
            or merged.get("was_prev_action_invalid")
            or merged.get("invalid_action")
        )
        target = (
            merged.get("target")
            or merged.get("target_object")
            or merged.get("object")
            or merged.get("focus_object")
            or ""
        )
        summary = (
            merged.get("summary")
            or merged.get("env_feedback")
            or raw_text
            or f"Feedback for {action_desc}"
        )
        risk_notes = self._coerce_to_list(merged.get("risk_notes"))
        state_vars = merged.get("state_vars") if isinstance(merged.get("state_vars"), dict) else {}
        constraints = merged.get("constraints") if isinstance(merged.get("constraints"), list) else []
        if "invalid_ratio" in merged and merged["invalid_ratio"] is not None:
            try:
                state_vars.setdefault("invalid_ratio_hint", float(merged["invalid_ratio"]))
            except (TypeError, ValueError):
                pass
        state_vars.setdefault("reward", reward)
        normalized = {
            "summary": summary,
            "invalid": invalid,
            "target": target,
            "risk_notes": risk_notes,
            "state_vars": state_vars,
            "constraints": constraints,
            "env_info": structured if isinstance(structured, dict) else {},
            "raw_text": raw_text or summary,
        }
        return normalized

    def _extract_feedback_with_llm(
        self,
        text: str,
        action_desc: str,
        instruction: str,
        reward: float,
    ) -> Dict[str, Any]:
        if not text:
            return self._fallback_feedback_from_text(text)
        payload: Optional[Dict[str, Any]] = None
        if self.llm_client:
            prompt = (
                "You are an execution-feedback parser for an embodied agent. "
                "Given the task instruction, action description, reward, and textual environment commentary, "
                "produce JSON with keys summary (string), invalid (bool), target (string), risk_notes (list of strings), "
                "constraints (list of dict), state_vars (dict), env_feedback (string). "
                "Summaries should be concise but informative, and invalid should be true when the action failed or was blocked."
                f"\nInstruction: {instruction}\nAction: {action_desc}\nReward: {reward}\nFeedback: {text}\nJSON:"
            )
            response = self._attempt_llm_completion(prompt)
            payload = self._safe_json_parse(response)
        if not payload:
            payload = self._fallback_feedback_from_text(text)
        payload.setdefault("raw_text", text)
        return payload

    def _fallback_feedback_from_text(self, text: str) -> Dict[str, Any]:
        lowered = (text or "").lower()
        invalid = any(keyword in lowered for keyword in ("invalid", "fail", "blocked", "unable"))
        sentences = [s.strip() for s in re.split(r"[.!?]", text or "") if s.strip()]
        summary = sentences[0] if sentences else (text.strip()[:160] if text else "")
        risk_notes = [sentence for sentence in sentences if "risk" in sentence.lower() or "hazard" in sentence.lower()]
        target_match = None
        target_pattern = re.search(r"(?:object|target|on)\s+([A-Za-z0-9_\-]+)", lowered)
        if target_pattern:
            target_match = target_pattern.group(1)
        else:
            tokens = re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", text or "")
            target_match = tokens[0] if tokens else ""
        return {
            "summary": summary,
            "invalid": invalid,
            "target": target_match or "",
            "risk_notes": risk_notes,
            "constraints": [],
            "state_vars": {},
            "env_feedback": text,
        }

    def _is_truthy(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0.0
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {"true", "yes", "1", "invalid", "failed"}
        return False

    # -- ingestion paths ---------------------------------------------------

    def ingest_perception(
        self,
        perception: Any,
        instruction: str,
        img_path: str,
        env_step: int,
    ) -> None:
        """摄取感知结果：支持自然语言输入，先经 LLM 抽取得到结构化槽位再写入 WM。"""
        perception_bundle = self._normalize_perception_input(perception, instruction)
        digest = perception_bundle.get("scene_summary") or perception_bundle.get("raw_text", "")
        clip_id = self._register_clip(
            kind="observation",
            uri=img_path,
            digest=digest,
            annotations=perception_bundle.get("environment_snapshot", {}),
        )
        candidates: List[SemanticUnitCandidate] = []
        object_count = 0
        relation_count = 0
        rule_count = 0
        place_added = False
        object_labels: List[str] = []
        relation_samples: List[str] = []
        rule_samples: List[str] = []
        timestamp = time.time()
        # objects
        # 逐个检测到的 objects 生成 object kind 节点，记录可视状态与时间戳
        for obj_entry in perception_bundle.get("detected_objects", []):
            if isinstance(obj_entry, dict):
                obj_data = dict(obj_entry)
            else:
                obj_data = {"label": str(obj_entry)}
            obj_name = (
                obj_data.get("label")
                or obj_data.get("name")
                or obj_data.get("id")
                or "object"
            )
            obj_conf = float(obj_data.get("confidence", 0.65))
            extra_state = obj_data.get("state_vars") or obj_data.get("attributes") or {}
            extra_state = extra_state if isinstance(extra_state, dict) else {}
            deterministic_key = obj_data.get("canonical_id") or obj_data.get("unit_id")
            if not obj_name:
                continue
            # Canonicalize object label to ensure consistent WM references
            canonical_obj_name = self._canonicalize_label(obj_name)
            state_vars = {"visible": True, "last_seen_ts": timestamp}
            state_vars.update(extra_state)
            # Standard properties from LTM or enrichment
            if obj_data.get("category"):
                state_vars.setdefault("category", obj_data["category"])
            if obj_data.get("materials"):
                state_vars.setdefault("materials", obj_data["materials"])
            if obj_data.get("affordances"):
                state_vars.setdefault("affordances", obj_data["affordances"])
            if obj_data.get("usage_notes"):
                state_vars.setdefault("usage_notes", obj_data["usage_notes"])
            if obj_data.get("temperature_tolerance"):
                state_vars.setdefault("temperature_tolerance", obj_data["temperature_tolerance"])
            # Visual attributes from perception (new fields)
            if obj_data.get("color"):
                state_vars.setdefault("colors", obj_data["color"] if isinstance(obj_data["color"], list) else [obj_data["color"]])
            if obj_data.get("size"):
                state_vars.setdefault("size_hint", obj_data["size"])
            elif obj_data.get("size_hint"):
                state_vars.setdefault("size_hint", obj_data["size_hint"])
            if obj_data.get("material"):
                mat = obj_data["material"]
                state_vars.setdefault("materials", mat if isinstance(mat, list) else [mat])
            if obj_data.get("shape"):
                state_vars.setdefault("shape_hint", obj_data["shape"])
            if obj_data.get("texture"):
                state_vars.setdefault("texture", obj_data["texture"])
            if obj_data.get("features"):
                feat = obj_data["features"]
                state_vars.setdefault("top_features", feat if isinstance(feat, list) else [feat])
            if obj_data.get("position"):
                state_vars.setdefault("spatial_position", obj_data["position"])
            obj_constraints = []
            for constraint in obj_data.get("constraints") or []:
                if isinstance(constraint, str):
                    obj_constraints.append({"text": constraint})
                elif isinstance(constraint, dict):
                    obj_constraints.append(constraint)
            symbol = SemanticSymbol(
                type_candidates=[TypeCandidate(label=canonical_obj_name, confidence=obj_conf)],
                state_vars=state_vars,
                constraints=obj_constraints,
            )
            metadata = SemanticMetadata(
                confidence=obj_conf,
                recency=env_step,
                ttl=80,
                provenance="perception",
            )
            candidates.append(
                SemanticUnitCandidate(
                    kind="object",
                    symbol=symbol,
                    metadata=metadata,
                    clip_refs=[clip_id],
                    key=deterministic_key,
                )
            )
            object_count += 1
            if symbol.type_candidates:
                object_labels.append(symbol.type_candidates[0].label)
        # spatial summary as place node
        if perception_bundle.get("scene_summary"):
            # place 节点用于概括场景，可帮助 navigation planner 形成上下文
            space_state = {"occupancy": "unknown", "last_seen_ts": timestamp}
            space_attrs = perception_bundle.get("space_semantics") or {}
            if isinstance(space_attrs, dict):
                for key, value in space_attrs.items():
                    space_state[key] = value
            place_symbol = SemanticSymbol(
                type_candidates=[TypeCandidate(label=perception_bundle["scene_summary"], confidence=0.7)],
                state_vars=space_state,
            )
            candidates.append(
                SemanticUnitCandidate(
                    kind="place",
                    symbol=place_symbol,
                    metadata=SemanticMetadata(confidence=0.7, recency=env_step, ttl=120, provenance="perception"),
                    clip_refs=[clip_id],
                )
            )
            place_added = True
        # object relations -> link semantics
        for relation_entry in perception_bundle.get("objects_relations", []):
            if isinstance(relation_entry, dict):
                relation_text = relation_entry.get("text") or relation_entry.get("label") or "relation"
                relation_type = relation_entry.get("relation") or relation_entry.get("type") or "related-to"
                relation_conf = float(relation_entry.get("confidence", 0.6))
                target_candidate = (
                    relation_entry.get("target")
                    or relation_entry.get("object_b")
                    or relation_entry.get("destination")
                )
                source_candidate = relation_entry.get("source") or relation_entry.get("object_a")
                state_hint = relation_entry.get("state_vars") or {}
            else:
                relation_text = str(relation_entry)
                source_candidate, target_candidate, relation_type = self._parse_relation_entities(relation_text)
                relation_conf = 0.6
                state_hint = {}
            if not relation_text:
                continue
            # Canonicalize both source and target
            if source_candidate:
                canonical_source = self._canonicalize_label(source_candidate)
            else:
                canonical_source = None
            if target_candidate:
                canonical_target = self._canonicalize_label(target_candidate)
            else:
                canonical_target = None
            # Build canonical relation description for link label
            if canonical_source and canonical_target:
                canonical_relation_text = f"{canonical_source} {relation_type} {canonical_target}"
            else:
                canonical_relation_text = relation_text
            relation_symbol = SemanticSymbol(
                type_candidates=[TypeCandidate(label=canonical_relation_text, confidence=relation_conf)],
                relations=[
                    RelationSpec(
                        relation=relation_type,
                        target=canonical_target or relation_text,
                        confidence=relation_conf,
                    )
                ],
                state_vars={
                    "passable": "unknown" if relation_type == "inside" else True,
                    **(state_hint if isinstance(state_hint, dict) else {}),
                },
            )
            candidates.append(
                SemanticUnitCandidate(
                    kind="link",
                    symbol=relation_symbol,
                    metadata=SemanticMetadata(
                        confidence=relation_conf,
                        recency=env_step,
                        ttl=60,
                        provenance="perception",
                    ),
                    clip_refs=[clip_id],
                )
            )
            relation_count += 1
            if relation_text:
                relation_samples.append(relation_text)
        # implicit rules from state_changes
        for change_entry in perception_bundle.get("state_changes", []):
            # state_changes 常对应环境规则或约束，写为 rule kind
            if isinstance(change_entry, dict):
                change_text = change_entry.get("text") or change_entry.get("label") or ""
                constraints = change_entry.get("constraints") or []
                confidence = float(change_entry.get("confidence", 0.55))
            else:
                change_text = str(change_entry)
                constraints = []
                confidence = 0.55
            if not change_text:
                continue
            rule_symbol = SemanticSymbol(
                type_candidates=[TypeCandidate(label=change_text, confidence=confidence)],
                constraints=constraints if constraints else [{"text": change_text}],
            )
            candidates.append(
                SemanticUnitCandidate(
                    kind="rule",
                    symbol=rule_symbol,
                    metadata=SemanticMetadata(
                        confidence=confidence,
                        recency=env_step,
                        ttl=70,
                        provenance="perception",
                    ),
                    clip_refs=[clip_id],
                )
            )
            rule_count += 1
            if change_text:
                rule_samples.append(change_text)
        logger.info(
            "[SemanticMemoryManager] Perception ingest clip=%s objects=%d relations=%d rules=%d place=%s sample_objects=%s",
            clip_id,
            object_count,
            relation_count,
            rule_count,
            place_added,
            object_labels[:4],
        )
        if relation_samples:
            logger.info(
                "[SemanticMemoryManager] Relation snippets: %s",
                relation_samples[:3],
            )
        if rule_samples:
            logger.info(
                "[SemanticMemoryManager] Rule snippets: %s",
                rule_samples[:3],
            )
        logs = self.operator.write(candidates, env_step)
        logger.debug(f"[SemanticMemoryManager] Perception ingestion produced {len(logs)} ops")
        
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
                # ✨ 保存对齐结果到历史记录（避免重复）
                for align in alignments:
                    existing = any(
                        h['abstract_term'] == align['abstract_term'] and
                        h['concrete_term'] == align['concrete_term']
                        for h in self._alignment_history
                    )
                    if not existing:
                        self._alignment_history.append(align)
                
                logger.info(
                    f"[SemanticMemoryManager] Semantic alignment: {len(alignments)} mappings established"
                )
                # ✨ 标记发生了重要的语义对齐（需要重新规划）
                self._significant_alignment_occurred = True
                logger.info(
                    "[SemanticMemoryManager] ⚠️ Significant alignment occurred - replan recommended"
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
        
        self.operator.integrate(env_step)
        self.operator.abstract(env_step)
        self.operator.simplify(env_step)
        
        # ⚠️ 改进：不在每个 step 同步到 LTM，只在 episode 结束时统一同步
        # self._sync_ltm_from_wm(perspective="perception", clip_refs=[clip_id])
        
        self._merge_ltm_attributes_into_wm(
            object_labels + ([perception_bundle.get("scene_summary")] if place_added else []),
            env_step,
        )

    def ingest_execution_feedback(
        self,
        action_id: int,
        action_desc: str,
        env_info: Dict[str, Any],
        reward: float,
        env_step: int,
    ) -> None:
        """摄取执行反馈：支持自然语言反馈，经 LLM 结构化后同步 phase 节点。"""
        feedback_bundle = self._normalize_feedback_input(
            action_desc=action_desc,
            env_info=env_info,
            instruction=self.instruction or "",
            reward=reward,
        )
        invalid = feedback_bundle.get("invalid", False)
        target = feedback_bundle.get("target", "")
        summary = feedback_bundle.get("summary") or (
            f"Action {action_id} {action_desc} invalid={invalid} reward={reward:.2f}"
        )
        clip_id = self._register_clip(
            kind="interaction",
            uri=f"step://{env_step}",
            digest=summary,
            annotations={
                "env_info": feedback_bundle.get("env_info", env_info),
                "risk_notes": feedback_bundle.get("risk_notes", []),
                "action": action_desc,
            },
        )
        phase_state = {
            "last_action": action_desc,
            "reward": reward,
            "step": env_step,
        }
        phase_state.update(feedback_bundle.get("state_vars", {}))
        phase_symbol = SemanticSymbol(
            type_candidates=[TypeCandidate(label="execution_phase", confidence=0.6)],
            state_vars=phase_state,
            constraints=feedback_bundle.get("constraints", []),
        )
        phase_candidate = SemanticUnitCandidate(
            kind="phase",
            symbol=phase_symbol,
            metadata=SemanticMetadata(confidence=0.6, recency=env_step, ttl=50, provenance="executor"),
            clip_refs=[clip_id],
            key=f"phase_execution",
        )
        logger.info(
            "[SemanticMemoryManager] Execution feedback ingest step=%d action=%s invalid=%s target=%s reward=%.2f",
            env_step,
            action_desc,
            invalid,
            target or "<none>",
            reward,
        )
        self.operator.write([phase_candidate], env_step)
        if target:
            self.operator.correctness(
                {"target": target, "invalid": invalid}, env_step
            )
        self.operator.simplify(env_step)
        
        # Track object holding state for pick/place actions
        if not invalid:
            # Update holding state after successful pick up action
            if "pick up" in action_desc.lower():
                self._update_holding_state_after_pick(action_desc, env_info, env_step)
            # Update object position after successful place action
            elif "place at" in action_desc.lower():
                self._update_object_position_after_place(action_desc, env_info, env_step)
        
        self._inject_action_references(action_desc, env_step)
        
        # ⚠️ 改进：不在每个 step 同步到 LTM，只在 episode 结束时统一同步
        # self._sync_ltm_from_wm(perspective="execution", clip_refs=[clip_id])
        
        refresh_labels = [target] if target else []
        self._merge_ltm_attributes_into_wm(refresh_labels, env_step)

    def ingest_step(
        self,
        perception: Any,
        instruction: str,
        img_path: str,
        action_id: int,
        action_desc: str,
        env_info: Dict[str, Any],
        reward: float,
        env_step: int,
    ) -> None:
        """
        摄取一个完整的 step：观察 + 动作执行同时处理。
        
        一个 step 包含:
        1. 观察(perception): 感知环境状态
        2. 动作(action): 执行动作并获得反馈
        
        这两者应该原子化处理,因为它们是同一个时间步的不同方面。
        """
        timestamp = time.time()
        logger.info(
            "[SemanticMemoryManager] ========== Step %d BEGIN: perception + action ==========",
            env_step
        )
        
        # ===== Part 1: Process Perception =====
        perception_bundle = self._normalize_perception_input(perception, instruction)
        perception_digest = perception_bundle.get("scene_summary") or perception_bundle.get("raw_text", "")
        perception_clip_id = self._register_clip(
            kind="observation",
            uri=img_path,
            digest=perception_digest,
            annotations=perception_bundle.get("environment_snapshot", {}),
        )
        
        candidates: List[SemanticUnitCandidate] = []
        object_count = 0
        relation_count = 0
        rule_count = 0
        place_added = False
        object_labels: List[str] = []
        
        # Process objects from perception
        for obj_entry in perception_bundle.get("detected_objects", []):
            if isinstance(obj_entry, dict):
                obj_data = dict(obj_entry)
            else:
                obj_data = {"label": str(obj_entry)}
            obj_name = (
                obj_data.get("label")
                or obj_data.get("name")
                or obj_data.get("id")
                or "object"
            )
            if not obj_name:
                continue
            
            obj_conf = float(obj_data.get("confidence", 0.65))
            canonical_obj_name = self._canonicalize_label(obj_name)
            
            extra_state = obj_data.get("state_vars") or obj_data.get("attributes") or {}
            extra_state = extra_state if isinstance(extra_state, dict) else {}
            
            state_vars = {"visible": True, "last_seen_ts": timestamp}
            state_vars.update(extra_state)
            
            # Add visual attributes
            if obj_data.get("category"):
                state_vars.setdefault("category", obj_data["category"])
            if obj_data.get("color"):
                state_vars.setdefault("colors", obj_data["color"] if isinstance(obj_data["color"], list) else [obj_data["color"]])
            if obj_data.get("size") or obj_data.get("size_hint"):
                state_vars.setdefault("size_hint", obj_data.get("size") or obj_data.get("size_hint"))
            if obj_data.get("material"):
                mat = obj_data["material"]
                state_vars.setdefault("materials", mat if isinstance(mat, list) else [mat])
            
            # Add spatial position if available
            if "spatial_position" in obj_data:
                state_vars["spatial_position"] = obj_data["spatial_position"]
            
            symbol = SemanticSymbol(
                type_candidates=[TypeCandidate(label=canonical_obj_name, confidence=obj_conf)],
                state_vars=state_vars,
            )
            
            deterministic_key = obj_data.get("canonical_id") or obj_data.get("unit_id")
            candidates.append(
                SemanticUnitCandidate(
                    kind="object",
                    symbol=symbol,
                    metadata=SemanticMetadata(
                        confidence=obj_conf,
                        recency=env_step,
                        ttl=100,
                        provenance="perception"
                    ),
                    clip_refs=[perception_clip_id],
                    key=deterministic_key,
                )
            )
            object_labels.append(canonical_obj_name)
            object_count += 1
        
        # Process relations
        for rel_entry in perception_bundle.get("spatial_relations", []):
            if not isinstance(rel_entry, dict):
                continue
            subj = rel_entry.get("subject")
            pred = rel_entry.get("predicate")
            obj = rel_entry.get("object")
            if not all([subj, pred, obj]):
                continue
            
            rel_conf = float(rel_entry.get("confidence", 0.6))
            candidates.append(
                SemanticUnitCandidate(
                    kind="link",
                    symbol=SemanticSymbol(
                        type_candidates=[TypeCandidate(label=pred, confidence=rel_conf)],
                        state_vars={"subject": subj, "object": obj},
                    ),
                    metadata=SemanticMetadata(confidence=rel_conf, recency=env_step, ttl=80, provenance="perception"),
                    clip_refs=[perception_clip_id],
                )
            )
            relation_count += 1
        
        # Process rules/constraints
        for rule_entry in perception_bundle.get("rules_or_constraints", []):
            rule_text = rule_entry if isinstance(rule_entry, str) else rule_entry.get("rule", "")
            if not rule_text:
                continue
            candidates.append(
                SemanticUnitCandidate(
                    kind="rule",
                    symbol=SemanticSymbol(
                        type_candidates=[TypeCandidate(label="constraint", confidence=0.65)],
                        constraints=[rule_text],
                    ),
                    metadata=SemanticMetadata(confidence=0.65, recency=env_step, ttl=100, provenance="perception"),
                    clip_refs=[perception_clip_id],
                )
            )
            rule_count += 1
        
        # Process place/scene
        scene_summary = perception_bundle.get("scene_summary")
        if scene_summary:
            candidates.append(
                SemanticUnitCandidate(
                    kind="place",
                    symbol=SemanticSymbol(
                        type_candidates=[TypeCandidate(label=scene_summary, confidence=0.7)],
                        state_vars={"description": scene_summary},
                    ),
                    metadata=SemanticMetadata(confidence=0.7, recency=env_step, ttl=100, provenance="perception"),
                    clip_refs=[perception_clip_id],
                    key="current_scene",
                )
            )
            place_added = True
        
        logger.info(
            "[SemanticMemoryManager] Perception processed: objects=%d relations=%d rules=%d place=%s",
            object_count, relation_count, rule_count, place_added
        )
        
        # Write perception candidates to WM
        if candidates:
            self.operator.write(candidates, env_step)
        
        # ===== Part 2: Process Action Execution =====
        feedback_bundle = self._normalize_feedback_input(
            action_desc=action_desc,
            env_info=env_info,
            instruction=instruction,
            reward=reward,
        )
        invalid = feedback_bundle.get("invalid", False)
        target = feedback_bundle.get("target", "")
        action_summary = feedback_bundle.get("summary") or (
            f"Action {action_id} {action_desc} invalid={invalid} reward={reward:.2f}"
        )
        
        action_clip_id = self._register_clip(
            kind="interaction",
            uri=f"step://{env_step}",
            digest=action_summary,
            annotations={
                "env_info": feedback_bundle.get("env_info", env_info),
                "risk_notes": feedback_bundle.get("risk_notes", []),
                "action": action_desc,
            },
        )
        
        phase_state = {
            "last_action": action_desc,
            "reward": reward,
            "step": env_step,
        }
        phase_state.update(feedback_bundle.get("state_vars", {}))
        
        phase_candidate = SemanticUnitCandidate(
            kind="phase",
            symbol=SemanticSymbol(
                type_candidates=[TypeCandidate(label="execution_phase", confidence=0.6)],
                state_vars=phase_state,
                constraints=feedback_bundle.get("constraints", []),
            ),
            metadata=SemanticMetadata(confidence=0.6, recency=env_step, ttl=50, provenance="executor"),
            clip_refs=[action_clip_id],
            key="phase_execution",
        )
        
        logger.info(
            "[SemanticMemoryManager] Action executed: step=%d action=%s invalid=%s target=%s reward=%.2f",
            env_step,
            action_desc,
            invalid,
            target or "<none>",
            reward,
        )
        
        self.operator.write([phase_candidate], env_step)
        
        if target:
            self.operator.correctness({"target": target, "invalid": invalid}, env_step)
        
        # Track object holding state for pick/place actions
        if not invalid:
            if "pick up" in action_desc.lower():
                self._update_holding_state_after_pick(action_desc, env_info, env_step)
            elif "place at" in action_desc.lower():
                self._update_object_position_after_place(action_desc, env_info, env_step)
        
        self._inject_action_references(action_desc, env_step)
        
        # ===== Part 3: Semantic Alignment & Integration =====
        # Check for entity gaps and attempt semantic alignment
        if not self._pending_entity_gaps and self.instruction:
            logger.info("[SemanticMemoryManager] Detecting entity gaps from instruction")
            temp_query = WMReadQuery(goal=self.instruction)
            temp_result = self.operator.read(temp_query)
            entity_gaps = [g for g in temp_result.gaps if g.startswith("entity:")]
            if entity_gaps:
                self._pending_entity_gaps = entity_gaps
                logger.info(f"[SemanticMemoryManager] Found {len(entity_gaps)} entity gaps for alignment")
        
        if self._pending_entity_gaps and object_labels:
            logger.info(
                f"[SemanticMemoryManager] Attempting semantic alignment: {len(self._pending_entity_gaps)} gaps, "
                f"{len(object_labels)} observed objects"
            )
            alignments = self.operator.align_semantics(
                pending_gaps=self._pending_entity_gaps,
                observed_labels=object_labels,
                step=env_step
            )
            if alignments:
                for align in alignments:
                    existing = any(
                        h['abstract_term'] == align['abstract_term'] and
                        h['concrete_term'] == align['concrete_term']
                        for h in self._alignment_history
                    )
                    if not existing:
                        self._alignment_history.append(align)
                
                logger.info(f"[SemanticMemoryManager] Semantic alignment: {len(alignments)} mappings established")
                self._significant_alignment_occurred = True
                
                for alignment in alignments:
                    logger.info(
                        f"  - '{alignment['abstract_term']}' → '{alignment['concrete_term']}' "
                        f"(conf={alignment['confidence']:.2f})"
                    )
                    aligned_gap = f"entity:{alignment.get('kind', 'place')}:{alignment['abstract_term']}"
                    if aligned_gap in self._pending_entity_gaps:
                        self._pending_entity_gaps.remove(aligned_gap)
                
                logger.info(f"[SemanticMemoryManager] Remaining pending gaps: {len(self._pending_entity_gaps)}")
        
        # Run WM operations: integrate, abstract, simplify
        self.operator.integrate(env_step)
        self.operator.abstract(env_step)
        self.operator.simplify(env_step)
        
        # ⚠️ 改进：不在每个 step 同步到 LTM，只在 episode 结束时统一同步
        # Merge LTM attributes into WM for enrichment
        refresh_labels = object_labels + ([target] if target else [])
        if scene_summary and place_added:
            refresh_labels.append(scene_summary)
        self._merge_ltm_attributes_into_wm(refresh_labels, env_step)
        
        logger.info(
            "[SemanticMemoryManager] ========== Step %d END: WM updated (no LTM sync) ==========",
            env_step
        )

    # -- planner support ----------------------------------------------------

    def prepare_planner_context(self, goal: str) -> WMReadResult:
        """组合 read + LTM 检索，为 planner 生成最终上下文。"""
        self._planner_query_step += 1
        query = WMReadQuery(
            goal=goal,
            focus_kinds=["object", "place", "link", "rule"],
            risk_tags=["safety"],
            required_fields=["state_vars", "constraints"],
            max_nodes=12,
        )
        
        logger.info("[SemanticMemoryManager] Preparing planner context for goal: %s", goal[:100])
        
        result = self.operator.read(query)
        
        logger.info(
            "[SemanticMemoryManager] WM read result: nodes=%d gaps=%d",
            len(result.nodes),
            len(result.gaps),
        )
        
        if result.gaps:
            # 分类显示 gaps
            entity_gaps = [g for g in result.gaps if g.startswith("entity:")]
            attr_gaps = [g for g in result.gaps if g.startswith("attribute:")]
            other_gaps = [g for g in result.gaps if not g.startswith("entity:") and not g.startswith("attribute:")]
            
            # ✨ 保存 entity gaps 用于后续语义对齐
            self._pending_entity_gaps = entity_gaps.copy()
            
            # ✨ 立即在规划阶段进行语义对齐（从 LTM 查询具体对象）
            if entity_gaps:
                logger.info("[SemanticMemoryManager] Entity gaps detected: %s", entity_gaps)
                # 尝试从 LTM 中查询并对齐
                # 注意：这里 observed_labels 为空，但 align_semantics 会主动查询 LTM
                alignments = self.operator.align_semantics(
                    pending_gaps=entity_gaps,
                    observed_labels=[],  # 规划阶段没有新观察，依赖 LTM 查询
                    step=self._planner_query_step
                )
                if alignments:
                    # ✨ 保存新对齐到历史记录（避免重复）
                    for align in alignments:
                        # 检查是否已经存在相同的对齐
                        existing = any(
                            h['abstract_term'] == align['abstract_term'] and
                            h['concrete_term'] == align['concrete_term']
                            for h in self._alignment_history
                        )
                        if not existing:
                            self._alignment_history.append(align)
                    
                    logger.info(
                        f"[SemanticMemoryManager] Planning-stage alignment: {len(alignments)} mappings from LTM"
                    )
                    for alignment in alignments:
                        logger.info(
                            f"  - '{alignment['abstract_term']}' → '{alignment['concrete_term']}' "
                            f"(conf={alignment['confidence']:.2f}, source=ltm)"
                        )
                        # 从待解析列表中移除已对齐的实体
                        aligned_gap = f"entity:{alignment.get('kind', 'place')}:{alignment['abstract_term']}"
                        if aligned_gap in self._pending_entity_gaps:
                            self._pending_entity_gaps.remove(aligned_gap)
                    
                    # ✨ 同步对齐后的节点（处理 _needs_ltm_sync 标记）
                    self._merge_ltm_attributes_into_wm([], self._planner_query_step)
                    
                    # 重新读取 WM，包含新对齐的实体
                    logger.info("[SemanticMemoryManager] Re-reading WM after planning-stage alignment...")
                    result = self.operator.read(query)
            
            if attr_gaps:
                logger.info("[SemanticMemoryManager] Attribute gaps detected: %s", attr_gaps[:3])
            if other_gaps:
                logger.info("[SemanticMemoryManager] Other gaps: %s", other_gaps[:3])
            
            hints = self.ltm.search(query, result.gaps)
            
            logger.info("[SemanticMemoryManager] LTM search returned %d hints", len(hints))
            
            updated, passthrough = self._apply_attribute_hints_to_wm(
                hints,
                step=self._planner_query_step,
            )
            
            if updated:
                logger.info("[SemanticMemoryManager] WM updated with LTM hints, re-reading...")
                result = self.operator.read(query)
                extra_hints = self.ltm.search(query, result.gaps)
                result.ltm_hints = passthrough + extra_hints
            else:
                result.ltm_hints = hints
        else:
            result.ltm_hints = []
            self._pending_entity_gaps = []
            logger.info("[SemanticMemoryManager] No gaps detected in WM read")
        
        # ✨ 将完整的对齐历史添加到返回结果中（包含本次和之前所有的对齐）
        result.alignments = list(self._alignment_history)
        
        logger.info(
            "[SemanticMemoryManager] Planner context prepared: nodes=%d gaps=%d hints=%d alignments=%d",
            len(result.nodes),
            len(result.gaps),
            len(result.ltm_hints),
            len(result.alignments),
        )
        return result

    # -- reporting ----------------------------------------------------------

    def export_digest(self, limit: int = 40) -> List[Dict[str, Any]]:
        """导出 WM 摘要（按 recency/utility 排序）。"""
        return self.graph.snapshot(limit=limit)

    def export_clips(self) -> List[Dict[str, Any]]:
        """导出所有 clip 元数据，方便 UI 或离线分析。"""
        return [clip.to_dict() for clip in self.clips.values()]

    def on_episode_end(self) -> Dict[str, Any]:
        """Episode 结束时统一同步 WM → LTM，并进行优化。
        
        这是 WM 和 LTM 交互的唯一时机：
        1. 导出完整 WM 快照
        2. 过滤临时状态，提取长期知识
        3. Consolidate：融合到 LTM
        4. Decompose：拆分模糊节点
        5. Forget：清理过期知识
        6. 持久化到磁盘
        """
        logger.info(
            "[SemanticMemoryManager] Episode %s ending, synchronizing WM → LTM...",
            self.episode_id or "unknown"
        )
        
        # 1. 导出完整 WM 快照（不限制节点数，获取所有信息）
        snapshot = self.export_digest(limit=256)  # 增加限制，确保获取所有节点
        logger.info(
            "[SemanticMemoryManager] WM snapshot exported: %d nodes, %d clips",
            len(snapshot),
            len(self.clips)
        )
        
        # 2. Consolidate：将 WM 知识融合到 LTM（内部会过滤临时状态）
        consolidate_stats = self.ltm.consolidate(snapshot)
        logger.info(
            "[SemanticMemoryManager] Consolidate complete: promoted=%d nodes, merged=%d pairs",
            consolidate_stats.get("promoted_count", 0),
            consolidate_stats.get("merged_pairs", 0)
        )
        
        # 3. Decompose：拆分高模糊度节点
        decompose_stats = self.ltm.decompose(snapshot)
        logger.info(
            "[SemanticMemoryManager] Decompose complete: %d nodes decomposed",
            decompose_stats.get("decomposed_count", 0)
        )
        
        # 4. Forget：移除长期未使用的节点
        forget_stats = self.ltm.remove_stale()
        logger.info(
            "[SemanticMemoryManager] Forget complete: removed %d nodes, %d edges",
            forget_stats.get("removed_nodes", 0),
            forget_stats.get("removed_edges", 0)
        )
        
        # 5. 记录操作到 LTM
        payload = {
            "episode_id": self.episode_id,
            "wm_snapshot_size": len(snapshot),
            "consolidate": consolidate_stats,
            "decompose": decompose_stats,
            "forget": forget_stats,
        }
        self.ltm.mutate("episode_end", payload)
        
        # 6. 持久化到磁盘
        self._save_ltm_store()
        logger.info(
            "[SemanticMemoryManager] Episode end complete. LTM size: %d nodes, %d edges",
            len(self.ltm.nodes),
            len(self.ltm.edges)
        )
        
        return payload
