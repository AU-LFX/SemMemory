"""
语义增强版 Memory Manager（WM + LTM）
====================================================
目标：
1) 在 memory 内部彻底去 LLM 化：WM/LTM 的更新、检索、融合、巩固全部使用“语义属性 + 图算法”
2) 与你给出的 unified_graph.py / semantic_retrieve.py 兼容
3) 代码注释全部中文，且尽量可直接替换你原来的 memory 文件

依赖：
- embodiedbench.evaluator.unified_graph.UnifiedMemoryGraph / Node / Edge / NodeMetadata / EdgeMetadata
- embodiedbench.evaluator.semantic_retrieve.MemoryRetriever / RetrievalConfig（你给的实现）

说明：
- LLM 仍可用于：perceive（生成结构化 perception）与 plan（基于上下文生成动作）
- 本文件只负责：存储、检索、融合、巩固（算法化）
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from embodiedbench.main import logger
from embodiedbench.evaluator.unified_graph import (
    UnifiedMemoryGraph,
    Node,
    Edge,
    NodeMetadata,
    EdgeMetadata,
)
from embodiedbench.evaluator.semantic_retrieve import MemoryRetriever, RetrievalConfig
from embodiedbench.evaluator.object_knowledge_seed import apply_seed_to_ltm


# ============================================================
# 0) 语义关系配置（用于写边、推理、扩展）
# ============================================================

RELATION_SEMANTICS: Dict[str, Dict[str, Any]] = {
    # taxonomic
    "is_a": {"relation_type": "taxonomic", "is_transitive": True, "is_symmetric": False, "strength": 1.0, "directionality": "directed"},
    "subclass_of": {"relation_type": "taxonomic", "is_transitive": True, "is_symmetric": False, "strength": 0.95, "directionality": "directed"},
    "instance_of": {"relation_type": "taxonomic", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},

    # partitive
    "part_of": {"relation_type": "partitive", "is_transitive": True, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},
    "has_part": {"relation_type": "partitive", "is_transitive": False, "is_symmetric": False, "strength": 0.85, "directionality": "directed"},
    "contains": {"relation_type": "partitive", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},

    # synonym
    "synonym_of": {"relation_type": "synonymy", "is_transitive": False, "is_symmetric": True, "strength": 0.95, "directionality": "undirected"},
    "similar_to": {"relation_type": "synonymy", "is_transitive": False, "is_symmetric": True, "strength": 0.7, "directionality": "undirected"},

    # functional
    "affords": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "used_for": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "has_function": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.75, "directionality": "directed"},

    # spatial（动态关系，通常不写回 LTM）
    "located_at": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed", "semantic_role": "location"},
    "on": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed", "semantic_role": "location"},
    "in": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed", "semantic_role": "location"},
    "near": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": True, "strength": 0.6, "directionality": "undirected", "semantic_role": "location"},
    "left_of": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.55, "directionality": "directed"},
    "right_of": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.55, "directionality": "directed"},

    # associative / attributes
    "has_color": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},
    "has_size": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "made_of": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},

    # 稳定位置先验（常驻/出生点）——可以写回 LTM
    "usually_found_at": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.85, "directionality": "directed", "semantic_role": "location"},
}

# WM 中哪些 properties 属于短期/动态信息（不写入 LTM）
EPHEMERAL_PROPERTY_KEYS: Set[str] = {
    "visible",
    "last_seen_ts",
    "step",
    "recency",
    "current_position",
    "holding",
    "ltm_retrieval",          # semantic_retrieve 写回 bundle
    "edge_consistency_bonus",
    "final_confidence",
}

# ============================================================
# 1) 轻量语义编码器（无外部依赖）
#    你若已有 embedding，可直接替换 encode()
# ============================================================

class SemanticEncoder:
    """
    无 embedding 版本：Hashing-BOW + cosine
    - 优点：无外部依赖、稳定、可运行
    - 缺点：语义泛化弱于真正 embedding，但已经足够支撑“语义属性 + 图结构”的增强检索
    """
    def __init__(self, dim: int = 2048):
        self.dim = dim
        self._pat = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+", re.U)

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        return self._pat.findall(text)

    def encode(self, text: str) -> List[float]:
        toks = self._tokenize(text)
        if not toks:
            return [0.0] * self.dim

        counts = Counter(toks)
        vec = [0.0] * self.dim
        for t, c in counts.items():
            idx = hash(t) % self.dim
            vec[idx] += 1.0 + math.log(1.0 + c)

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# ============================================================
# 2) 语义索引：将 Node 转换成可检索的文本表示 + 向量
# ============================================================

class SemanticIndex:
    """
    为 UnifiedMemoryGraph 维护 node_text / node_vec
    - node_text = label + properties 摘要 + 1-hop 邻域摘要
    """
    def __init__(self, graph: UnifiedMemoryGraph, encoder: SemanticEncoder):
        self.graph = graph
        self.encoder = encoder
        self.node_text: Dict[str, str] = {}
        self.node_vec: Dict[str, List[float]] = {}
        self._dirty: Set[str] = set()

    def mark_dirty(self, node_id: str) -> None:
        self._dirty.add(node_id)

    def _build_text(self, node: Node, max_neighbors: int = 10) -> str:
        # 1) label
        parts = [node.label]

        # 2) properties（过滤动态字段）
        props = []
        for k, v in (node.properties or {}).items():
            if k in EPHEMERAL_PROPERTY_KEYS:
                continue
            if isinstance(v, (str, int, float, bool)):
                props.append(f"{k}:{v}")
            elif isinstance(v, list):
                props.append(f"{k}:" + ",".join([str(x) for x in v[:5]]))
            elif isinstance(v, dict):
                # dict 不展开太深，避免噪声
                props.append(f"{k}:{list(v.keys())[:5]}")
        if props:
            parts.append(" ".join(props))

        # 3) 1-hop 邻域：出边 + 入边摘要
        neigh = []
        out_eids = list(self.graph.out_edges.get(node.id, []))[:max_neighbors]
        in_eids = list(self.graph.in_edges.get(node.id, []))[:max_neighbors]

        for eid in out_eids:
            e = self.graph.edges.get(eid)
            if not e:
                continue
            t = self.graph.nodes.get(e.target)
            if t:
                neigh.append(f"{e.relation}:{t.label}")

        for eid in in_eids:
            e = self.graph.edges.get(eid)
            if not e:
                continue
            s = self.graph.nodes.get(e.source)
            if s:
                neigh.append(f"in:{s.label}:{e.relation}")

        if neigh:
            parts.append(" ".join(neigh))

        return " | ".join([p for p in parts if p]).strip()

    def refresh(self) -> None:
        """
        仅重建 dirty 或缺失的节点向量
        """
        for nid, node in list(self.graph.nodes.items()):
            if nid not in self.node_vec or nid in self._dirty:
                text = self._build_text(node)
                self.node_text[nid] = text
                self.node_vec[nid] = self.encoder.encode(text)
        self._dirty.clear()

    def query(self, query_text: str, topk: int = 10) -> List[Tuple[str, float]]:
        """
        返回 [(node_id, cosine_score)] 按分数降序
        """
        self.refresh()
        qv = self.encoder.encode(query_text)
        scored: List[Tuple[str, float]] = []
        for nid, nv in self.node_vec.items():
            scored.append((nid, self.encoder.cosine(qv, nv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]


# ============================================================
# 3) 图传播：Personalized PageRank（使用边语义属性作为权重）
# ============================================================

def _edge_weight(graph: UnifiedMemoryGraph, edge: Edge) -> float:
    """
    统一边权重计算：
    - 优先使用 graph._edge_weight_score（你 unified_graph 内已实现）
    - 若未来改名或不可用，就退化使用 metadata 字段估算
    """
    try:
        return float(graph._edge_weight_score(edge))  # type: ignore
    except Exception:
        # 兜底：relation_type 权重 * (strength/confidence)
        t = edge.metadata.relation_type or graph.relation_name_fallback_type.get(edge.relation, "associative")
        type_w = graph.relation_type_weight.get(t, 0.6)
        return max(0.0, min(1.5, type_w * (0.55 * edge.metadata.strength + 0.35 * edge.metadata.confidence + 0.10)))

def personalized_pagerank(
    graph: UnifiedMemoryGraph,
    seeds: Dict[str, float],
    steps: int = 10,
    restart: float = 0.35,
) -> Dict[str, float]:
    """
    简化版 PPR：
    - seeds: {node_id: weight}
    - 使用边权重 + 对称边双向传播
    """
    if not seeds:
        return {}

    # 构建邻接（双向传播：对称边 / undirected / bidirectional）
    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for e in graph.edges.values():
        w = _edge_weight(graph, e)
        adj[e.source].append((e.target, w))
        if e.metadata.is_symmetric or e.metadata.directionality in ("undirected", "bidirectional"):
            adj[e.target].append((e.source, w))

    # 初始分布
    score = defaultdict(float)
    for nid, w in seeds.items():
        score[nid] += float(w)

    for _ in range(steps):
        new = defaultdict(float)

        # restart 部分：回到 seeds
        for nid, w in seeds.items():
            new[nid] += restart * float(w)

        # walk 部分：沿边传播
        for nid, v in score.items():
            nbrs = adj.get(nid, [])
            if not nbrs:
                continue
            Z = sum(w for _, w in nbrs) or 1.0
            for nb, w in nbrs:
                new[nb] += (1.0 - restart) * float(v) * (w / Z)

        score = new

    return dict(score)


# ============================================================
# 4) 返回给 Planner 的上下文结构
# ============================================================

@dataclass
class PlannerContext:
    """
    给 planner 的上下文（尽量结构化，不依赖 LLM 总结）
    """
    goal: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    gaps: List[str]
    ltm_hints: List[Dict[str, Any]]
    
    def to_text_block(self) -> str:
        """
        将PlannerContext转换为文本块，用于LLM prompt
        """
        lines = []
        
        # Goal
        if self.goal:
            lines.append(f"Goal: {self.goal}")
            lines.append("")
        
        # Nodes (objects, locations, etc.)
        if self.nodes:
            lines.append(f"Relevant objects and locations ({len(self.nodes)} items):")
            for node in self.nodes[:20]:  # 限制数量
                label = node.get('label', 'unknown')
                node_type = node.get('properties', {}).get('type', 'object')
                confidence = node.get('metadata', {}).get('confidence', 0.0)
                lines.append(f"  - {label} ({node_type}, confidence: {confidence:.2f})")
            if len(self.nodes) > 20:
                lines.append(f"  ... and {len(self.nodes) - 20} more")
            lines.append("")
        
        # Edges (relations)
        if self.edges:
            lines.append(f"Known relations ({len(self.edges)} items):")
            for edge in self.edges[:30]:  # 限制数量
                source_label = edge.get('source_label', edge.get('source', '?'))
                target_label = edge.get('target_label', edge.get('target', '?'))
                relation = edge.get('relation', 'related_to')
                lines.append(f"  - {source_label} --[{relation}]--> {target_label}")
            if len(self.edges) > 30:
                lines.append(f"  ... and {len(self.edges) - 30} more")
            lines.append("")
        
        # Gaps (missing information)
        if self.gaps:
            lines.append("Information gaps:")
            for gap in self.gaps[:10]:
                lines.append(f"  - {gap}")
            if len(self.gaps) > 10:
                lines.append(f"  ... and {len(self.gaps) - 10} more")
            lines.append("")
        
        # LTM hints (prior knowledge)
        if self.ltm_hints:
            lines.append(f"Prior knowledge from LTM ({len(self.ltm_hints)} hints):")
            for hint in self.ltm_hints[:15]:
                # hints结构: {"wm_node": str, "selected_anchor": str, "final_confidence": float}
                wm_node = hint.get('wm_node', 'unknown')
                selected_anchor = hint.get('selected_anchor', 'no anchor')
                confidence = hint.get('final_confidence', 0.0)
                lines.append(f"  - WM:{wm_node} ← LTM:{selected_anchor} (conf={confidence:.2f})")
            if len(self.ltm_hints) > 15:
                lines.append(f"  ... and {len(self.ltm_hints) - 15} more")
        
        return "\n".join(lines) if lines else "No semantic memory context available"


# ============================================================
# 5) WM 算子：write/read/correctness/abstract/simplify/integrate
# ============================================================

class WMOperators:
    """
    WM（工作记忆）算子：
    - write_perception: 写入 perception（对象/属性/关系）
    - read: 面向 goal 的语义检索 + 子图扩展
    - correctness: 根据 action valid/reward 更新置信度
    - abstract: 从 WM 挖稳定模式，生成 rule 节点
    - simplify: 语义感知剪枝（保持 max_nodes）
    - integrate_from_ltm: 用 semantic_retrieve 从 LTM 拉先验证据到 WM
    """
    def __init__(
        self,
        wm: UnifiedMemoryGraph,
        ltm: UnifiedMemoryGraph,
        encoder: SemanticEncoder,
        retriever: MemoryRetriever,
        max_nodes: int = 256,
    ):
        self.wm = wm
        self.ltm = ltm
        self.encoder = encoder
        self.wm_index = SemanticIndex(wm, encoder)
        self.retriever = retriever
        self.max_nodes = max_nodes

    # -------------------------
    # (1) WM.write：写入 perception
    # -------------------------

    def write_perception(self, perception: Dict[str, Any], step: int, clip_id: Optional[str] = None) -> None:
        """
        纯算法写入：
        - detected_objects: 支持 list[str] 或 list[dict]
        - objects_relations: 支持 list[str] 形如 "mug is on table"
        - spatial 关系默认 ephemeral（动态）
        - step<=1 的位置会额外写 usually_found_at（稳定先验）
        """
        self.wm.current_step = step
        now = time.time()

        detected = perception.get("detected_objects", []) or []
        relations = perception.get("objects_relations", []) or []
        
        # 🔥 新增：获取对象属性和指令实体信息
        obj_attrs = perception.get("detected_object_attributes", []) or []
        instruction_entities = perception.get("instruction_entities_and_attributes", []) or []

        # 1) 写对象节点
        for obj in detected:
            if isinstance(obj, str):
                label = obj.strip()
                props = {"type": "object"}
            else:
                label = str(obj.get("label", "unknown")).strip()
                props = {"type": obj.get("type", "object")}
                # 常见属性字段
                for k in ("color", "size", "material", "category", "shape"):
                    if obj.get(k) is not None and obj.get(k) != "":
                        props[k] = obj.get(k)
                if obj.get("position") is not None:
                    props["current_position"] = obj.get("position")
                    if step <= 1:
                        props.setdefault("initial_location", obj.get("position"))

            # 🔥 标准化标签（去除冠词）
            label = self._normalize_label(label)
            
            if not label or label.lower() == "unknown":
                continue

            # 节点 metadata：WM 节点一般 ephemeral=True（工作记忆随时间衰减）
            # 但“对象身份”本身可以不 ephemeral（你可按需要调整）
            nm = NodeMetadata(
                confidence=0.85,
                provenance="perception",
                ephemeral=False,   # 对象节点建议非 ephemeral，便于 episode 内稳定使用
                ttl=200,
                timestamp=float(step),
            )
            node = self.wm.add_node(label=label, properties=props, metadata=nm, clip_refs=[clip_id] if clip_id else None)

            # 标记动态属性
            node.properties["visible"] = True
            node.properties["last_seen_ts"] = now

            self.wm_index.mark_dirty(node.id)

            # 属性节点 + 边
            self._attach_attributes(node, step=step)

            # step<=1：写出生点（稳定先验）
            if step <= 1 and node.properties.get("initial_location"):
                loc_label = str(node.properties["initial_location"])
                loc_node = self._get_or_create_node(loc_label, ntype="location", step=step, persistent=True)
                self._add_edge(node.id, loc_node.id, "usually_found_at", step=step, persistent=True)
        
        # 🔥 1.5) 写入detected_object_attributes（额外的属性信息）
        self._write_object_attributes(obj_attrs, step=step)
        
        # 🔥 1.6) 写入instruction_entities_and_attributes（指令语义信息）
        if step <= 1:  # 仅在初始步骤存储指令实体
            self._write_instruction_entities(instruction_entities, step=step)

        # 2) 写关系（文本规则解析）
        for r in relations:
            if not isinstance(r, str):
                continue
            self._write_relation_text(r, step=step)

    def _attach_attributes(self, obj_node: Node, step: int) -> None:
        """
        将对象属性转为 attribute 节点 + 边（稳定信息，适合写回 LTM）
        """
        # color
        if "color" in obj_node.properties and obj_node.properties["color"]:
            colors = obj_node.properties["color"]
            if not isinstance(colors, list):
                colors = [colors]
            for c in colors[:5]:
                a = self._get_or_create_node(str(c), ntype="attribute", step=step, persistent=True)
                self._add_edge(obj_node.id, a.id, "has_color", step=step, persistent=True)

        # size
        if "size" in obj_node.properties and obj_node.properties["size"]:
            a = self._get_or_create_node(str(obj_node.properties["size"]), ntype="attribute", step=step, persistent=True)
            self._add_edge(obj_node.id, a.id, "has_size", step=step, persistent=True)

        # material
        if "material" in obj_node.properties and obj_node.properties["material"]:
            mats = obj_node.properties["material"]
            if not isinstance(mats, list):
                mats = [mats]
            for m in mats[:5]:
                a = self._get_or_create_node(str(m), ntype="attribute", step=step, persistent=True)
                self._add_edge(obj_node.id, a.id, "made_of", step=step, persistent=True)
    
    def _write_object_attributes(self, obj_attrs: List[Any], step: int) -> None:
        """
        写入detected_object_attributes（来自感知模块输出的详细对象属性）
        🔥 将属性同时存储到节点properties和创建属性边
        
        Args:
            obj_attrs: 对象属性列表，可能是字符串或字典格式
            step: 当前步数
        """
        if not obj_attrs:
            return
        
        for attr_item in obj_attrs:
            # 支持字符串格式："red apple" 或字典格式 {"label": "shelf", "attributes": {...}}
            if isinstance(attr_item, str):
                # 简单解析 "color object" 模式
                parts = attr_item.strip().split(maxsplit=1)
                if len(parts) == 2:
                    attr_value, obj_label = parts
                    # 标准化标签
                    obj_label = self._normalize_label(obj_label)
                    # 查找或创建对象节点
                    obj_nodes = self.wm.find_nodes(label=obj_label, exact=False, limit=1)
                    if obj_nodes:
                        attr_node = self._get_or_create_node(attr_value, ntype="attribute", step=step, persistent=True)
                        self._add_edge(obj_nodes[0].id, attr_node.id, "has_attribute", step=step, persistent=True)
            elif isinstance(attr_item, dict):
                # 字典格式处理（来自感知模块的详细输出）
                obj_label = attr_item.get("object") or attr_item.get("label")
                if not obj_label:
                    continue
                
                # 标准化标签
                obj_label = self._normalize_label(str(obj_label))
                
                # 查找已存在的对象节点（优先精确匹配，再模糊匹配）
                obj_nodes = self.wm.find_nodes(label=obj_label, exact=True, limit=1)
                if not obj_nodes:
                    obj_nodes = self.wm.find_nodes(label=obj_label, exact=False, limit=1)
                
                if not obj_nodes:
                    # 如果找不到对象节点，创建一个
                    obj_node = self._get_or_create_node(obj_label, ntype="object", step=step, persistent=True)
                else:
                    obj_node = obj_nodes[0]
                
                # 🔥 获取attributes字典（感知模块输出的完整属性）
                attributes = attr_item.get("attributes", {})
                if isinstance(attributes, dict):
                    # 🔥 直接将属性存储到节点properties中
                    for key, value in attributes.items():
                        if value and key not in ["container_or_surface", "relative_position"]:
                            # 存储到节点properties
                            obj_node.properties[key] = value
                    
                    # 同时创建属性节点和边（用于图检索）
                    for key in ["color", "size", "shape", "material", "state"]:
                        if key in attributes and attributes[key]:
                            values = attributes[key] if isinstance(attributes[key], list) else [attributes[key]]
                            for val in values[:3]:
                                if val:
                                    attr_node = self._get_or_create_node(str(val), ntype="attribute", step=step, persistent=True)
                                    relation = f"has_{key}"
                                    self._add_edge(obj_node.id, attr_node.id, relation, step=step, persistent=True)
    
    def _write_instruction_entities(self, instruction_entities: List[Any], step: int) -> None:
        """
        写入instruction_entities_and_attributes（来自指令解析的目标实体和属性）
        这些信息在检索时会用于精确匹配
        
        Args:
            instruction_entities: 指令中的实体列表，例如 ["red apple", "wooden table"]
            step: 当前步数（应该 <=1，仅初始步骤调用）
        """
        if not instruction_entities:
            return
        
        # 创建一个特殊的"instruction_goal"节点用于存储所有目标实体
        goal_node = self._get_or_create_node(
            "instruction_requirements", 
            ntype="goal", 
            step=step, 
            persistent=True,
            confidence=0.95
        )
        
        for entity_item in instruction_entities:
            # 支持字符串格式或字典格式
            if isinstance(entity_item, str):
                # 简单格式："red apple"
                entity_label = entity_item.strip()
                if not entity_label:
                    continue
                
                # 创建实体节点
                entity_node = self._get_or_create_node(
                    entity_label, 
                    ntype="instruction_entity", 
                    step=step, 
                    persistent=True,
                    confidence=0.9
                )
                # 链接到goal节点
                self._add_edge(goal_node.id, entity_node.id, "requires", step=step, persistent=True, confidence=0.95)
                
                # 尝试解析属性（简单模式："adjective noun"）
                parts = entity_label.split(maxsplit=1)
                if len(parts) == 2:
                    attr_value, obj_label = parts
                    # 创建属性节点并链接
                    attr_node = self._get_or_create_node(attr_value, ntype="attribute", step=step, persistent=True)
                    self._add_edge(entity_node.id, attr_node.id, "has_attribute", step=step, persistent=True)
                    
            elif isinstance(entity_item, dict):
                # 字典格式：{"entity": "apple", "attributes": ["red", "fresh"]}
                entity_label = entity_item.get("entity") or entity_item.get("label")
                if not entity_label:
                    continue
                
                entity_node = self._get_or_create_node(
                    str(entity_label), 
                    ntype="instruction_entity", 
                    step=step, 
                    persistent=True,
                    confidence=0.9
                )
                self._add_edge(goal_node.id, entity_node.id, "requires", step=step, persistent=True, confidence=0.95)
                
                # 处理属性列表
                attributes = entity_item.get("attributes", [])
                if not isinstance(attributes, list):
                    attributes = [attributes]
                
                for attr_val in attributes:
                    if attr_val:
                        attr_node = self._get_or_create_node(str(attr_val), ntype="attribute", step=step, persistent=True)
                        self._add_edge(entity_node.id, attr_node.id, "has_attribute", step=step, persistent=True)

    def _get_or_create_node(self, label: str, ntype: str, step: int, persistent: bool, confidence: float = 0.8) -> Node:
        """
        若 WM 中已有同名节点则复用，否则创建
        
        Args:
            label: 节点标签
            ntype: 节点类型
            step: 当前步数
            persistent: 是否持久化
            confidence: 置信度（默认0.8）
        """
        label = (label or "").strip()
        if not label:
            # 创建一个兜底节点（极少发生）
            label = "unknown"

        exist = self.wm.find_nodes(label=label, exact=True, limit=1)
        if exist:
            return exist[0]

        nm = NodeMetadata(
            confidence=confidence,
            provenance="wm_infer",
            ephemeral=not persistent,
            ttl=200 if persistent else 120,
            timestamp=float(step),
        )
        node = self.wm.add_node(label=label, properties={"type": ntype}, metadata=nm)
        self.wm_index.mark_dirty(node.id)
        return node

    def _add_edge(self, src: str, tgt: str, rel: str, step: int, persistent: bool, confidence: float = None) -> None:
        """
        去重写边，并把 RELATION_SEMANTICS 写入 EdgeMetadata
        
        Args:
            src: 源节点ID
            tgt: 目标节点ID
            rel: 关系类型
            step: 当前步数
            persistent: 是否持久化
            confidence: 置信度（可选，默认根据persistent决定）
        """
        # 去重：source+target+relation
        exist = self.wm.find_edges(source=src, target=tgt, relation=rel, limit=1)
        if exist:
            return

        sem = RELATION_SEMANTICS.get(rel, {})
        
        # 如果没有指定confidence，使用默认值
        if confidence is None:
            confidence = 0.85 if persistent else 0.75
            
        em = EdgeMetadata(
            confidence=confidence,
            provenance="perception" if persistent else "wm_dynamic",
            ephemeral=not persistent,
            ttl=250 if persistent else 80,
            timestamp=float(step),

            relation_type=str(sem.get("relation_type", "associative")),
            is_symmetric=bool(sem.get("is_symmetric", False)),
            is_transitive=bool(sem.get("is_transitive", False)),
            strength=float(sem.get("strength", 0.5)),
            directionality=str(sem.get("directionality", "directed")),
            semantic_role=sem.get("semantic_role", None),
        )
        self.wm.add_edge(src, tgt, rel, metadata=em)

    def _write_relation_text(self, text: str, step: int, from_llm_summary: bool = False) -> None:
        """
        规则解析常见空间关系：
        - "A is on B" / "A is in B" / "A near B" / "A is at B"
        - 🔥 自动去除冠词(the/a/an)避免重复节点
        
        Args:
            text: 关系描述文本
            step: 当前步数
            from_llm_summary: 是否来自LLM episode总结（影响置信度和TTL）
        """
        t = (text or "").strip()
        if not t:
            return

        patterns = [
            (re.compile(r"^(.+?)\s+is\s+on\s+(?:the\s+)?(.+)$", re.I), "on"),
            (re.compile(r"^(.+?)\s+is\s+in\s+(?:the\s+)?(.+)$", re.I), "in"),
            (re.compile(r"^(.+?)\s+(?:is\s+)?near\s+(?:the\s+)?(.+)$", re.I), "near"),
            (re.compile(r"^(.+?)\s+is\s+at\s+(?:the\s+)?(.+)$", re.I), "located_at"),
            # 新增：支持"usually"等模式（来自LLM总结）
            (re.compile(r"^(.+?)\s+(?:is\s+)?usually\s+(?:on|in|at)\s+(?:the\s+)?(.+)$", re.I), "usually_found_at"),
            # 新增：支持 "A left of B", "A right of B" 等位置关系
            (re.compile(r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(?:the\s+)?(.+)$", re.I), "left_of"),
            (re.compile(r"^(.+?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(?:the\s+)?(.+)$", re.I), "right_of"),
        ]

        for pat, rel in patterns:
            m = pat.match(t)
            if not m:
                continue
            s_label = m.group(1).strip()
            o_label = m.group(2).strip()
            
            # 🔥 标准化标签：去除开头的冠词
            s_label = self._normalize_label(s_label)
            o_label = self._normalize_label(o_label)

            # 如果来自LLM总结，使用更高的置信度和持久化
            if from_llm_summary:
                s = self._get_or_create_node(s_label, ntype="object", step=step, persistent=True, confidence=0.85)
                loc = self._get_or_create_node(o_label, ntype="location", step=step, persistent=True, confidence=0.85)
                # LLM总结的知识应该是持久的
                self._add_edge(s.id, loc.id, rel, step=step, persistent=True, confidence=0.85)
            else:
                s = self._get_or_create_node(s_label, ntype="object", step=step, persistent=True)
                loc = self._get_or_create_node(o_label, ntype="location", step=step, persistent=True)
                # 动态空间边（ephemeral）
                self._add_edge(s.id, loc.id, rel, step=step, persistent=False)

                # 初始 step 写稳定位置先验
                if step <= 1 and rel in ("on", "in", "located_at"):
                    self._add_edge(s.id, loc.id, "usually_found_at", step=step, persistent=True)
                    s.properties.setdefault("initial_location", loc.label)

            return
    
    def _normalize_label(self, label: str) -> str:
        """
        标准化节点标签：去除冠词、多余空格
        
        Args:
            label: 原始标签
        
        Returns:
            标准化后的标签
        """
        if not label:
            return label
        
        # 去除开头的冠词 (the, a, an)
        label = re.sub(r"^(the|a|an)\s+", "", label, flags=re.I).strip()
        # 去除多余空格
        label = re.sub(r"\s+", " ", label).strip()
        
        return label

    # -------------------------
    # (2) WM.correctness：正确性校验（基于环境反馈）
    # -------------------------

    def correctness(self, action_desc: str, env_info: Dict[str, Any], reward: float, step: int) -> None:
        """
        纯算法置信度更新：
        - 若 was_prev_action_invalid=True：降低涉及实体的 confidence
        - 若 action 有效：略微提高涉及实体 confidence（reward>0 再额外加一点）
        """
        invalid = bool(env_info.get("was_prev_action_invalid", False))
        entities = self._extract_entities_from_action(action_desc)

        # 找到可能关联的 WM 节点
        affected: List[Node] = []
        for ent in entities:
            # 非精确：容忍 label 差异
            hits = self.wm.find_nodes(label=ent, exact=False, limit=5)
            affected.extend(hits)

        # 更新 node.metadata.confidence
        for n in affected:
            old = float(n.metadata.confidence)
            if invalid:
                new = max(0.05, old - 0.15)
            else:
                boost = 0.05 + (0.05 if reward > 0 else 0.0)
                new = min(0.99, old + boost)
            n.metadata.confidence = new
            n.metadata.timestamp = float(step)
            self.wm_index.mark_dirty(n.id)

    def _extract_entities_from_action(self, action_desc: str) -> List[str]:
        """
        简单规则抽取动作中可能的实体名
        （如果你动作格式更固定，可以增强这个解析）
        """
        text = (action_desc or "").lower()
        ents: List[str] = []

        # pick up X
        m = re.search(r"pick\s+up\s+(?:the\s+)?(.+)", text)
        if m:
            ents.append(m.group(1).strip())

        # place X on/in/at Y（这里只抓 Y）
        m = re.search(r"place\s+.+?\s+(?:on|in|at)\s+(?:the\s+)?(.+)", text)
        if m:
            ents.append(m.group(1).strip())

        # navigate to X
        m = re.search(r"navigate\s+to\s+(?:the\s+)?(.+)", text)
        if m:
            ents.append(m.group(1).strip())

        # fallback：取最后一个词（保守）
        toks = re.findall(r"[a-z0-9_]+", text)
        if toks:
            ents.append(toks[-1])

        # 去重
        out: List[str] = []
        for e in ents:
            e = e.strip()
            if e and e not in out:
                out.append(e)
        return out[:5]

    # -------------------------
    # (3) WM.integrate：用 semantic_retrieve 从 LTM 拉证据（不调用 LLM）
    # -------------------------

    def integrate_from_ltm(self, goal: str, step: int) -> List[Dict[str, Any]]:
        """
        用 semantic_retrieve 的 MemoryRetriever：
        - 先根据 goal 在 WM 找一批相关节点（anchors）
        - 🔥 优先检索instruction_entity节点（来自指令解析）
        - 对这些 WM 节点执行 retrieve_batch：为每个节点写回 ltm_retrieval bundle，并可将 LTM 子图复制进 WM（ephemeral 证据）
        """
        # 1) 从 goal 中提取关键词（兼容中文/英文）
        keywords = extract_goal_keywords(goal)
        if not keywords:
            keywords = [goal]  # 至少用 goal 本身

        # 2) 找 WM anchor 节点：semantic_find_nodes 会用认知属性排序
        anchor_ids: Set[str] = set()
        
        # 🔥 2.1) 优先添加所有instruction_entity节点（这些是指令中的关键目标）
        instruction_entities = self.wm.find_nodes(property_filters={"type": "instruction_entity"}, limit=50)
        for entity_node in instruction_entities:
            anchor_ids.add(entity_node.id)
        
        # 🔥 2.2) 添加所有object节点（确保检索到所有已感知的对象）
        object_nodes = self.wm.find_nodes(property_filters={"type": "object"}, limit=50)
        for obj_node in object_nodes:
            anchor_ids.add(obj_node.id)
        
        # 2.3) 基于关键词的检索
        for kw in keywords[:8]:
            ranked = self.wm.semantic_find_nodes(kw, exact=False, limit=6)
            for n, _sc in ranked:
                anchor_ids.add(n.id)

        if not anchor_ids:
            return []

        # 3) 批量检索写回（MemoryRetriever 内部会写 wm_node.properties["ltm_retrieval"]）
        self.retriever.retrieve_batch(self.wm, list(anchor_ids))

        # 4) 返回给上层一些提示信息（用于日志或 planner 特殊策略）
        hints: List[Dict[str, Any]] = []
        for nid in anchor_ids:
            n = self.wm.get_node(nid)
            if not n:
                continue
            b = n.properties.get(self.retriever.cfg.writeback_key, {})
            if b and b.get("selected_anchor"):
                hints.append({
                    "wm_node": n.label,
                    "selected_anchor": b.get("selected_anchor"),
                    "final_confidence": b.get("final_confidence", b.get("aggregated_confidence", b.get("base_score", 0.0))),
                })

        return hints

    # -------------------------
    # (4) WM.read：面向 goal 的语义读取（返回子图）
    # -------------------------

    def read(self, goal: str, step: int, max_nodes: int = 30) -> Tuple[List[Node], List[Edge], List[str]]:
        """
        语义读取策略：
        1) semantic index 召回 topk 节点作为 seeds
        2) PPR 扩展出连贯的子图（更“语义理解”）
        3) 返回 nodes/edges，同时做轻量 gap 检测
        """
        self.wm.current_step = step

        # 1) seeds：用 index + label recall 双通道
        hits = self.wm_index.query(goal, topk=max(10, max_nodes))
        seed_ids: Dict[str, float] = {}
        for nid, sc in hits[:10]:
            seed_ids[nid] = max(seed_ids.get(nid, 0.0), float(sc))

        # 🔥 1.5) 优先检索 instruction_entity 节点（来自指令解析）
        instruction_entities = self.wm.find_nodes(property_filters={"type": "instruction_entity"}, limit=20)
        for entity_node in instruction_entities:
            # 给予较高的初始分数
            seed_ids[entity_node.id] = max(seed_ids.get(entity_node.id, 0.0), 0.9)
            # 同时加入该实体的所有属性邻居
            for edge in self.wm.edges.values():
                if edge.source == entity_node.id and edge.relation in ["has_attribute", "has_color", "has_size"]:
                    attr_node = self.wm.nodes.get(edge.target)
                    if attr_node:
                        seed_ids[attr_node.id] = max(seed_ids.get(attr_node.id, 0.0), 0.85)

        # 额外：用 goal 关键词再召回一些 label 相近节点
        for kw in extract_goal_keywords(goal)[:6]:
            ranked = self.wm.semantic_find_nodes(kw, exact=False, limit=5)
            for n, sc in ranked:
                seed_ids[n.id] = max(seed_ids.get(n.id, 0.0), float(sc))
                
                # 🔥 如果匹配到的是对象节点，也加入其属性邻居
                if n.properties.get("type") == "object":
                    for edge in self.wm.edges.values():
                        if edge.source == n.id and "has_" in edge.relation:
                            attr_node = self.wm.nodes.get(edge.target)
                            if attr_node:
                                seed_ids[attr_node.id] = max(seed_ids.get(attr_node.id, 0.0), 0.7)

        if not seed_ids:
            # 兜底：返回当前 WM 的部分节点
            nodes = list(self.wm.nodes.values())[:max_nodes]
            edges = list(self.wm.edges.values())[:80]
            gaps = self._detect_gaps(goal, nodes)
            return nodes, edges, gaps

        # 2) PPR 扩展
        ppr = personalized_pagerank(self.wm, seed_ids, steps=8, restart=0.4)
        ranked = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep_ids = {nid for nid, _ in ranked}

        nodes = [self.wm.nodes[nid] for nid in keep_ids if nid in self.wm.nodes]
        edges = [e for e in self.wm.edges.values() if e.source in keep_ids and e.target in keep_ids]

        gaps = self._detect_gaps(goal, nodes)
        return nodes, edges, gaps

    def _detect_gaps(self, goal: str, nodes: List[Node]) -> List[str]:
        """
        无 LLM gap 检测（轻量版）：
        - 将 goal token 与 WM node.label 做覆盖检查
        - 未出现的关键词 -> gap
        """
        toks = extract_goal_keywords(goal)
        if not toks:
            return []
        label_text = " ".join([n.label.lower() for n in nodes])
        gaps: List[str] = []
        for t in toks:
            if len(t) <= 1:
                continue
            if t.lower() not in label_text:
                gaps.append(f"missing_concept:{t}")
        return gaps[:10]

    # -------------------------
    # (5) WM.abstract：抽象（从 WM 挖稳定模式）
    # -------------------------

    def abstract(self, step: int, min_support: int = 3) -> int:
        """
        简单可用的抽象：
        - 挖 (object -> usually_found_at -> location) 的频繁模式，形成 rule 节点
        """
        pair_count = Counter()

        for e in self.wm.edges.values():
            if e.relation != "usually_found_at":
                continue
            s = self.wm.nodes.get(e.source)
            t = self.wm.nodes.get(e.target)
            if not s or not t:
                continue
            pair_count[(s.label.lower(), t.label.lower())] += 1

        created = 0
        for (obj, loc), c in pair_count.items():
            if c < min_support:
                continue
            rule_label = f"rule:{obj}:usually_found_at:{loc}"
            if self.wm.find_nodes(label=rule_label, exact=True, limit=1):
                continue

            nm = NodeMetadata(
                confidence=min(0.99, 0.6 + 0.1 * c),
                provenance="wm_abstract",
                ephemeral=False,
                ttl=9999,
                timestamp=float(step),
                centrality=0.6,
                family_resemblance=0.6,
                prototypicality=0.6,
            )
            self.wm.add_node(
                label=rule_label,
                properties={"type": "rule", "support": int(c), "kind": "usually_found_at"},
                metadata=nm,
            )
            created += 1

        return created

    # -------------------------
    # (6) WM.simplify：剪枝（保持 WM 可控大小）
    # -------------------------

    def simplify(self, step: int) -> int:
        """
        语义感知剪枝：
        - 评分 = confidence + centrality + recency + degree - prior_penalty
        - 优先删除：ltm_copy/ltm_read 导入但长期未被确认、低置信、低连接度、过旧节点
        """
        if len(self.wm.nodes) <= self.max_nodes:
            return 0

        # 统计 degree
        indeg = Counter()
        outdeg = Counter()
        for e in self.wm.edges.values():
            outdeg[e.source] += 1
            indeg[e.target] += 1

        # 计算每个节点分数
        scored: List[Tuple[str, float]] = []
        for nid, n in self.wm.nodes.items():
            deg = float(indeg[nid] + outdeg[nid])
            conf = float(n.metadata.confidence)
            cent = float(n.metadata.centrality)

            # step 是 int，timestamp 在 unified_graph 里是 float；这里把 timestamp 当“step”用
            age = float(step - n.metadata.timestamp)
            rec = max(0.0, 1.0 - min(1.0, age / 200.0))

            # ltm_read/ltmcopy 证据节点如果不可见，惩罚（鼓励淘汰没用的先验）
            prov = (n.metadata.provenance or "").lower()
            prior_penalty = 0.15 if ("ltm" in prov and not bool(n.properties.get("visible", False))) else 0.0

            score = 0.55 * conf + 0.20 * cent + 0.15 * rec + 0.10 * min(1.0, deg / 10.0) - prior_penalty
            scored.append((nid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep = set([nid for nid, _ in scored[: self.max_nodes]])
        remove = [nid for nid, _ in scored[self.max_nodes :]]

        removed = 0
        for nid in remove:
            if nid in keep:
                continue
            ok = self.wm.remove_node(nid)
            if ok:
                removed += 1

        return removed


# ============================================================
# 6) LTM 算子：add/remove/decompose/consolidate/search
# ============================================================

class LTMOperators:
    """
    LTM（长期记忆）算子：
    - add_from_wm: 将 WM 的稳定节点/边写回 LTM（含合并）
    - remove: 删除节点
    - search: 语义检索（index + PPR）
    - consolidate: 去重合并
    - decompose: 多义分解（按关系类型桶）
    """
    def __init__(self, ltm: UnifiedMemoryGraph, encoder: SemanticEncoder):
        self.ltm = ltm
        self.encoder = encoder
        self.index = SemanticIndex(ltm, encoder)

    def add_from_wm(self, wm: UnifiedMemoryGraph, step: int, sim_th: float = 0.90, prop_th: float = 0.30) -> Dict[str, int]:
        """
        将 WM 中“稳定内容”写回 LTM：
        - 只写入 metadata.ephemeral=False 的节点
        - 不写入动态边（ephemeral=True），也不写入 ltm_read/ltmcopy 证据
        - 合并策略：
            1) label 精确匹配 -> 更新 properties/metadata.confidence
            2) 否则：语义向量相似 >= sim_th 且 properties token Jaccard >= prop_th -> 合并
            3) 否则：新增节点
        """
        added_nodes = 0
        merged_nodes = 0
        added_edges = 0

        # 1) 写节点
        wm_nodes = [n for n in wm.nodes.values() if not n.metadata.ephemeral]
        for wn in wm_nodes:
            # 🔥 修复：区分"纯LTM证据"和"被agent更新过的节点"
            # - ltmcopy_ 开头的节点：纯证据，跳过（避免重复写回）
            # - ltm_read provenance 但有 perception 相关属性：agent观测过，应该回写
            prov = (wn.metadata.provenance or "").lower()
            
            # 跳过纯 ltmcopy 证据节点（它们是子图导入的副本）
            if wn.id.startswith("ltmcopy_"):
                continue
            
            # 如果是 ltm_read，检查是否有 agent 新增的内容
            if "ltm" in prov:
                # 检查是否有"非LTM"的属性（说明被agent更新过）
                has_agent_update = False
                agent_props = ["visible", "last_seen_ts", "location", "current_position", 
                               "holding", "step", "recency"]
                for key in agent_props:
                    if key in wn.properties:
                        has_agent_update = True
                        break
                
                # 如果没有agent更新，说明是纯ltm证据，跳过
                if not has_agent_update:
                    continue
                
                # 如果有agent更新，改写provenance为混合来源，然后继续处理
                # 这样它会被合并回LTM，且能清除动态字段
                wn.metadata.provenance = "wm_ltm_updated"

            target = self._add_or_merge_node(wn, step=step, sim_th=sim_th, prop_th=prop_th)
            if target["action"] == "added":
                added_nodes += 1
            else:
                merged_nodes += 1

        # 2) 写边（只写稳定边）
        for we in wm.edges.values():
            if we.metadata.ephemeral:
                continue
            
            # 🔥 修复：区分纯ltm证据边和有效关系边
            # 跳过 ltmcopy 相关的边（子图导入的副本）
            s = wm.nodes.get(we.source)
            t = wm.nodes.get(we.target)
            if not s or not t:
                continue
            if s.id.startswith("ltmcopy_") or t.id.startswith("ltmcopy_"):
                continue
            
            # 如果边的provenance是ltm但连接的节点不是ltmcopy，可能是有价值的关系
            # 例如：perception节点与ltm_read节点之间的新发现关系
            # 这种情况应该保留并回写
            
            if s.metadata.ephemeral or t.metadata.ephemeral:
                continue

            # 在 LTM 找对应节点（按 label 精确）
            s_l = self.ltm.find_nodes(label=s.label, exact=True, limit=1)
            t_l = self.ltm.find_nodes(label=t.label, exact=True, limit=1)
            if not s_l or not t_l:
                continue

            # 去重
            if self.ltm.find_edges(source=s_l[0].id, target=t_l[0].id, relation=we.relation, limit=1):
                continue

            # 写入边：provenance=ltm
            sem = RELATION_SEMANTICS.get(we.relation, {})
            em = EdgeMetadata(
                confidence=float(we.metadata.confidence),
                provenance="ltm",
                ephemeral=False,
                ttl=9999,
                timestamp=float(step),

                relation_type=str(we.metadata.relation_type or sem.get("relation_type", "associative")),
                is_symmetric=bool(we.metadata.is_symmetric or sem.get("is_symmetric", False)),
                is_transitive=bool(we.metadata.is_transitive or sem.get("is_transitive", False)),
                strength=float(we.metadata.strength or sem.get("strength", 0.5)),
                directionality=str(we.metadata.directionality or sem.get("directionality", "directed")),
                semantic_role=we.metadata.semantic_role or sem.get("semantic_role", None),
            )
            self.ltm.add_edge(s_l[0].id, t_l[0].id, we.relation, properties=dict(we.properties or {}), metadata=em)
            added_edges += 1

        return {"added_nodes": added_nodes, "merged_nodes": merged_nodes, "added_edges": added_edges}

    def _add_or_merge_node(self, wn: Node, step: int, sim_th: float, prop_th: float) -> Dict[str, Any]:
        """
        对单个节点做 add/merge
        """
        # 先精确 label 匹配
        exact = self.ltm.find_nodes(label=wn.label, exact=True, limit=1)
        if exact:
            ln = exact[0]
            ln.properties.update(self._clean_props_for_ltm(wn.properties))
            ln.metadata.confidence = min(0.99, max(float(ln.metadata.confidence), float(wn.metadata.confidence)))
            ln.metadata.timestamp = float(step)
            self.index.mark_dirty(ln.id)
            return {"action": "merged", "node_id": ln.id}

        # 语义相似匹配：用向量检索 topk
        hits = self.index.query(wn.label, topk=8)
        if hits:
            wn_text = self._node_text_for_sim(wn)
            wn_vec = self.encoder.encode(wn_text)
            wn_tok = flatten_props_tokens(wn.properties)

            best_id = None
            best_sim = 0.0
            best_prop = 0.0
            for nid, _ in hits:
                ln = self.ltm.nodes.get(nid)
                if not ln:
                    continue
                ln_text = self._node_text_for_sim(ln)
                ln_vec = self.encoder.encode(ln_text)
                sim = self.encoder.cosine(wn_vec, ln_vec)

                prop = jaccard(wn_tok, flatten_props_tokens(ln.properties))
                if sim > best_sim:
                    best_sim = sim
                    best_prop = prop
                    best_id = nid

            if best_id and best_sim >= sim_th and best_prop >= prop_th:
                ln = self.ltm.nodes[best_id]
                ln.properties.update(self._clean_props_for_ltm(wn.properties))
                ln.metadata.confidence = min(0.99, max(float(ln.metadata.confidence), float(wn.metadata.confidence)))
                ln.metadata.timestamp = float(step)
                self.index.mark_dirty(ln.id)
                return {"action": "merged", "node_id": ln.id, "sim": best_sim, "prop": best_prop}

        # 新增节点
        nm = NodeMetadata(
            confidence=float(wn.metadata.confidence),
            provenance="ltm",
            ephemeral=False,
            ttl=9999,
            timestamp=float(step),
            centrality=float(wn.metadata.centrality),
            family_resemblance=float(wn.metadata.family_resemblance),
            prototypicality=float(wn.metadata.prototypicality),
            vagueness=float(wn.metadata.vagueness),
            gradience=float(wn.metadata.gradience),
        )
        ln = self.ltm.add_node(
            label=wn.label,
            properties=self._clean_props_for_ltm(wn.properties),
            metadata=nm,
        )
        self.index.mark_dirty(ln.id)
        return {"action": "added", "node_id": ln.id}

    def _clean_props_for_ltm(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理 WM 中的动态字段，避免污染 LTM
        """
        out: Dict[str, Any] = {}
        for k, v in (props or {}).items():
            if k in EPHEMERAL_PROPERTY_KEYS:
                continue
            out[k] = v
        return out

    def _node_text_for_sim(self, node: Node) -> str:
        """
        用于节点相似度的文本（更短更稳）
        """
        base = [node.label]
        # 只取少量关键属性，避免噪声
        for k in ("type", "category", "color", "size", "material", "shape"):
            if k in (node.properties or {}) and node.properties[k] not in (None, "", []):
                base.append(f"{k}:{node.properties[k]}")
        return " | ".join(base)

    # -------------------------
    # LTM.search：语义检索（index + PPR）
    # -------------------------

    def search(self, query: str, topk: int = 12) -> List[Dict[str, Any]]:
        """
        返回可解释的检索结果：
        - 先 semantic index 召回
        - 再用 PPR 扩展成连贯证据（返回 topk）
        """
        hits = self.index.query(query, topk=max(10, topk))
        if not hits:
            return []

        seeds = {nid: max(0.01, float(sc)) for nid, sc in hits[:8]}
        ppr = personalized_pagerank(self.ltm, seeds, steps=10, restart=0.35)
        ranked = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:topk]

        out: List[Dict[str, Any]] = []
        for nid, s in ranked:
            n = self.ltm.nodes.get(nid)
            if not n:
                continue
            # evidence：取少量邻域边
            ev = []
            for eid in list(self.ltm.out_edges.get(nid, []))[:6]:
                e = self.ltm.edges.get(eid)
                if not e:
                    continue
                t = self.ltm.nodes.get(e.target)
                if t:
                    ev.append(f"{e.relation}:{t.label}")
            out.append({"node_id": nid, "label": n.label, "score": float(s), "evidence": ev})
        return out

    # -------------------------
    # LTM.consolidate：去重合并（语义相似 + 结构重合）
    # -------------------------

    def consolidate(self, sim_th: float = 0.92, struct_th: float = 0.35, max_pairs: int = 500) -> Dict[str, int]:
        """
        去重合并：
        - 对候选 pairs 做判断：语义向量相似 >= sim_th 且邻域结构 jaccard >= struct_th
        - 合并：把 drop 节点边重定向到 keep 节点，合并 properties
        注意：此实现为可用版本，复杂度 O(N * k)，避免全对全爆炸
        """
        self.index.refresh()
        nodes = list(self.ltm.nodes.values())
        if len(nodes) <= 1:
            return {"merged_pairs": 0}

        merged = 0
        checked = 0

        # 预缓存邻域集合（用于结构相似度）
        neigh_cache: Dict[str, Set[str]] = {}

        def neigh_set(nid: str) -> Set[str]:
            if nid in neigh_cache:
                return neigh_cache[nid]
            s: Set[str] = set()
            # 出边
            for eid in list(self.ltm.out_edges.get(nid, []))[:25]:
                e = self.ltm.edges.get(eid)
                if not e:
                    continue
                t = self.ltm.nodes.get(e.target)
                if t:
                    s.add(f"{e.relation}:{t.label.lower()}")
            # 入边
            for eid in list(self.ltm.in_edges.get(nid, []))[:25]:
                e = self.ltm.edges.get(eid)
                if not e:
                    continue
                ss = self.ltm.nodes.get(e.source)
                if ss:
                    s.add(f"in:{ss.label.lower()}:{e.relation}")
            neigh_cache[nid] = s
            return s

        # 用 semantic index 召回候选邻居，避免全对全
        for a in nodes:
            if a.id not in self.ltm.nodes:
                continue
            # 以 label 查询，找潜在相似节点
            cands = self.index.query(a.label, topk=8)
            for b_id, _ in cands:
                if b_id == a.id:
                    continue
                if b_id not in self.ltm.nodes:
                    continue
                b = self.ltm.nodes[b_id]

                checked += 1
                if checked > max_pairs:
                    break

                va = self.index.node_vec.get(a.id)
                vb = self.index.node_vec.get(b.id)
                if not va or not vb:
                    continue
                sim = self.encoder.cosine(va, vb)
                if sim < sim_th:
                    continue

                na = neigh_set(a.id)
                nb = neigh_set(b.id)
                j = jaccard(na, nb)
                if j < struct_th:
                    continue

                # 选择 keep：更高 centrality + 更短 label（更一般）作为保留
                keep_id, drop_id = self._choose_keep_drop(a, b)
                self._merge_nodes(keep_id, drop_id)
                merged += 1

            if checked > max_pairs:
                break

        return {"merged_pairs": merged}

    def _choose_keep_drop(self, a: Node, b: Node) -> Tuple[str, str]:
        """
        决定保留哪个节点：
        - centrality 高的优先保留
        - centrality 相近时，label 更短的保留（更通用）
        """
        ca = float(a.metadata.centrality)
        cb = float(b.metadata.centrality)
        if ca > cb + 0.05:
            return a.id, b.id
        if cb > ca + 0.05:
            return b.id, a.id
        # centrality 接近：保留 label 更短者
        if len(a.label) <= len(b.label):
            return a.id, b.id
        return b.id, a.id

    def _merge_nodes(self, keep_id: str, drop_id: str) -> None:
        """
        合并 drop -> keep
        """
        keep = self.ltm.nodes.get(keep_id)
        drop = self.ltm.nodes.get(drop_id)
        if not keep or not drop:
            return

        # 合并 properties
        for k, v in (drop.properties or {}).items():
            if k not in keep.properties:
                keep.properties[k] = v

        # 重定向边
        for e in list(self.ltm.edges.values()):
            if e.source == drop_id:
                # 避免重复
                if not self.ltm.find_edges(source=keep_id, target=e.target, relation=e.relation, limit=1):
                    self.ltm.add_edge(keep_id, e.target, e.relation, properties=dict(e.properties or {}), metadata=e.metadata)
                self.ltm.remove_edge(e.id)
            elif e.target == drop_id:
                if not self.ltm.find_edges(source=e.source, target=keep_id, relation=e.relation, limit=1):
                    self.ltm.add_edge(e.source, keep_id, e.relation, properties=dict(e.properties or {}), metadata=e.metadata)
                self.ltm.remove_edge(e.id)

        # 建立同义边（可选）：保留可解释性
        # 注意：如果你不想增加边，可以删掉这段
        try:
            sem = RELATION_SEMANTICS.get("synonym_of", {})
            em = EdgeMetadata(
                confidence=0.8,
                provenance="ltm",
                ephemeral=False,
                ttl=9999,
                timestamp=time.time(),
                relation_type=sem.get("relation_type", "synonymy"),
                is_symmetric=True,
                is_transitive=False,
                strength=sem.get("strength", 0.95),
                directionality=sem.get("directionality", "undirected"),
            )
            # 用 label 的不同来判断是否加 synonym_of
            if keep.label.lower() != drop.label.lower():
                if not self.ltm.find_edges(source=drop_id, target=keep_id, relation="synonym_of", limit=1) and \
                   not self.ltm.find_edges(source=keep_id, target=drop_id, relation="synonym_of", limit=1):
                    # drop 即将被删，所以先用 keep<->keep（无意义），这里改为：新建一个 alias 节点太重
                    # 直接把 synonym_of 省略也行。这里选择不加 synonym_of，避免残留 dangling。
                    pass
        except Exception:
            pass

        # 删除 drop 节点
        self.ltm.remove_node(drop_id)

        # 标记索引 dirty
        self.index.mark_dirty(keep_id)

    # -------------------------
    # LTM.decompose：多义分解（按 relation_type 桶）
    # -------------------------

    def decompose(self, label: str, min_edges: int = 8) -> Dict[str, Any]:
        """
        简易多义分解：
        - 若某节点出边很多且关系类型分布分散，则按 relation_type 分桶生成 sense 节点
        """
        nodes = self.ltm.find_nodes(label=label, exact=True, limit=1)
        if not nodes:
            return {"decomposed": 0, "reason": "node_not_found"}
        n = nodes[0]

        out_eids = list(self.ltm.out_edges.get(n.id, []))
        if len(out_eids) < min_edges:
            return {"decomposed": 0, "reason": "not_enough_edges"}

        buckets: Dict[str, List[Edge]] = defaultdict(list)
        for eid in out_eids:
            e = self.ltm.edges.get(eid)
            if not e:
                continue
            rt = e.metadata.relation_type or self.ltm.relation_name_fallback_type.get(e.relation, "associative")
            buckets[rt].append(e)

        # 至少 2 个桶才有“多义”意义
        if len(buckets) < 2:
            return {"decomposed": 0, "reason": "single_bucket"}

        created = []
        for i, (rt, edges) in enumerate(buckets.items(), 1):
            sense_label = f"{n.label}#sense{i}"
            nm = NodeMetadata(
                confidence=float(n.metadata.confidence),
                provenance="ltm",
                ephemeral=False,
                ttl=9999,
                timestamp=time.time(),
            )
            s_node = self.ltm.add_node(label=sense_label, properties={**(n.properties or {}), "sense": rt}, metadata=nm)
            created.append(s_node.label)
            self.index.mark_dirty(s_node.id)

            # 将该桶的边复制到 sense 节点
            for e in edges:
                if not self.ltm.find_edges(source=s_node.id, target=e.target, relation=e.relation, limit=1):
                    self.ltm.add_edge(s_node.id, e.target, e.relation, properties=dict(e.properties or {}), metadata=e.metadata)

            # 原节点与 sense 的 part_of 连接（可选）
            sem = RELATION_SEMANTICS.get("part_of", {"relation_type": "partitive", "strength": 0.8})
            em = EdgeMetadata(
                confidence=0.8,
                provenance="ltm",
                ephemeral=False,
                ttl=9999,
                timestamp=time.time(),
                relation_type=sem.get("relation_type", "partitive"),
                is_symmetric=False,
                is_transitive=True,
                strength=sem.get("strength", 0.9),
                directionality="directed",
            )
            self.ltm.add_edge(n.id, s_node.id, "part_of", metadata=em)

        n.properties["polysemy_decomposed"] = True
        self.index.mark_dirty(n.id)
        return {"decomposed": len(created), "senses": created}

    def remove(self, label: str) -> bool:
        nodes = self.ltm.find_nodes(label=label, exact=True, limit=1)
        if not nodes:
            return False
        return bool(self.ltm.remove_node(nodes[0].id))


# ============================================================
# 7) 总控：SemanticMemoryManager（对外接口）
# ============================================================

class SemanticMemoryManager:
    """
    对外统一入口：
    - wm / ltm 两张 unified graph
    - retriever 用于 LTM->WM 语义融合（你 semantic_retrieve 的 MemoryRetriever）
    """
    def __init__(
        self,
        max_wm_nodes: int = 256,
        ltm_save_path: str = "output/semantic_ltm.json",
        retriever_config: Optional[RetrievalConfig] = None,
        llm_model_name: Optional[str] = None,  # 新增：用于未来LLM调用
    ):
        self.wm = UnifiedMemoryGraph()
        self.ltm = UnifiedMemoryGraph()

        self.encoder = SemanticEncoder(dim=2048)

        self.ltm_save_path = ltm_save_path
        self.current_step: int = 0
        self.instruction: str = ""
        self.llm_model_name = llm_model_name  # 保存LLM模型名称

        # 启动时尝试加载 LTM
        self._try_load_ltm()

        # semantic_retrieve: 用 LTM 做检索器
        self.retriever = MemoryRetriever(self.ltm, config=retriever_config or RetrievalConfig())

        self.wm_ops = WMOperators(self.wm, self.ltm, self.encoder, self.retriever, max_nodes=max_wm_nodes)
        self.ltm_ops = LTMOperators(self.ltm, self.encoder)
    # -------------------------
    # 生命周期
    # -------------------------

    def reset(self, instruction: str = "") -> None:
        """
        新 episode：清空 WM，保留 LTM
        """
        self.wm = UnifiedMemoryGraph()
        self.wm_ops = WMOperators(self.wm, self.ltm, self.encoder, self.retriever, max_nodes=self.wm_ops.max_nodes)
        self.current_step = 0
        self.instruction = instruction or ""

    def ingest_perception(self, perception: Dict[str, Any], step: int, clip_id: Optional[str] = None) -> None:
        """
        每步 perception 输入：写 WM
        """
        self.current_step = int(step)
        logger.info(f"[SemanticMemory] 📝 Ingesting perception at step {step}...")
        logger.info(f"  - Detected objects: {perception.get('detected_objects', [])}")
        logger.info(f"  - Object attributes: {perception.get('detected_object_attributes', [])}")
        logger.info(f"  - Instruction entities: {perception.get('instruction_entities_and_attributes', [])}")
        logger.info(f"  - Objects relations: {perception.get('objects_relations', [])}")
        logger.info(f"  - State changes: {perception.get('state_changes', [])}")
        
        self.wm_ops.write_perception(perception, step=int(step), clip_id=clip_id)
        
        logger.info(f"[SemanticMemory] ✅ Perception ingested. Current WM size: {len(self.wm.nodes)} nodes, {len(self.wm.edges)} edges")

    def ingest_action_feedback(self, action_desc: str, env_info: Dict[str, Any], reward: float, step: int) -> None:
        """
        每步 action 执行后：做 correctness check（更新置信度）
        """
        self.current_step = int(step)
        self.wm_ops.correctness(action_desc, env_info, reward, step=int(step))

    def prepare_planner_context(self, goal: str) -> PlannerContext:
        """
        planner 前调用：
        1) integrate_from_ltm：从 LTM 拉“相关先验子图”到 WM（ephemeral 证据）
        2) read：从 WM 读出“与 goal 相关的连贯子图”
        """
        step = int(self.current_step)
        
        logger.info(f"[SemanticMemory] 🔄 Preparing planner context for goal: {goal[:80]}...")
        logger.info(f"  - Current WM state: {len(self.wm.nodes)} nodes, {len(self.wm.edges)} edges")
        logger.info(f"  - Current LTM state: {len(self.ltm.nodes)} nodes, {len(self.ltm.edges)} edges")
        
        logger.info(f"[SemanticMemory] 📚 Step 1: Integrating relevant knowledge from LTM...")
        ltm_hints = self.wm_ops.integrate_from_ltm(goal, step=step)
        logger.info(f"  - Checked {len(self.wm.nodes)} WM nodes for LTM retrieval")
        logger.info(f"  - Retrieved {len(ltm_hints)} hints from LTM")
        for i, hint in enumerate(ltm_hints[:5], 1):
            wm_node = hint.get('wm_node', 'unknown')
            selected_anchor = hint.get('selected_anchor', 'no anchor')
            confidence = hint.get('final_confidence', 0.0)
            
            # 将LTM节点ID转换为标签
            ltm_label = selected_anchor
            if selected_anchor.startswith('n_'):
                ltm_node = self.ltm.get_node(selected_anchor)
                if ltm_node:
                    ltm_label = ltm_node.label
            
            logger.info(f"    {i}. WM:{wm_node} ← LTM:{ltm_label} (conf={confidence:.2f})")
        if len(ltm_hints) > 5:
            logger.info(f"    ... and {len(ltm_hints) - 5} more hints")
        
        logger.info(f"[SemanticMemory] 📖 Step 2: Reading relevant subgraph from WM...")
        nodes, edges, gaps = self.wm_ops.read(goal, step=step, max_nodes=30)
        logger.info(f"  - Read {len(nodes)} nodes, {len(edges)} edges")
        logger.info(f"  - Detected {len(gaps)} information gaps: {gaps[:5]}")

        # 🔥 构建节点ID到标签的映射
        node_map = {n.id: n.label for n in nodes}
        
        # 🔥 转换edges为dict，同时添加source_label和target_label
        edges_dict = []
        for e in edges:
            edge_dict = e.to_dict()
            edge_dict['source_label'] = node_map.get(e.source, e.source)
            edge_dict['target_label'] = node_map.get(e.target, e.target)
            edges_dict.append(edge_dict)
        
        # 🔥 转换ltm_hints中的节点ID为标签
        ltm_hints_with_labels = []
        for hint in ltm_hints:
            hint_copy = hint.copy()
            selected_anchor = hint.get('selected_anchor', '')
            
            # 如果是节点ID（以n_开头），转换为标签
            if selected_anchor.startswith('n_'):
                ltm_node = self.ltm.get_node(selected_anchor)
                if ltm_node:
                    hint_copy['selected_anchor'] = ltm_node.label
                    hint_copy['selected_anchor_id'] = selected_anchor  # 保留ID用于调试
            
            ltm_hints_with_labels.append(hint_copy)

        return PlannerContext(
            goal=goal,
            nodes=[n.to_dict() for n in nodes],
            edges=edges_dict,
            gaps=gaps,
            ltm_hints=ltm_hints_with_labels,
        )

    def on_episode_end(self, llm_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        episode 结束：
        0) [新增] 如果提供了LLM总结，先将其写入WM
        1) WM.abstract（抽象稳定模式）
        2) WM.simplify（剪枝）
        3) LTM.add_from_wm（把 WM 稳定内容写回 LTM）
        4) LTM.consolidate（去重合并）
        5) 保存 LTM
        
        Args:
            llm_summary: LLM对整个episode的总结（可选）
                包含learned_patterns、spatial_knowledge等字段
        """
        step = int(self.current_step)
        
        # 🔥 新增：如果有LLM总结，先写入WM
        llm_insights = {}
        if llm_summary:
            try:
                llm_insights = self._ingest_llm_episode_summary(llm_summary, step=step)
                logger.info(
                    f"[SemanticMemory] Ingested LLM episode summary: "
                    f"{len(llm_insights.get('patterns_added', []))} patterns, "
                    f"{len(llm_insights.get('knowledge_added', []))} knowledge items"
                )
            except Exception as e:
                logger.warning(f"[SemanticMemory] Failed to ingest LLM summary: {e}")

        logger.info(f"[SemanticMemory] 🔧 Running WM.abstract() to extract patterns...")
        created_rules = self.wm_ops.abstract(step=step, min_support=3)
        logger.info(f"  - Created {created_rules} abstract rules")
        
        logger.info(f"[SemanticMemory] 🧹 Running WM.simplify() to prune ephemeral nodes...")
        removed_wm = self.wm_ops.simplify(step=step)
        logger.info(f"  - Removed {removed_wm} ephemeral nodes from WM")

        logger.info(f"[SemanticMemory] 💾 Writing back stable WM knowledge to LTM...")
        writeback = self.ltm_ops.add_from_wm(self.wm, step=step, sim_th=0.90, prop_th=0.30)
        logger.info(f"  - Added {writeback} nodes/edges to LTM from WM")
        
        logger.info(f"[SemanticMemory] 🔄 Consolidating LTM (merge duplicates)...")
        consolidated = self.ltm_ops.consolidate(sim_th=0.92, struct_th=0.35, max_pairs=600)
        logger.info(f"  - Merged {consolidated} duplicate nodes/edges in LTM")

        logger.info(f"[SemanticMemory] 💿 Saving LTM to disk...")
        self.save_ltm()
        
        final_wm_stats = self.wm.stats()
        final_ltm_stats = self.ltm.stats()
        logger.info(f"[SemanticMemory] ✅ Episode end processing complete:")
        logger.info(f"  - Final WM: {final_wm_stats}")
        logger.info(f"  - Final LTM: {final_ltm_stats}")

        return {
            "llm_insights": llm_insights,
            "wm_rules_created": created_rules,
            "wm_removed": removed_wm,
            "ltm_writeback": writeback,
            "ltm_consolidate": consolidated,
            "wm_stats": final_wm_stats,
            "ltm_stats": final_ltm_stats,
        }
    
    def _ingest_llm_episode_summary(self, llm_summary: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        将LLM的episode总结写入WM图
        
        Args:
            llm_summary: LLM生成的总结，包含learned_patterns、spatial_knowledge等
            step: 当前步数
            
        Returns:
            Dict包含写入WM的统计信息
        """
        patterns_added = []
        knowledge_added = []
        
        # 1. 处理学到的模式（behavioral patterns）
        learned_patterns = llm_summary.get("learned_patterns", []) or []
        for pattern in learned_patterns[:10]:  # 限制数量
            if not pattern or not isinstance(pattern, str):
                continue
            
            # 创建pattern节点
            nm = NodeMetadata(
                confidence=0.8,
                provenance="llm_episode_summary",
                ephemeral=False,
                ttl=1000,  # 长期保留
                timestamp=float(step),
            )
            pattern_node = self.wm.add_node(
                label=f"pattern:{pattern[:100]}",  # 限制长度
                properties={"type": "pattern", "description": pattern},
                metadata=nm,
            )
            patterns_added.append(pattern_node.id)
        
        # 2. 处理空间知识（spatial knowledge）
        spatial_knowledge = llm_summary.get("spatial_knowledge", []) or []
        for knowledge in spatial_knowledge[:20]:  # 限制数量
            if not knowledge or not isinstance(knowledge, str):
                continue
            
            # 尝试解析空间关系（例如 "spoon is usually on the dining table"）
            self._write_relation_text(knowledge, step=step, from_llm_summary=True)
            knowledge_added.append(knowledge)
        
        # 3. 处理成功因素（作为高置信度知识）
        success_factors = llm_summary.get("success_factors", []) or []
        for factor in success_factors[:10]:
            if not factor or not isinstance(factor, str):
                continue
            
            nm = NodeMetadata(
                confidence=0.85,
                provenance="llm_episode_summary",
                ephemeral=False,
                ttl=1000,
                timestamp=float(step),
            )
            factor_node = self.wm.add_node(
                label=f"success_factor:{factor[:100]}",
                properties={"type": "success_factor", "description": factor},
                metadata=nm,
            )
            knowledge_added.append(factor)
        
        # 4. 处理失败分析（作为警示知识）
        failure_analysis = llm_summary.get("failure_analysis", []) or []
        for failure in failure_analysis[:10]:
            if not failure or not isinstance(failure, str):
                continue
            
            nm = NodeMetadata(
                confidence=0.75,
                provenance="llm_episode_summary",
                ephemeral=False,
                ttl=800,
                timestamp=float(step),
            )
            failure_node = self.wm.add_node(
                label=f"failure_case:{failure[:100]}",
                properties={"type": "failure_analysis", "description": failure},
                metadata=nm,
            )
            knowledge_added.append(failure)
        
        return {
            "patterns_added": patterns_added,
            "knowledge_added": knowledge_added,
            "summary": llm_summary.get("episode_summary", ""),
        }

    def export_digest(self, limit: int = 32) -> List[Dict[str, Any]]:
        """
        导出 WM 的摘要信息供 agent 使用
        """
        nodes = list(self.wm.nodes.values())[:limit]
        return [n.to_dict() for n in nodes]

    # -------------------------
    # LTM 的辅助接口
    # -------------------------

    def search_ltm(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        """
        用于调试或上层策略：直接语义检索 LTM
        """
        return self.ltm_ops.search(query, topk=topk)

    def decompose_ltm(self, label: str) -> Dict[str, Any]:
        """
        多义分解：按 relation_type 分桶生成 sense 节点
        """
        out = self.ltm_ops.decompose(label)
        self.save_ltm()
        return out

    def remove_ltm(self, label: str) -> bool:
        ok = self.ltm_ops.remove(label)
        if ok:
            self.save_ltm()
        return ok

    # -------------------------
    # 持久化
    # -------------------------

    def save_ltm(self) -> None:
        """
        保存 LTM 到 json
        """
        path = self.ltm_save_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ltm.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"[SemanticMemoryManager] LTM saved to: {path}")
        except Exception as e:
            logger.warning(f"[SemanticMemoryManager] Failed to save LTM: {e}")

    def _try_load_ltm(self) -> None:
        """
        初始化时加载 LTM：
        - 若磁盘中存在 LTM，则从磁盘加载
        - 否则从 seed KG 构建一个可用的初始 LTM
        注意：此函数可能在 __init__ 早期被调用，此时 self.retriever 可能尚未创建，
        所以对 retriever 的重绑定必须做 hasattr 检查。
        """
        path = getattr(self, "ltm_save_path", None)

        # -------------------------
        # 1) 磁盘存在：优先加载
        # -------------------------
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 覆盖式加载（from_dict 会清空并重建索引）
                self.ltm.from_dict(data)

                # retriever 可能尚未创建，做安全检查
                if hasattr(self, "retriever") and self.retriever is not None:
                    self.retriever.ltm = self.ltm

                logger.info(f"[SemanticMemoryManager] LTM loaded from: {path}, stats={self.ltm.stats()}")
                return
            except Exception as e:
                logger.warning(f"[SemanticMemoryManager] Failed to load LTM from disk: {e}. Fallback to seed...")

        # -------------------------
        # 2) 磁盘不存在或加载失败：用 seed 构建
        # -------------------------
        try:


            # 确保 seed 写入的节点 provenance 含 "ltm"
            # 你 seed 构建函数里如果 provenance 叫 "seed_knowledge"，检索器会过滤掉
            # apply_seed_to_ltm 内部应把 provenance 设成 "ltm_seed" 或至少包含 "ltm"
            apply_seed_to_ltm(self.ltm, perspective="seed")

            if hasattr(self, "retriever") and self.retriever is not None:
                self.retriever.ltm = self.ltm

            logger.info(f"[SemanticMemoryManager] No disk LTM. Built from seed, stats={self.ltm.stats()}")
        except Exception as e:
            logger.warning(f"[SemanticMemoryManager] Failed to build seed LTM: {e}")



# ============================================================
# 8) 工具函数：goal 关键词提取、属性 token、jaccard
# ============================================================

_STOPWORDS_EN = {
    "the", "a", "an", "to", "of", "in", "on", "at", "near", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "for", "from", "into",
    "then", "now", "it", "this", "that", "there", "here",
    "please", "go", "get", "find", "look", "search", "move", "pick", "place",
}


def extract_goal_keywords(goal: str, max_kw: int = 12) -> List[str]:
    """
    从 goal / instruction 里抽取“可能代表实体/概念”的关键词：
    - 英文：字母数字下划线 token
    - 中文：连续中文片段（粗略）
    """
    goal = (goal or "").strip()
    if not goal:
        return []

    kws: List[str] = []

    # 2) 英文 token
    en_parts = re.findall(r"[a-zA-Z0-9_]{2,}", goal.lower())
    for p in en_parts:
        if p in _STOPWORDS_EN:
            continue
        kws.append(p)

    # 去重，保持顺序
    out: List[str] = []
    for k in kws:
        k = k.strip()
        if not k:
            continue
        if k not in out:
            out.append(k)

    return out[:max_kw]

def flatten_props_tokens(props: Dict[str, Any]) -> Set[str]:
    """
    将 properties 粗暴扁平化为 token 集合（用于属性相似度）
    """
    tokens: Set[str] = set()
    for k, v in (props or {}).items():
        if k in EPHEMERAL_PROPERTY_KEYS:
            continue
        tokens.add(str(k).lower())
        if isinstance(v, (list, tuple, set)):
            for it in v:
                tokens.add(str(it).lower())
        elif isinstance(v, dict):
            for kk, vv in v.items():
                tokens.add(str(kk).lower())
                tokens.add(str(vv).lower())
        else:
            tokens.add(str(v).lower())
    return tokens

def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))
