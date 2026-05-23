# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import uuid
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union, Iterable

from embodiedbench.main import logger


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ----------------------------
# Metadata
# ----------------------------

@dataclass
class NodeMetadata:
    # 基础属性
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    ttl: int = 100
    provenance: str = "perception"
    ephemeral: bool = False
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    # 认知属性
    centrality: float = 0.5
    family_resemblance: float = 0.5
    perspective: Optional[str] = None
    vagueness: float = 0.0
    gradience: float = 0.0

    # 🔥 修复: 原代码 find_prototypes 用了 prototypicality 但字段不存在
    # 你也可以把它当作 “原型程度/典型性”
    prototypicality: float = 0.5


@dataclass
class EdgeMetadata:
    # 基础属性
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    ttl: int = 100
    provenance: str = "perception"
    ephemeral: bool = False
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    # 语义关系属性
    relation_type: str = "associative"
    is_symmetric: bool = False
    is_transitive: bool = False
    strength: float = 0.5
    directionality: str = "directed"   # directed / undirected / bidirectional
    semantic_role: Optional[str] = None


# ----------------------------
# Node / Edge
# ----------------------------

@dataclass
class Node:
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    clip_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "properties": dict(self.properties),
            "metadata": asdict(self.metadata),
            "clip_refs": list(self.clip_refs),
        }

    def __repr__(self) -> str:
        props_str = ", ".join(f"{k}={v}" for k, v in list(self.properties.items())[:3])
        return f"Node({self.label}, {props_str})"


@dataclass
class Edge:
    id: str
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: EdgeMetadata = field(default_factory=EdgeMetadata)
    clip_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "properties": dict(self.properties),
            "metadata": asdict(self.metadata),
            "clip_refs": list(self.clip_refs),
        }

    def __repr__(self) -> str:
        return f"Edge({self.source} --{self.relation}--> {self.target})"


# ----------------------------
# Unified Memory Graph (Enhanced)
# ----------------------------

class UnifiedMemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}

        self.label_index: Dict[str, Set[str]] = defaultdict(set)
        self.out_edges: Dict[str, Set[str]] = defaultdict(set)
        self.in_edges: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)

        self.current_step: int = 0

        # 🔥 可配置：不同语义关系类型的默认优先级（用于检索/扩展排序）
        # 你可以按任务类型动态调整
        self.relation_type_weight: Dict[str, float] = {
            "taxonomic": 1.00,
            "partitive": 0.95,
            "synonymy": 0.90,
            "functional": 0.85,
            "causal": 0.80,
            "associative": 0.65,
            "spatial": 0.55,
            "temporal": 0.50,
            "metaphorical": 0.40,
            "antonymy": 0.30,
        }

        # 常见 relation 名称到 relation_type 的轻量映射（兜底）
        self.relation_name_fallback_type: Dict[str, str] = {
            "is_a": "taxonomic",
            "subclass_of": "taxonomic",
            "instance_of": "taxonomic",
            "has_part": "partitive",
            "part_of": "partitive",
            "synonym_of": "synonymy",
            "antonym_of": "antonymy",
            "opposite_of": "antonymy",
            "metaphor_of": "metaphorical",
            "symbolizes": "metaphorical",
            "causes": "causal",
            "enables": "causal",
            "prevents": "causal",
            "before": "temporal",
            "after": "temporal",
            "during": "temporal",
            "on": "spatial",
            "in": "spatial",
            "near": "spatial",
            "left_of": "spatial",
            "right_of": "spatial",
            "located_at": "spatial",
            "has_color": "associative",
            "used_for": "functional",
            "has_function": "functional",
        }

    # ----------------------------
    # Helper: normalization + similarity
    # ----------------------------

    @staticmethod
    def _norm_label(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[_\-\s]+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s

    @staticmethod
    def _tokenize(s: str) -> Set[str]:
        s = UnifiedMemoryGraph._norm_label(s)
        return set([t for t in s.split(" ") if t])

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / max(union, 1)

    def _label_similarity(self, query: str, cand: str) -> float:
        """
        无 embedding 的简易语义相似：substring + token jaccard
        """
        q = self._norm_label(query)
        c = self._norm_label(cand)
        if not q or not c:
            return 0.0
        if q == c:
            return 1.0
        if q in c or c in q:
            # 子串匹配比 jaccard 更强
            return 0.85
        return 0.65 * self._jaccard(self._tokenize(q), self._tokenize(c))

    def _touch_node(self, node: Node) -> None:
        node.metadata.last_access = time.time()
        node.metadata.access_count += 1

    def _touch_edge(self, edge: Edge) -> None:
        edge.metadata.last_access = time.time()
        edge.metadata.access_count += 1

    def _edge_effective_type(self, edge: Edge) -> str:
        # 优先用 metadata.relation_type，其次用 relation 名称兜底
        if edge.metadata.relation_type:
            return edge.metadata.relation_type
        return self.relation_name_fallback_type.get(edge.relation, "associative")

    # ----------------------------
    # Helper: metadata match (keep yours but add small improvements)
    # ----------------------------

    def _match_metadata(
        self,
        metadata: Union[NodeMetadata, EdgeMetadata],
        filters: Dict[str, Any]
    ) -> bool:
        for key, value in filters.items():
            if not hasattr(metadata, key):
                return False
            attr_value = getattr(metadata, key)

            if isinstance(value, tuple) and len(value) == 2:
                operator, threshold = value
                if operator == ">":
                    if not (attr_value > threshold):
                        return False
                elif operator == ">=":
                    if not (attr_value >= threshold):
                        return False
                elif operator == "<":
                    if not (attr_value < threshold):
                        return False
                elif operator == "<=":
                    if not (attr_value <= threshold):
                        return False
                elif operator == "==":
                    if not (attr_value == threshold):
                        return False
                elif operator == "!=":
                    if not (attr_value != threshold):
                        return False
                else:
                    logger.warning(f"[UnifiedGraph] Unknown operator: {operator}")
                    return False
            else:
                if attr_value != value:
                    return False
        return True

    # ----------------------------
    # Node CRUD (mostly same)
    # ----------------------------

    def add_node(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[NodeMetadata] = None,
        clip_refs: Optional[List[str]] = None,
        node_id: Optional[str] = None,
    ) -> Node:
        if node_id is None:
            # 🔥 修复：直接用 label_index 查找，避免循环调用 find_nodes
            existing_ids = self.label_index.get(label, set())
            if existing_ids:
                node_id = list(existing_ids)[0]
                logger.debug(f"[UnifiedGraph] Merging into existing node: {node_id}")

        if node_id and node_id in self.nodes:
            node = self.nodes[node_id]
            if properties:
                node.properties.update(properties)
            if metadata:
                node.metadata = metadata
            if clip_refs:
                # 🔥 修复：去重 clip_refs，保持顺序
                existing_refs = set(node.clip_refs)
                for ref in clip_refs:
                    if ref not in existing_refs:
                        node.clip_refs.append(ref)
                        existing_refs.add(ref)
            self._touch_node(node)
            logger.debug(f"[UnifiedGraph] Updated node: {node}")
        else:
            node_id = node_id or _generate_id("n")
            node = Node(
                id=node_id,
                label=label,
                properties=properties or {},
                metadata=metadata or NodeMetadata(timestamp=time.time()),
                clip_refs=clip_refs or [],
            )
            self.nodes[node_id] = node
            self.label_index[label].add(node_id)
            self._touch_node(node)
            logger.debug(f"[UnifiedGraph] Added node: {node}")
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        n = self.nodes.get(node_id)
        if n:
            self._touch_node(n)
        return n

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        for edge_id in list(self.out_edges.get(node_id, [])):
            self.remove_edge(edge_id)
        for edge_id in list(self.in_edges.get(node_id, [])):
            self.remove_edge(edge_id)

        self.label_index[node.label].discard(node_id)
        del self.nodes[node_id]
        logger.debug(f"[UnifiedGraph] Removed node: {node_id}")
        return True

    # ----------------------------
    # Edge CRUD (mostly same)
    # ----------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[EdgeMetadata] = None,
        clip_refs: Optional[List[str]] = None,
        edge_id: Optional[str] = None,
    ) -> Edge:
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"[UnifiedGraph] Cannot add edge: node not found (source={source}, target={target})")
            raise ValueError(f"Source or target node not found: {source}, {target}")

        existing_edges = self.find_edges(source=source, target=target, relation=relation)
        if existing_edges and edge_id is None:
            edge = existing_edges[0]
            if properties:
                edge.properties.update(properties)
            if metadata:
                edge.metadata = metadata
            if clip_refs:
                edge.clip_refs.extend(clip_refs)
            self._touch_edge(edge)
            logger.debug(f"[UnifiedGraph] Updated edge: {edge}")
            return edge

        edge_id = edge_id or _generate_id("e")
        edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            relation=relation,
            properties=properties or {},
            metadata=metadata or EdgeMetadata(timestamp=time.time()),
            clip_refs=clip_refs or [],
        )

        self.edges[edge_id] = edge
        self.out_edges[source].add(edge_id)
        self.in_edges[target].add(edge_id)
        self.relation_index[relation].add(edge_id)

        self._touch_edge(edge)
        logger.debug(f"[UnifiedGraph] Added edge: {edge}")
        return edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        e = self.edges.get(edge_id)
        if e:
            self._touch_edge(e)
        return e

    def remove_edge(self, edge_id: str) -> bool:
        if edge_id not in self.edges:
            return False

        edge = self.edges[edge_id]
        self.out_edges[edge.source].discard(edge_id)
        self.in_edges[edge.target].discard(edge_id)
        self.relation_index[edge.relation].discard(edge_id)

        del self.edges[edge_id]
        logger.debug(f"[UnifiedGraph] Removed edge: {edge_id}")
        return True

    # ----------------------------
    # 🔥 Enhanced Node Search: semantic ranking with cognitive attrs
    # ----------------------------

    def _node_prior_score(
        self,
        node: Node,
        perspective_bias: Optional[str] = None
    ) -> float:
        """
        把认知属性真正用起来：
        - confidence / centrality / family_resemblance / prototypicality 强化
        - vagueness 对具体概念是惩罚，对模糊查询可放宽（这里先默认惩罚一点）
        - perspective_bias 匹配时加分
        """
        m = node.metadata
        base = 0.40 * m.confidence + 0.25 * m.centrality + 0.15 * m.family_resemblance + 0.20 * m.prototypicality
        vag_penalty = 0.15 * m.vagueness
        score = base - vag_penalty

        if perspective_bias and m.perspective:
            if self._norm_label(m.perspective) == self._norm_label(perspective_bias):
                score += 0.10  # 视角匹配奖励
        return max(0.0, min(1.2, score))

    def _edge_weight_score(self, edge: Edge) -> float:
        """
        用 relation_type / strength / confidence 做边权重
        """
        t = self._edge_effective_type(edge)
        type_w = self.relation_type_weight.get(t, 0.6)
        m = edge.metadata
        # directionality: undirected/bidirectional 更“稳”的关联，可略加分
        dir_bonus = 0.05 if m.directionality in ("undirected", "bidirectional") else 0.0
        return max(
            0.0,
            min(1.5, type_w * (0.55 * m.strength + 0.35 * m.confidence + 0.10) + dir_bonus)
        )

    def semantic_find_nodes(
        self,
        query_label: str,
        *,
        exact: bool = False,
        property_filters: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        perspective_bias: Optional[str] = None,
        allow_synonym_expansion: bool = True,
        allow_taxonomic_expansion: bool = True,
    ) -> List[Tuple[Node, float]]:
        """
        返回带分数的节点检索结果：
        score = label_similarity * 语义扩展加成 + prior(认知属性)
        """
        if not query_label:
            return []

        # 1) 初始候选：label_index 模糊/精确
        base_candidates: Set[str] = set()
        if exact:
            base_candidates = set(self.label_index.get(query_label, set()))
        else:
            qn = self._norm_label(query_label)
            for lbl, ids in self.label_index.items():
                if qn in self._norm_label(lbl):
                    base_candidates.update(ids)

        # 🔥 修复：如果是精确匹配，不进行语义扩展
        if exact:
            allow_synonym_expansion = False
            allow_taxonomic_expansion = False

        # 2) 扩展候选：同义 / 上下位（用边的语义属性）
        expanded_candidates: Set[str] = set(base_candidates)

        # 为了扩展，需要找到“最像 query 的一批节点”作为 anchor
        anchors = []
        for nid in base_candidates:
            n = self.nodes[nid]
            anchors.append((nid, self._label_similarity(query_label, n.label)))
        anchors.sort(key=lambda x: x[1], reverse=True)
        anchor_ids = [nid for nid, _ in anchors[:5]]

        if allow_synonym_expansion:
            # 沿 synonymy（is_symmetric=True）扩展
            for aid in anchor_ids:
                for eid in self.out_edges.get(aid, set()):
                    e = self.edges[eid]
                    if self._edge_effective_type(e) == "synonymy" or e.metadata.is_symmetric:
                        expanded_candidates.add(e.target)
                for eid in self.in_edges.get(aid, set()):
                    e = self.edges[eid]
                    if self._edge_effective_type(e) == "synonymy" or e.metadata.is_symmetric:
                        expanded_candidates.add(e.source)

        if allow_taxonomic_expansion:
            # 沿 taxonomic / partitive 做一跳扩展（不做闭包，闭包用 infer_transitive_closure_enhanced）
            for aid in anchor_ids:
                for eid in self.out_edges.get(aid, set()):
                    e = self.edges[eid]
                    t = self._edge_effective_type(e)
                    if t in ("taxonomic", "partitive") and (e.metadata.strength >= 0.5):
                        expanded_candidates.add(e.target)
                for eid in self.in_edges.get(aid, set()):
                    e = self.edges[eid]
                    t = self._edge_effective_type(e)
                    if t in ("taxonomic", "partitive") and (e.metadata.strength >= 0.5):
                        expanded_candidates.add(e.source)

        # 3) 过滤 + 打分
        scored: List[Tuple[Node, float]] = []
        for nid in expanded_candidates:
            node = self.nodes[nid]

            if property_filters:
                ok = True
                for k, v in property_filters.items():
                    if node.properties.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            if metadata_filters and not self._match_metadata(node.metadata, metadata_filters):
                continue

            sim = self._label_similarity(query_label, node.label)

            # gradience：如果节点是连续属性（gradience高），对“近似匹配”放宽一点
            # 例如 “reddish” vs “red”，token overlap 低但可接受
            if node.metadata.gradience >= 0.5 and sim < 0.6:
                sim = min(0.7, sim + 0.10)

            prior = self._node_prior_score(node, perspective_bias=perspective_bias)

            # 同义扩展来源：给一个小 bonus（通过 edge 扩展进来的节点通常更可信）
            bonus = 0.0
            if nid not in base_candidates:
                bonus += 0.05

            score = 0.60 * sim + 0.40 * prior + bonus
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # 保留原 find_nodes 语义，但内部可以复用 semantic_find_nodes
    def find_nodes(
        self,
        label: Optional[str] = None,
        exact: bool = False,
        property_filters: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Node]:
        if label is None:
            # 原逻辑：全量过滤
            candidates = set(self.nodes.keys())
            results: List[Node] = []
            for nid in candidates:
                node = self.nodes[nid]
                if property_filters:
                    if not all(node.properties.get(k) == v for k, v in property_filters.items()):
                        continue
                if metadata_filters and not self._match_metadata(node.metadata, metadata_filters):
                    continue
                results.append(node)
                if len(results) >= limit:
                    break
            return results

        ranked = self.semantic_find_nodes(
            query_label=label,
            exact=exact,
            property_filters=property_filters,
            metadata_filters=metadata_filters,
            limit=limit,
        )
        return [n for n, _ in ranked]

    # ----------------------------
    # Edge search (keep yours)
    # ----------------------------

    def find_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relation: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Edge]:
        candidates: Set[str] = set(self.edges.keys())
        if source:
            candidates &= self.out_edges.get(source, set())
        if target:
            candidates &= self.in_edges.get(target, set())
        if relation:
            candidates &= self.relation_index.get(relation, set())

        results: List[Edge] = []
        for eid in list(candidates):
            edge = self.edges[eid]
            if metadata_filters and not self._match_metadata(edge.metadata, metadata_filters):
                continue
            results.append(edge)
            if len(results) >= limit:
                break
        return results

    # ----------------------------
    # 🔥 Enhanced neighbors: respect directionality / symmetry / semantic_role
    # ----------------------------

    def get_neighbors_semantic(
        self,
        node_id: str,
        *,
        relation: Optional[str] = None,
        relation_type: Optional[str] = None,
        semantic_role: Optional[str] = None,
        direction: str = "out",  # out/in/both
        min_strength: float = 0.0,
        include_symmetric_back: bool = True,
        limit: int = 50,
    ) -> List[Tuple[Node, Edge, float]]:
        """
        返回 (neighbor_node, edge, edge_score)，并用语义属性过滤与排序。
        """
        if node_id not in self.nodes:
            return []

        candidates: List[Tuple[Node, Edge, float]] = []

        def consider_edge(edge: Edge, neighbor_id: str) -> None:
            if relation and edge.relation != relation:
                return
            t = self._edge_effective_type(edge)
            if relation_type and t != relation_type:
                return
            if semantic_role and edge.metadata.semantic_role != semantic_role:
                return
            if edge.metadata.strength < min_strength:
                return

            w = self._edge_weight_score(edge)
            neighbor = self.nodes[neighbor_id]
            candidates.append((neighbor, edge, w))

        # 出边
        if direction in ("out", "both"):
            for eid in self.out_edges.get(node_id, set()):
                e = self.edges[eid]
                consider_edge(e, e.target)

        # 入边（directionality=directed 时，很多关系不希望反向用；这里给个保守策略）
        if direction in ("in", "both"):
            for eid in self.in_edges.get(node_id, set()):
                e = self.edges[eid]
                # 若是严格 directed 且非对称关系，反向使用会污染语义
                if e.metadata.directionality == "directed" and not (include_symmetric_back and e.metadata.is_symmetric):
                    continue
                consider_edge(e, e.source)

        # 对称关系补全：如果边标了 is_symmetric，就允许 “反向邻居”
        if include_symmetric_back:
            # 这里已在 in/both 中处理了，out-only 场景也补一下
            if direction == "out":
                for eid in self.in_edges.get(node_id, set()):
                    e = self.edges[eid]
                    if e.metadata.is_symmetric:
                        consider_edge(e, e.source)

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:limit]

    # ----------------------------
    # 🔥 Enhanced traverse: semantic traversal using relation_type/transitivity
    # ----------------------------

    def traverse_semantic(
        self,
        start_node: str,
        *,
        max_depth: int = 3,
        allowed_relation_types: Optional[Set[str]] = None,
        allowed_relations: Optional[Set[str]] = None,
        prefer_perspective: Optional[str] = None,
        # spatial/temporal等动态关系是否只取一跳不扩张
        dynamic_types_one_hop: Set[str] = frozenset({"spatial", "temporal"}),
        # 过滤：边强度阈值
        min_edge_strength: float = 0.0,
        # 是否跟随可传递边做闭包式扩张
        expand_transitive: bool = True,
        # 对称边是否双向扩张
        expand_symmetric: bool = True,
        # 结果数量限制
        limit: int = 200,
    ) -> List[Node]:
        if start_node not in self.nodes:
            return []

        visited: Set[str] = set()
        results: List[Node] = []

        q = deque([(start_node, 0)])
        while q and len(results) < limit:
            nid, depth = q.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)

            node = self.nodes[nid]
            self._touch_node(node)
            results.append(node)

            if depth == max_depth:
                continue

            # 邻居候选：先收集，再按 edge_score+node_prior 排序
            neighs: List[Tuple[str, float, Edge]] = []

            # 出边
            for eid in self.out_edges.get(nid, set()):
                e = self.edges[eid]
                t = self._edge_effective_type(e)

                if allowed_relations and e.relation not in allowed_relations:
                    continue
                if allowed_relation_types and t not in allowed_relation_types:
                    continue
                if e.metadata.strength < min_edge_strength:
                    continue

                # dynamic one-hop：spatial/temporal 默认不再扩张到更深层
                if t in dynamic_types_one_hop and depth >= 1:
                    continue

                # transitive control：如果不允许 expand_transitive，则跳过 transitive 边的深层扩张
                if (not expand_transitive) and e.metadata.is_transitive and depth >= 1:
                    continue

                w = self._edge_weight_score(e)
                neighs.append((e.target, w, e))

                # 对称边：补反向
                if expand_symmetric and e.metadata.is_symmetric:
                    neighs.append((e.source, w * 0.95, e))

            # 入边（保守：只用 bidirectional/undirected 或 symmetric）
            for eid in self.in_edges.get(nid, set()):
                e = self.edges[eid]
                t = self._edge_effective_type(e)

                if allowed_relations and e.relation not in allowed_relations:
                    continue
                if allowed_relation_types and t not in allowed_relation_types:
                    continue
                if e.metadata.strength < min_edge_strength:
                    continue

                if e.metadata.directionality == "directed" and not (expand_symmetric and e.metadata.is_symmetric):
                    continue

                if t in dynamic_types_one_hop and depth >= 1:
                    continue

                if (not expand_transitive) and e.metadata.is_transitive and depth >= 1:
                    continue

                w = self._edge_weight_score(e)
                neighs.append((e.source, w * 0.9, e))

            # 排序：边权 + 邻居节点 prior（centrality/视角/置信度等）
            def score_neighbor(item: Tuple[str, float, Edge]) -> float:
                nb_id, ew, _e = item
                nb = self.nodes.get(nb_id)
                if not nb:
                    return -1e9
                prior = self._node_prior_score(nb, perspective_bias=prefer_perspective)
                return 0.65 * ew + 0.35 * prior

            neighs.sort(key=score_neighbor, reverse=True)

            for nb_id, _, _ in neighs[:50]:
                if nb_id not in visited:
                    q.append((nb_id, depth + 1))

        return results

    # ----------------------------
    # 🔥 Enhanced transitive closure: respect is_transitive + relation_type
    # ----------------------------

    def infer_transitive_closure_enhanced(
        self,
        start_node: str,
        relation: Optional[str] = None,
        relation_type: Optional[str] = None,
        max_depth: int = 5,
        min_strength: float = 0.0,
    ) -> List[Node]:
        """
        更严格的闭包推理：
        - 仅沿 is_transitive=True 的边扩张
        - 可指定 relation 或 relation_type（建议 relation_type="taxonomic"/"partitive"/"causal"）
        """
        if start_node not in self.nodes:
            return []

        visited: Set[str] = set()
        q = deque([(start_node, 0)])
        out: List[Node] = []

        while q:
            nid, depth = q.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            out.append(self.nodes[nid])

            if depth == max_depth:
                continue

            # 只沿 transitive 边扩张
            for eid in self.out_edges.get(nid, set()):
                e = self.edges[eid]
                if not e.metadata.is_transitive:
                    continue
                if e.metadata.strength < min_strength:
                    continue
                if relation and e.relation != relation:
                    continue
                t = self._edge_effective_type(e)
                if relation_type and t != relation_type:
                    continue
                q.append((e.target, depth + 1))

        return out

    # ----------------------------
    # Advanced query methods (fix + use cognitive attrs)
    # ----------------------------

    def find_central_concepts(self, threshold: float = 0.7, limit: int = 10) -> List[Node]:
        nodes = self.find_nodes(metadata_filters={"centrality": (">", threshold)}, limit=limit)
        nodes.sort(key=lambda n: n.metadata.centrality, reverse=True)
        return nodes

    def find_prototypes(self, label_pattern: str = None, threshold: float = 0.7) -> List[Node]:
        """
        原型：优先 prototypicality，其次 family_resemblance + centrality
        """
        candidates = self.find_nodes(label=label_pattern) if label_pattern else list(self.nodes.values())
        filtered = [n for n in candidates if n.metadata.prototypicality >= threshold]
        filtered.sort(key=lambda n: (n.metadata.prototypicality, n.metadata.family_resemblance, n.metadata.centrality), reverse=True)
        return filtered

    def find_vague_concepts(self, threshold: float = 0.5) -> List[Node]:
        return self.find_nodes(metadata_filters={"vagueness": (">", threshold)})

    def find_gradient_attributes(self, threshold: float = 0.5) -> List[Node]:
        return self.find_nodes(metadata_filters={"gradience": (">", threshold)})

    def find_symmetric_relations(self) -> List[Edge]:
        return self.find_edges(metadata_filters={"is_symmetric": True})

    def find_transitive_relations(self, relation_type: str = None) -> List[Edge]:
        filters = {"is_transitive": True}
        if relation_type:
            filters["relation_type"] = relation_type
        return self.find_edges(metadata_filters=filters)

    def find_strong_relations(self, threshold: float = 0.8, relation_type: str = None) -> List[Edge]:
        filters = {"strength": (">", threshold)}
        if relation_type:
            filters["relation_type"] = relation_type
        return self.find_edges(metadata_filters=filters)

    # ----------------------------
    # Maintenance / Export / Import / Stats (keep yours)
    # ----------------------------

    def prune(self, current_step: int) -> Tuple[int, int]:
        self.current_step = current_step
        nodes_removed = 0
        edges_removed = 0

        expired_nodes = []
        for node_id, node in self.nodes.items():
            if "ltm" in node.metadata.provenance.lower():
                continue
            age = current_step - node.metadata.timestamp
            if age > node.metadata.ttl:
                expired_nodes.append(node_id)

        for node_id in expired_nodes:
            self.remove_node(node_id)
            nodes_removed += 1

        orphan_edges = []
        for edge_id, edge in self.edges.items():
            if edge.source not in self.nodes or edge.target not in self.nodes:
                orphan_edges.append(edge_id)

        for edge_id in orphan_edges:
            self.remove_edge(edge_id)
            edges_removed += 1

        logger.info(f"[UnifiedGraph] Pruned: {nodes_removed} nodes, {edges_removed} edges")
        return nodes_removed, edges_removed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
            "current_step": self.current_step,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.label_index.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.relation_index.clear()

        for nid, node_data in data.get("nodes", {}).items():
            node = Node(
                id=node_data["id"],
                label=node_data["label"],
                properties=node_data.get("properties", {}),
                metadata=NodeMetadata(**node_data.get("metadata", {})),
                clip_refs=node_data.get("clip_refs", []),
            )
            self.nodes[nid] = node
            self.label_index[node.label].add(nid)

        for eid, edge_data in data.get("edges", {}).items():
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                relation=edge_data["relation"],
                properties=edge_data.get("properties", {}),
                metadata=EdgeMetadata(**edge_data.get("metadata", {})),
                clip_refs=edge_data.get("clip_refs", []),
            )
            self.edges[eid] = edge
            self.out_edges[edge.source].add(eid)
            self.in_edges[edge.target].add(eid)
            self.relation_index[edge.relation].add(eid)

        self.current_step = data.get("current_step", 0)

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "labels": len(self.label_index),
            "relations": len(self.relation_index),
            "avg_out_degree": sum(len(e) for e in self.out_edges.values()) / max(len(self.nodes), 1),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"UnifiedGraph(nodes={s['nodes']}, edges={s['edges']}, labels={s['labels']})"
