import time
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class RetrievalConfig:
    # Anchor候选召回
    top_k: int = 10
    require_ltm_provenance: bool = True  # 仅从 provenance 含 ltm 的节点中检索

    # 子图展开
    max_hops: int = 2
    max_nodes: int = 60
    max_edges: int = 120

    # 关系优先级（小=更重要）
    relation_priority: Dict[str, int] = field(default_factory=lambda: {
        # taxonomic / stable
        "is_a": 0, "subclass_of": 0, "instance_of": 0,
        # stable attributes
        "has_color": 0, "made_of": 0, "has_part": 0, "part_of": 0,
        "used_for": 1, "has_function": 1,
        # synonym
        "synonym_of": 1,
        # spatial / dynamic
        "located_at": 2, "on": 2, "in": 2, "near": 2, "left_of": 2, "right_of": 2,
        "adjacent": 2
    })

    # 哪些关系算“位置候选”
    location_relations: Set[str] = field(default_factory=lambda: {
        "located_at", "on", "in", "near"
    })

    # 哪些 relation_type 算“稳定关系类型”
    stable_relation_types: Set[str] = field(default_factory=lambda: {
        "taxonomic", "partitive", "functional", "synonymy", "associative"
    })

    # 哪些 relation_type 算“动态/空间关系”
    dynamic_relation_types: Set[str] = field(default_factory=lambda: {
        "spatial", "temporal"
    })

    # 动态关系时间衰减
    time_decay_tau_sec: float = 600.0  # 10分钟示例，可按任务节奏改

    # 是否将 LTM 子图复制进 WM（ephemeral证据）
    import_subgraph_into_wm: bool = True
    imported_node_provenance: str = "ltm_read"
    imported_edge_provenance: str = "ltm_read"
    imported_ephemeral: bool = True

    # Multi-hit聚合参数
    use_noisy_or: bool = True

    # 一致性加分（利用 WM 边）
    use_wm_edge_consistency: bool = True
    edge_consistency_weight: float = 0.15  # 对 anchor 最终分的加成权重

    # 写回字段名
    writeback_key: str = "ltm_retrieval"


# ------------------------------------------------------------
# Helper scoring
# ------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _time_decay(ts: float, now: float, tau: float) -> float:
    dt = max(0.0, now - ts)
    return math.exp(-dt / max(1e-6, tau))

def _noisy_or(scores: List[float]) -> float:
    p = 1.0
    for s in scores:
        p *= (1.0 - _clamp01(s))
    return 1.0 - p

def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def _flatten_props(props: Dict[str, Any]) -> Set[str]:
    """
    将 properties 粗暴扁平化为 token 集合，用于简单相似度
    """
    tokens: Set[str] = set()
    for k, v in (props or {}).items():
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


# ------------------------------------------------------------
# MemoryRetriever
# ------------------------------------------------------------

class MemoryRetriever:
    """
    用你的 UnifiedMemoryGraph 做：
    - 输入 WM 节点 -> 在 LTM 中找到可能的 anchor -> 展开子图 -> 汇总为 bundle -> 写回 WM
    - 批量：multi-hit 聚合 conf，回填到每个 WM 节点
    """

    def __init__(self, ltm_graph, config: Optional[RetrievalConfig] = None):
        self.ltm = ltm_graph
        self.cfg = config or RetrievalConfig()

    # -------------------------
    # Public APIs
    # -------------------------

    def retrieve_one(self, wm_graph, wm_node_id: str) -> Dict[str, Any]:
        """
        对单个 WM node 执行检索并写回 WM。
        返回写回的 bundle。
        """
        wm_node = wm_graph.get_node(wm_node_id)
        if wm_node is None:
            return {"error": "wm_node_not_found", "wm_node_id": wm_node_id}

        now = time.time()
        anchors = self.anchor_candidates(wm_node)

        if not anchors:
            bundle = {
                "anchor_candidates": [],
                "selected_anchor": None,
                "base_score": 0.0,
                "subgraph": {"nodes": [], "edges": []},
                "stable_facts": {},
                "likely_locations": [],
                "ts": now,
                "note": "no anchor found in ltm"
            }
            wm_node.properties[self.cfg.writeback_key] = bundle
            return bundle

        # 先用 base_score 选 top-1
        selected_anchor_id, base_score, explain = anchors[0]

        # 可选：用 WM 边一致性做轻量 re-rank（需要 WM 邻居的检索结果，所以通常在 batch 里做更好）
        # 这里先不改选中 anchor，只记录候选。
        subgraph = self.expand_subgraph(selected_anchor_id, now)
        stable_facts = self.extract_stable_facts(selected_anchor_id, subgraph)
        likely_locations = self.extract_locations(selected_anchor_id, now)

        bundle = {
            "anchor_candidates": [
                {"ltm_node_id": a_id, "score": float(sc), "evidence": ev}
                for (a_id, sc, ev) in anchors
            ],
            "selected_anchor": selected_anchor_id,
            "base_score": float(base_score),
            "subgraph": subgraph,
            "stable_facts": stable_facts,
            "likely_locations": likely_locations,
            "aggregated_confidence": float(base_score),  # batch 后会回填
            "supporting_wm_nodes_for_anchor": [],
            "ts": now
        }

        # 写回 WM
        wm_node.properties[self.cfg.writeback_key] = bundle

        # 可选：把子图拷进 WM（作为 ephemeral 证据）
        if self.cfg.import_subgraph_into_wm:
            self.import_subgraph_into_wm(wm_graph, subgraph)

        return bundle

    def retrieve_batch(self, wm_graph, wm_node_ids: List[str]) -> Dict[str, Any]:
        """
        批量检索：
        1) 对每个 WM node 做 retrieve_one（写回 bundle）
        2) multi-hit 聚合：多个 WM node 命中同一 LTM anchor -> aggregated_confidence 提升
        3) 可选：用 WM 边一致性对 selected_anchor 做加分与重排（轻量版本）
        """
        # 1) 单个检索
        for wid in wm_node_ids:
            self.retrieve_one(wm_graph, wid)

        # 2) multi-hit聚合
        anchor_support_scores = defaultdict(list)   # anchor -> [base_scores...]
        anchor_support_nodes = defaultdict(list)    # anchor -> [wm_ids...]

        for wid in wm_node_ids:
            w = wm_graph.get_node(wid)
            if not w:
                continue
            b = w.properties.get(self.cfg.writeback_key, {})
            a = b.get("selected_anchor")
            sc = float(b.get("base_score", 0.0))
            if a:
                anchor_support_scores[a].append(sc)
                anchor_support_nodes[a].append(wid)

        aggregated = {}
        for a, scores in anchor_support_scores.items():
            aggregated[a] = _noisy_or(scores) if self.cfg.use_noisy_or else _clamp01(sum(scores) / max(1, len(scores)))

        # 3) 回填到每个 WM node
        for wid in wm_node_ids:
            w = wm_graph.get_node(wid)
            if not w:
                continue
            b = w.properties.get(self.cfg.writeback_key, {})
            a = b.get("selected_anchor")
            if not a:
                continue
            b["aggregated_confidence"] = float(aggregated.get(a, b.get("base_score", 0.0)))
            b["supporting_wm_nodes_for_anchor"] = anchor_support_nodes.get(a, [])
            w.properties[self.cfg.writeback_key] = b

        # 4) 可选：利用 WM 边一致性，对 anchor 进行轻量加分/调整
        if self.cfg.use_wm_edge_consistency:
            self.apply_wm_edge_consistency(wm_graph, wm_node_ids)

        return {"status": "ok", "num_nodes": len(wm_node_ids)}

    # -------------------------
    # Anchor candidates
    # -------------------------
    def _is_ltm_node(self, node) -> bool:
        """判断是否 LTM 节点：provenance 里包含 ltm 即可（兼容 ltm_seed / ltm_write / ltm_consolidated 等）"""
        if not node or not getattr(node, "metadata", None):
            return False
        prov = str(getattr(node.metadata, "provenance", "") or "").lower()
        return ("ltm" in prov)

    def anchor_candidates(self, wm_node) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        返回 [(ltm_node_id, score, evidence_dict)] 按 score 降序
        - 修复点：不再用 metadata_filters={"provenance":"ltm"} 这种严格匹配
        - 改为：候选先召回，再用 provenance contains 'ltm' 过滤
        """
        candidates: List[Any] = []

        # 1) exact label
        exact = self.ltm.find_nodes(label=wm_node.label, exact=True, limit=self.cfg.top_k)
        # 过滤 provenance
        exact = [n for n in exact if (not self.cfg.require_ltm_provenance) or self._is_ltm_node(n)]
        candidates.extend(exact)

        # 2) fuzzy label（补充）
        if len(candidates) < self.cfg.top_k:
            fuzzy = self.ltm.find_nodes(label=wm_node.label, exact=False, limit=self.cfg.top_k)
            fuzzy = [n for n in fuzzy if (not self.cfg.require_ltm_provenance) or self._is_ltm_node(n)]
            exist = {x.id for x in candidates}
            candidates.extend([n for n in fuzzy if n.id not in exist])

        # 3) synonym 扩展（保持你原逻辑）
        synonym_labels = set()
        for n in exact:
            for e_id in self.ltm.out_edges.get(n.id, []):
                e = self.ltm.get_edge(e_id)
                if e and e.relation == "synonym_of":
                    nb = self.ltm.get_node(e.target)
                    if nb:
                        synonym_labels.add(nb.label)
            for e_id in self.ltm.in_edges.get(n.id, []):
                e = self.ltm.get_edge(e_id)
                if e and e.relation == "synonym_of":
                    nb = self.ltm.get_node(e.source)
                    if nb:
                        synonym_labels.add(nb.label)

        for sl in list(synonym_labels)[:3]:
            syn_nodes = self.ltm.find_nodes(label=sl, exact=True, limit=5)
            syn_nodes = [n for n in syn_nodes if (not self.cfg.require_ltm_provenance) or self._is_ltm_node(n)]
            exist = {x.id for x in candidates}
            for sn in syn_nodes:
                if sn.id not in exist:
                    candidates.append(sn)

        # ---- 下面打分逻辑保持不变 ----
        wm_tokens = _flatten_props(wm_node.properties)
        results: List[Tuple[str, float, Dict[str, Any]]] = []

        for n in candidates[: self.cfg.top_k * 3]:
            ltm_tokens = _flatten_props(n.properties)

            if n.label == wm_node.label:
                name_score = 1.0
            elif wm_node.label.lower() in n.label.lower() or n.label.lower() in wm_node.label.lower():
                name_score = 0.75
            elif n.label in synonym_labels:
                name_score = 0.70
            else:
                name_score = 0.55

            prop_score = _jaccard(wm_tokens, ltm_tokens)
            prior = _clamp01(n.metadata.confidence)
            score = _clamp01(0.45 * name_score + 0.35 * prop_score + 0.20 * prior)

            evidence = {
                "name_score": float(name_score),
                "prop_score": float(prop_score),
                "prior_confidence": float(prior),
                "wm_label": wm_node.label,
                "ltm_label": n.label,
            }
            results.append((n.id, score, evidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.cfg.top_k]

    # -------------------------
    # Subgraph expansion
    # -------------------------

    def expand_subgraph(self, anchor_id: str, now: float) -> Dict[str, Any]:
        """
        从 LTM anchor 展开一个受控子图（nodes+edges），用于写回 WM。
        """
        visited: Set[str] = set([anchor_id])
        queue: List[Tuple[str, int]] = [(anchor_id, 0)]

        nodes_out: Dict[str, Dict[str, Any]] = {}
        edges_out: Dict[str, Dict[str, Any]] = {}

        anchor_node = self.ltm.get_node(anchor_id)
        if not anchor_node:
            return {"nodes": [], "edges": []}
        nodes_out[anchor_id] = anchor_node.to_dict()

        def edge_prio(edge) -> Tuple[int, float]:
            # priority: relation_priority -> then stronger/ more confident first
            p = self.cfg.relation_priority.get(edge.relation, 1)
            # stronger first: use edge.metadata.strength & confidence
            strength = getattr(edge.metadata, "strength", 0.5)
            conf = edge.metadata.confidence
            return (p, -(0.6 * strength + 0.4 * conf))

        while queue:
            nid, depth = queue.pop(0)
            if depth >= self.cfg.max_hops:
                continue

            # 收集出边+入边
            eids = list(self.ltm.out_edges.get(nid, [])) + list(self.ltm.in_edges.get(nid, []))
            edges = []
            for eid in eids:
                e = self.ltm.get_edge(eid)
                if e:
                    edges.append(e)

            edges.sort(key=edge_prio)

            for e in edges:
                if len(edges_out) >= self.cfg.max_edges:
                    break
                edges_out[e.id] = e.to_dict()

                nb = e.target if e.source == nid else e.source
                if nb in visited:
                    continue

                if len(nodes_out) >= self.cfg.max_nodes:
                    continue

                # 是否允许继续扩展到下一跳：动态关系在 depth>=1 时不扩展
                rel_type = getattr(e.metadata, "relation_type", "associative")
                is_dynamic = (rel_type in self.cfg.dynamic_relation_types) or (e.relation in self.cfg.location_relations)

                allow_expand = True
                if depth + 1 >= 2 and is_dynamic:
                    allow_expand = False

                visited.add(nb)
                nb_node = self.ltm.get_node(nb)
                if nb_node:
                    nodes_out[nb] = nb_node.to_dict()

                if allow_expand:
                    queue.append((nb, depth + 1))

        return {"nodes": list(nodes_out.values()), "edges": list(edges_out.values())}

    # -------------------------
    # Extract stable facts & locations
    # -------------------------

    def extract_stable_facts(self, anchor_id: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """
        从子图中抽取稳定事实（例如：is_a / has_color / made_of / used_for）
        输出结构化摘要，供规划与识别使用。
        """
        # 把 nodes quick map
        node_map = {n["id"]: n for n in subgraph.get("nodes", [])}
        facts = defaultdict(list)

        for e in subgraph.get("edges", []):
            rel = e["relation"]
            meta = e.get("metadata", {})
            rel_type = meta.get("relation_type", "associative")

            # 只拿稳定关系
            if rel_type not in self.cfg.stable_relation_types:
                continue

            if e["source"] == anchor_id:
                tgt = node_map.get(e["target"], {})
                facts[rel].append({"target_id": e["target"], "target_label": tgt.get("label"), "conf": meta.get("confidence", 0.8)})
            elif e["target"] == anchor_id:
                src = node_map.get(e["source"], {})
                facts[f"inv:{rel}"].append({"source_id": e["source"], "source_label": src.get("label"), "conf": meta.get("confidence", 0.8)})

        return dict(facts)

    def extract_locations(self, anchor_id: str, now: float) -> List[Dict[str, Any]]:
        """
        从 LTM 直接抽取 anchor 的位置候选（on/in/located_at/near），并加时间衰减。
        """
        loc_scores = defaultdict(float)
        loc_evidence = defaultdict(list)

        for eid in self.ltm.out_edges.get(anchor_id, []):
            e = self.ltm.get_edge(eid)
            if not e:
                continue
            if e.relation not in self.cfg.location_relations:
                # 也支持 semantic_role=location 或 relation_type=spatial
                if e.metadata.semantic_role != "location" and e.metadata.relation_type != "spatial":
                    continue

            base = _clamp01(e.metadata.confidence)
            # 空间/动态做衰减
            decay = _time_decay(e.metadata.timestamp, now, self.cfg.time_decay_tau_sec)
            score = base * decay

            loc_scores[e.target] = max(loc_scores[e.target], score)
            loc_evidence[e.target].append({
                "relation": e.relation,
                "edge_confidence": float(e.metadata.confidence),
                "edge_timestamp": float(e.metadata.timestamp),
                "decay": float(decay)
            })

        locs = []
        for loc_id, sc in loc_scores.items():
            loc_node = self.ltm.get_node(loc_id)
            locs.append({
                "location_node_id": loc_id,
                "location_label": loc_node.label if loc_node else None,
                "confidence": float(_clamp01(sc)),
                "evidence": loc_evidence[loc_id]
            })
        locs.sort(key=lambda x: x["confidence"], reverse=True)
        return locs[:5]

    # -------------------------
    # Optional: import subgraph into WM as ephemeral evidence
    # -------------------------

    def import_subgraph_into_wm(self, wm_graph, subgraph: Dict[str, Any]) -> None:
        """
        将 LTM 子图复制进 WM 图，作为 ephemeral 证据节点/边。
        - 节点ID：用 "ltmcopy_<ltm_id>" 防冲突
        - 节点properties 增加 {"ltm_id": original_id}
        - provenance 标记为 ltm_read，ephemeral=True，避免同步回 LTM
        """
        id_map = {}  # ltm_node_id -> wm_node_id

        # 1) import nodes
        for n in subgraph.get("nodes", []):
            ltm_id = n["id"]
            wm_id = f"ltmcopy_{ltm_id}"
            id_map[ltm_id] = wm_id

            # 如果 WM 已经有这个copy节点，就更新
            props = dict(n.get("properties", {}))
            props["ltm_id"] = ltm_id
            props["ltm_label"] = n.get("label")

            meta = n.get("metadata", {}) or {}
            # 复制时：ephemeral + provenance
            from embodiedbench.evaluator.unified_graph import NodeMetadata  # 你项目里已有
            nm = NodeMetadata(
                confidence=float(meta.get("confidence", 0.8)),
                provenance=self.cfg.imported_node_provenance,
                ephemeral=self.cfg.imported_ephemeral,
                ttl=meta.get("ttl", 100),
            )
            wm_graph.add_node(label=n.get("label"), properties=props, metadata=nm, node_id=wm_id)

        # 2) import edges
        for e in subgraph.get("edges", []):
            src = id_map.get(e["source"])
            tgt = id_map.get(e["target"])
            if not src or not tgt:
                continue

            meta = e.get("metadata", {}) or {}
            from embodiedbench.evaluator.unified_graph import EdgeMetadata
            em = EdgeMetadata(
                confidence=float(meta.get("confidence", 0.8)),
                provenance=self.cfg.imported_edge_provenance,
                ephemeral=self.cfg.imported_ephemeral,
                ttl=meta.get("ttl", 100),
                relation_type=meta.get("relation_type", "associative"),
                is_symmetric=bool(meta.get("is_symmetric", False)),
                is_transitive=bool(meta.get("is_transitive", False)),
                strength=float(meta.get("strength", 0.5)),
                directionality=meta.get("directionality", "directed"),
                semantic_role=meta.get("semantic_role", None),
            )
            wm_graph.add_edge(src, tgt, e["relation"], properties=dict(e.get("properties", {})), metadata=em)

    # -------------------------
    # WM edge consistency (optional)
    # -------------------------

    def apply_wm_edge_consistency(self, wm_graph, wm_node_ids: List[str]) -> None:
        """
        用 WM 的边关系给 anchor 选择做轻量加分/修正：
        若 WM 中 w_i --rel--> w_j，而 LTM 中 anchor_i 与 anchor_j 也存在相同 rel（或同 relation_type），则加分。
        这里不做复杂全局优化，只做“当前 selected_anchor 的一致性加分”，并回写到 bundle。
        """
        # 取每个 WM node 的 selected_anchor
        selected_anchor = {}
        base_score = {}
        for wid in wm_node_ids:
            w = wm_graph.get_node(wid)
            if not w:
                continue
            b = w.properties.get(self.cfg.writeback_key, {})
            a = b.get("selected_anchor")
            if a:
                selected_anchor[wid] = a
                base_score[wid] = float(b.get("aggregated_confidence", b.get("base_score", 0.0)))

        # 计算一致性支持
        consistency_bonus = defaultdict(float)

        # 遍历 WM edges：只看这批节点间的边
        for eid, edge in wm_graph.edges.items():
            if edge.source not in selected_anchor or edge.target not in selected_anchor:
                continue

            a_src = selected_anchor[edge.source]
            a_tgt = selected_anchor[edge.target]

            # 检查 LTM 是否有对应关系
            ok = False
            # 直接查 a_src -> a_tgt 的 relation
            for le in self.ltm.find_edges(source=a_src, target=a_tgt, relation=edge.relation, limit=5):
                ok = True
                break
            # 也允许对称关系反向
            if not ok:
                for le in self.ltm.find_edges(source=a_tgt, target=a_src, relation=edge.relation, limit=5):
                    # 若关系在 LTM 标记为对称，则也算 ok
                    if le.metadata.is_symmetric:
                        ok = True
                        break

            if ok:
                # 给两个端点都加一些一致性分
                consistency_bonus[edge.source] += self.cfg.edge_consistency_weight
                consistency_bonus[edge.target] += self.cfg.edge_consistency_weight

        # 回写
        for wid in wm_node_ids:
            w = wm_graph.get_node(wid)
            if not w:
                continue
            b = w.properties.get(self.cfg.writeback_key, {})
            if not b.get("selected_anchor"):
                continue
            base = float(b.get("aggregated_confidence", b.get("base_score", 0.0)))
            bonus = float(consistency_bonus.get(wid, 0.0))
            b["edge_consistency_bonus"] = bonus
            b["final_confidence"] = float(_clamp01(base + bonus))
            w.properties[self.cfg.writeback_key] = b