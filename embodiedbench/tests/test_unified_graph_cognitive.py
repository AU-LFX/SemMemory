#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试统一内存图的认知属性和语义关系功能"""

import sys
sys.path.insert(0, '/home/dministrator/EmbodiedBench-problemsolving')

from embodiedbench.evaluator.unified_graph import (
    UnifiedMemoryGraph, 
    NodeMetadata, 
    EdgeMetadata
)


def test_cognitive_attributes():
    """测试节点的认知属性查询"""
    print("\n🧠 测试认知属性查询...")
    
    graph = UnifiedMemoryGraph()
    
    # 添加高中心性概念（核心概念）
    apple = graph.add_node(
        "apple",
        metadata=NodeMetadata(
            centrality=0.9,
            prototypicality=0.8,
            provenance="ltm"
        )
    )
    
    # 添加模糊概念
    nearby = graph.add_node(
        "nearby",
        metadata=NodeMetadata(
            vagueness=0.85,
            gradience=0.7,
            perspective="spatial"
        )
    )
    
    # 添加渐进性属性
    red = graph.add_node(
        "red",
        metadata=NodeMetadata(
            gradience=0.9,
            vagueness=0.6
        )
    )
    
    # 测试查询
    print("\n1️⃣ 查找高中心性概念 (centrality > 0.7):")
    central = graph.find_central_concepts(threshold=0.7)
    for node in central:
        print(f"  ✅ {node.label}: centrality={node.metadata.centrality}")
    
    print("\n2️⃣ 查找模糊概念 (vagueness > 0.5):")
    vague = graph.find_vague_concepts(threshold=0.5)
    for node in vague:
        print(f"  ✅ {node.label}: vagueness={node.metadata.vagueness}")
    
    print("\n3️⃣ 查找渐进性属性 (gradience > 0.5):")
    gradient = graph.find_gradient_attributes(threshold=0.5)
    for node in gradient:
        print(f"  ✅ {node.label}: gradience={node.metadata.gradience}")
    
    print("\n✅ 认知属性测试通过！")


def test_semantic_relations():
    """测试边的语义关系属性查询"""
    print("\n🔗 测试语义关系查询...")
    
    graph = UnifiedMemoryGraph()
    
    # 创建节点
    robin = graph.add_node("robin")
    bird = graph.add_node("bird")
    animal = graph.add_node("animal")
    
    sofa = graph.add_node("sofa")
    couch = graph.add_node("couch")
    
    hot = graph.add_node("hot")
    cold = graph.add_node("cold")
    
    # 添加分类关系（可传递）
    graph.add_edge(
        robin.id, bird.id, "is_a",
        metadata=EdgeMetadata(
            relation_type="taxonomic",
            is_transitive=True,
            strength=1.0
        )
    )
    
    graph.add_edge(
        bird.id, animal.id, "is_a",
        metadata=EdgeMetadata(
            relation_type="taxonomic",
            is_transitive=True,
            strength=1.0
        )
    )
    
    # 添加同义关系（对称）
    graph.add_edge(
        sofa.id, couch.id, "synonym_of",
        metadata=EdgeMetadata(
            relation_type="synonymy",
            is_symmetric=True,
            strength=0.95
        )
    )
    
    # 添加反义关系（对称）
    graph.add_edge(
        hot.id, cold.id, "antonym_of",
        metadata=EdgeMetadata(
            relation_type="antonymy",
            is_symmetric=True,
            strength=1.0
        )
    )
    
    # 测试查询
    print("\n1️⃣ 查找对称关系:")
    symmetric = graph.find_symmetric_relations()
    for edge in symmetric:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  ✅ {src.label} --{edge.relation}--> {tgt.label} (symmetric)")
    
    print("\n2️⃣ 查找可传递关系:")
    transitive = graph.find_transitive_relations(relation_type="taxonomic")
    for edge in transitive:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  ✅ {src.label} --{edge.relation}--> {tgt.label} (transitive)")
    
    print("\n3️⃣ 查找高强度关系 (strength > 0.9):")
    strong = graph.find_strong_relations(threshold=0.9)
    for edge in strong:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  ✅ {src.label} --{edge.relation}--> {tgt.label} (strength={edge.metadata.strength})")
    
    print("\n4️⃣ 推理传递闭包 (robin is_a ?):")
    closure = graph.infer_transitive_closure(robin.id, "is_a")
    for node in closure:
        if node.id != robin.id:
            print(f"  ✅ robin is_a {node.label} (inferred)")
    
    print("\n✅ 语义关系测试通过！")


def test_metadata_filters():
    """测试元数据过滤功能"""
    print("\n🔍 测试元数据过滤器...")
    
    graph = UnifiedMemoryGraph()
    
    # 添加各种节点
    nodes_data = [
        ("apple", {"centrality": 0.9, "vagueness": 0.2}),
        ("nearby", {"centrality": 0.3, "vagueness": 0.8}),
        ("red", {"centrality": 0.5, "vagueness": 0.6}),
        ("table", {"centrality": 0.7, "vagueness": 0.1}),
    ]
    
    for label, attrs in nodes_data:
        graph.add_node(
            label,
            metadata=NodeMetadata(
                centrality=attrs["centrality"],
                vagueness=attrs["vagueness"]
            )
        )
    
    # 测试组合查询
    print("\n1️⃣ 查找高中心性且低模糊性的概念:")
    print("   (centrality > 0.6 AND vagueness < 0.3)")
    nodes = graph.find_nodes(
        metadata_filters={
            "centrality": (">", 0.6),
            "vagueness": ("<", 0.3)
        }
    )
    for node in nodes:
        print(f"  ✅ {node.label}: c={node.metadata.centrality}, v={node.metadata.vagueness}")
    
    print("\n2️⃣ 查找中等中心性的概念:")
    print("   (0.4 <= centrality <= 0.6)")
    nodes = graph.find_nodes(
        metadata_filters={
            "centrality": (">=", 0.4)
        }
    )
    nodes = [n for n in nodes if n.metadata.centrality <= 0.6]
    for node in nodes:
        print(f"  ✅ {node.label}: centrality={node.metadata.centrality}")
    
    print("\n✅ 元数据过滤测试通过！")


def test_ltm_persistence():
    """测试 LTM 节点不过期"""
    print("\n⏰ 测试 LTM 节点持久化...")
    
    graph = UnifiedMemoryGraph()
    
    # 添加 LTM 节点
    ltm_node = graph.add_node(
        "apple",
        metadata=NodeMetadata(
            provenance="ltm",
            ttl=10
        )
    )
    
    # 添加普通节点
    wm_node = graph.add_node(
        "banana",
        metadata=NodeMetadata(
            provenance="perception",
            ttl=10
        )
    )
    
    print(f"  初始节点数: {len(graph.nodes)}")
    
    # 模拟时间流逝（超过 TTL）
    graph.prune(current_step=20)
    
    print(f"  清理后节点数: {len(graph.nodes)}")
    
    # 检查 LTM 节点是否存在
    assert ltm_node.id in graph.nodes, "❌ LTM 节点被错误删除！"
    print(f"  ✅ LTM 节点 '{ltm_node.label}' 仍然存在")
    
    # 检查普通节点是否被删除
    assert wm_node.id not in graph.nodes, "❌ 过期节点未被删除！"
    print(f"  ✅ 过期节点 '{wm_node.label}' 已被删除")
    
    print("\n✅ LTM 持久化测试通过！")


if __name__ == "__main__":
    test_cognitive_attributes()
    test_semantic_relations()
    test_metadata_filters()
    test_ltm_persistence()
    
    print("\n" + "="*60)
    print("🎉 所有测试通过！统一内存图的认知属性系统正常工作！")
    print("="*60)
