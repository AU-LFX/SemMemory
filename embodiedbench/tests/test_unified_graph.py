#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试统一内存图 (Unified Memory Graph)"""

import sys
sys.path.insert(0, '/home/dministrator/EmbodiedBench-problemsolving')

from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph, Node, Edge, NodeMetadata


def test_basic_operations():
    """测试基本操作"""
    print("\n" + "=" * 60)
    print("Test 1: 基本操作 (添加节点和边)")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # 添加节点
    apple = graph.add_node("apple", properties={"type": "fruit"})
    red = graph.add_node("red", properties={"type": "color"})
    table = graph.add_node("table", properties={"type": "furniture"})
    kitchen = graph.add_node("kitchen", properties={"type": "location"})
    
    print(f"✓ 添加了 4 个节点")
    print(f"  - {apple}")
    print(f"  - {red}")
    print(f"  - {table}")
    print(f"  - {kitchen}")
    
    # 添加边
    e1 = graph.add_edge(apple.id, red.id, "has_color")
    e2 = graph.add_edge(apple.id, table.id, "located_at")
    e3 = graph.add_edge(table.id, kitchen.id, "in")
    
    print(f"\n✓ 添加了 3 条边")
    print(f"  - {e1}")
    print(f"  - {e2}")
    print(f"  - {e3}")
    
    print(f"\n图统计: {graph.stats()}")


def test_query_nodes():
    """测试节点查询"""
    print("\n" + "=" * 60)
    print("Test 2: 节点查询")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # 添加多个节点
    apple1 = graph.add_node("apple", properties={"color": "red", "size": "medium"})
    apple2 = graph.add_node("apple", properties={"color": "green", "size": "small"})
    banana = graph.add_node("banana", properties={"color": "yellow", "size": "medium"})
    
    # 精确查询
    results = graph.find_nodes(label="apple", exact=True)
    print(f"\n查询 label='apple' (exact): 找到 {len(results)} 个节点")
    for node in results:
        print(f"  - {node}")
    
    # 模糊查询
    results = graph.find_nodes(label="app")
    print(f"\n查询 label='app' (fuzzy): 找到 {len(results)} 个节点")
    for node in results:
        print(f"  - {node}")
    
    # 属性过滤
    results = graph.find_nodes(property_filters={"color": "red"})
    print(f"\n查询 color='red': 找到 {len(results)} 个节点")
    for node in results:
        print(f"  - {node}")


def test_query_edges():
    """测试边查询"""
    print("\n" + "=" * 60)
    print("Test 3: 边查询")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # 构建场景
    apple = graph.add_node("apple")
    banana = graph.add_node("banana")
    red = graph.add_node("red")
    yellow = graph.add_node("yellow")
    table = graph.add_node("table")
    
    graph.add_edge(apple.id, red.id, "has_color")
    graph.add_edge(banana.id, yellow.id, "has_color")
    graph.add_edge(apple.id, table.id, "located_at")
    graph.add_edge(banana.id, table.id, "located_at")
    
    # 查询特定关系
    results = graph.find_edges(relation="has_color")
    print(f"\n查询 relation='has_color': 找到 {len(results)} 条边")
    for edge in results:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  - {src.label} --{edge.relation}--> {tgt.label}")
    
    # 查询从特定节点出发的边
    results = graph.find_edges(source=apple.id)
    print(f"\n查询 source='apple': 找到 {len(results)} 条边")
    for edge in results:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  - {src.label} --{edge.relation}--> {tgt.label}")
    
    # 查询到特定节点的边
    results = graph.find_edges(target=table.id)
    print(f"\n查询 target='table': 找到 {len(results)} 条边")
    for edge in results:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        print(f"  - {src.label} --{edge.relation}--> {tgt.label}")


def test_traversal():
    """测试图遍历"""
    print("\n" + "=" * 60)
    print("Test 4: 图遍历")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # 构建层次结构: apple -> table -> kitchen -> house
    apple = graph.add_node("apple")
    table = graph.add_node("table")
    kitchen = graph.add_node("kitchen")
    house = graph.add_node("house")
    
    graph.add_edge(apple.id, table.id, "on")
    graph.add_edge(table.id, kitchen.id, "in")
    graph.add_edge(kitchen.id, house.id, "part_of")
    
    # 遍历
    results = graph.traverse(apple.id, relations=["on", "in", "part_of"], max_depth=3)
    print(f"\n从 'apple' 开始遍历 (relations=['on', 'in', 'part_of']):")
    print(f"遍历到 {len(results)} 个节点:")
    for node in results:
        print(f"  - {node.label}")
    
    # 获取邻居
    neighbors = graph.get_neighbors(table.id, direction="both")
    print(f"\n获取 'table' 的所有邻居 (双向):")
    for node in neighbors:
        print(f"  - {node.label}")


def test_complex_scenario():
    """测试复杂场景：厨房中的物体"""
    print("\n" + "=" * 60)
    print("Test 5: 复杂场景 - 厨房中的物体")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # === 创建节点 ===
    # 对象
    apple = graph.add_node("apple")
    spatula = graph.add_node("spatula")
    strawberry = graph.add_node("strawberry")
    
    # 颜色
    red = graph.add_node("red")
    silver = graph.add_node("silver")
    
    # 位置
    table = graph.add_node("table")
    fridge = graph.add_node("fridge")
    drawer = graph.add_node("drawer")
    kitchen = graph.add_node("kitchen")
    
    # 材质
    metal = graph.add_node("metal")
    
    # 功能
    flip = graph.add_node("flip")
    spread = graph.add_node("spread")
    
    # 类型
    fruit = graph.add_node("fruit")
    tool = graph.add_node("tool")
    
    # === 创建边（关系）===
    # 对象属性
    graph.add_edge(apple.id, red.id, "has_color")
    graph.add_edge(strawberry.id, red.id, "has_color")
    graph.add_edge(spatula.id, silver.id, "has_color")
    graph.add_edge(spatula.id, metal.id, "made_of")
    
    # 位置关系
    graph.add_edge(apple.id, table.id, "located_at")
    graph.add_edge(strawberry.id, fridge.id, "located_at")
    graph.add_edge(spatula.id, drawer.id, "located_at")
    
    # 容器关系
    graph.add_edge(table.id, kitchen.id, "in")
    graph.add_edge(fridge.id, kitchen.id, "in")
    graph.add_edge(drawer.id, kitchen.id, "in")
    
    # 功能关系
    graph.add_edge(spatula.id, flip.id, "affords")
    graph.add_edge(spatula.id, spread.id, "affords")
    
    # 类型关系
    graph.add_edge(apple.id, fruit.id, "is_a")
    graph.add_edge(strawberry.id, fruit.id, "is_a")
    graph.add_edge(spatula.id, tool.id, "is_a")
    
    print(f"\n✓ 构建完成: {graph.stats()}")
    
    # === 查询示例 ===
    
    # Q1: 找到所有红色的东西
    print("\n【查询1】找到所有红色的东西:")
    red_edges = graph.find_edges(target=red.id, relation="has_color")
    for edge in red_edges:
        obj = graph.get_node(edge.source)
        print(f"  - {obj.label} (红色)")
    
    # Q2: 找到所有在厨房的东西（通过两跳查询）
    print("\n【查询2】找到所有在厨房的东西:")
    # 先找到所有在厨房中的容器
    in_kitchen_edges = graph.find_edges(target=kitchen.id, relation="in")
    for container_edge in in_kitchen_edges:
        container = graph.get_node(container_edge.source)
        # 再找到在这些容器中的物体
        located_edges = graph.find_edges(target=container.id, relation="located_at")
        for obj_edge in located_edges:
            obj = graph.get_node(obj_edge.source)
            print(f"  - {obj.label} (在 {container.label})")
    
    # Q3: 找到所有工具及其功能
    print("\n【查询3】找到所有工具及其功能:")
    tool_edges = graph.find_edges(target=tool.id, relation="is_a")
    for edge in tool_edges:
        obj = graph.get_node(edge.source)
        # 查找该工具的功能
        afford_edges = graph.find_edges(source=obj.id, relation="affords")
        functions = [graph.get_node(e.target).label for e in afford_edges]
        print(f"  - {obj.label} 可以: {', '.join(functions)}")
    
    # Q4: spatula 在哪里？
    print("\n【查询4】spatula 在哪里？")
    loc_edges = graph.find_edges(source=spatula.id, relation="located_at")
    if loc_edges:
        location = graph.get_node(loc_edges[0].target)
        # 继续查询容器的位置
        container_edges = graph.find_edges(source=location.id, relation="in")
        if container_edges:
            room = graph.get_node(container_edges[0].target)
            print(f"  - spatula 在 {location.label} (位于 {room.label})")


def test_update_scenario():
    """测试更新场景：移动对象"""
    print("\n" + "=" * 60)
    print("Test 6: 更新场景 - 移动对象")
    print("=" * 60)
    
    graph = UnifiedMemoryGraph()
    
    # 初始状态
    apple = graph.add_node("apple")
    table = graph.add_node("table")
    fridge = graph.add_node("fridge")
    
    # apple 在 table 上
    old_edge = graph.add_edge(apple.id, table.id, "located_at")
    print(f"\n初始状态: apple 在 table")
    
    # 移动 apple 到 fridge
    print(f"\n执行动作: 将 apple 移动到 fridge")
    
    # 1. 删除旧的位置边
    graph.remove_edge(old_edge.id)
    
    # 2. 添加新的位置边
    new_edge = graph.add_edge(apple.id, fridge.id, "located_at")
    
    print(f"\n新状态: apple 在 fridge")
    
    # 验证
    loc_edges = graph.find_edges(source=apple.id, relation="located_at")
    if loc_edges:
        location = graph.get_node(loc_edges[0].target)
        print(f"✓ 验证成功: apple 当前位置 = {location.label}")


def main():
    """运行所有测试"""
    print("\n")
    print("=" * 60)
    print("统一内存图 (Unified Memory Graph) 测试")
    print("=" * 60)
    
    test_basic_operations()
    test_query_nodes()
    test_query_edges()
    test_traversal()
    test_complex_scenario()
    test_update_scenario()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
