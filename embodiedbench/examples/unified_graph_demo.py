#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一内存图实际应用示例

展示如何使用统一图来处理具身AI任务中的常见场景
"""

import sys
sys.path.insert(0, '/home/dministrator/EmbodiedBench-problemsolving')

from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph, NodeMetadata, EdgeMetadata


class EmbodiedMemoryManager:
    """基于统一图的具身AI内存管理器"""
    
    def __init__(self):
        self.graph = UnifiedMemoryGraph()
        self.current_step = 0
    
    def process_observation(self, observation_data):
        """处理观察数据
        
        Args:
            observation_data: {
                "objects": [
                    {"label": "apple", "color": "red", "position": "table"},
                    {"label": "spatula", "color": "silver", "position": "drawer"},
                ],
                "scene": "kitchen"
            }
        """
        print(f"\n[Step {self.current_step}] 处理观察...")
        
        # 处理场景
        if "scene" in observation_data:
            scene_node = self.graph.add_node(
                observation_data["scene"],
                properties={"type": "location"},
                metadata=NodeMetadata(timestamp=self.current_step, provenance="perception"),
            )
            print(f"  ✓ 场景: {scene_node.label}")
        
        # 处理对象
        for obj_data in observation_data.get("objects", []):
            obj_label = obj_data["label"]
            
            # 添加对象节点
            obj_node = self.graph.add_node(
                obj_label,
                properties={"type": "object"},
                metadata=NodeMetadata(timestamp=self.current_step, provenance="perception"),
            )
            
            # 添加颜色
            if "color" in obj_data:
                color_node = self.graph.add_node(
                    obj_data["color"],
                    properties={"type": "attribute"},
                    metadata=NodeMetadata(timestamp=self.current_step, provenance="perception"),
                )
                self.graph.add_edge(
                    obj_node.id, color_node.id, "has_color",
                    metadata=EdgeMetadata(timestamp=self.current_step, provenance="perception"),
                )
                print(f"  ✓ {obj_label} --has_color--> {obj_data['color']}")
            
            # 添加位置
            if "position" in obj_data:
                pos_node = self.graph.add_node(
                    obj_data["position"],
                    properties={"type": "location"},
                    metadata=NodeMetadata(timestamp=self.current_step, provenance="perception"),
                )
                self.graph.add_edge(
                    obj_node.id, pos_node.id, "located_at",
                    metadata=EdgeMetadata(timestamp=self.current_step, ephemeral=True),  # 位置是临时的
                )
                print(f"  ✓ {obj_label} --located_at--> {obj_data['position']}")
        
        self.current_step += 1
        print(f"  当前图: {self.graph.stats()}")
    
    def execute_action(self, action_data):
        """执行动作并更新图
        
        Args:
            action_data: {
                "type": "pick_place",
                "object": "apple",
                "from": "table",
                "to": "fridge"
            }
        """
        print(f"\n[Step {self.current_step}] 执行动作: {action_data['type']}")
        
        if action_data["type"] == "pick_place":
            obj_label = action_data["object"]
            from_loc = action_data.get("from")
            to_loc = action_data["to"]
            
            # 查找对象节点
            obj_nodes = self.graph.find_nodes(label=obj_label, exact=True)
            if not obj_nodes:
                print(f"  ✗ 未找到对象: {obj_label}")
                return
            
            obj_node = obj_nodes[0]
            
            # 删除旧位置边
            old_edges = self.graph.find_edges(source=obj_node.id, relation="located_at")
            for edge in old_edges:
                old_loc = self.graph.get_node(edge.target)
                self.graph.remove_edge(edge.id)
                print(f"  ✓ 删除: {obj_label} --located_at--> {old_loc.label}")
            
            # 添加新位置边
            to_node = self.graph.add_node(
                to_loc,
                properties={"type": "location"},
                metadata=NodeMetadata(timestamp=self.current_step, provenance="action"),
            )
            self.graph.add_edge(
                obj_node.id, to_node.id, "located_at",
                metadata=EdgeMetadata(timestamp=self.current_step, ephemeral=True, provenance="action"),
            )
            print(f"  ✓ 添加: {obj_label} --located_at--> {to_loc}")
        
        self.current_step += 1
        print(f"  当前图: {self.graph.stats()}")
    
    def query_object_location(self, object_label):
        """查询对象位置"""
        obj_nodes = self.graph.find_nodes(label=object_label, exact=True)
        if not obj_nodes:
            return f"{object_label} 未找到"
        
        obj_node = obj_nodes[0]
        loc_edges = self.graph.find_edges(source=obj_node.id, relation="located_at")
        
        if not loc_edges:
            return f"{object_label} 位置未知"
        
        loc_node = self.graph.get_node(loc_edges[0].target)
        return f"{object_label} 在 {loc_node.label}"
    
    def query_objects_by_color(self, color):
        """查询指定颜色的所有对象"""
        # 找到颜色节点
        color_nodes = self.graph.find_nodes(label=color, exact=True)
        if not color_nodes:
            return []
        
        color_node = color_nodes[0]
        
        # 找到所有有该颜色的对象
        color_edges = self.graph.find_edges(target=color_node.id, relation="has_color")
        objects = [self.graph.get_node(edge.source).label for edge in color_edges]
        
        return objects
    
    def query_objects_in_location(self, location):
        """查询指定位置的所有对象"""
        # 找到位置节点
        loc_nodes = self.graph.find_nodes(label=location, exact=True)
        if not loc_nodes:
            return []
        
        loc_node = loc_nodes[0]
        
        # 找到所有在该位置的对象
        loc_edges = self.graph.find_edges(target=loc_node.id, relation="located_at")
        objects = [self.graph.get_node(edge.source).label for edge in loc_edges]
        
        return objects
    
    def export_for_ltm(self):
        """导出非临时知识到LTM"""
        print("\n导出知识到LTM...")
        
        # 过滤掉临时节点和边
        persistent_nodes = []
        persistent_edges = []
        
        for node in self.graph.nodes.values():
            if not node.metadata.ephemeral:
                persistent_nodes.append(node)
        
        for edge in self.graph.edges.values():
            if not edge.metadata.ephemeral:
                # 确保源和目标都不是临时的
                source_node = self.graph.get_node(edge.source)
                target_node = self.graph.get_node(edge.target)
                if source_node and target_node:
                    if not source_node.metadata.ephemeral and not target_node.metadata.ephemeral:
                        persistent_edges.append(edge)
        
        print(f"  长期节点: {len(persistent_nodes)} / {len(self.graph.nodes)}")
        print(f"  长期边: {len(persistent_edges)} / {len(self.graph.edges)}")
        
        # 示例：打印长期知识
        print("\n  长期知识示例:")
        for edge in persistent_edges[:5]:
            src = self.graph.get_node(edge.source)
            tgt = self.graph.get_node(edge.target)
            print(f"    - {src.label} --{edge.relation}--> {tgt.label}")
        
        return {
            "nodes": [n.to_dict() for n in persistent_nodes],
            "edges": [e.to_dict() for e in persistent_edges],
        }
    
    def visualize_graph(self):
        """简单的图可视化"""
        print("\n图结构可视化:")
        print("=" * 60)
        
        # 按类型分组节点
        by_type = {}
        for node in self.graph.nodes.values():
            node_type = node.properties.get("type", "unknown")
            by_type.setdefault(node_type, []).append(node)
        
        for node_type, nodes in sorted(by_type.items()):
            print(f"\n【{node_type}】")
            for node in nodes[:5]:  # 最多显示5个
                print(f"  • {node.label}")
        
        print(f"\n【关系】")
        # 显示关系类型统计
        relation_counts = {}
        for edge in self.graph.edges.values():
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
        
        for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  • {relation}: {count} 条")
        
        print("=" * 60)


def demo_scenario_1():
    """演示场景1：基本观察和查询"""
    print("\n" + "=" * 60)
    print("场景1：基本观察和查询")
    print("=" * 60)
    
    manager = EmbodiedMemoryManager()
    
    # 初始观察
    observation = {
        "scene": "kitchen",
        "objects": [
            {"label": "apple", "color": "red", "position": "table"},
            {"label": "banana", "color": "yellow", "position": "table"},
            {"label": "spatula", "color": "silver", "position": "drawer"},
        ]
    }
    
    manager.process_observation(observation)
    
    # 查询
    print("\n=== 查询 ===")
    print(f"Q: apple 在哪里？")
    print(f"A: {manager.query_object_location('apple')}")
    
    print(f"\nQ: 有哪些红色的东西？")
    red_objects = manager.query_objects_by_color("red")
    print(f"A: {', '.join(red_objects) if red_objects else '无'}")
    
    print(f"\nQ: table 上有什么？")
    objects_on_table = manager.query_objects_in_location("table")
    print(f"A: {', '.join(objects_on_table) if objects_on_table else '无'}")


def demo_scenario_2():
    """演示场景2：动作执行和位置更新"""
    print("\n" + "=" * 60)
    print("场景2：动作执行和位置更新")
    print("=" * 60)
    
    manager = EmbodiedMemoryManager()
    
    # 初始观察
    observation = {
        "objects": [
            {"label": "apple", "color": "red", "position": "table"},
        ]
    }
    manager.process_observation(observation)
    
    # 查询初始位置
    print("\n=== 动作前 ===")
    print(manager.query_object_location("apple"))
    
    # 执行动作：将 apple 移动到 fridge
    action = {
        "type": "pick_place",
        "object": "apple",
        "from": "table",
        "to": "fridge"
    }
    manager.execute_action(action)
    
    # 查询新位置
    print("\n=== 动作后 ===")
    print(manager.query_object_location("apple"))


def demo_scenario_3():
    """演示场景3：LTM导出（过滤临时状态）"""
    print("\n" + "=" * 60)
    print("场景3：LTM导出（过滤临时状态）")
    print("=" * 60)
    
    manager = EmbodiedMemoryManager()
    
    # 观察（包含临时位置和永久属性）
    observation = {
        "objects": [
            {"label": "apple", "color": "red", "position": "table"},
            {"label": "spatula", "color": "silver", "position": "drawer"},
        ]
    }
    manager.process_observation(observation)
    
    # 添加一些长期知识
    apple = manager.graph.find_nodes(label="apple")[0]
    fruit = manager.graph.add_node("fruit", metadata=NodeMetadata(ephemeral=False))
    manager.graph.add_edge(apple.id, fruit.id, "is_a", metadata=EdgeMetadata(ephemeral=False))
    
    spatula = manager.graph.find_nodes(label="spatula")[0]
    tool = manager.graph.add_node("tool", metadata=NodeMetadata(ephemeral=False))
    manager.graph.add_edge(spatula.id, tool.id, "is_a", metadata=EdgeMetadata(ephemeral=False))
    
    # 可视化当前图
    manager.visualize_graph()
    
    # 导出到LTM（只包含长期知识）
    ltm_data = manager.export_for_ltm()


def main():
    """运行所有演示场景"""
    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    
    print("\n" + "=" * 60)
    print("✓ 所有场景演示完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
