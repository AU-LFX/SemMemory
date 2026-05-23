# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
LTM 初始 KG（Seed Knowledge Graph）
========================================
你给的文件需要同步适配我上面那套“语义增强版 MemoryManager + UnifiedMemoryGraph”。

关键适配点（非常重要）：
1) provenance：
   - 上面代码的检索器 RetrievalConfig 默认 require_ltm_provenance=True，
     且 anchor_candidates 用 metadata_filters={"provenance":"ltm"} 过滤。
   - 因此 Seed 图的 NodeMetadata.provenance 必须包含 "ltm"（建议直接 "ltm"），否则 LTM 检索永远找不到 seed 节点。

2) metadata 字段：
   - UnifiedMemoryGraph.NodeMetadata / EdgeMetadata 已固定字段：
     NodeMetadata: confidence/timestamp/ttl/provenance/ephemeral/last_access/access_count/
                   centrality/family_resemblance/perspective/vagueness/gradience/prototypicality
     EdgeMetadata: confidence/timestamp/ttl/provenance/ephemeral/last_access/access_count/
                   relation_type/is_symmetric/is_transitive/strength/directionality/semantic_role
   - 你原来 seed 里用了 graduality/fuzziness 等不匹配字段，要改为 gradience/vagueness。
   - 节点 centrality 这种应该放 metadata，而不是 properties（上面的 graph prior_score 用 metadata）。

3) edges 的语义属性：
   - 需要填 relation_type / is_symmetric / is_transitive / strength / directionality / semantic_role
   - 这些要与我上面 memory 文件里的 RELATION_SEMANTICS 对齐（至少覆盖 is_a/synonym_of/part_of/has_color/made_of/affords/on/in/near/...）

4) 输出结构：
   - 你原来的 build_ltm_seed_graph 返回 “nodes/edges 列表”，其中 node 以 label 表示。
   - 我建议继续返回 label-based list（方便），同时提供 build_ltm_seed_unified_graph() 直接构建 UnifiedMemoryGraph，
     这样上层初始化时一句话把 seed 写进 LTM。

下面是**可直接替换**你原文件的适配版本（保留了你原 API：get_concept_definition/resolve_canonical_label/.../build_ltm_seed_graph/build_abstract_concrete_mapping），
并新增：
- build_ltm_seed_unified_graph(...) -> UnifiedMemoryGraph
- apply_seed_to_ltm(ltm_graph, ...)  -> None（将 seed 写入现有 LTM）

注释全部中文。
"""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from embodiedbench.evaluator.unified_graph import UnifiedMemoryGraph, NodeMetadata, EdgeMetadata

__all__ = [
    "get_concept_definition",
    "resolve_canonical_label",
    "iter_concept_labels",
    "list_concept_templates",
    "build_ltm_seed_graph",
    "build_ltm_seed_unified_graph",
    "apply_seed_to_ltm",
    "build_abstract_concrete_mapping",
]


# ============================================================
# 基础工具
# ============================================================

def _constraint(text: str) -> Dict[str, Any]:
    return {"text": text}

def _normalize(label: Optional[str]) -> str:
    return (label or "").strip().lower()


# ============================================================
# 属性字段 -> 图关系映射
# 说明：raw template 的 state_vars 里可能含 colors/materials/affordances...
# 我们会将这些展开为 attribute node + edge（has_color/made_of/affords 等）
# ============================================================

_ATTRIBUTE_KEY_RELATIONS: Dict[str, Tuple[str, str]] = {
    "color": ("has_color", "color"),
    "colors": ("has_color", "color"),
    "size_hint": ("has_size", "size"),
    "shape_hint": ("has_shape", "shape"),
    "top_features": ("has_feature", "feature"),
    "features": ("has_feature", "feature"),
    "texture": ("has_texture", "texture"),
    "materials": ("made_of", "material"),
    "material": ("made_of", "material"),
    "affordances": ("affords", "affordance"),
    "usage_notes": ("has_usage", "usage"),
}


# ============================================================
# 关系语义配置：对齐上面 MemoryManager 的 RELATION_SEMANTICS（最小覆盖集）
# Seed 文件是静态知识，建议：
# - taxonomic/partitive/synonymy/functional 等设置为稳定（ephemeral=False）
# - spatial(on/in/near/located_at) 在 seed 中一般作为“概念定义”也可稳定存
#   但在 episode 运行时 WM 的空间关系是动态的，是否写入 LTM 由上层控制
# ============================================================

_RELATION_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "is_a": {"relation_type": "taxonomic", "is_transitive": True, "is_symmetric": False, "strength": 1.0, "directionality": "directed"},
    "subclass_of": {"relation_type": "taxonomic", "is_transitive": True, "is_symmetric": False, "strength": 0.95, "directionality": "directed"},
    "instance_of": {"relation_type": "taxonomic", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},

    "part_of": {"relation_type": "partitive", "is_transitive": True, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},
    "has_part": {"relation_type": "partitive", "is_transitive": False, "is_symmetric": False, "strength": 0.85, "directionality": "directed"},
    "contains": {"relation_type": "partitive", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},

    "synonym_of": {"relation_type": "synonymy", "is_transitive": False, "is_symmetric": True, "strength": 0.95, "directionality": "undirected"},
    "similar_to": {"relation_type": "synonymy", "is_transitive": False, "is_symmetric": True, "strength": 0.7, "directionality": "undirected"},

    "affords": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "used_for": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "has_function": {"relation_type": "functional", "is_transitive": False, "is_symmetric": False, "strength": 0.75, "directionality": "directed"},

    "has_color": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},
    "has_size": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed"},
    "has_shape": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.75, "directionality": "directed"},
    "has_feature": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.7, "directionality": "directed"},
    "has_texture": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.7, "directionality": "directed"},
    "has_usage": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.65, "directionality": "directed"},
    "made_of": {"relation_type": "associative", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed"},

    # seed 中的空间关系：作为“概念定义”可以稳定写入
    "on": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed", "semantic_role": "location"},
    "in": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.8, "directionality": "directed", "semantic_role": "location"},
    "near": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": True, "strength": 0.6, "directionality": "undirected", "semantic_role": "location"},
    "located_at": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.9, "directionality": "directed", "semantic_role": "location"},
    "usually_found_at": {"relation_type": "spatial", "is_transitive": False, "is_symmetric": False, "strength": 0.85, "directionality": "directed", "semantic_role": "location"},
}


# ============================================================
# canonical label index
# ============================================================

_CANONICAL_LABEL_INDEX: Dict[str, str] = {}


# ============================================================
# 你原来的 RAW_TEMPLATES（我保留结构，内容可继续扩展）
# 注意：此处省略了后续大量模板内容，你只需保留你原文件中的完整 _RAW_TEMPLATES 即可。
# ============================================================

_RAW_TEMPLATES: Dict[str, Dict[str, Any]] = {
	# ==================================================================================
	# ABSTRACT CONCEPTS & SEMANTIC CATEGORIES
	# ==================================================================================
	
	# --- Spatial Concepts ---
	"receptacle": {
		"kinds": ["concept", "spatial_category"],
		"attributes": {
			"state_vars": {
				"category": "spatial_concept",
				"definition": "A physical space or container that can hold, contain, or receive objects or liquids",
				"semantic_type": "abstract",
				"concrete_instances": ["sink", "basin", "bowl", "container", "bin", "drawer", "cabinet"],
				"abstract_properties": ["containment", "volume", "accessibility"],
			},
		},
	},
	"container": {
		"kinds": ["concept", "spatial_category"],
		"attributes": {
			"state_vars": {
				"category": "spatial_concept",
				"definition": "An object or space designed to store, hold, or transport other objects",
				"semantic_type": "abstract",
				"concrete_instances": ["box", "basket", "bag", "bin", "jar", "cabinet", "drawer"],
				"abstract_properties": ["enclosure", "portability", "storage_capacity"],
			},
		},
	},
	"surface": {
		"kinds": ["concept", "spatial_category"],
		"attributes": {
			"state_vars": {
				"category": "spatial_concept",
				"definition": "A flat or horizontal area suitable for placing, resting, or working with objects",
				"semantic_type": "abstract",
				"concrete_instances": ["counter", "table", "desk", "shelf", "floor", "tray"],
				"abstract_properties": ["flatness", "support", "accessibility"],
			},
		},
	},
	
	# --- Functional Concepts ---
	"appliance": {
		"kinds": ["concept", "functional_category"],
		"attributes": {
			"state_vars": {
				"category": "functional_concept",
				"definition": "An electrical or mechanical device designed to perform specific household tasks",
				"semantic_type": "abstract",
				"concrete_instances": ["microwave", "oven", "stove", "refrigerator", "dishwasher", "toaster"],
				"abstract_properties": ["powered", "task_specific", "automated"],
			},
		},
	},
	"tool": {
		"kinds": ["concept", "functional_category"],
		"attributes": {
			"state_vars": {
				"category": "functional_concept",
				"definition": "A handheld object designed to perform specific tasks or operations",
				"semantic_type": "abstract",
				"concrete_instances": ["spatula", "knife", "screwdriver", "hammer", "wrench", "pliers"],
				"abstract_properties": ["graspable", "purpose_built", "manipulable"],
			},
		},
	},
	"utensil": {
		"kinds": ["concept", "functional_category"],
		"attributes": {
			"state_vars": {
				"category": "functional_concept",
				"definition": "A tool or implement used for eating, cooking, or food preparation",
				"semantic_type": "abstract",
				"concrete_instances": ["spatula", "spoon", "fork", "knife", "ladle", "tongs"],
				"abstract_properties": ["food_related", "handheld", "cleanable"],
			},
		},
	},
	
	# --- Material Properties ---
	"fragile": {
		"kinds": ["property", "material_attribute"],
		"attributes": {
			"state_vars": {
				"category": "material_property",
				"definition": "Easily broken, damaged, or destroyed; requires careful handling",
				"semantic_type": "property",
				"applies_to": ["glass", "ceramic", "porcelain", "crystal"],
				"constraints": ["handle_with_care", "avoid_dropping", "gentle_placement"],
			},
		},
	},
	"heat_resistant": {
		"kinds": ["property", "material_attribute"],
		"attributes": {
			"state_vars": {
				"category": "material_property",
				"definition": "Capable of withstanding high temperatures without damage or deformation",
				"semantic_type": "property",
				"applies_to": ["metal", "ceramic", "glass", "stone"],
				"affordances": ["hot_object_placement", "cooking", "heating"],
			},
		},
	},
	"waterproof": {
		"kinds": ["property", "material_attribute"],
		"attributes": {
			"state_vars": {
				"category": "material_property",
				"definition": "Impervious to water; not damaged by moisture or liquid exposure",
				"semantic_type": "property",
				"applies_to": ["plastic", "metal", "rubber", "sealed_wood"],
				"affordances": ["washing", "outdoor_use", "liquid_storage"],
			},
		},
	},
	
	# --- State Concepts ---
	"open": {
		"kinds": ["state", "spatial_state"],
		"attributes": {
			"state_vars": {
				"category": "object_state",
				"definition": "A state where access is unobstructed; doors, lids, or barriers are not closed",
				"semantic_type": "state",
				"opposite": "closed",
				"applies_to": ["door", "drawer", "cabinet", "container", "window"],
				"preconditions": ["accessible_interior", "visible_contents"],
			},
		},
	},
	"closed": {
		"kinds": ["state", "spatial_state"],
		"attributes": {
			"state_vars": {
				"category": "object_state",
				"definition": "A state where access is obstructed; doors, lids, or barriers are shut",
				"semantic_type": "state",
				"opposite": "open",
				"applies_to": ["door", "drawer", "cabinet", "container", "window"],
				"effects": ["contents_protected", "access_blocked"],
			},
		},
	},
	"clean": {
		"kinds": ["state", "hygiene_state"],
		"attributes": {
			"state_vars": {
				"category": "object_state",
				"definition": "Free from dirt, contaminants, or unwanted substances",
				"semantic_type": "state",
				"opposite": "dirty",
				"applies_to": ["dish", "utensil", "surface", "floor", "appliance"],
				"maintenance": ["wash", "wipe", "sanitize"],
			},
		},
	},
	"dirty": {
		"kinds": ["state", "hygiene_state"],
		"attributes": {
			"state_vars": {
				"category": "object_state",
				"definition": "Soiled, contaminated, or covered with unwanted substances",
				"semantic_type": "state",
				"opposite": "clean",
				"applies_to": ["dish", "utensil", "surface", "floor", "appliance"],
				"required_action": ["clean", "wash"],
			},
		},
	},
	
	# --- Action Concepts ---
	"pick_up": {
		"kinds": ["action", "manipulation_action"],
		"attributes": {
			"state_vars": {
				"category": "action_concept",
				"definition": "Grasp and lift an object from a surface or location",
				"semantic_type": "action",
				"preconditions": ["object_graspable", "object_reachable", "hands_free"],
				"effects": ["object_held", "object_removed_from_surface"],
				"constraints": ["object_weight_manageable", "object_not_fixed"],
			},
		},
	},
	"place": {
		"kinds": ["action", "manipulation_action"],
		"attributes": {
			"state_vars": {
				"category": "action_concept",
				"definition": "Put down or position an object at a specific location",
				"semantic_type": "action",
				"preconditions": ["object_held", "target_surface_accessible"],
				"effects": ["object_at_location", "hands_free"],
				"constraints": ["surface_supports_object", "sufficient_space"],
			},
		},
	},
	"navigate": {
		"kinds": ["action", "locomotion_action"],
		"attributes": {
			"state_vars": {
				"category": "action_concept",
				"definition": "Move from current position to a target location",
				"semantic_type": "action",
				"preconditions": ["path_exists", "target_reachable"],
				"effects": ["position_changed", "at_target_location"],
				"constraints": ["no_obstacles", "sufficient_clearance"],
			},
		},
	},
	
	# --- Spatial Relations ---
	"on": {
		"kinds": ["relation", "spatial_relation"],
		"attributes": {
			"state_vars": {
				"category": "spatial_relation",
				"definition": "Supported by and in contact with the upper surface of another object",
				"semantic_type": "relation",
				"inverse": "under",
				"examples": ["book on table", "plate on counter"],
				"constraints": ["surface_contact", "gravity_support"],
			},
		},
	},
	"in": {
		"kinds": ["relation", "spatial_relation"],
		"attributes": {
			"state_vars": {
				"category": "spatial_relation",
				"definition": "Contained within the interior or boundaries of another object or space",
				"semantic_type": "relation",
				"inverse": "contains",
				"examples": ["food in bowl", "utensil in drawer"],
				"constraints": ["spatial_enclosure", "boundary_defined"],
			},
		},
	},
	"near": {
		"kinds": ["relation", "spatial_relation"],
		"attributes": {
			"state_vars": {
				"category": "spatial_relation",
				"definition": "Close in proximity but not in direct contact",
				"semantic_type": "relation",
				"opposite": "far",
				"examples": ["chair near table", "sink near counter"],
				"fuzzy_measure": "distance < threshold",
			},
		},
	},
	
	# --- Constraints & Rules ---
	"handle_with_care": {
		"kinds": ["constraint", "safety_rule"],
		"attributes": {
			"state_vars": {
				"category": "behavioral_constraint",
				"definition": "Requires gentle, careful manipulation to avoid damage",
				"semantic_type": "constraint",
				"applies_to": ["fragile", "valuable", "delicate"],
				"violations": ["dropping", "rough_handling", "impact"],
				"recommendations": ["slow_movement", "secure_grip", "padded_surface"],
			},
		},
	},
	"temperature_safety": {
		"kinds": ["constraint", "safety_rule"],
		"attributes": {
			"state_vars": {
				"category": "safety_constraint",
				"definition": "Precautions required when handling hot or cold objects",
				"semantic_type": "constraint",
				"applies_to": ["hot_object", "cold_object", "temperature_extreme"],
				"violations": ["burns", "frostbite", "material_damage"],
				"recommendations": ["use_protection", "wait_for_cooling", "insulated_handling"],
			},
		},
	},
	"food_hygiene": {
		"kinds": ["rule", "hygiene_rule"],
		"attributes": {
			"state_vars": {
				"category": "hygiene_rule",
				"definition": "Standards for cleanliness and safety in food handling",
				"semantic_type": "rule",
				"applies_to": ["food", "utensil", "cooking_surface", "dish"],
				"requirements": ["clean_before_use", "sanitize_after_raw_food", "separate_raw_cooked"],
				"violations": ["contamination", "cross_contamination", "spoilage"],
			},
		},
	},
	
	# ==================================================================================
	# CONCRETE OBJECTS (storage / navigation targets)
	# ==================================================================================
	"cabinet": {
		"kinds": ["place", "storage"],
		"attributes": {
			"state_vars": {
				"category": "storage",
				"materials": ["wood", "composite"],
				"affordances": ["open", "store_items"],
				"door_state": "closed",
				"supports_hot_items": False,
				"reachable_surfaces": ["shelves"],
			},
			"constraints": [
				_constraint("open the cabinet door before placing or retrieving objects"),
			],
		},
	},
	"cabinet 4": {"extends": "cabinet"},
	"cabinet 5": {"extends": "cabinet"},
	"cabinet 6": {"extends": "cabinet"},
	"cabinet 7": {"extends": "cabinet"},
	"refrigerator": {
		"kinds": ["place", "appliance"],
		"attributes": {
			"state_vars": {
				"category": "appliance",
				"materials": ["metal", "plastic"],
				"affordances": ["store_perishables", "cool"],
				"supports_hot_items": False,
				"door_state": "closed",
				"temperature_controlled": True,
			},
			"constraints": [
				_constraint("close the refrigerator door after taking or placing items"),
			],
		},
	},
	"refrigerator push point": {
		"kinds": ["place"],
		"attributes": {
			"state_vars": {
				"category": "navigation_anchor",
				"affordances": ["navigate", "stand"],
				"description": "Safe standing spot for pushing the refrigerator door",
			},
		},
		"aliases": ["fridge push point"],
	},
	"counter": {
		"kinds": ["place", "surface"],
		"attributes": {
			"state_vars": {
				"category": "surface",
				"materials": ["stone", "laminate"],
				"affordances": ["prepare_food", "place_items"],
				"supports_hot_items": True,
				"is_surface": True,
			},
			"constraints": [
				_constraint("keep fragile items away from the counter edge"),
			],
		},
		"relations": [
			{"relation": "is_a", "target": "surface", "confidence": 1.0},
		],
	},
	"right counter in the kitchen": {"extends": "counter"},
	"left counter in the kitchen": {"extends": "counter"},
	"drawer": {
		"kinds": ["place", "storage"],
		"attributes": {
			"state_vars": {
				"category": "drawer",
				"materials": ["wood", "composite"],
				"affordances": ["open", "store_utensils"],
			},
		},
	},
	"left drawer of the kitchen counter": {"extends": "drawer"},
	"right drawer of the kitchen counter": {"extends": "drawer"},
	"sink in the kitchen": {
		"kinds": ["place", "surface", "receptacle"],
		"attributes": {
			"state_vars": {
				"category": "sink",
				"materials": ["ceramic", "steel"],
				"affordances": ["wash", "rinse", "hold_water"],
				"supports_hot_items": True,
				"requires_plumbing": True,
				"is_receptacle": True,
			},
		},
		"relations": [
			{"relation": "is_a", "target": "receptacle", "confidence": 1.0},
			{"relation": "synonym_of", "target": "basin", "confidence": 0.8},
		],
	},
	"sink": {"extends": "sink in the kitchen"},
	"chair": {
		"kinds": ["place", "furniture"],
		"attributes": {
			"state_vars": {
				"category": "seating",
				"materials": ["fabric", "wood"],
				"affordances": ["sit", "place_soft_items"],
				"supports_hot_items": False,
			},
			"constraints": [
				_constraint("avoid placing liquids directly on the cushion"),
			],
		},
	},
	"chair 1": {"extends": "chair"},
	"table": {
		"kinds": ["place", "surface"],
		"attributes": {
			"state_vars": {
				"category": "surface",
				"materials": ["wood", "glass"],
				"affordances": ["serve_food", "place_items"],
				"supports_hot_items": True,
				"is_surface": True,
			},
			"constraints": [
				_constraint("avoid scratching the tabletop"),
			],
		},
		"relations": [
			{"relation": "is_a", "target": "surface", "confidence": 1.0},
		],
	},
	"table 1": {"extends": "table"},
	"table 2": {"extends": "table"},
	"tv stand": {
		"kinds": ["place", "surface"],
		"attributes": {
			"state_vars": {
				"category": "surface",
				"materials": ["wood", "metal"],
				"affordances": ["hold_electronics", "place_decor"],
			},
		},
	},
	"sofa": {
		"kinds": ["place", "furniture"],
		"attributes": {
			"state_vars": {
				"category": "seating",
				"materials": ["fabric", "wood"],
				"affordances": ["sit", "recline"],
				"supports_hot_items": False,
			},
			"constraints": [
				_constraint("avoid placing sharp or wet objects directly on the sofa"),
			],
		},
	},

	# --- manipulable objects ---------------------------------------------------------
	"ball": {
		"kinds": ["object", "toy"],
		"attributes": {
			"state_vars": {
				"category": "toy",
				"materials": ["rubber", "fabric"],
				"affordances": ["roll", "throw"],
				"size_hint": "medium",
			},
		},
	},
	"clamp": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "fastener",
				"materials": ["metal", "plastic"],
				"affordances": ["grip", "secure"],
				"hazard": "pinch_points",
			},
		},
	},
	"hammer": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "tool",
				"materials": ["metal", "wood"],
				"affordances": ["hit", "drive_nails"],
				"hazard": "impact",
			},
		},
	},
	"screwdriver": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "tool",
				"materials": ["metal", "plastic"],
				"affordances": ["turn_screws"],
			},
		},
	},
	"padlock": {
		"kinds": ["object", "hardware"],
		"attributes": {
			"state_vars": {
				"category": "lock",
				"materials": ["metal"],
				"affordances": ["secure"],
			},
		},
	},
	"scissors": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "cutting_tool",
				"materials": ["metal", "plastic"],
				"affordances": ["cut"],
				"hazard": "sharp_edges",
			},
			"constraints": [
				_constraint("keep blades closed while carrying scissors"),
			],
		},
	},
	"block": {
		"kinds": ["object", "toy"],
		"attributes": {
			"state_vars": {
				"category": "toy",
				"materials": ["wood", "plastic"],
				"affordances": ["stack", "build"],
			},
		},
	},
	"drill": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "power_tool",
				"materials": ["metal", "plastic"],
				"affordances": ["drill"],
				"hazard": "rotating_bit",
			},
			"constraints": [
				_constraint("ensure the drill is powered off before handling"),
			],
		},
	},
	"spatula": {
		"kinds": ["object", "utensil"],
		"attributes": {
			"state_vars": {
				"category": "cooking_utensil",
				"materials": ["metal", "silicone"],
				"affordances": ["flip", "spread"],
				"supports_hot_items": True,
			},
		},
	},
	"knife": {
		"kinds": ["object", "utensil"],
		"attributes": {
			"state_vars": {
				"category": "cutlery",
				"materials": ["metal", "wood"],
				"affordances": ["slice"],
				"hazard": "sharp_edge",
			},
		},
	},
	"spoon": {
		"kinds": ["object", "utensil"],
		"attributes": {
			"state_vars": {
				"category": "cutlery",
				"materials": ["metal", "plastic"],
				"affordances": ["scoop"],
			},
		},
	},
	"plate": {
		"kinds": ["object", "tableware"],
		"attributes": {
			"state_vars": {
				"category": "tableware",
				"materials": ["ceramic", "glass"],
				"affordances": ["serve_food"],
				"supports_hot_items": True,
			},
		},
	},
	"sponge": {
		"kinds": ["object", "cleaning_tool"],
		"attributes": {
			"state_vars": {
				"category": "cleaning_tool",
				"materials": ["foam", "cellulose"],
				"affordances": ["absorb", "scrub"],
			},
		},
	},
	"cleanser": {
		"kinds": ["object", "cleaning_supply"],
		"attributes": {
			"state_vars": {
				"category": "chemical",
				"materials": ["liquid"],
				"affordances": ["clean"],
				"hazard": "avoid_eye_contact",
			},
		},
	},
	"can": {
		"kinds": ["object", "container"],
		"attributes": {
			"state_vars": {
				"category": "container",
				"materials": ["metal"],
				"affordances": ["store_liquid"],
			},
		},
	},
	"box": {
		"kinds": ["object", "container"],
		"attributes": {
			"state_vars": {
				"category": "container",
				"materials": ["cardboard", "plastic"],
				"affordances": ["store_items", "transport"],
				"is_container": True,
			},
		},
		"relations": [
			{"relation": "is_a", "target": "container", "confidence": 1.0},
		],
	},
	"lego": {
		"kinds": ["object", "toy"],
		"attributes": {
			"state_vars": {
				"category": "construction_toy",
				"materials": ["plastic"],
				"affordances": ["build", "connect"],
			},
		},
	},
	"rubriks cube": {
		"kinds": ["object", "puzzle"],
		"attributes": {
			"state_vars": {
				"category": "puzzle",
				"materials": ["plastic"],
				"affordances": ["rotate", "solve"],
			},
		},
		"aliases": ["rubik's cube"],
	},
	"book": {
		"kinds": ["object", "media"],
		"attributes": {
			"state_vars": {
				"category": "reading_material",
				"materials": ["paper"],
				"affordances": ["read", "stack"],
			},
		},
	},
	"bowl": {
		"kinds": ["object", "tableware", "receptacle"],
		"attributes": {
			"state_vars": {
				"category": "tableware",
				"materials": ["ceramic", "glass"],
				"affordances": ["hold_food", "hold_liquid"],
				"is_receptacle": True,
			},
		},
		"relations": [
			{"relation": "is_a", "target": "receptacle", "confidence": 0.9},
		],
	},
	"cup": {
		"kinds": ["object", "tableware"],
		"attributes": {
			"state_vars": {
				"category": "drinkware",
				"materials": ["ceramic", "glass"],
				"affordances": ["hold_beverage"],
			},
		},
	},
	"mug": {
		"kinds": ["object", "tableware"],
		"attributes": {
			"state_vars": {
				"category": "drinkware",
				"materials": ["ceramic"],
				"affordances": ["hold_hot_beverage"],
				"supports_hot_items": True,
			},
		},
	},
	"orange": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["orange"],
				"size_hint": "medium",
				"edibility": "edible",
			},
		},
	},
	"banana": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["yellow"],
				"shape_hint": "curved",
				"edibility": "edible",
			},
		},
	},
	"strawberry": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["red"],
				"top_features": ["green leaves"],
				"texture": "seeded surface",
				"size_hint": "small",
				"edibility": "edible",
			},
			"constraints": [
				_constraint("handle gently; bruises easily"),
			],
		},
	},
	"apple": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["red", "green", "yellow"],
				"edibility": "edible",
			},
		},
	},
	"pear": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["green", "yellow"],
				"shape_hint": "narrow neck",
				"edibility": "edible",
			},
		},
	},
	"peach": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["orange", "pink"],
				"texture": "fuzzy skin",
				"edibility": "edible",
			},
		},
	},
	"plum": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["purple", "red"],
				"edibility": "edible",
			},
		},
	},
	"lemon": {
		"kinds": ["object", "food"],
		"attributes": {
			"state_vars": {
				"category": "fruit",
				"colors": ["yellow"],
				"flavor_profile": "citrus",
				"edibility": "edible",
			},
		},
	},
	"lid": {
		"kinds": ["object", "kitchenware"],
		"attributes": {
			"state_vars": {
				"category": "kitchenware",
				"materials": ["metal", "glass"],
				"affordances": ["cover"],
			},
		},
	},
	"toy airplane": {
		"kinds": ["object", "toy"],
		"attributes": {
			"state_vars": {
				"category": "toy",
				"materials": ["plastic"],
				"affordances": ["play", "display"],
			},
		},
	},
	"wrench": {
		"kinds": ["object", "tool"],
		"attributes": {
			"state_vars": {
				"category": "tool",
				"materials": ["metal"],
				"affordances": ["tighten", "loosen"],
			},
		},
	},
}


# ============================================================
# 模板解析（继承/合并）
# ============================================================

def _register_canonical_label(label: Optional[str], canonical: str) -> None:
    norm = _normalize(label)
    if not norm:
        return
    _CANONICAL_LABEL_INDEX.setdefault(norm, canonical)

def _bootstrap_canonical_index() -> None:
    for label, data in _RAW_TEMPLATES.items():
        canonical = data.get("canonical_label") or label
        _register_canonical_label(label, canonical)
        for alias in data.get("aliases", []) or []:
            _register_canonical_label(alias, canonical)

_bootstrap_canonical_index()

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = merged[key] + [item for item in value if item not in merged[key]]
        else:
            merged[key] = deepcopy(value)
    return merged

def _resolve_template(label: str, seen: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    normalized = _normalize(label)
    data = _RAW_TEMPLATES.get(normalized)
    if not data:
        # try exact key without normalization
        data = _RAW_TEMPLATES.get(label)
    if not data:
        return {}

    seen = list(seen or [])
    if normalized in seen:
        raise ValueError(f"circular concept inheritance detected for '{label}'")
    seen.append(normalized)

    template = {k: deepcopy(v) for k, v in data.items() if k != "extends"}
    parent = data.get("extends")
    if parent:
        parent_template = _resolve_template(parent, seen)
        template = _merge_dicts(parent_template, template)

    template.setdefault("kinds", ["object"])
    attributes = template.setdefault("attributes", {})
    attributes.setdefault("state_vars", {})
    attributes.setdefault("constraints", [])
    template.setdefault("aliases", [])
    return template


# ============================================================
# 公开 API
# ============================================================

def get_concept_definition(label: str) -> Dict[str, Any]:
    """返回 label 的解析后模板（大小写不敏感）"""
    if not label:
        return {}
    template = _resolve_template(label)
    if not template:
        return {}
    aliases = template.get("aliases", [])
    normalized = _normalize(label)
    if normalized not in {_normalize(a) for a in aliases}:
        aliases.append(label)
    template["aliases"] = aliases
    template.setdefault("canonical_label", label)
    return template

def resolve_canonical_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    norm = _normalize(label)
    if not norm:
        return None
    canonical = _CANONICAL_LABEL_INDEX.get(norm)
    if canonical:
        return canonical
    template = get_concept_definition(label)
    if not template:
        return None
    canonical = template.get("canonical_label") or template.get("label")
    if not canonical:
        return None
    _register_canonical_label(label, canonical)
    return canonical

def iter_concept_labels() -> Iterable[str]:
    return sorted(set(_RAW_TEMPLATES.keys()))

def list_concept_templates() -> Dict[str, Dict[str, Any]]:
    return {label: get_concept_definition(label) for label in iter_concept_labels()}


# ============================================================
# 辅助：属性值 flatten
# ============================================================

def _flatten_attribute_values(value: Any) -> Iterable[str]:
    if value in (None, "") or isinstance(value, bool):
        return []
    if isinstance(value, (list, tuple, set)):
        flattened: List[str] = []
        for item in value:
            flattened.extend(_flatten_attribute_values(item))
        return flattened
    if isinstance(value, dict):
        # dict 不在 seed 的属性节点展开范围内（避免爆炸）
        return []
    return [str(value).strip()]


# ============================================================
# Seed graph 输出：label-based（兼容你原 build_ltm_seed_graph）
# ============================================================

def build_ltm_seed_graph(perspective: str = "seed") -> Dict[str, Any]:
	"""构建 LTM 种子图（适配 UnifiedMemoryGraph / MemoryRetriever）

	关键适配点：
	1) provenance 必须包含 "ltm"，否则 retriever 会过滤掉（比如用 ltm_seed）
	2) centrality / vagueness / gradience / perspective 等应写进 metadata（不是 properties）
	3) edge.metadata 要带 relation_type / is_symmetric / is_transitive / strength（尽量提供）
	"""
	nodes: List[Dict[str, Any]] = []
	edges: List[Dict[str, Any]] = []
	attribute_cache: Dict[Tuple[str, str], str] = {}
	node_labels: Set[str] = set()
	timestamp = time.time()

	def _edge_semantic_meta(rel: str) -> Dict[str, Any]:
		"""把 relation 映射到 unified_graph 的语义元信息（轻量规则版）"""
		rel_n = _normalize(rel)
		# 默认
		relation_type = "associative"
		is_sym = False
		is_trans = False
		strength = 0.7
		directionality = "directed"
		semantic_role = None

		if rel_n in ("is_a", "subclass_of", "instance_of"):
			relation_type = "taxonomic"
			is_trans = True
			strength = 0.95
		elif rel_n in ("has_part", "part_of"):
			relation_type = "partitive"
			is_trans = True
			strength = 0.9
		elif rel_n in ("synonym_of",):
			relation_type = "synonymy"
			is_sym = True
			directionality = "bidirectional"
			strength = 0.9
		elif rel_n in ("on", "in", "near", "located_at", "left_of", "right_of"):
			relation_type = "spatial"
			strength = 0.7
			semantic_role = "location"
		elif rel_n in ("before", "after", "during"):
			relation_type = "temporal"
			strength = 0.7

		return {
			"relation_type": relation_type,
			"is_symmetric": is_sym,
			"is_transitive": is_trans,
			"strength": strength,
			"directionality": directionality,
			"semantic_role": semantic_role,
		}

	# -------- pass1：节点 --------
	for label in iter_concept_labels():
		concept = get_concept_definition(label)
		canonical = concept.get("canonical_label", label)
		if canonical in node_labels:
			continue
		node_labels.add(canonical)

		attributes = concept.get("attributes", {}) or {}
		state_vars = attributes.get("state_vars", {}) or {}
		kinds = concept.get("kinds", ["object"])

		nodes.append(
			{
				"label": canonical,
				"properties": {
					"kinds": kinds,
					# 把 state_vars 展开进 properties（用于检索 token）
					**state_vars,
				},
				"metadata": {
					# ⚠️ provenance 必须带 ltm，否则检索过滤会丢
					"provenance": "ltm_seed",
					"ephemeral": False,
					"timestamp": timestamp,
					"confidence": 0.95,
					# 认知属性：统一写 metadata
					"centrality": 0.55 if "concept" in kinds else 0.45,
					"family_resemblance": 0.55,
					"prototypicality": 0.55,
					"perspective": perspective,
					"vagueness": 0.25,
					"gradience": 0.25,
				}
			}
		)

	# -------- pass2：边（属性边 + 显式 relations）--------
	for label in iter_concept_labels():
		concept = get_concept_definition(label)
		canonical = concept.get("canonical_label", label)
		attributes = concept.get("attributes", {}) or {}
		state_vars = attributes.get("state_vars", {}) or {}

		# 1) state_vars -> 属性节点 + 属性边
		for key, raw_value in state_vars.items():
			relation_meta = _ATTRIBUTE_KEY_RELATIONS.get(key)
			if not relation_meta:
				continue
			relation, attr_kind = relation_meta
			for attr_label in _flatten_attribute_values(raw_value):
				if not attr_label:
					continue

				cache_key = (attr_kind, attr_label.lower())
				if cache_key not in attribute_cache:
					attribute_cache[cache_key] = attr_label
					if attr_label not in node_labels:
						node_labels.add(attr_label)
						nodes.append(
							{
								"label": attr_label,
								"properties": {
									"kinds": ["attribute", attr_kind],
									"attribute_type": attr_kind,
								},
								"metadata": {
									"provenance": "ltm_seed",
									"ephemeral": False,
									"timestamp": timestamp,
									"confidence": 0.85,
									"centrality": 0.25,
									"family_resemblance": 0.4,
									"prototypicality": 0.4,
									"perspective": f"{perspective}_attr",
									"vagueness": 0.2,
									"gradience": 0.3,
								}
							}
						)

				meta_extra = _edge_semantic_meta(relation)
				edges.append(
					{
						"source_label": canonical,
						"target_label": attr_label,
						"relation": relation,
						"properties": {},
						"metadata": {
							"provenance": "ltm_seed",
							"ephemeral": False,
							"timestamp": timestamp,
							"confidence": 0.80,
							**meta_extra,
						}
					}
				)

		# 2) concept["relations"] 显式语义边
		explicit_relations = concept.get("relations", []) or []
		for rel_spec in explicit_relations:
			rel_type = rel_spec.get("relation")
			target_label = rel_spec.get("target")
			confidence = float(rel_spec.get("confidence", 0.9))

			if not rel_type or not target_label:
				continue

			if target_label not in node_labels:
				node_labels.add(target_label)
				nodes.append(
					{
						"label": target_label,
						"properties": {
							"kinds": ["concept", "abstract"],
						},
						"metadata": {
							"provenance": "ltm_seed",
							"ephemeral": False,
							"timestamp": timestamp,
							"confidence": 0.9,
							"centrality": 0.5,
							"family_resemblance": 0.55,
							"prototypicality": 0.55,
							"perspective": f"{perspective}_implicit",
							"vagueness": 0.25,
							"gradience": 0.25,
						}
					}
				)

			meta_extra = _edge_semantic_meta(rel_type)
			edges.append(
				{
					"source_label": canonical,
					"target_label": target_label,
					"relation": rel_type,
					"properties": {},
					"metadata": {
						"provenance": "ltm_seed",
						"ephemeral": False,
						"timestamp": timestamp,
						"confidence": confidence,
						**meta_extra,
					}
				}
			)

	return {"nodes": nodes, "edges": edges}


# ============================================================
# 直接构建 UnifiedMemoryGraph（推荐给上层初始化用）
# ============================================================

def build_ltm_seed_unified_graph(perspective: str = "seed") -> UnifiedMemoryGraph:
    """
    直接返回 UnifiedMemoryGraph，适合在 SemanticMemoryManager 初始化时加载：
        g = build_ltm_seed_unified_graph()
        manager.ltm = g
    """
    data = build_ltm_seed_graph(perspective=perspective)
    g = UnifiedMemoryGraph()

    # label -> node_id
    label_to_id: Dict[str, str] = {}

    # 1) nodes
    for n in data.get("nodes", []) or []:
        label = n["label"]
        props = dict(n.get("properties", {}) or {})
        meta_d = dict(n.get("metadata", {}) or {})

        # meta_d 是 NodeMetadata.__dict__，字段名应匹配 NodeMetadata
        nm = NodeMetadata(**_filter_node_metadata_kwargs(meta_d))
        node = g.add_node(label=label, properties=props, metadata=nm)
        label_to_id[label] = node.id

    # 2) edges
    for e in data.get("edges", []) or []:
        s_lbl = e["source_label"]
        t_lbl = e["target_label"]
        rel = e["relation"]

        sid = label_to_id.get(s_lbl)
        tid = label_to_id.get(t_lbl)
        if not sid or not tid:
            # 少数情况下 edges 里出现未建节点：补建
            if not sid:
                nm = NodeMetadata(confidence=0.75, provenance="ltm", ephemeral=False, ttl=9999, timestamp=time.time())
                node = g.add_node(label=s_lbl, properties={"kinds": ["concept"]}, metadata=nm)
                sid = node.id
                label_to_id[s_lbl] = sid
            if not tid:
                nm = NodeMetadata(confidence=0.75, provenance="ltm", ephemeral=False, ttl=9999, timestamp=time.time())
                node = g.add_node(label=t_lbl, properties={"kinds": ["concept"]}, metadata=nm)
                tid = node.id
                label_to_id[t_lbl] = tid

        meta_d = dict(e.get("metadata", {}) or {})
        em = EdgeMetadata(**_filter_edge_metadata_kwargs(meta_d))

        g.add_edge(sid, tid, rel, properties=dict(e.get("properties", {}) or {}), metadata=em)

    return g


def apply_seed_to_ltm(ltm_graph: UnifiedMemoryGraph, perspective: str = "seed") -> None:
    """
    将 seed 写入已有的 LTM 图：
    - 已存在同 label 节点：更新 properties（不覆盖全部，采用 update）
    - 边按 (source,target,relation) 去重
    """
    data = build_ltm_seed_graph(perspective=perspective)

    # label->node_id（复用已有节点）
    label_to_id: Dict[str, str] = {}

    for n in data.get("nodes", []) or []:
        label = n["label"]
        props = dict(n.get("properties", {}) or {})
        meta_d = dict(n.get("metadata", {}) or {})

        existing = ltm_graph.find_nodes(label=label, exact=True, limit=1)
        if existing:
            node = existing[0]
            node.properties.update(props)
            # provenance 强制包含 ltm
            node.metadata.provenance = "ltm"
            # 置信度取更大
            node.metadata.confidence = min(0.99, max(float(node.metadata.confidence), float(meta_d.get("confidence", 0.85))))
            label_to_id[label] = node.id
        else:
            nm = NodeMetadata(**_filter_node_metadata_kwargs(meta_d))
            node = ltm_graph.add_node(label=label, properties=props, metadata=nm)
            label_to_id[label] = node.id

    for e in data.get("edges", []) or []:
        s_lbl = e["source_label"]
        t_lbl = e["target_label"]
        rel = e["relation"]

        sid = label_to_id.get(s_lbl)
        tid = label_to_id.get(t_lbl)
        if not sid or not tid:
            continue

        if ltm_graph.find_edges(source=sid, target=tid, relation=rel, limit=1):
            continue

        meta_d = dict(e.get("metadata", {}) or {})
        em = EdgeMetadata(**_filter_edge_metadata_kwargs(meta_d))
        ltm_graph.add_edge(sid, tid, rel, properties=dict(e.get("properties", {}) or {}), metadata=em)


# ============================================================
# 抽象->具体映射（你原函数，保持）
# ============================================================

def build_abstract_concrete_mapping() -> Dict[str, List[str]]:
    """构建抽象概念到具体实例的映射表"""
    mapping: Dict[str, List[str]] = {}
    for label in iter_concept_labels():
        concept = get_concept_definition(label)
        state_vars = concept.get("attributes", {}).get("state_vars", {}) or {}
        concrete_instances = state_vars.get("concrete_instances", [])
        if concrete_instances:
            mapping[label] = list(concrete_instances)
    return mapping


# ============================================================
# 内部：构造 EdgeMetadata / 过滤 metadata kwargs
# ============================================================

def _make_edge_metadata(relation: str, confidence: float, ts: float, provenance: str = "ltm") -> EdgeMetadata:
    """
    为某个 relation 构造一套一致的语义属性（对齐上层语义检索/遍历）
    """
    sem = _RELATION_SEMANTICS.get(relation, {})
    return EdgeMetadata(
        confidence=float(confidence),
        provenance=provenance,   # ✅ 必须含 ltm（建议就是 ltm）
        ephemeral=False,
        ttl=9999,
        timestamp=float(ts),

        relation_type=str(sem.get("relation_type", "associative")),
        is_symmetric=bool(sem.get("is_symmetric", False)),
        is_transitive=bool(sem.get("is_transitive", False)),
        strength=float(sem.get("strength", 0.6)),
        directionality=str(sem.get("directionality", "directed")),
        semantic_role=sem.get("semantic_role", None),
    )

def _filter_node_metadata_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    防止 seed 的 metadata 字段和 NodeMetadata 不一致导致构造报错。
    """
    allowed = {
        "confidence", "timestamp", "ttl", "provenance", "ephemeral", "last_access", "access_count",
        "centrality", "family_resemblance", "perspective", "vagueness", "gradience", "prototypicality",
    }
    out = {k: v for k, v in (d or {}).items() if k in allowed}
    # 兜底：确保 provenance 含 ltm
    out["provenance"] = "ltm"
    out.setdefault("ttl", 9999)
    out.setdefault("ephemeral", False)
    out.setdefault("confidence", 0.85)
    out.setdefault("timestamp", time.time())
    out.setdefault("centrality", 0.45)
    out.setdefault("family_resemblance", 0.45)
    out.setdefault("prototypicality", 0.45)
    out.setdefault("vagueness", 0.20)
    out.setdefault("gradience", 0.20)
    return out

def _filter_edge_metadata_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    防止 seed 的 metadata 字段和 EdgeMetadata 不一致导致构造报错。
    """
    allowed = {
        "confidence", "timestamp", "ttl", "provenance", "ephemeral", "last_access", "access_count",
        "relation_type", "is_symmetric", "is_transitive", "strength", "directionality", "semantic_role",
    }
    out = {k: v for k, v in (d or {}).items() if k in allowed}
    out["provenance"] = "ltm"
    out.setdefault("ttl", 9999)
    out.setdefault("ephemeral", False)
    out.setdefault("confidence", 0.85)
    out.setdefault("timestamp", time.time())
    out.setdefault("relation_type", "associative")
    out.setdefault("is_symmetric", False)
    out.setdefault("is_transitive", False)
    out.setdefault("strength", 0.6)
    out.setdefault("directionality", "directed")
    out.setdefault("semantic_role", None)
    return out
