# EmbodiedBench 完整工作流程详解

## 任务示例
**指令**：`Deliver a small red object with green top to the intended a large gray piece of furniture with a backrest by physically moving it there.`

**目标**：找到草莓（小红色物体+绿色顶部），搬到沙发（大灰色家具+靠背）上。

---

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                   MetaFlatHabitatAgent                      │
│  (主控制器：感知-记忆-规划-执行循环)                          │
└─────────────────────────────────────────────────────────────┘
                          ▼
        ┌─────────────────┬──────────────┬─────────────┐
        ▼                 ▼              ▼             ▼
  PerceptionModule  SemanticMemory  VLMPlanner  FeedbackController
  (多模态感知)      (WM+LTM记忆)    (规划器)     (反馈控制)
```

---

## 完整工作流程（Episode级别）

### 阶段0：Episode初始化

**时间点**：Episode开始

**代码位置**：`meta_flat_habitat_agent.py::run_episode()` Line 1530-1560

```python
# 1. 重置环境
obs = env.reset()

# 2. 创建WorkingMemory（工作记忆 - episode级别草稿纸）
wm = WorkingMemory(instruction=instruction)

# 3. 初始化FeedbackState（反馈控制状态）
wm.feedback_state = FeedbackState()

# 4. 重置SemanticMemoryManager（WM清空，LTM保留）
self.semantic_memory.reset_wm()  # WM每个episode重置
# LTM跨episode持久化

# 5. 初始感知（看一眼当前场景）
initial_perception = self.perceive_initial(env, instruction)
```

**WM状态**：空图，0个节点，0条边  
**LTM状态**：持久化图，423个节点，1163条边（来自之前episodes）

---

### 阶段1：初始感知与记忆摄入

#### 1.1 初始感知（Vision → Text）

**时间点**：Step 0（执行任何动作前）

**代码位置**：`meta_flat_habitat_agent.py::perceive_initial()` Line 440-590

**流程**：

```python
def perceive_initial(env, instruction) -> PerceptionOutput:
    # 1. 获取RGB图像
    rgb_img = env.render()  # shape: (H, W, 3)
    
    # 2. 保存图像到磁盘
    img_path = save_image(rgb_img, episode_id, step=0)
    
    # 3. 构建初始感知提示词
    system_prompt = """
    You are a perception module for a household robot.
    Analyze the RGB image and extract:
    - Scene summary
    - Detected objects
    - Object attributes (color, size, material, shape, state, location)
    - Spatial relations
    - Instruction entities (objects mentioned in the task)
    
    Output JSON format:
    {
      "scene_summary": "...",
      "detected_objects": ["object1", "object2", ...],
      "detected_object_attributes": [
        {
          "label": "object1",
          "attributes": {
            "color": "red",
            "size": "small",
            "container_or_surface": "on the table",
            "relative_position": "left of X"
          },
          "confidence": 0.9
        }
      ],
      "objects_relations": ["X is near Y", ...],
      "instruction_entities_and_attributes": [
        "small red object with green top",
        "large gray piece of furniture with a backrest"
      ]
    }
    """
    
    # 4. 调用VLM（GPT-4o-mini with vision）
    response = call_vlm(
        system_prompt=system_prompt,
        user_text=f"Instruction: {instruction}",
        image_path=img_path
    )
    
    # 5. 解析JSON输出
    parsed = json.loads(response)
    
    # 6. 🔥 关键转换：从detected_object_attributes提取object_locations
    object_locations = []
    for attr_dict in parsed["detected_object_attributes"]:
        container = attr_dict["attributes"].get("container_or_surface", "")
        
        if container and container != "n/a":
            # 解析 "on the table" -> relation="on", container="table"
            spatial_rel = "located_at"
            container_name = container
            
            for prefix, rel in [("on the ", "on"), ("in the ", "in"), 
                               ("inside the ", "in"), ("near the ", "near")]:
                if container.startswith(prefix):
                    spatial_rel = rel
                    container_name = container[len(prefix):].strip()
                    break
            
            object_locations.append({
                "object": attr_dict["label"],
                "container": container_name,
                "spatial_relation": spatial_rel
            })
    
    # 7. 返回结构化感知
    return PerceptionOutput(
        scene_summary=parsed["scene_summary"],
        detected_objects=parsed["detected_objects"],
        detected_object_attributes=parsed["detected_object_attributes"],
        objects_relations=parsed["objects_relations"],
        object_locations=object_locations,  # 🔥 新增
        instruction_entities_and_attributes=parsed["instruction_entities_and_attributes"]
    )
```

**初始感知输出示例**：
```json
{
  "scene_summary": "Living room with gray couch, small table, shelf, and gray rug",
  "detected_objects": ["gray piece of furniture", "small table", "shelf", "rug"],
  "detected_object_attributes": [
    {
      "label": "gray piece of furniture",
      "attributes": {
        "color": "gray",
        "size": "large",
        "shape": "rectangular",
        "container_or_surface": "in the living area",
        "relative_position": "near the wall"
      },
      "confidence": 0.9
    }
  ],
  "object_locations": [
    {"object": "rug", "container": "floor", "spatial_relation": "on"}
  ],
  "instruction_entities_and_attributes": [
    "small red object with green top",
    "large gray piece of furniture with a backrest"
  ]
}
```

#### 1.2 感知摄入到WM（Perception → WM Graph）

**时间点**：Step 0，初始感知后立即执行

**代码位置**：`semantic_memory.py::ingest_perception()` Line 1445-1610

**流程图**：

```
PerceptionOutput
    ↓
┌─────────────────────────────────────────────────┐
│  SemanticMemoryManager.ingest_perception()      │
├─────────────────────────────────────────────────┤
│  1. 摄入指令实体（instruction entities）         │
│     ├─ 创建节点：type="instruction_entity"       │
│     └─ 创建边：instruction_requirements --[requires]--> entity │
│                                                 │
│  2. 摄入检测到的对象（detected objects）          │
│     ├─ 创建节点：type="object"                   │
│     ├─ 标记：visible=True, last_seen_ts=当前时间  │
│     └─ 如果有ltm_id，链接到LTM                   │
│                                                 │
│  3. 摄入对象属性（object attributes）            │
│     ├─ 解析颜色、大小、材质、形状、状态           │
│     ├─ 创建属性节点：type="attribute"            │
│     └─ 创建边：object --[has_color/has_size]--> attribute │
│                                                 │
│  4. 摄入空间关系（spatial relations）            │
│     ├─ 解析 "X is near Y"                       │
│     └─ 创建边：X --[near]--> Y                  │
│                                                 │
│  5. 摄入状态变化（state changes）                │
│     └─ 记录到WM的change log                     │
└─────────────────────────────────────────────────┘
    ↓
WM Graph (UnifiedMemoryGraph)
```

**详细算子工作流程**：

```python
def ingest_perception(perception: PerceptionOutput, step: int):
    """
    将PerceptionOutput转换为WM图节点和边
    """
    
    # ===== 算子1: 摄入指令实体 =====
    # 目的：将任务要求的目标对象固化到WM
    for entity_text in perception.instruction_entities_and_attributes:
        # 1. 创建instruction_entity节点
        entity_node = self.wm.add_node(
            label=entity_text,
            properties={
                "type": "instruction_entity",
                "source": "instruction_parsing"
            },
            metadata=NodeMetadata(
                confidence=0.90,
                provenance="initial_perception",
                timestamp=step
            )
        )
        
        # 2. 链接到任务需求节点
        req_node = self.wm.get_or_create_node(
            label="instruction_requirements",
            properties={"type": "goal"}
        )
        
        self.wm.add_edge(
            source=req_node.id,
            target=entity_node.id,
            relation="requires",
            metadata=EdgeMetadata(confidence=0.95)
        )
    
    # ===== 算子2: 摄入检测对象 =====
    for obj_label in perception.detected_objects:
        obj_node = self.wm.add_node(
            label=obj_label,
            properties={
                "type": "object",
                "visible": True,
                "last_seen_ts": time.time()
            },
            metadata=NodeMetadata(
                confidence=0.85,
                provenance="vision",
                timestamp=step
            )
        )
    
    # ===== 算子3: 摄入对象属性 =====
    for attr_dict in perception.detected_object_attributes:
        obj_label = attr_dict["label"]
        attrs = attr_dict["attributes"]
        
        # 3.1 颜色属性
        if "color" in attrs:
            colors = attrs["color"] if isinstance(attrs["color"], list) else [attrs["color"]]
            for color in colors:
                color_node = self.wm.add_node(
                    label=color,
                    properties={"type": "attribute", "attribute_type": "color"}
                )
                self.wm.add_edge(
                    source=obj_label,
                    target=color,
                    relation="has_color",
                    metadata=EdgeMetadata(confidence=0.85)
                )
        
        # 3.2 大小属性
        if "size" in attrs:
            size_node = self.wm.add_node(
                label=attrs["size"],
                properties={"type": "attribute", "attribute_type": "size"}
            )
            self.wm.add_edge(
                source=obj_label,
                target=attrs["size"],
                relation="has_size"
            )
        
        # 3.3 材质、形状、状态...（同理）
    
    # ===== 算子4: 摄入空间关系 =====
    for relation_text in perception.objects_relations:
        # 解析 "shelf is to the left of gray piece of furniture"
        # → shelf --[left_of]--> gray piece of furniture
        parsed = parse_spatial_relation(relation_text)
        
        self.wm.add_edge(
            source=parsed.subject,
            target=parsed.object,
            relation=parsed.relation,
            metadata=EdgeMetadata(confidence=0.75)
        )
    
    # ===== 算子5: 摄入对象位置 =====
    # 🔥 关键！用于后续更新到LTM
    for loc_item in perception.object_locations:
        obj_label = loc_item["object"]
        container = loc_item["container"]
        spatial_rel = loc_item["spatial_relation"]
        
        # 创建位置节点
        location_node = self.wm.add_node(
            label=container,
            properties={"type": "location"}
        )
        
        # 创建空间关系边
        self.wm.add_edge(
            source=obj_label,
            target=container,
            relation=spatial_rel,  # "on", "in", "near"
            metadata=EdgeMetadata(confidence=0.75)
        )
```

**WM状态变化**：

**摄入前**：
- 节点数：0
- 边数：0

**摄入后**（Step 0）：
- 节点数：19
  - 2个 instruction_entity：`small red object with green top`, `large gray piece of furniture with a backrest`
  - 4个 object：`gray piece of furniture`, `small table`, `shelf`, `rug`
  - 11个 attribute：`gray`, `large`, `brown`, `small`, `medium`, `rectangular`, `wood`, `fabric`, `empty`, `clean`
  - 1个 location：`front of the gray piece of furniture`
  - 1个 goal：`instruction_requirements`
- 边数：28
  - 2条 requires：`instruction_requirements --[requires]--> entity`
  - 4条 has_color：`object --[has_color]--> color`
  - 4条 has_size：`object --[has_size]--> size`
  - 4条 has_material：`object --[has_material]--> material`
  - 3条 spatial：`shelf --[left_of]--> gray piece of furniture`
  - 1条 usually_found_at：`rug --[usually_found_at]--> front of...`

---

### 阶段2：WM更新到LTM（跨episode知识积累）

**时间点**：每个感知摄入后立即执行

**代码位置**：`semantic_memory.py::update_ltm_from_perception()` Line 1915-2110

**目的**：将当前episode的WM中稳定、高置信度的知识写入LTM，供未来episodes使用

**流程图**：

```
WM Graph (当前episode)
    ↓
┌─────────────────────────────────────────────────┐
│  update_ltm_from_perception()                   │
├─────────────────────────────────────────────────┤
│  1. 提取object_locations（对象位置信息）         │
│     ├─ 读取：perception.object_locations        │
│     ├─ 格式：{object, container, spatial_relation} │
│     └─ 目的：建立 object --[usually_found_at]--> location 知识 │
│                                                 │
│  2. 创建/更新LTM节点                            │
│     ├─ 对象节点：type="object"                  │
│     ├─ 位置节点：type="location"                │
│     └─ 检查是否已存在（去重）                    │
│                                                 │
│  3. 创建空间关系边                              │
│     ├─ 映射关系：                               │
│     │   on/in/inside → usually_found_at       │
│     │   near/next_to → spatial_near           │
│     │   at → located_at                        │
│     └─ 写入LTM图                                │
│                                                 │
│  4. 摄入对象属性到LTM                           │
│     ├─ 从detected_object_attributes提取         │
│     ├─ 创建属性边：has_color, has_size, etc.    │
│     └─ 仅写入高置信度属性（>0.7）                │
└─────────────────────────────────────────────────┘
    ↓
LTM Graph (持久化)
```

**详细算子流程**：

```python
def update_ltm_from_perception(perception: PerceptionOutput, step: int):
    """
    将感知中的稳定知识写入LTM
    """
    
    # ===== 算子1: 处理对象位置（最重要！）=====
    object_locations = perception.object_locations  # 已在perceive阶段提取
    
    logger.info(f"[LTM Update] Processing {len(object_locations)} object locations...")
    
    for loc_item in object_locations:
        obj_label = loc_item["object"]  # "yellow cup"
        container_label = loc_item["container"]  # "counter"
        spatial_rel = loc_item["spatial_relation"]  # "on"
        
        # 1.1 在LTM中创建/查找对象节点
        obj_nodes = self.ltm.find_nodes(label=obj_label, exact=False, limit=1)
        if not obj_nodes:
            obj_node = self.ltm.add_node(
                label=obj_label,
                properties={"type": "object", "kinds": ["object"]},
                metadata=NodeMetadata(
                    confidence=0.80,
                    provenance="perception_to_ltm",
                    timestamp=step
                )
            )
        else:
            obj_node = obj_nodes[0]
        
        # 1.2 在LTM中创建/查找位置节点
        container_nodes = self.ltm.find_nodes(label=container_label, exact=False, limit=1)
        if not container_nodes:
            container_node = self.ltm.add_node(
                label=container_label,
                properties={"type": "location", "kinds": ["location", "container"]},
                metadata=NodeMetadata(confidence=0.80)
            )
        else:
            container_node = container_nodes[0]
        
        # 1.3 映射空间关系到LTM边类型
        relation_mapping = {
            "on": "usually_found_at",
            "in": "usually_found_at",
            "inside": "usually_found_at",
            "at": "located_at",
            "near": "spatial_near",
            "next_to": "spatial_near"
        }
        
        ltm_relation = relation_mapping.get(spatial_rel, "located_at")
        
        # 1.4 在LTM中创建空间关系边
        self.ltm.add_edge(
            source=obj_node.id,
            target=container_node.id,
            relation=ltm_relation,
            metadata=EdgeMetadata(
                confidence=0.85,
                provenance="perception_spatial",
                timestamp=step,
                relation_type="spatial"
            )
        )
        
        logger.info(f"  ✓ {obj_label} --[{ltm_relation}]--> {container_label}")
    
    # ===== 算子2: 摄入对象属性 =====
    for attr_dict in perception.detected_object_attributes:
        obj_label = attr_dict["label"]
        attrs = attr_dict["attributes"]
        confidence = attr_dict.get("confidence", 0.8)
        
        # 只摄入高置信度属性
        if confidence < 0.7:
            continue
        
        # 2.1 查找/创建对象节点
        obj_nodes = self.ltm.find_nodes(label=obj_label, exact=False, limit=1)
        if not obj_nodes:
            obj_node = self.ltm.add_node(
                label=obj_label,
                properties={"type": "object"}
            )
        else:
            obj_node = obj_nodes[0]
        
        # 2.2 添加属性
        for attr_type in ["color", "size", "material", "shape", "state"]:
            if attr_type in attrs and attrs[attr_type]:
                attr_value = attrs[attr_type]
                if isinstance(attr_value, list):
                    attr_value = attr_value[0]  # 取第一个
                
                # 创建属性节点
                attr_node = self.ltm.add_node(
                    label=str(attr_value),
                    properties={"type": "attribute", "attribute_type": attr_type}
                )
                
                # 创建属性边
                self.ltm.add_edge(
                    source=obj_node.id,
                    target=attr_node.id,
                    relation=f"has_{attr_type}",
                    metadata=EdgeMetadata(confidence=0.85)
                )
```

**LTM状态变化**：

**更新前**（Episode开始）：
- 节点数：423
- 边数：1163
- 包含之前episodes学到的知识

**更新后**（Step 0感知后）：
- 节点数：430（+7）
  - 新增：`rug`, `gray piece of furniture`, `small table`, `shelf`, `floor`, `living area`, `wall`
- 边数：1185（+22）
  - 新增：`rug --[usually_found_at]--> floor`
  - 新增：`rug --[has_color]--> gray`
  - 新增：`shelf --[has_material]--> wood`
  - 等等...

---

### 阶段3：从LTM检索先验知识（规划准备）

**时间点**：每次规划前

**代码位置**：`semantic_memory.py::prepare_planner_context()` Line 1645-1710

**目的**：从LTM中检索与当前任务相关的先验知识，辅助规划

**流程图**：

```
Goal: "Deliver a small red object with green top..."
    ↓
┌─────────────────────────────────────────────────┐
│  prepare_planner_context()                      │
├─────────────────────────────────────────────────┤
│  Step 1: 从LTM检索（integrate_from_ltm）        │
│  ├─ 提取goal关键词：["deliver", "small", "red",│
│  │   "object", "green", "top", "large", "gray", │
│  │   "furniture", "backrest"]                   │
│  │                                              │
│  ├─ 在WM中找anchor节点：                        │
│  │   ├─ 优先：instruction_entity节点            │
│  │   ├─ 次优：object节点                        │
│  │   └─ 基于关键词：semantic_find_nodes         │
│  │                                              │
│  ├─ 对每个anchor，调用MemoryRetriever：         │
│  │   └─ 在LTM中找相似节点                       │
│  │       ├─ 语义相似度（embedding cosine）      │
│  │       ├─ 标签匹配                            │
│  │       └─ 返回topk最相关节点                  │
│  │                                              │
│  └─ 写回WM：                                    │
│      └─ wm_node.properties["ltm_retrieval"] = { │
│           "selected_anchor": ltm_node_id,       │
│           "final_confidence": 0.95,             │
│           "neighbors": [...]                    │
│         }                                       │
│                                                 │
│  Step 2: 从WM读取子图（read）                   │
│  ├─ 语义索引检索seeds                           │
│  ├─ Personalized PageRank扩展                  │
│  └─ 返回相关节点和边                            │
└─────────────────────────────────────────────────┘
    ↓
PlannerContext: {
  nodes: 30个,
  edges: 41条,
  ltm_hints: 8条
}
```

**详细算子流程**：

```python
def prepare_planner_context(goal: str, step: int) -> PlannerContext:
    """
    为规划器准备上下文
    """
    
    # ===== Step 1: 从LTM检索先验知识 =====
    ltm_hints = self.integrate_from_ltm(goal, step)
    
    # integrate_from_ltm 内部流程：
    def integrate_from_ltm(goal: str, step: int):
        # 1.1 提取goal关键词
        keywords = extract_goal_keywords(goal)
        # → ["deliver", "small", "red", "object", "green", "top", ...]
        
        # 1.2 在WM中找anchor节点
        anchor_ids = set()
        
        # 优先1：所有instruction_entity节点
        instruction_entities = self.wm.find_nodes(
            property_filters={"type": "instruction_entity"},
            limit=50
        )
        for entity_node in instruction_entities:
            anchor_ids.add(entity_node.id)
            # 加入：small red object with green top
            #      large gray piece of furniture with a backrest
        
        # 优先2：所有object节点
        object_nodes = self.wm.find_nodes(
            property_filters={"type": "object"},
            limit=50
        )
        for obj_node in object_nodes:
            anchor_ids.add(obj_node.id)
            # 加入：gray piece of furniture, small table, shelf, rug
        
        # 优先3：基于关键词的语义检索
        for kw in keywords[:8]:
            ranked = self.wm.semantic_find_nodes(kw, limit=6)
            for node, score in ranked:
                anchor_ids.add(node.id)
        
        # 1.3 批量检索（调用MemoryRetriever）
        self.retriever.retrieve_batch(self.wm, list(anchor_ids))
        
        # retriever.retrieve_batch 内部流程：
        def retrieve_batch(wm: UnifiedMemoryGraph, anchor_ids: List[str]):
            for anchor_id in anchor_ids:
                anchor_node = wm.get_node(anchor_id)
                
                # 在LTM中搜索相似节点
                ltm_candidates = self.ltm.semantic_find_nodes(
                    query=anchor_node.label,
                    limit=10
                )
                
                # 选择最佳匹配
                best_match = ltm_candidates[0]  # (node, score)
                
                # 写回WM节点的properties
                anchor_node.properties["ltm_retrieval"] = {
                    "selected_anchor": best_match[0].id,
                    "final_confidence": best_match[1],
                    "query_label": anchor_node.label,
                    "retrieved_label": best_match[0].label
                }
        
        # 1.4 收集hints返回
        hints = []
        for anchor_id in anchor_ids:
            node = self.wm.get_node(anchor_id)
            bundle = node.properties.get("ltm_retrieval", {})
            
            if bundle.get("selected_anchor"):
                # 🔥 转换节点ID为标签
                ltm_node_id = bundle["selected_anchor"]
                ltm_node = self.ltm.get_node(ltm_node_id)
                
                hints.append({
                    "wm_node": node.label,
                    "selected_anchor": ltm_node.label if ltm_node else ltm_node_id,
                    "selected_anchor_id": ltm_node_id,
                    "final_confidence": bundle.get("final_confidence", 0.0)
                })
        
        return hints
    
    # ===== Step 2: 从WM读取相关子图 =====
    nodes, edges, gaps = self.read(goal, step, max_nodes=30)
    
    # read 内部流程：
    def read(goal: str, step: int, max_nodes: int):
        # 2.1 语义索引检索seeds
        hits = self.wm_index.query(goal, topk=10)
        seed_ids = {node_id: score for node_id, score in hits}
        
        # 2.2 优先instruction_entity节点
        instruction_entities = self.wm.find_nodes(
            property_filters={"type": "instruction_entity"},
            limit=20
        )
        for entity in instruction_entities:
            seed_ids[entity.id] = 0.9  # 高初始分数
            
            # 同时加入该实体的属性邻居
            for edge in self.wm.edges.values():
                if edge.source == entity.id and "has_" in edge.relation:
                    seed_ids[edge.target] = 0.85
        
        # 2.3 Personalized PageRank扩展
        ppr_scores = personalized_pagerank(
            graph=self.wm,
            seeds=seed_ids,
            steps=10,
            restart=0.35
        )
        
        # 2.4 选择topk节点
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [nid for nid, _ in sorted_nodes[:max_nodes]]
        
        nodes = [self.wm.get_node(nid) for nid in selected_ids]
        
        # 2.5 提取相关边
        edges = []
        for edge in self.wm.edges.values():
            if edge.source in selected_ids and edge.target in selected_ids:
                edges.append(edge)
        
        # 2.6 检测信息缺口
        gaps = []
        for keyword in extract_goal_keywords(goal):
            if not self.wm.find_nodes(label=keyword, exact=False):
                gaps.append(f"missing_concept:{keyword}")
        
        return nodes, edges, gaps
    
    # ===== Step 3: 转换ltm_hints的节点ID为标签 =====
    ltm_hints_with_labels = []
    for hint in ltm_hints:
        hint_copy = hint.copy()
        selected_anchor = hint.get('selected_anchor', '')
        
        if selected_anchor.startswith('n_'):
            ltm_node = self.ltm.get_node(selected_anchor)
            if ltm_node:
                hint_copy['selected_anchor'] = ltm_node.label
        
        ltm_hints_with_labels.append(hint_copy)
    
    # ===== Step 4: 构建PlannerContext =====
    return PlannerContext(
        goal=goal,
        nodes=[n.to_dict() for n in nodes],
        edges=[e.to_dict() for e in edges],
        gaps=gaps,
        ltm_hints=ltm_hints_with_labels
    )
```

**检索结果示例**：

```
LTM检索hints (8条):
  1. WM:rug ← LTM:rug (conf=0.95)
  2. WM:gray piece of furniture ← LTM:gray piece of furniture (conf=0.97)
  3. WM:small red object with green top ← LTM:small red object with green top near table (conf=1.00)
  4. WM:small ← LTM:small (conf=1.00)
  5. WM:shelf ← LTM:shelf (conf=0.85)
  6. WM:large gray piece of furniture with a backrest ← LTM:couch (conf=0.85)
  7. WM:large ← LTM:large (conf=1.00)
  8. WM:small table ← LTM:small table (conf=1.00)

WM子图 (30节点, 41边):
  节点：instruction_entity, object, attribute, location
  边：requires, has_color, has_size, usually_found_at, near, left_of, ...

信息缺口 (2条):
  - missing_concept:deliver
  - missing_concept:intended
```

---

### 阶段4：规划（Planning）

**时间点**：每次需要新plan时（初始 + replan）

**代码位置**：`meta_flat_habitat_agent.py::_plan_with_vlm()` Line 1064-1097

**当前实现**：固定68动作序列（用于baseline测试）

```python
def _plan_with_vlm(img_path, instruction, wm) -> ActionPlan:
    """
    🔥 当前版本：返回固定动作序列（0-67）
    用于建立确定性基线，不调用LLM规划
    """
    return ActionPlan(
        action_ids=list(range(68)),  # [0, 1, 2, ..., 67]
        reasoning="Fixed action sequence: executing all 68 actions (0-67) in order."
    )
```

**原始LLM规划流程**（已注释）：

```python
def _plan_with_vlm_original(img_path, instruction, wm) -> ActionPlan:
    """
    原始VLM规划流程
    """
    
    # 1. 准备语义上下文
    semantic_ctx = wm.semantic_memory_digest[-1] if wm.semantic_memory_digest else None
    
    # 2. 构建规划提示词
    planner_prompt = f"""
    Instruction: {instruction}
    
    ### Scenario context for planning
    {semantic_ctx.to_text()}  # 包含：goal, 相关对象, 关系, 信息缺口, LTM先验知识
    
    Current view and environment:
    - Scene overview: {wm.latest_perception().scene_summary}
    
    ### Planning heuristics
    - GOAL-FIRST SEARCH: 理解目标 → 确认目标位置 → 直接导航
    - WM-FIRST: 优先使用WM已知位置
    - NO REVISITS: 不重复访问相同位置
    - NO INVALID REPEAT: 不重复invalid动作
    
    Output JSON:
    {
      "visual_state_description": "...",
      "reasoning_and_reflection": "...",
      "executable_plan": [
        {"action_id": 6, "action_name": "navigate to the table 1"},
        {"action_id": 16, "action_name": "pick up the ball"},
        ...
      ],
      "language_plan": "..."
    }
    """
    
    # 3. 调用VLM
    response = call_vlm(planner_prompt, image_path=img_path)
    
    # 4. 解析JSON
    parsed = json.loads(response)
    action_ids = [item["action_id"] for item in parsed["executable_plan"]]
    
    return ActionPlan(
        action_ids=action_ids,
        reasoning=json.dumps(parsed)
    )
```

---

### 阶段5：执行与后置感知（Execution & Post-Action Perception）

**时间点**：每个step

**代码位置**：`meta_flat_habitat_agent.py::run_episode()` Line 1600-1800

**流程**：

```python
# 主循环
for step in range(max_steps):
    
    # ===== 5.1 执行动作 =====
    action_id = current_plan.action_ids[plan_pointer]
    obs, reward, done, info = env.step(action_id)
    
    # 记录执行轨迹
    wm.execution_trace.append(ExecutionStepTrace(
        env_step=step,
        action_id=action_id,
        action_desc=action_descriptions[action_id],
        reward=reward,
        info=info,
        invalid=info.get("invalid_action", False)
    ))
    
    # ===== 5.2 后置感知 =====
    post_perception = perceive_with_action_history(
        env=env,
        instruction=instruction,
        last_action=action_descriptions[action_id],
        env_feedback=info.get("env_feedback", ""),
        action_history=wm.execution_trace[-5:]  # 最近5步
    )
    
    # perceive_with_action_history 流程：
    def perceive_with_action_history(...):
        # 1. 保存当前RGB图像
        rgb_img = env.render()
        img_path = save_image(rgb_img, episode_id, step)
        
        # 2. 构建提示词（包含动作历史）
        action_history_text = ""
        for trace in action_history[-5:]:
            action_history_text += f"- step {trace.env_step}: {trace.action_desc}, "
            action_history_text += f"reward={trace.reward:.2f}, invalid={trace.invalid}\n"
        
        system_prompt = """
        You are a perception module analyzing execution results.
        
        Input:
        - Current RGB image (after action)
        - Just executed action
        - Recent action history
        - Task instruction
        
        Analyze:
        - What changed after the action?
        - Did the action succeed or fail? Why?
        - What's the current state of relevant objects?
        
        Output JSON (same format as initial perception)
        """
        
        user_text = f"""
        [Task instruction]: {instruction}
        [Just executed action]: {last_action}
        [Environment feedback]: {env_feedback}
        {action_history_text}
        [Current robot state]:
        - Episode: {env._current_episode_num}
        - Step: {env._current_step}
        - Holding object: {env.is_holding}
        
        Analyze the current image and tell me:
        1) What is the result of the last action (success/failure + why)?
        2) What changed in the environment?
        3) What objects are visible and how are they arranged?
        4) For each visible object, extract attributes (especially color and size).
        """
        
        # 3. 调用VLM
        response = call_vlm(system_prompt, user_text, image_path=img_path)
        
        # 4. 解析并提取object_locations（同初始感知）
        parsed = json.loads(response)
        object_locations = extract_object_locations(parsed["detected_object_attributes"])
        
        return PerceptionOutput(
            scene_summary=parsed["scene_summary"],
            detected_objects=parsed["detected_objects"],
            detected_object_attributes=parsed["detected_object_attributes"],
            objects_relations=parsed["objects_relations"],
            object_locations=object_locations,  # 🔥 关键
            state_changes=parsed["state_changes"],
            environment_snapshot=env_snapshot,
            instruction_entities_and_attributes=parsed["instruction_entities_and_attributes"]
        )
    
    # ===== 5.3 摄入后置感知到WM =====
    wm.perception_history.append(post_perception)
    semantic_memory.ingest_perception(post_perception, step)
    # → WM图更新：新对象、新属性、新关系
    
    # ===== 5.4 更新LTM =====
    semantic_memory.update_ltm_from_perception(post_perception, step)
    # → LTM图更新：新的空间知识、对象属性
    
    # ===== 5.5 反馈控制决策 =====
    if step % feedback_window == 0:
        decision = feedback_controller.decide(
            wm=wm,
            invalid_ratio=recent_invalid_ratio,
            progress_delta=recent_progress_delta
        )
        
        if decision.action == "reperceive":
            # 重新感知当前场景
            post_perception = perceive_initial(env, instruction)
            semantic_memory.ingest_perception(post_perception, step)
        
        elif decision.action == "replan":
            # 重新规划
            semantic_ctx = semantic_memory.prepare_planner_context(instruction, step)
            wm.semantic_memory_digest.append(semantic_ctx)
            current_plan = _plan_with_vlm(img_path, instruction, wm)
            wm.plans.append(current_plan)
            plan_pointer = 0
        
        elif decision.action == "reperceive_and_replan":
            # 先感知再规划
            post_perception = perceive_initial(env, instruction)
            semantic_memory.ingest_perception(post_perception, step)
            semantic_ctx = semantic_memory.prepare_planner_context(instruction, step)
            current_plan = _plan_with_vlm(img_path, instruction, wm)
            plan_pointer = 0
    
    # ===== 5.6 推进plan指针 =====
    plan_pointer += 1
    if plan_pointer >= len(current_plan.action_ids):
        # Plan执行完，需要重新规划
        semantic_ctx = semantic_memory.prepare_planner_context(instruction, step)
        current_plan = _plan_with_vlm(img_path, instruction, wm)
        plan_pointer = 0
```

---

### 阶段6：Episode结束处理

**时间点**：Episode结束（done=True 或 达到max_steps）

**代码位置**：`semantic_memory.py::on_episode_end()` Line 1712-1850

**流程**：

```python
def on_episode_end(llm_summary: Optional[Dict] = None) -> Dict:
    """
    Episode结束后的记忆整理
    """
    
    # ===== Step 1: 可选LLM总结摄入 =====
    if llm_summary:
        # 将LLM生成的episode总结写入WM
        summary_text = llm_summary.get("summary", "")
        key_learnings = llm_summary.get("key_learnings", [])
        
        for learning in key_learnings:
            # 创建知识节点
            knowledge_node = self.wm.add_node(
                label=learning,
                properties={"type": "knowledge", "source": "llm_summary"}
            )
    
    # ===== Step 2: WM抽象化（abstract）=====
    abstraction_report = self.wm.abstract(
        min_confidence=0.7,
        min_mentions=2
    )
    
    # abstract 内部流程：
    def abstract(min_confidence, min_mentions):
        # 找到稳定的模式：
        # - 多次被提到的对象
        # - 高置信度的关系
        # - 反复出现的空间配置
        
        stable_nodes = []
        for node in self.nodes.values():
            mentions = node.properties.get("mention_count", 0)
            confidence = node.metadata.confidence
            
            if mentions >= min_mentions and confidence >= min_confidence:
                stable_nodes.append(node)
                # 标记为"稳定"
                node.properties["stable"] = True
        
        return {"stable_nodes": len(stable_nodes)}
    
    # ===== Step 3: WM简化（simplify）=====
    simplify_report = self.wm.simplify(
        keep_recent_steps=10,
        min_confidence=0.5
    )
    
    # simplify 内部流程：
    def simplify(keep_recent_steps, min_confidence):
        # 删除低置信度、不稳定的节点
        removed_nodes = []
        
        for node in list(self.nodes.values()):
            # 保留条件：
            # 1. 标记为stable
            # 2. 最近步骤创建
            # 3. 置信度足够高
            
            is_stable = node.properties.get("stable", False)
            is_recent = node.metadata.timestamp >= current_step - keep_recent_steps
            is_confident = node.metadata.confidence >= min_confidence
            
            if not (is_stable or is_recent or is_confident):
                removed_nodes.append(node.id)
                self.remove_node(node.id)
        
        return {"removed_nodes": len(removed_nodes)}
    
    # ===== Step 4: WM知识写入LTM（add_from_wm）=====
    ltm_report = self.ltm.add_from_wm(
        wm=self.wm,
        min_confidence=0.75,
        only_stable=True
    )
    
    # add_from_wm 内部流程：
    def add_from_wm(wm, min_confidence, only_stable):
        # 将WM中的稳定知识复制到LTM
        
        added_nodes = 0
        added_edges = 0
        
        for wm_node in wm.nodes.values():
            # 筛选条件
            if only_stable and not wm_node.properties.get("stable", False):
                continue
            
            if wm_node.metadata.confidence < min_confidence:
                continue
            
            # 在LTM中查找是否已存在
            existing = self.find_nodes(label=wm_node.label, exact=False, limit=1)
            
            if existing:
                # 更新置信度（取最大值）
                ltm_node = existing[0]
                ltm_node.metadata.confidence = max(
                    ltm_node.metadata.confidence,
                    wm_node.metadata.confidence
                )
            else:
                # 创建新节点
                self.add_node(
                    label=wm_node.label,
                    properties=wm_node.properties.copy(),
                    metadata=wm_node.metadata
                )
                added_nodes += 1
        
        # 复制边
        for wm_edge in wm.edges.values():
            # 检查source和target是否都在LTM中
            src_exists = self.find_nodes(label=wm.get_node(wm_edge.source).label, limit=1)
            tgt_exists = self.find_nodes(label=wm.get_node(wm_edge.target).label, limit=1)
            
            if src_exists and tgt_exists:
                # 检查边是否已存在
                existing_edge = self.find_edges(
                    source=src_exists[0].id,
                    target=tgt_exists[0].id,
                    relation=wm_edge.relation
                )
                
                if not existing_edge:
                    self.add_edge(
                        source=src_exists[0].id,
                        target=tgt_exists[0].id,
                        relation=wm_edge.relation,
                        metadata=wm_edge.metadata
                    )
                    added_edges += 1
        
        return {"added_nodes": added_nodes, "added_edges": added_edges}
    
    # ===== Step 5: LTM去重合并（consolidate）=====
    consolidate_report = self.ltm.consolidate(
        similarity_threshold=0.85
    )
    
    # consolidate 内部流程：
    def consolidate(similarity_threshold):
        # 合并相似节点
        merged_count = 0
        
        # 找到所有相似节点对
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1.id >= node2.id:  # 避免重复比较
                    continue
                
                # 计算相似度
                similarity = self.compute_similarity(node1, node2)
                
                if similarity >= similarity_threshold:
                    # 合并node2到node1
                    self.merge_nodes(node1.id, node2.id)
                    merged_count += 1
        
        return {"merged_nodes": merged_count}
    
    # ===== Step 6: 保存LTM到磁盘 =====
    self.ltm.save_to_file("output/semantic_ltm.json")
    
    # ===== Step 7: 生成报告 =====
    return {
        "wm_abstraction": abstraction_report,
        "wm_simplification": simplify_report,
        "ltm_addition": ltm_report,
        "ltm_consolidation": consolidate_report,
        "final_wm_size": {"nodes": len(self.wm.nodes), "edges": len(self.wm.edges)},
        "final_ltm_size": {"nodes": len(self.ltm.nodes), "edges": len(self.ltm.edges)}
    }
```

---

## 完整时序图

```
Time ─────────────────────────────────────────────────────────────►

Episode Start
│
├─ 初始化
│  ├─ env.reset()
│  ├─ WM = WorkingMemory()  [空图]
│  └─ LTM.load()            [423 nodes from previous episodes]
│
├─ Step 0: 初始感知
│  ├─ VLM: RGB → Perception JSON
│  ├─ WM.ingest_perception()  [WM: 0→19 nodes]
│  └─ LTM.update_from_perception()  [LTM: 423→430 nodes]
│
├─ Step 0: 首次规划
│  ├─ LTM.integrate_from_ltm()  [检索8条hints]
│  ├─ WM.read()  [读取30节点子图]
│  ├─ prepare_planner_context()
│  └─ VLM: Planning  [返回action_ids: [0,1,2,...,67]]
│
├─ Step 1: 执行 action_id=0
│  ├─ env.step(0)
│  ├─ VLM: Post-action Perception
│  ├─ WM.ingest_perception()  [WM: 19→175 nodes]
│  └─ LTM.update_from_perception()  [LTM: 430→445 nodes]
│
├─ Step 2: 执行 action_id=1
│  ├─ env.step(1)
│  ├─ VLM: Post-action Perception
│  ├─ WM.ingest_perception()  [WM: 175→280 nodes]
│  └─ LTM.update_from_perception()  [LTM: 445→460 nodes]
│
├─ ...
│
├─ Step 2: 反馈控制决策（每2步检查一次）
│  ├─ FeedbackController.decide()
│  ├─ Decision: "reperceive_and_replan"
│  ├─ VLM: Re-perception
│  ├─ WM.ingest_perception()
│  ├─ LTM.integrate_from_ltm()  [检索新hints]
│  └─ VLM: Re-planning
│
├─ ...
│
└─ Episode End
   ├─ WM.abstract()  [标记稳定节点]
   ├─ WM.simplify()  [删除低置信度节点]
   ├─ LTM.add_from_wm()  [WM→LTM知识转移]
   ├─ LTM.consolidate()  [合并相似节点]
   ├─ LTM.save()  [保存到semantic_ltm.json]
   └─ Generate EpisodeSummary
```

---

## 关键算子工作时机总结

| 算子 | 所属模块 | 触发时机 | 输入 | 输出 | 目的 |
|------|---------|---------|------|------|------|
| **perceive_initial** | PerceptionModule | Episode开始 | RGB图像 | PerceptionOutput | 获取初始场景理解 |
| **perceive_with_action_history** | PerceptionModule | 每个step执行后 | RGB + 动作历史 | PerceptionOutput | 理解动作效果 |
| **ingest_perception** | SemanticMemory | 每次感知后 | PerceptionOutput | WM图更新 | 将感知转为图节点/边 |
| **update_ltm_from_perception** | SemanticMemory | 每次感知后 | PerceptionOutput | LTM图更新 | 积累空间知识 |
| **integrate_from_ltm** | WMOperators | 规划前 | Goal文本 | LTM hints | 检索先验知识 |
| **read** | WMOperators | 规划前 | Goal文本 | WM子图 | 提取相关上下文 |
| **prepare_planner_context** | SemanticMemory | 规划前 | Goal文本 | PlannerContext | 组装规划输入 |
| **_plan_with_vlm** | MetaFlatAgent | 需要plan时 | 图像+上下文 | ActionPlan | 生成动作序列 |
| **abstract** | UnifiedMemoryGraph | Episode结束 | WM图 | 稳定节点标记 | 识别稳定模式 |
| **simplify** | UnifiedMemoryGraph | Episode结束 | WM图 | 节点删除 | 清理低价值信息 |
| **add_from_wm** | UnifiedMemoryGraph | Episode结束 | WM图 | LTM节点/边增加 | WM→LTM知识转移 |
| **consolidate** | UnifiedMemoryGraph | Episode结束 | LTM图 | 节点合并 | 去重合并 |

---

## 数据流动完整示意

```
RGB Image (Habitat环境)
    ↓ VLM感知
PerceptionOutput (JSON)
    ↓ ingest_perception
WM Graph (episode临时记忆)
    ├→ update_ltm_from_perception → LTM Graph (持久化记忆)
    ├→ read → PlannerContext
    └→ integrate_from_ltm ← LTM Graph
            ↓
    PlannerContext (规划输入)
            ↓ _plan_with_vlm
    ActionPlan (动作序列)
            ↓ env.step
    Observation + Reward + Info
            ↓ perceive_with_action_history
    PerceptionOutput (后置感知)
            ↓ [循环]

Episode结束:
    WM → abstract → simplify → add_from_wm → LTM → consolidate → 保存
```

这就是整个系统的完整工作流程！每个阶段都有明确的输入输出和目的，通过精心设计的算子协同工作，实现了从感知到记忆到规划到执行的完整闭环。
