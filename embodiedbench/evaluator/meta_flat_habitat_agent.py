"""
Feedback-driven 扁平规划 Agent，专门适配 EBHabEnv + VLMPlanner（Habitat 数据集场景）

流水线：
    感知(Perception) → Memory-Web Fusion → TaskModel & FeedbackState（问题求解视角）
    → Planner（VLM，输出扁平 action_id 序列） → Executor → Memory / 日志

特点：
- 不使用层级计划（没有 HierarchicalPlan / HighLevelStep）
- 只用 Habitat 的动作 id（0~68），动作描述用 env.language_skill_set[action_id] 即可
- 复用你已有的 VLMPlanner 作为扁平 Planner
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import time
import json
import os

from embodiedbench.planner.planner_utils import local_image_to_data_url
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.main import logger
from embodiedbench.evaluator.semantic_memory import SemanticMemoryManager

def fix_json(raw: str) -> str:
    """
    一个极简版的 JSON 清洗函数：
    - 去掉 ```json / ``` 代码块包裹
    - 截取第一个 '{' 到最后一个 '}' 之间的子串

    不做引号替换，不做复杂 regex。
    主要用途：让大模型输出里多余前后文字 / markdown 不影响 json.loads。
    """
    if not isinstance(raw, str):
        raw = str(raw)

    s = raw.strip()

    # 1) 去掉 ```json / ``` 包裹
    if s.startswith("```"):
        # 去掉前面的 ```json 或 ``` 这一行
        s = s.replace("```json", "").replace("```", "").strip()

    # 2) 只保留第一个 '{' 到最后一个 '}' 之间的内容
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    # 这里不强行尝试 json.loads，只是尽量清洗一下；
    # 真正的解析在外层 try: json.loads(fixed) 里做。
    return s


# ===================== WM/LTM 详细打印工具 =====================

def print_wm_state(semantic_memory, step: int, context: str = ""):
    """
    打印WM的详细状态：节点、边、统计信息
    
    Args:
        semantic_memory: SemanticMemoryManager实例
        step: 当前步数
        context: 打印上下文（如"After perception", "After step execution"）
    """
    wm = semantic_memory.wm
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 WM STATE [{context}] - Step {step}")
    logger.info(f"{'='*80}")
    
    # 统计信息
    num_nodes = len(wm.nodes)
    num_edges = len(wm.edges)
    logger.info(f"📈 Statistics: {num_nodes} nodes, {num_edges} edges")
    
    # 按类型分组节点
    nodes_by_kind = {}
    for node in wm.nodes.values():
        # 优先使用'type'字段，fallback到'kind'字段
        kind = node.properties.get('type') or node.properties.get('kind', 'unknown')
        if kind not in nodes_by_kind:
            nodes_by_kind[kind] = []
        nodes_by_kind[kind].append(node)
    
    logger.info(f"\n🔷 Nodes by type:")
    for kind, nodes in sorted(nodes_by_kind.items()):
        logger.info(f"  {kind}: {len(nodes)} nodes")
        for node in nodes[:5]:  # 只显示前5个
            props_str = ""
            if node.properties:
                key_props = {k: v for k, v in list(node.properties.items())[:3]}
                props_str = f", props={key_props}"
            logger.info(f"    - {node.label} (conf={node.metadata.confidence:.2f}{props_str})")
        if len(nodes) > 5:
            logger.info(f"    ... and {len(nodes) - 5} more {kind} nodes")
    
    # 按关系类型分组边
    edges_by_relation = {}
    for edge in wm.edges.values():
        rel = edge.relation
        if rel not in edges_by_relation:
            edges_by_relation[rel] = []
        edges_by_relation[rel].append(edge)
    
    logger.info(f"\n🔗 Edges by relation type:")
    for relation, edges in sorted(edges_by_relation.items()):
        logger.info(f"  {relation}: {len(edges)} edges")
        for edge in edges[:5]:  # 只显示前5个
            source_node = wm.nodes.get(edge.source)
            target_node = wm.nodes.get(edge.target)
            if source_node and target_node:
                source_label = source_node.label
                target_label = target_node.label
                logger.info(f"    - '{source_label}' --[{relation}]--> '{target_label}' (conf={edge.metadata.confidence:.2f})")
            else:
                logger.info(f"    - {edge.source} --[{relation}]--> {edge.target} (broken reference)")
        if len(edges) > 5:
            logger.info(f"    ... and {len(edges) - 5} more {relation} edges")
    
    logger.info(f"{'='*80}\n")


def print_ltm_retrieval(nodes, edges, goal: str, context: str = ""):
    """
    打印从LTM检索到的内容
    
    Args:
        nodes: 检索到的节点列表
        edges: 检索到的边列表
        goal: 查询目标
        context: 上下文信息
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 LTM RETRIEVAL [{context}]")
    logger.info(f"{'='*80}")
    logger.info(f"Query goal: {goal}")
    logger.info(f"Retrieved: {len(nodes)} nodes, {len(edges)} edges")
    
    if nodes:
        logger.info(f"\n📦 Retrieved Nodes (top 10):")
        for i, node in enumerate(nodes[:10], 1):
            # 处理node可能是dict或Node对象的情况
            if isinstance(node, dict):
                # 优先使用'type'字段，fallback到'kind'字段
                node_kind = node.get('properties', {}).get('type') or node.get('properties', {}).get('kind', 'unknown')
                node_label = node.get('label', 'N/A')
                node_props = node.get('properties', {})
                node_conf = node.get('metadata', {}).get('confidence', 0.0)
            else:
                # 优先使用'type'字段，fallback到'kind'字段
                node_kind = node.properties.get('type') or node.properties.get('kind', 'unknown')
                node_label = node.label
                node_props = node.properties
                node_conf = node.metadata.confidence
            
            props_str = ""
            if node_props:
                key_props = {k: v for k, v in list(node_props.items())[:2] if k != 'kind'}
                props_str = f", props={key_props}"
            logger.info(f"  {i}. [{node_kind}] {node_label} (conf={node_conf:.2f}{props_str})")
        if len(nodes) > 10:
            logger.info(f"  ... and {len(nodes) - 10} more nodes")
    
    if edges:
        logger.info(f"\n🔗 Retrieved Edges (top 10):")
        # 需要传入node_id到label的映射，处理dict和Node对象两种情况
        node_map = {}
        for n in nodes:
            if isinstance(n, dict):
                node_map[n.get('id', '')] = n.get('label', 'N/A')
            else:
                node_map[n.id] = n.label
        
        for i, edge in enumerate(edges[:10], 1):
            # 处理edge可能是dict或Edge对象的情况
            if isinstance(edge, dict):
                edge_source = edge.get('source', '')
                edge_target = edge.get('target', '')
                edge_relation = edge.get('relation', 'unknown')
                edge_conf = edge.get('metadata', {}).get('confidence', 0.0)
            else:
                edge_source = edge.source
                edge_target = edge.target
                edge_relation = edge.relation
                edge_conf = edge.metadata.confidence
            
            source_label = node_map.get(edge_source, edge_source)
            target_label = node_map.get(edge_target, edge_target)
            logger.info(f"  {i}. '{source_label}' --[{edge_relation}]--> '{target_label}' (conf={edge_conf:.2f})")
        if len(edges) > 10:
            logger.info(f"  ... and {len(edges) - 10} more edges")
    
    logger.info(f"{'='*80}\n")


def print_memory_operation(operation: str, result: any, details: dict = None):
    """
    打印内存操作的结果
    
    Args:
        operation: 操作名称（如"ingest_perception", "read", "write"）
        result: 操作结果
        details: 额外的详细信息
    """
    logger.info(f"\n{'─'*80}")
    logger.info(f"⚙️  MEMORY OPERATION: {operation}")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")
    logger.info(f"   Result: {result}")
    logger.info(f"{'─'*80}\n")


# ===================== 基本数据结构 =====================

@dataclass
class PerceptionOutput:
    """对当前 Habitat 场景的抽象感知（这里不调用额外 LLM，先用规则抽象）。"""
    scene_summary: str
    detected_objects: List[str] = field(default_factory=list)
    detected_object_attributes: List[Dict[str, Any]] = field(default_factory=list)
    objects_relations: List[str] = field(default_factory=list)
    state_changes: List[str] = field(default_factory=list)
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    instruction_entities_and_attributes: List[str] = field(default_factory=list)



@dataclass
class TaskModel:
    """
    问题求解视角下的任务模型：
    - 目标、关键物体、约束等
    """
    goal_description: str
    key_objects: List[str]
    constraints: List[str]
    difficulty_factors: List[str]


@dataclass
class FeedbackState:
    """记录最近一次由大模型给出的控制决策结果。"""
    last_decision: str = "init"
    last_reason: str = ""
    last_confidence: float = 0.0
    last_invalid_action_ratio: float = 0.0
    last_progress_delta: float = 0.0
    raw_response: str = ""


@dataclass
class FeedbackDecision:
    """
    反馈控制决策：
    action:
        - "continue": 继续执行当前 plan
        - "reperceive": 重新感知，保持 plan
        - "replan": 重规划（重新调用 VLMPlanner）
        - "reperceive_and_replan": 先感知再重规划
    """
    action: str
    reason: str


@dataclass
class ActionPlan:
    """
    扁平动作计划：
    - 只保留 action_id 序列 + 一段整体 reasoning（就是 VLM 输出的 json 字符串）
    """
    action_ids: List[int]
    reasoning: str  # 原始 JSON 输出或解释字符串


@dataclass
class ExecutionStepTrace:
    """单步执行日志。"""
    env_step: int
    action_id: int
    action_desc: str
    reward: float
    info: Dict[str, Any]
    invalid: bool
    env_feedback: Optional[Any] = None


@dataclass
class WorkingMemory:
    """
    工作记忆（当前 episode 的「思维草稿纸」）：
      - 感知历史
      - 融合上下文历史
      - plan 历史
      - 执行轨迹
      - TaskModel
      - 反馈状态
      - LLM episode总结
    """
    instruction: str
    perception_history: List[PerceptionOutput] = field(default_factory=list)
    plans: List[ActionPlan] = field(default_factory=list)
    execution_trace: List[ExecutionStepTrace] = field(default_factory=list)

    task_model: Optional[TaskModel] = None
    feedback_state: Optional[FeedbackState] = None
    semantic_memory_digest: List[Dict[str, Any]] = field(default_factory=list) # WM摘要
    semantic_memory_gaps: List[str] = field(default_factory=list) #缺口文本
    semantic_memory_hints: List[Dict[str, Any]] = field(default_factory=list) #LTM提示
    semantic_memory_report: Dict[str, Any] = field(default_factory=dict) # episode结束报告
    llm_episode_summary: Dict[str, Any] = field(default_factory=dict) # LLM生成的episode总结

    def latest_perception(self) -> Optional[PerceptionOutput]:
        return self.perception_history[-1] if self.perception_history else None

    def latest_plan(self) -> Optional[ActionPlan]:
        return self.plans[-1] if self.plans else None


@dataclass
class EpisodeSummary:
    """整条 episode 的总结（方便转成原来的 episode_info）。"""
    instruction: str
    eval_set: str
    task_success: float
    task_progress: float
    subgoal_reward: float
    num_steps: int
    planner_steps: int
    planner_output_error: int
    num_invalid_actions: int
    num_invalid_action_ratio: float
    reward_mean: float
    episode_elapsed_seconds: float
    empty_plan: int
    working_memory: WorkingMemory

def summarize_recent_trace(wm: WorkingMemory, max_steps: int = 5) -> str:
    """
    把最近几步执行/感知/计划压缩成一段文字，让反馈 LLM 有素材。
    TODO:只有上一步的感知/计划/执行
    """
    lines = []
    if wm.task_model:
        lines.append(f"Instruction: {wm.task_model.goal_description}")
    if wm.latest_perception():
        p = wm.latest_perception()
        lines.append(f"Last perception scene_summary: {p.scene_summary}")
        if p.detected_objects:
            lines.append(f"Detected objects: {p.detected_objects[:8]}")

    if wm.latest_plan():
        plan = wm.latest_plan()
        lines.append(f"Current flat action plan: {plan.action_ids[:10]}")

    if wm.semantic_memory_digest:
        highlights = []
        for node in wm.semantic_memory_digest[:2]:
            label = node.get("label") or node.get("kind")
            state_vars = node.get("state_vars", {})
            state_str = ", ".join(f"{k}={v}" for k, v in list(state_vars.items())[:3])
            highlights.append(f"{node.get('kind')}:{label} ({state_str})")
        if highlights:
            lines.append("Semantic memory: " + "; ".join(highlights))

    steps = wm.execution_trace[-max_steps:]
    if steps:
        lines.append(f"Recent {len(steps)} execution steps:")
        for s in steps:
            feedback_str = ""
            if s.env_feedback:
                feedback_str = f", env_feedback={s.env_feedback}"
            lines.append(
                f"- step={s.env_step}, action_id={s.action_id}, "
                f"desc='{s.action_desc}', reward={s.reward:.3f}, invalid={s.invalid}"
                f"{feedback_str}"
            )

    return "\n".join(lines)


# ===================== 模块：感知 / 融合 / 反馈 =====================

class PerceptionRemoteModel(RemoteModel):
    """
    继承你现有的 RemoteModel，只复用 __init__ 里的“选后端”逻辑。
    感知任务不需要 planning 的 JSON schema，因此提供一个自由格式的调用。
    """

    def respond_freeform(self, message_history: list, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        # local 模型目前按你原来的设定是用 lmdeploy 的 pipeline，
        # 这里为了简单，先只支持 remote 类型（与 VLM planner 一致）
        if self.model_type == "local":
            raise NotImplementedError("PerceptionRemoteModel currently supports only remote models.")

        # gpt / qwen / 其他 OpenAI-compatible 后端统一走 chat.completions
        if isinstance(message_history, list):
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=message_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # OpenAI 风格：message.content 是 list[Text/Other] or just str
            content = response.choices[0].message.content
            # 在新的 openai==1.x 里 content 是 list[dict] or str，这里统一成 string
            if isinstance(content, str):
                return content
            else:
                # content 是 list[{"type": "text", "text": "..."}]
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts)
        else:
            raise ValueError("message_history must be a list of messages.")


class PerceptionModule:
    """
    Habitat 版感知模块（使用多模态大模型）：
    - 输入：环境状态 + 当前观测图片路径 + 语言指令
    - 调 RemoteModel backend（和 VLMPlanner 一致的模型 family）
    - 输出：结构化的 PerceptionOutput
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        model_type: str = "remote",
        tp: int = 1,
    ):
        # language_only=False，task_type=None：我们不需要 planner 的 JSON schema
        self.vision_model = PerceptionRemoteModel(
            model_name=model_name,
            model_type=model_type,
            language_only=False,
            tp=tp,
            task_type=None,
        )

    def _build_messages(self, env: EBHabEnv, img_path: str, instruction: str, feedback_hint: str | None = None) -> list:
        """
        构建给感知大模型的消息：
        - system: 角色 + 要求输出 JSON
        - user: 一张图 + 一段文字描述环境反馈信息
        """
        data_url = local_image_to_data_url(image_path=img_path)

        system_prompt = (
            "You are a perception module for a household robot in a simulated Habitat environment. "
            "You receive one RGB image from the robot's head camera and the current task instruction. "
            "You must output a concise but spatially detailed JSON description of the scene.\n\n"

            "First, explicitly analyze the instruction to extract (1) task-relevant entities/objects and (2) their attributes/constraints "
            "(e.g., color, size, material, quantity, target receptacle, room/location, ordering constraints). "
            "Include them in a dedicated output field.\n\n"

            "Second, for the CURRENT IMAGE, you MUST extract object-level attributes (at least color and size when visible) for each detected object. "
            "If you are uncertain, provide your best guess and add a short uncertainty note in the attribute value (e.g., 'red(?)'). "
            "Do NOT skip attributes just because they are hard.\n\n"

            "Your JSON MUST have the following fields:\n"
            "{\n"
            '  "scene_summary": string,  // describe the current room (kitchen, dining room, etc.), dominant furniture, and task-relevant situation\n'
            '  "detected_objects": [string],  // list salient objects you can see, e.g., ["spoon", "dining_table", "sink"]\n'
            '  "detected_object_attributes": [\n'
            '    {\n'
            '      "label": string,  // must match one of detected_objects\n'
            '      "attributes": {\n'
            '        "color": string | [string],\n'
            '        "size": string,  // e.g., small/medium/large\n'
            '        "material": string | [string],\n'
            '        "shape": string,\n'
            '        "state": string | [string],  // e.g., open/closed/filled/empty/dirty/clean\n'
            '        "container_or_surface": string,  // e.g., "on the counter", "inside the drawer", "in the sink"\n'
            '        "relative_position": string  // e.g., "left of the sink", "near the fridge"\n'
            '      },\n'
            '      "confidence": float  // 0~1, optional but recommended\n'
            '    }\n'
            '  ],\n'
            '  "objects_relations": [string],  // explicit spatial/functional relations, e.g., ["the spoon is on the dining table", "the fridge is left of the sink"]\n'
            '  "state_changes": [string],     // notable state descriptions or recent changes (drawer open, cup moved); if unsure, use []\n'
            '  "environment_snapshot": string,  // paragraph summarizing room layout, pathways, containers/receptacles, and where key objects sit relative to anchors\n'
            '  "instruction_entities_and_attributes": [string]  // extracted entities & attributes/constraints from the instruction\n'
            "}\n\n"
            "Highlight which counters, shelves, or containers hold objects, note obstacles/occlusions, and mention any nearby rooms worth exploring next. "
            "Do NOT include any other top-level keys. Do NOT wrap the JSON in markdown."
        )

        if feedback_hint:
            system_prompt += (
                "\n\n[Feedback hints from previous steps]\n"
                f"{feedback_hint}"
            )

        user_text = (
            "Here is the current observation of the robot in a household rearrangement task.\n"
            f"- Episode index: {env._current_episode_num}\n"
            f"- Current step: {env._current_step}\n"
            f"- Is the robot currently holding an object: {env.is_holding}\n"
            f"- Max episode steps: {env._max_episode_steps}\n"
            f"- Language instruction for this episode: \"{instruction}\"\n\n"

            # ✅ 再强调一次：别漏掉 image 属性
            "Requirements:\n"
            "1) Parse the instruction and output 'instruction_entities_and_attributes'.\n"
            "2) From THIS IMAGE, output 'detected_object_attributes' including color and size for each object when visible.\n"
            "3) Provide explicit relations using on/in/inside/left/right/front/behind/near.\n\n"
            "Now output exactly the JSON described above."
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ]
        return messages


    def perceive(self, env: EBHabEnv, img_path: str, instruction: str, feedback_hint: str | None = None) -> PerceptionOutput:
        """Run the perception stage and emit a structured :class:`PerceptionOutput`.

        Pipeline / 契约：
            1. Use :meth:`_build_messages` to inject instruction, spatial goals, and feedback hints.
            2. Query the remote VLM (metacognition tap point #1) and log raw JSON-like text for audits.
            3. Normalize the response into ``scene_summary``/``objects_relations``/``environment_snapshot`` so that
               downstream fusion + planning always receive the same schema.
            4. If parsing fails, fall back to a deterministic textual snapshot to keep the loop alive.

        Args:
            env (EBHabEnv): 当前 Habitat 环境，提供 episode / step / is_holding 等硬指标。
            img_path (str): 最新保存的第一人称图像路径（会被转成 data URL 发送给 VLM）。
            instruction (str): 原始语言指令，用于提示和 fallback 摘要。
            feedback_hint (str | None): 最近一次来自反馈/元认知的感知提示；None 时自动拼接全局提示。

        Returns:
            PerceptionOutput: 规范化的感知结果，带有 JSON 字段和 environment snapshot。

        Notes:
            - ``scene_summary`` + ``environment_snapshot`` 被 FeedbackController 和 planner 视作 metacognition evidence。
            - 任何异常都会记录 warning 并触发安全 fallback，保证 WorkingMemory 至少有一条感知记录。
        """
        messages = self._build_messages(env, img_path, instruction, feedback_hint=feedback_hint)

        try:
            raw_output = self.vision_model.respond_freeform(messages)
            logger.info(f"[PerceptionModule] LLM perception output:\n{raw_output}\n")

            # 尝试把输出修成合法 JSON
            try:
                fixed = fix_json(raw_output)
            except Exception:
                fixed = raw_output

            parsed = json.loads(fixed)

            scene_summary = parsed.get("scene_summary", "").strip()
            detected_objects = parsed.get("detected_objects", []) or []
            objects_relations = parsed.get("objects_relations", []) or []
            state_changes = parsed.get("state_changes", []) or []
            env_comment = parsed.get("environment_snapshot", "").strip()
            obj_attrs = parsed.get("detected_object_attributes", []) or []
            instruction_ea = parsed.get("instruction_entities_and_attributes", []) or []

            # 我们仍然保留一些来自 env 的 hard feedback info
            snapshot = {
                "episode": int(env._current_episode_num),
                "step": int(env._current_step),
                "is_holding": bool(env.is_holding),
                "max_steps": int(env._max_episode_steps),
                "env_comment_from_model": env_comment,
            }

            # 如果 scene_summary 太空，就 fallback 一下
            if not scene_summary:
                scene_summary = (
                    f"Episode {env._current_episode_num}, step {env._current_step}, "
                    f"holding={env.is_holding}, instruction='{instruction}'. "
                    f"Image saved at {img_path}."
                )

            return PerceptionOutput(
                scene_summary=scene_summary,
                detected_objects=[str(o) for o in detected_objects],
                detected_object_attributes=obj_attrs if isinstance(obj_attrs, list) else [],
                objects_relations=[str(r) for r in objects_relations],
                state_changes=[str(s) for s in state_changes],
                environment_snapshot=snapshot,
                instruction_entities_and_attributes=[str(x) for x in instruction_ea],
            )

        except Exception as e:
            logger.warning(f"[PerceptionModule] Perception LLM failed, fallback to rule-based summary: {e}")
            # 回退：用最开始的简单版本，保证不会崩
            scene_summary = (
                f"Episode {env._current_episode_num}, step {env._current_step}, "
                f"holding={env.is_holding}, instruction='{instruction}'. "
                f"Image saved at {img_path}."
            )
            snapshot = {
                "episode": int(env._current_episode_num),
                "step": int(env._current_step),
                "is_holding": bool(env.is_holding),
                "max_steps": int(env._max_episode_steps),
                "env_comment_from_model": "fallback_without_model",
            }
            return PerceptionOutput(
                scene_summary=scene_summary,
                detected_objects=[],
                objects_relations=[],
                state_changes=[],
                environment_snapshot=snapshot,
            )

    def perceive_with_action_history(
        self,
        env: EBHabEnv,
        img_path: str,
        instruction: str,
        last_action: str,
        recent_actions: List[Dict[str, Any]],
        env_feedback: Optional[str] = None,
    ) -> PerceptionOutput:
        """
        每步执行后的LLM感知：分析执行动作+环境反馈+新图像
        同时输出：
        - instruction_entities_and_attributes：从 instruction 抽实体/属性/约束
        - detected_object_attributes：从当前图像抽取每个可见对象的属性（颜色/大小/材质/状态等）
        """
        data_url = local_image_to_data_url(image_path=img_path)

        # 构建动作历史摘要
        action_history_text = ""
        if recent_actions:
            action_history_text = "\n[Recent action history (last 5 steps)]:\n"
            for i, act in enumerate(recent_actions[-5:], 1):
                action_history_text += (
                    f"  {i}. Action: {act.get('action_desc', 'N/A')}, "
                    f"Reward: {act.get('reward', 0):.2f}, "
                    f"Invalid: {act.get('invalid', False)}"
                )
                if act.get("env_feedback"):
                    action_history_text += f", Feedback: {act['env_feedback']}"
                action_history_text += "\n"

        system_prompt = (
            "You are a perception module for a household robot analyzing execution results. "
            "You receive:\n"
            "1) The CURRENT RGB image after executing an action\n"
            "2) The action that was just executed\n"
            "3) Recent action history and environment feedback\n"
            "4) The overall task instruction\n\n"

            "Your job is to analyze:\n"
            "- What changed after executing the action?\n"
            "- Did the action succeed or fail? Why?\n"
            "- What's the current state of relevant objects?\n"
            "- What should the robot know for next steps?\n\n"

            "IMPORTANT: First, explicitly analyze the instruction to extract (1) task-relevant entities/objects and "
            "(2) their attributes/constraints (e.g., color, size, material, quantity, target receptacle, room/location, ordering constraints). "
            "Put them in a dedicated output field.\n\n"

            "IMPORTANT: For the CURRENT IMAGE, you MUST extract object-level attributes for each detected object. "
            "At minimum, include color and size when visible. "
            "If uncertain, provide your best guess and mark with '(?)' (e.g., 'red(?)').\n\n"

            "Output a JSON with these exact fields:\n"
            "{\n"
            '  "scene_summary": string,  // current situation after the action\n'
            '  "detected_objects": [string],  // visible objects now\n'
            '  "detected_object_attributes": [\n'
            '    {\n'
            '      "label": string,  // must match one of detected_objects\n'
            '      "attributes": {\n'
            '        "color": string | [string],\n'
            '        "size": string,  // e.g., small/medium/large\n'
            '        "material": string | [string],\n'
            '        "shape": string,\n'
            '        "state": string | [string],  // e.g., open/closed/filled/empty/dirty/clean/holding\n'
            '        "container_or_surface": string,  // e.g., "on the counter", "inside the drawer", "in the sink"\n'
            '        "relative_position": string  // e.g., "left of the sink", "near the fridge"\n'
            '      },\n'
            '      "confidence": float  // 0~1, optional but recommended\n'
            '    }\n'
            '  ],\n'
            '  "objects_relations": [string],  // spatial relations after action\n'
            '  "state_changes": [string],  // what changed due to the action\n'
            '  "environment_snapshot": string,  // detailed analysis of execution result and current state\n'
            '  "instruction_entities_and_attributes": [string]  // extracted entities & attributes/constraints from the instruction\n'
            "}\n\n"
            "Focus on action consequences, state changes, and information needed for replanning. "
            "Do NOT include any other keys. Do NOT wrap in markdown."
        )

        user_text = (
            f"[Task instruction]: {instruction}\n\n"
            f"[Just executed action]: {last_action}\n"
        )
        if env_feedback:
            user_text += f"[Environment feedback]: {env_feedback}\n"
        user_text += action_history_text
        user_text += (
            f"\n[Current robot state]:\n"
            f"- Episode: {env._current_episode_num}\n"
            f"- Step: {env._current_step}\n"
            f"- Holding object: {env.is_holding}\n\n"
            "Analyze the current image and tell me:\n"
            "1) What is the result of the last action (success/failure + why)?\n"
            "2) What changed in the environment?\n"
            "3) What objects are visible and how are they arranged?\n"
            "4) For each visible object, extract attributes (especially color and size).\n"
            "5) Also extract entities and attributes/constraints from the instruction.\n\n"
            "Output the JSON described above."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        try:
            raw_output = self.vision_model.respond_freeform(messages)
            logger.info(f"[PerceptionModule] Post-action LLM perception output:\n{raw_output}\n")

            try:
                fixed = fix_json(raw_output)
            except Exception:
                fixed = raw_output

            parsed = json.loads(fixed)

            scene_summary = parsed.get("scene_summary", "").strip()
            detected_objects = parsed.get("detected_objects", []) or []
            obj_attrs = parsed.get("detected_object_attributes", []) or []
            objects_relations = parsed.get("objects_relations", []) or []
            state_changes = parsed.get("state_changes", []) or []
            env_comment = parsed.get("environment_snapshot", "").strip()
            instr_ea = parsed.get("instruction_entities_and_attributes", []) or []

            snapshot = {
                "episode": int(env._current_episode_num),
                "step": int(env._current_step),
                "is_holding": bool(env.is_holding),
                "max_steps": int(env._max_episode_steps),
                "last_action": last_action,
                "env_comment_from_model": env_comment,
            }

            if not scene_summary:
                scene_summary = (
                    f"Post-action perception at step {env._current_step}: "
                    f"executed '{last_action}', holding={env.is_holding}"
                )

            return PerceptionOutput(
                scene_summary=scene_summary,
                detected_objects=[str(o) for o in detected_objects],
                detected_object_attributes=obj_attrs if isinstance(obj_attrs, list) else [],
                objects_relations=[str(r) for r in objects_relations],
                state_changes=[str(s) for s in state_changes],
                environment_snapshot=snapshot,
                instruction_entities_and_attributes=[str(x) for x in instr_ea] if isinstance(instr_ea, list) else [],
            )

        except Exception as e:
            logger.warning(
                f"[PerceptionModule] Post-action perception failed, using fallback: {e}"
            )
            scene_summary = (
                f"Fallback perception at step {env._current_step}: "
                f"executed '{last_action}', holding={env.is_holding}"
            )
            snapshot = {
                "episode": int(env._current_episode_num),
                "step": int(env._current_step),
                "is_holding": bool(env.is_holding),
                "max_steps": int(env._max_episode_steps),
                "last_action": last_action,
                "env_comment_from_model": "fallback_due_to_error",
            }
            return PerceptionOutput(
                scene_summary=scene_summary,
                detected_objects=[],
                detected_object_attributes=[],
                objects_relations=[],
                state_changes=[],
                environment_snapshot=snapshot,
                instruction_entities_and_attributes=[],
            )


    def summarize_episode(
        self,
        instruction: str,
        action_history: List[Dict[str, Any]],
        final_state: Dict[str, Any],
        task_success: float,
        task_progress: float,
    ) -> Dict[str, Any]:
        """
        Episode结束时的LLM总结：分析整个episode的执行过程
        
        Args:
            instruction: 原始任务指令
            action_history: 完整的动作执行历史
            final_state: 最终环境状态
            task_success: 任务成功率
            task_progress: 任务进度
            
        Returns:
            Dict包含episode总结、关键发现、经验教训等
        """
        # 构建动作历史摘要
        action_summary = f"\n[Complete action history ({len(action_history)} steps)]:\n"
        for i, act in enumerate(action_history, 1):
            action_summary += (
                f"  Step {i}: {act.get('action_desc', 'N/A')} "
                f"(Reward: {act.get('reward', 0):.2f}, "
                f"Invalid: {act.get('invalid', False)}"
            )
            if act.get('env_feedback'):
                action_summary += f", Feedback: {act['env_feedback']}"
            action_summary += ")\n"
            if i >= 100:  # 限制长度
                remaining = len(action_history) - i
                if remaining > 0:
                    action_summary += f"  ... and {remaining} more steps\n"
                break
        
        system_prompt = (
            "You are an episodic memory consolidation module for a household robot. "
            "You receive the complete execution history of an episode and must extract valuable insights.\n\n"
            "Your job is to analyze:\n"
            "1. What strategies worked well?\n"
            "2. What mistakes were made and why?\n"
            "3. What patterns or rules can be learned?\n"
            "4. What object relations or spatial knowledge should be remembered?\n"
            "5. What can be improved in future episodes?\n\n"
            "Output a JSON with these exact fields:\n"
            "{\n"
            '  "episode_summary": string,  // brief summary of what happened\n'
            '  "success_factors": [string],  // what contributed to success/failure\n'
            '  "learned_patterns": [string],  // behavioral patterns discovered\n'
            '  "spatial_knowledge": [string],  // object locations, relations learned\n'
            '  "failure_analysis": [string],  // why certain actions failed\n'
            '  "future_recommendations": [string]  // suggestions for similar tasks\n'
            "}\n\n"
            "Focus on generalizable knowledge that can help in future similar tasks. "
            "Do NOT include any other keys. Do NOT wrap in markdown."
        )
        
        user_text = (
            f"[Original task]: {instruction}\n\n"
            f"[Final result]:\n"
            f"- Task success: {task_success:.2f}\n"
            f"- Task progress: {task_progress:.2f}\n"
            f"- Total steps: {len(action_history)}\n"
            f"- Final state: {final_state}\n\n"
        )
        
        user_text += action_summary
        
        user_text += (
            "\nAnalyze this episode and extract:\n"
            "1. Key insights about object locations and relations\n"
            "2. Successful strategies vs. failed approaches\n"
            "3. Patterns in action sequences that worked\n"
            "4. Common mistakes to avoid\n"
            "5. Knowledge that should be remembered for future tasks\n\n"
            "Output the JSON described above."
        )
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]
        
        try:
            raw_output = self.vision_model.respond_freeform(messages)
            logger.info(f"[PerceptionModule] Episode summary LLM output:\n{raw_output}\n")
            
            # 解析JSON
            try:
                fixed = fix_json(raw_output)
            except Exception:
                fixed = raw_output
            
            parsed = json.loads(fixed)
            
            return {
                "episode_summary": parsed.get("episode_summary", "").strip(),
                "success_factors": parsed.get("success_factors", []) or [],
                "learned_patterns": parsed.get("learned_patterns", []) or [],
                "spatial_knowledge": parsed.get("spatial_knowledge", []) or [],
                "failure_analysis": parsed.get("failure_analysis", []) or [],
                "future_recommendations": parsed.get("future_recommendations", []) or [],
                "raw_llm_output": raw_output,
            }
            
        except Exception as e:
            logger.warning(
                f"[PerceptionModule] Episode summarization failed, using fallback: {e}"
            )
            return {
                "episode_summary": f"Episode completed with success={task_success:.2f}, progress={task_progress:.2f}",
                "success_factors": [],
                "learned_patterns": [],
                "spatial_knowledge": [],
                "failure_analysis": [],
                "future_recommendations": [],
                "error": str(e),
            }


# MemoryWebFusionEngine 类已删除 - 不再需要 fusion 中间层
# 现在直接使用 SemanticMemory 进行感知结果的存储和检索


class FeedbackController:
    """
    反馈控制器：
    - 构建 TaskModel（规则版）
    - 通过 LLM 根据 WorkingMemory/近期表现判断下一步控制动作
    """

    ALLOWED_DECISIONS = (
        "continue",
        "replan",
        "reperceive",
        "reperceive_and_replan",
    )

    def build_task_model(
        self,
        instruction: str,
        env: EBHabEnv,
    ) -> TaskModel:
        """Derive a lightweight :class:`TaskModel` contract from the instruction.

        Responsibilities:
            1. Inspect the instruction and action vocabulary to guess key objects / affordances.
            2. Generate coarse constraints + difficulty hints for the feedback loop.
            3. Provide a stable structure for WorkingMemory so metacognition can compare “plan vs. reality”.

        Args:
            instruction (str): episode 指令；作为 goal_description 的来源。
            env (EBHabEnv): 暴露 ``language_skill_set``，用于抽取关键对象 tokens。

        Returns:
            TaskModel: 包含 goal、key_objects、constraints、difficulty_factors 的概括。

        Notes:
            - 当前实现是启发式版本，但保持字段稳定，方便之后喂给 LLM 决策。
            - ``key_objects``/``constraints`` 会在 ``summarize_recent_trace`` 中被引用，成为 metacognition evidence。
        """
        # 简单从动作空间里挖一些关键词作为 key_objects（非常粗糙，但够用来体现结构）
        lower_instr = instruction.lower()
        key_objects: List[str] = []
        for a in env.language_skill_set:
            # 找到出现在动作描述中的名词片段
            for token in a.replace("_", " ").split():
                t = token.lower()
                if len(t) > 3 and t in lower_instr and t not in key_objects:
                    key_objects.append(t)

        constraints: List[str] = [
            "Max steps limited by environment.",
            "Avoid too many invalid actions.",
        ]
        difficulty_factors: List[str] = []
        if len(key_objects) >= 3:
            difficulty_factors.append("Many objects are involved.")
        if "rearrange" in lower_instr:
            difficulty_factors.append("Long-horizon rearrangement task.")

        return TaskModel(
            goal_description=instruction,
            key_objects=key_objects,
            constraints=constraints,
            difficulty_factors=difficulty_factors,
        )

    def init_feedback_state(self) -> FeedbackState:
        return FeedbackState()

    def assess_step(
        self,
        feedback_state: FeedbackState,
        working_memory: WorkingMemory,
        info: Dict[str, Any],
        invalid_ratio_window: float,
        progress_delta: float,
        decision_model: RemoteModel,
    ) -> FeedbackDecision:
        """Run the controller-level metacognition check and pick the next action.

        Workflow:
            1. Persist最新无效率 / 进度增量到 ``feedback_state``（方便调试或提示输出）。
            2. 若 ``decision_model`` 可用，构造包含 WorkingMemory、plan snapshot、执行轨迹的消息。
            3. 解析 LLM JSON，选择 {continue|replan|reperceive|reperceive_and_replan} 中的一个动作。
            4. 如遇异常，则回退到规则策略，确保控制环不断裂。

        Args:
            feedback_state (FeedbackState): 累积控制决策的状态对象。
            working_memory (WorkingMemory): 包含指令、plan、执行轨迹等 metacognition 证据。
            info (Dict[str, Any]): 来自 env.step 的原始 info。
            invalid_ratio_window (float): 最近窗口的 invalid 比例。
            progress_delta (float): 最近窗口中 task_progress 的增量。
            decision_model (RemoteModel): 用于推理控制动作的 LLM，None 时触发 fallback。

        Returns:
            FeedbackDecision: action + reason，用于 Agent 主循环。

        Notes:
            - ``feedback_state.raw_response`` 保存了原始 JSON，方便日后 prompt 追踪。
            - 该函数是 metacognition tap point #3：它根据执行表现动态调整 perception/planning。
        """
        feedback_state.last_invalid_action_ratio = invalid_ratio_window
        feedback_state.last_progress_delta = progress_delta

        if decision_model is None:
            logger.warning("[FeedbackController] No decision model provided, fallback to rules.")
            return self._fallback_decision(feedback_state, invalid_ratio_window, progress_delta)

        messages = self._build_decision_messages(
            feedback_state=feedback_state,
            working_memory=working_memory,
            info=info,
            invalid_ratio_window=invalid_ratio_window,
            progress_delta=progress_delta,
        )
        logger.info(f"[FeedbackController] Decision LLM messages:\n{messages}\n")
        try:
            raw = decision_model.respond_freeform(messages, temperature=0.0, max_tokens=512)
            logger.info(f"[FeedbackController] Decision LLM output:\n{raw}\n")
            try:
                fixed = fix_json(raw)
            except Exception:
                fixed = raw
            parsed = json.loads(fixed)
        except Exception as e:
            logger.warning(f"[FeedbackController] Decision LLM failed, fallback to rules: {e}")
            return self._fallback_decision(feedback_state, invalid_ratio_window, progress_delta)

        decision = str(parsed.get("decision", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip() or "LLM suggested decision."
        confidence = parsed.get("confidence", parsed.get("confidence_score", 0.0))
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        if decision not in self.ALLOWED_DECISIONS:
            logger.warning(
                f"[FeedbackController] Invalid decision '{decision}' from LLM, fallback to rules."
            )
            return self._fallback_decision(feedback_state, invalid_ratio_window, progress_delta)

        feedback_state.last_decision = decision
        feedback_state.last_reason = reason
        feedback_state.last_confidence = confidence
        feedback_state.raw_response = json.dumps(parsed, ensure_ascii=False)

        return FeedbackDecision(action=decision, reason=reason)

    def _build_decision_messages(
        self,
        feedback_state: FeedbackState,   # 保留参数以兼容调用，但这里不再使用
        working_memory: WorkingMemory,
        info: Dict[str, Any],
        invalid_ratio_window: float,
        progress_delta: float,
    ) -> List[Dict[str, Any]]:
        allowed = ", ".join(self.ALLOWED_DECISIONS)
        system_prompt = (
            "You are the feedback controller for a household rearrangement agent. "
            "Given recent execution traces and simple metrics, choose the next controller action.\n\n"
            "You should mainly use:\n"
            "- The recent invalid action ratio.\n"
            "- Whether task_progress has increased or stayed flat over recent steps.\n"
            "- The detailed execution history (reward, invalid flag, and env_feedback at each step).\n\n"
            "A step can still be valid even when task_progress does not change. "
            "However, if task_progress has recently increased, that is a strong signal "
            "to prefer 'continue'. When the invalid action ratio is high and task_progress "
            "stays near zero for many steps, and env_feedback repeatedly indicates invalid behaviour "
            "(e.g., too far from object, wrong room, wrong container), consider 'reperceive', 'replan', "
            "or 'reperceive_and_replan' to escape bad loops.\n\n"
            "- If the last 2-3 steps are ALL 'navigate' actions (action descriptions contain 'navigate to'), "
            "you should almost ALWAYS choose 'continue' instead of 'replan'.\n"
            "Always respond with ONLY JSON in the form:\n"
            "{"
            "\"decision\": string in {"
            + allowed
            + "}, "
            "\"reason\": string explanation, "
            "\"confidence\": float between 0 and 1"
            "}\n"
            "Do not wrap the JSON in markdown."
        )

        # ===== 1. 高层：任务目标 / 指令（不再放最新感知） =====
        wm_lines: List[str] = []

        if working_memory.task_model is not None:
            wm_lines.append(f"Goal: {working_memory.task_model.goal_description}")
        else:
            wm_lines.append(f"Goal (raw instruction): {working_memory.instruction}")

        wm_summary = "\n".join(wm_lines) if wm_lines else "N/A"

        # ===== 2. 最新 plan snapshot：只保留 action_ids，不要 reasoning =====
        latest_plan = working_memory.latest_plan()
        if latest_plan:
            plan_ids = latest_plan.action_ids
            plan_text = f"Action IDs : {plan_ids}"
        else:
            plan_text = "No plan available yet."

        # ===== 3. 执行轨迹：完整，带 reward / invalid / task_progress / env_feedback =====
        steps = working_memory.execution_trace
        if steps:
            action_lines: List[str] = []
            for s in steps:
                # info 里取 task_progress 和 env_feedback
                if isinstance(s.info, dict):
                    step_progress = s.info.get("task_progress", None)
                    fb = s.info.get("env_feedback", None)
                else:
                    step_progress = None
                    fb = None

                # 任务进度
                if step_progress is None:
                    prog_str = "NA"
                else:
                    try:
                        prog_str = f"{float(step_progress):.3f}"
                    except Exception:
                        prog_str = str(step_progress)

                # env_feedback 文本
                if isinstance(fb, (dict, list)):
                    fb_str = json.dumps(fb, ensure_ascii=False)
                else:
                    fb_str = str(fb) if fb not in (None, "") else "none"

                action_lines.append(
                    f"- step {s.env_step}: action {s.action_id} '{s.action_desc}', "
                    f"reward={s.reward:.2f}, invalid={s.invalid}, "
                    f"task_progress={prog_str}, env_feedback={fb_str}"
                )

            action_history = "\n".join(action_lines)
        else:
            action_history = "(no executed actions yet)"

        # ===== 4. 当前这一时刻的简要指标（不用 env_feedback 文本） =====
        current_progress = info.get("task_progress", 0.0)
        current_success = info.get("task_success", 0.0)

        user_text = (
            f"Instruction: {working_memory.instruction}\n"
            f"Recent invalid action ratio (window): {invalid_ratio_window:.3f}\n"
            f"Recent progress delta (window): {progress_delta:.3f}\n"
            f"Current task_progress: {float(current_progress):.3f}, "
            f"task_success: {float(current_success):.3f}\n"
            "\n=== High-level goal summary ===\n"
            f"{wm_summary}\n"
            "\n=== Latest plan snapshot ===\n"
            f"{plan_text}\n"
            "\n=== Full action history (from the beginning of this episode, most recent last) ===\n"
            f"{action_history}\n"
            "\nUse the invalid flags, rewards, task_progress trend, and env_feedback texts to choose "
            f"exactly one decision from {{{allowed}}} that best helps the agent either continue "
            "making progress or escape ineffective behaviour."
        )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]


    def _fallback_decision(
        self,
        feedback_state: FeedbackState,
        invalid_ratio_window: float,
        progress_delta: float,
    ) -> FeedbackDecision:
        if invalid_ratio_window > 0.4 and progress_delta <= 0.1:
            feedback_state.last_decision = "reperceive"
            feedback_state.last_reason = "Fallback: high invalid ratio with little progress."
            return FeedbackDecision(
                action="reperceive",
                reason=feedback_state.last_reason,
            )

        if invalid_ratio_window > 0.2:
            feedback_state.last_decision = "replan"
            feedback_state.last_reason = "Fallback: invalid actions accumulating, need better plan."
            return FeedbackDecision(
                action="replan",
                reason=feedback_state.last_reason,
            )

        feedback_state.last_decision = "continue"
        feedback_state.last_reason = "Fallback: progress acceptable."
        return FeedbackDecision(
            action="continue",
            reason=feedback_state.last_reason,
        )

# ===================== 顶层 Agent：Habitat 场景 + 扁平计划 =====================

class HabitatMetaFlatAgent:
    """
    顶层 Agent：
      - 复用 EBHabEnv + VLMPlanner
    - 把「问题求解结构 + 反馈」包在外层
      - 对外暴露一个 run_single_episode()，直接返回 EpisodeSummary 和 episode_info dict
    """

    def __init__(self, env: EBHabEnv, planner: VLMPlanner):
        self.env = env
        self.planner = planner

        self.perception = PerceptionModule(
            model_name=planner.model_name,
            model_type=planner.model_type,
            tp=1,  # 或者从 config 里传
        )
        # self.fusion 已删除 - 不再需要 fusion 中间层
        self.feedback_controller = FeedbackController()
        # 🔥 传入 planner 的 model_name，让 SemanticMemory 使用相同的 LLM
        self.semantic_memory = SemanticMemoryManager(
            max_wm_nodes=256,
            llm_model_name=planner.model_name
        )
        # === 反馈相关 ===
        # 1) 记录 feedback 对感知 / 规划各自给出的 prompt 调整
        self.feedback_adjustments: Dict[str, List[Dict[str, Any]]] = {
            "perception": [],
            "planning": [],
        }
        # 2) 复用同一个 remote 模型做“文字版 feedback 反思”
        self.feedback_llm = self.perception.vision_model  # 已经是 RemoteModel
        self._last_plan_log_index: int = 0

    # ---- 把 VLMPlanner 的一次输出封装成 ActionPlan（扁平序列） ----

    def _plan_with_vlm(
        self,
        img_path: str,
        instruction: str,
        wm: Optional[WorkingMemory] = None,
        feedback_hint: str | None = None,
    ) -> ActionPlan:
        """Invoke :class:`VLMPlanner` with full metacognitive context to get flat plans.

        Responsibilities:
            1. Compose scenario blocks（最新感知 + 执行轨迹）和 heuristics，让 VLM 对当前状态有更准确认知。
            2. 自动拼接 perception / planning feedback hints（metacognition tap point #4）。
            3. 调用 VLMPlanner.act 并把结果转成 ``ActionPlan``（扁平 action_id 序列 + reasoning）。
            4. 统一处理 special codes (-2 空计划、-1 全部无效) 以简化主循环。

        Args:
            img_path (str): 当前画面路径，作为 planner 输入。
            instruction (str): Episode 指令。
            wm (WorkingMemory | None): 若存在，则注入最新感知/执行上下文。
            feedback_hint (str | None): 显式规划提示；None 时自动挖掘累计反馈。

        Returns:
            ActionPlan: action id 列表 + reasoning 文本。

        Notes:
            - 该函数保留统一日志，方便复盘 metacognition 是如何影响 prompt 的。
            - 任何空计划都会提前终止循环并设置 ``empty_plan_flag``，避免执行垃圾动作。
        """
        # 如果调用方没给，就自动用累积的 planning feedback hint
        if feedback_hint is None:
            feedback_hint = self._compose_planning_feedback_hint()
        else:
            # 如果调用方给了（比如你以后想叠加临时提示），也可以在这里再叠加全局的：
            extra = self._compose_planning_feedback_hint()
            if extra:
                feedback_hint = feedback_hint + "\n\n" + extra

        scenario_blocks: List[str] = []
        semantic_result = None
        if self.semantic_memory is not None:
            try:
                logger.info(f"[MetaFlat] 🔍 Retrieving relevant context from LTM for planning...")
                semantic_result = self.semantic_memory.prepare_planner_context(instruction)
                # 🔥 打印LTM检索结果
                if semantic_result:
                    print_ltm_retrieval(
                        semantic_result.nodes,
                        semantic_result.edges,
                        goal=instruction,
                        context="For Planning"
                    )
            except Exception as e:
                logger.warning(f"[MetaFlat] Semantic memory read failed: {e}")
            else:
                if semantic_result.nodes:
                    scenario_blocks.append(
                        "Semantic memory context:\n" + semantic_result.to_text_block()
                    )
                if wm is not None:
                    wm.semantic_memory_digest = self.semantic_memory.export_digest(limit=32)
                    wm.semantic_memory_gaps = list(semantic_result.gaps)
                    wm.semantic_memory_hints = list(semantic_result.ltm_hints)
        if wm is not None:
            perception = wm.latest_perception()
            if perception is not None:
                # 🔥 简化perception：只保留场景概览，不列举对象（已在WM中）
                snapshot_comment = perception.environment_snapshot.get("env_comment_from_model", "") if isinstance(perception.environment_snapshot, dict) else ""
                scenario_blocks.append(
                    "Current view and environment:\n"
                    f"- Scene overview: {perception.scene_summary}\n"
                    # f"- Robot's current view: {snapshot_comment}\n"
                    # f"(Note: Detected objects and their relations are already in Working Memory above)"
                )

            recent_steps = wm.execution_trace
            if recent_steps:
                step_lines = []
                for step in recent_steps:
                    progress = step.info.get("task_progress", "NA") if isinstance(step.info, dict) else "NA"
                    step_lines.append(
                        f"step {step.env_step}: action {step.action_id} '{step.action_desc}', "
                        f"reward={step.reward:.2f}, invalid={step.invalid}, task_progress={progress}"
                    )
                
                # 🔥 添加执行分析指导
                analysis_guide = (
                    "\n\n🔍 **ANALYZE the execution snapshot carefully**:\n"
                    "- **Successful actions** (reward > 0, invalid=False, task_progress increased):\n"
                    "  * These actions were CORRECT and made progress toward the goal\n"
                    "  * REMEMBER which locations/objects worked and REUSE this knowledge\n"
                )
                
                scenario_blocks.append(
                    "Recent execution snapshot:\n" + "\n".join(step_lines) + analysis_guide
                )

        scenario_context = "\n\n".join(block for block in scenario_blocks if block).strip()
        if not scenario_context:
            scenario_context = "(No additional perception or execution context available.)"

        heuristics_text = (
            "\n🔎 **GOAL-FIRST SEARCH POLICY (must follow before any long plan):**\n"
            "A. **Fully understand the instruction first**: extract the TRUE target object (what to pick) and the TRUE destination (where to place).\n"
            "B. **Confirm target identity**: The location of the target object must be confirmed first.\n"
            "C. **If WM gives locations**: go to them in priority order before doing other exploration.\n"
            "D. **If WM has NO location evidence**: explore systematically by navigating to NEW, unvisited anchor areas and scan.\n"
            "E. **STOP exploring as soon as target is found**\n"
            "F. **NO REVISITS**: do NOT navigate to the same named viewpoint/anchor area repeatedly \n"
            
            "🎯 **CRITICAL: Always follow Task Understanding and Working Memory first!**\n"
            "- **READ Task Understanding carefully**: It tells you what objects in the instruction map to.\n"
            "- **CHECK Working Memory**: consult WM for known locations and evidence.\n"
            "- **When WM lists a location for an object, ALSO check that location's immediate neighbors (left/right).** "
            "For example, if WM indicates 'counter', also try 'left counter' and 'right counter'; if WM indicates 'drawer of the kitchen counter', also try 'left drawer of the kitchen counter' and 'right drawer of the kitchen counter'.\n"
            "- **WM-FIRST: If WM lists multiple locations for the target, prioritize WM and try each location in turn.\n"
            "If the target is not found after trying all WM locations, navigate to a new area and scan from a new viewpoint.\n"
            "- **NEVER explore randomly** if Task Understanding + WM already tells you where the object is.\n\n"
            "Planning sequence:\n"
            "1. Read Task Understanding → identify what object you need\n"
            "2. Check Working Memory → see if object location is known\n"
            "3. If location known → navigate there directly\n"
            "4. If location unknown → explore systematically\n"
            "- **CHECK recent action history carefully**: If an action was marked as INVALID (was_prev_action_invalid=True), DO NOT repeat the same action (or the same action template on the same target) again. Instead, change the room/target and gather new evidence before trying a different action.\n"
            "- NO REVISITS: Do NOT navigate/scan the same named anchor/viewpoint again if it was already checked and the target was not found.\n"
            "- NO INVALID REPEAT: If was_prev_action_invalid=True, NEVER repeat the same action (or same action on the same target/container). Change room/anchor/target first.\n"
            "- NO VIEW TOGGLING: Do not switch viewpoints/camera angles in the same place. At each anchor, do at most ONE quick check (one look/scan). If not found, MOVE to a new anchor.\n"
            "Additional heuristics:\n"
            "ACTION-ID MATCH: The action_name description and action_id MUST exactly correspond to the same action in the available actions list. Double-check before outputting.\n"
            "- When repeated invalid actions occur near the same furniture, explore a different room to gather new observations.\n"
        )

        planning_prompt = (
            f"Instruction: {instruction.strip()}\n\n"
            "### Scenario context for planning\n"
            f"{scenario_context}\n\n"
            "### Planning heuristics\n"
            f"{heuristics_text}"
        )

        if feedback_hint:
            effective_instruction = (
                planning_prompt
                + "\n\n[Feedback hints about planning]\n"
                + feedback_hint
            )
        else:
            effective_instruction = planning_prompt

        logger.info(
            f"[MetaFlat] Planning with VLM. episode={self.env._current_episode_num}, "
            f"step={self.env._current_step}, instruction={instruction}"
        )
        logger.info(f"[MetaFlat] Effective instruction for VLM:\n{effective_instruction}\n")
        
        # 🔥 传递semantic memory context到planner，用于提取task-relevant actions
        wm_context_for_planner = None
        if semantic_result is not None and semantic_result.nodes:
            wm_context_for_planner = semantic_result.to_text_block()
        
        action, reasoning = self.planner.act(
            img_path, 
            effective_instruction, 
            wm_context=wm_context_for_planner
        )
        # VLMPlanner.json_to_action 已经把 -1/-2/列表处理过：
        # -2: 空 plan
        # -1: 全部无效
        if action == -2:
            logger.info("[MetaFlat] VLMPlanner returned EMPTY PLAN (-2).")
            return ActionPlan(action_ids=[], reasoning=reasoning)
        if action == -1:
            # 视为「只输出一个 invalid 占位」，后续逻辑会统计 invalid
            logger.info("[MetaFlat] VLMPlanner returned INVALID PLAN (-1).")
            return ActionPlan(action_ids=[], reasoning=reasoning)

        # 否则一定是 list[int]
        if isinstance(action, int):
            action_ids = [int(action)]
        else:
            action_ids = [int(a) for a in action]

        logger.info(
            f"[MetaFlat] Planned action ids: {action_ids}; reasoning_len={len(reasoning) if isinstance(reasoning, str) else 'n/a'}"
        )
        
        # 🔥 打印planner的reasoning内容
        if reasoning and isinstance(reasoning, str):
            logger.info(f"[MetaFlat] Planner reasoning:\n{reasoning}")
        
        return ActionPlan(action_ids=action_ids, reasoning=reasoning)

    # ---- 单条 episode 的完整流程 ----

    def run_single_episode(self) -> Tuple[EpisodeSummary, Dict[str, Any]]:
        """Execute the full perception→fusion→plan→feedback loop for one episode.

        Responsibilities:
            1. Reset env / planner, capture first observation, and initialize :class:`WorkingMemory`。
            2. Sequentially run perception, fusion, task modeling, planning, and execution with embedded feedback hooks.
            3. Maintain statistics（rewards、invalid ratio、elapsed time）并写入 ``EpisodeSummary``。
            4. Emit ``episode_info`` dict，保持与历史 evaluator 输出兼容，便于日志/评测。

        Returns:
            Tuple[EpisodeSummary, Dict[str, Any]]: (rich summary, legacy episode_info)

        Notes:
            - 这是元认知主循环的“实车路测”：perception/planning feedback hints都会在此被触发。
            - WorkingMemory 被保存在 summary 中，便于后续调试或训练数据蒸馏。
        """
        # reset 环境
        obs = self.env.reset()
        img_path = self.env.save_image(obs)
        instruction = self.env.episode_language_instruction
        self.planner.reset()

        logger.info(
            f"[MetaFlat] Start episode {self.env._current_episode_num} "
            f"with instruction: {instruction}"
        )

        wm = WorkingMemory(instruction=instruction)
        self.semantic_memory.reset(instruction=instruction)

        # ===== Stage 1: 感知 =====
        feedback_hint_for_perception = self._compose_perception_feedback_hint()
        p_out = self.perception.perceive(
            self.env, img_path, instruction,
            feedback_hint=feedback_hint_for_perception,
        )
        wm.perception_history.append(p_out)
        try:
            logger.info(f"[MetaFlat] 📥 Ingesting initial perception into WM...")
            self.semantic_memory.ingest_perception(
                asdict(p_out), 
                step=self.env._current_step, 
                clip_id=img_path
            )
            # 🔥 打印感知后的WM状态
            print_wm_state(self.semantic_memory, self.env._current_step, "After Initial Perception")
        except Exception as e:
            logger.warning(f"[MetaFlat] Semantic memory perception ingest failed: {e}")
        else:
            wm.semantic_memory_digest = self.semantic_memory.export_digest(limit=32)

        # ===== Stage 2: 任务建模 + 反馈初始化 =====
        # (Fusion 阶段已删除 - 直接使用 SemanticMemory)
        task_model = self.feedback_controller.build_task_model(instruction, self.env)
        wm.task_model = task_model
        wm.feedback_state = self.feedback_controller.init_feedback_state()

        # ===== Stage 3/4: 计划 + 执行 + 反馈循环 =====
        rewards: List[float] = []
        num_invalid_actions = 0
        empty_plan_flag = 0

        current_plan: Optional[ActionPlan] = None
        plan_pointer = 0
        done = False
        info: Dict[str, Any] = {
            "task_success": 0.0,
            "task_progress": 0.0,
            "subgoal_reward": 0.0,
            "env_step": 0,
        }

        recent_invalid_flags: List[bool] = []
        steps_since_feedback = 0
        last_task_progress = 0.0
        episode_start_time = self.env._episode_start_time or time.time()

        while not done and self.env._current_step < self.env._max_episode_steps:
            # 如果当前 plan 用完了，重新规划
            if current_plan is None or plan_pointer >= len(current_plan.action_ids):
                current_plan = self._plan_with_vlm(img_path, instruction, wm=wm)
                logger.info(
                    f"[MetaFlat] New plan received: n_actions={len(current_plan.action_ids)}; "
                    f"first_actions={current_plan.action_ids}"
                )
                wm.plans.append(current_plan)
                plan_pointer = 0
                # 新 plan 产生：从现在开始，后面的反馈统计都只看这个时刻之后的 step
                self._last_plan_log_index = len(getattr(self.env, "episode_log", []) or [])
                recent_invalid_flags = []  # 清空“最近 invalid”统计，只看新 plan 之后
                last_task_progress = info.get("task_progress", last_task_progress)

                if len(current_plan.action_ids) == 0:
                    # 相当于 VLM 说「我不知道该干嘛」
                    empty_plan_flag = 1
                    logger.info("[MetaFlat] Empty action plan, stop episode early.")
                    break

            # 取下一个 action_id
            action_id = current_plan.action_ids[plan_pointer]
            action_desc = self.env.language_skill_set[action_id]

            logger.info(
                f"[MetaFlat] Step {self.env._current_step+1}: "
                f"execute action_id={action_id}, desc='{action_desc}'"
            )

            try:
                obs, reward, done, info = self.env.step(
                    action_id, reasoning=current_plan.reasoning
                )
            except Exception as e:
                logger.error(f"[MetaFlat] env.step error: {e}")
                # 当成 invalid
                reward = -1.0
                done = False
                info = {
                    "env_step": self.env._current_step,
                    "task_success": 0.0,
                    "task_progress": last_task_progress,
                    "subgoal_reward": 0.0,
                    "was_prev_action_invalid": True,
                }
            else:
                logger.info(
                    f"[MetaFlat] env.step result: step={info.get('env_step', self.env._current_step)}, "
                    f"reward={reward:.3f}, task_success={info.get('task_success', 0.0):.3f}, "
                    f"task_progress={info.get('task_progress', 0.0):.3f}, invalid={bool(info.get('was_prev_action_invalid', False))}"
                )

            rewards.append(reward)
            invalid_flag = bool(info.get("was_prev_action_invalid", False))
            recent_invalid_flags.append(invalid_flag)
            if invalid_flag:
                num_invalid_actions += 1

            # planner 用环境 feedback 更新内部状态
            try:
                self.planner.update_info(info)
            except Exception as e:
                logger.warning(f"[MetaFlat] planner.update_info error: {e}")

            # 存日志
            wm.execution_trace.append(
                ExecutionStepTrace(
                    env_step=int(info.get("env_step", self.env._current_step)),
                    action_id=action_id,
                    action_desc=action_desc,
                    reward=float(reward),
                    info=info,
                    invalid=invalid_flag,
                    env_feedback=info.get("env_feedback"),
                )
            )
            
            # 🔥 先保存当前图像（执行action后的新视图）
            img_path = self.env.save_image(obs)
            logger.info(f"[MetaFlat] Saved post-action image to {img_path}")
            
            # 🔥🔥🔥 每步执行后调用LLM感知新图像，分析动作执行结果
            # 构建最近动作历史
            recent_actions = [
                {
                    "action_desc": step.action_desc,
                    "reward": step.reward,
                    "invalid": step.invalid,
                    "env_feedback": step.env_feedback,
                }
                for step in wm.execution_trace[-5:]  # 最近5步
            ]
            
            try:
                # 调用新的感知方法：分析动作执行结果
                logger.info(f"[MetaFlat] 📥 Running post-action LLM perception...")
                p_out_post_action = self.perception.perceive_with_action_history(
                    env=self.env,
                    img_path=img_path,
                    instruction=instruction,
                    last_action=action_desc,
                    recent_actions=recent_actions,
                    env_feedback=info.get("env_feedback"),
                )
                
                # 添加到感知历史
                wm.perception_history.append(p_out_post_action)
                
                # 将LLM感知结果写入WM图
                logger.info(f"[MetaFlat] 💾 Ingesting post-action perception into WM...")
                self.semantic_memory.ingest_perception(
                    asdict(p_out_post_action),
                    step=self.env._current_step,
                    clip_id=img_path,
                )
                
                logger.info(
                    f"[MetaFlat] Post-action LLM perception: "
                    f"detected {len(p_out_post_action.detected_objects)} objects, "
                    f"{len(p_out_post_action.state_changes)} state changes"
                )
                
                # 🔥 打印step执行后的WM状态
                print_wm_state(self.semantic_memory, self.env._current_step, f"After Step {self.env._current_step} Execution")
                
            except Exception as e:
                logger.warning(f"[MetaFlat] Post-action LLM perception failed: {e}")
                # 失败时仍然做基于规则的correctness更新
                try:
                    self.semantic_memory.ingest_action_feedback(
                        action_desc=action_desc,
                        env_info=info,
                        reward=float(reward),
                        step=int(info.get("env_step", self.env._current_step)),
                    )
                except Exception as e2:
                    logger.warning(f"[MetaFlat] Fallback correctness update also failed: {e2}")
            
            # 更新WM摘要供后续使用
            try:
                wm.semantic_memory_digest = self.semantic_memory.export_digest(limit=32)
            except Exception as e:
                logger.warning(f"[MetaFlat] Failed to export WM digest: {e}")

            steps_since_feedback += 1
            plan_completed = (
                current_plan is not None
                and len(current_plan.action_ids) > 0
                and (plan_pointer + 1) >= len(current_plan.action_ids)
            )

            # ---- 反馈监控：把最近表现交给 LLM 决策下一步 ----
            if (
                wm.feedback_state is not None
                and self.env._current_step > 0
                and (steps_since_feedback >= 2 or plan_completed)
            ):
                window = min(10, len(recent_invalid_flags))
                invalid_ratio_window = (
                    sum(recent_invalid_flags[-window:]) / window if window > 0 else 0.0
                )
                cur_progress = info.get("task_progress", 0.0)
                progress_delta = cur_progress - last_task_progress
                last_task_progress = cur_progress
                logger.info(
                    f"[MetaFlat] Feedback monitor window={window}, invalid_ratio_window={invalid_ratio_window:.3f}, "
                    f"progress_delta={progress_delta:.3f}, cur_progress={cur_progress:.3f}"
                )

                feedback_decision = self.feedback_controller.assess_step(
                    wm.feedback_state,
                    working_memory=wm,
                    info=info,
                    invalid_ratio_window=invalid_ratio_window,
                    progress_delta=progress_delta,
                    decision_model=self.feedback_llm
                )

                steps_since_feedback = 0

                if feedback_decision.action in ("replan", "reperceive", "reperceive_and_replan"):
                    need_replan = False

                    # === 先让反馈模块根据最近轨迹产出 prompt hints ===
                    if feedback_decision.action in ("reperceive", "reperceive_and_replan"):
                        # 为感知阶段更新 feedback 提示
                        self._run_feedback_update(stage="perception", wm=wm, last_info=info)
                        # 重新感知时，用“从感知开始后的最新 feedback 提示”
                        feedback_hint_for_perception = self._compose_perception_feedback_hint()
                        p_out = self.perception.perceive(
                            self.env, img_path, instruction,
                            feedback_hint=feedback_hint_for_perception,
                        )
                        wm.perception_history.append(p_out)
                        try:
                            self.semantic_memory.ingest_perception(
                                asdict(p_out), 
                                step=self.env._current_step, 
                                clip_id=img_path
                            )
                        except Exception as e:
                            logger.warning(
                                f"[MetaFlat] Semantic memory ingest failed after reperceive: {e}"
                            )
                        else:
                            wm.semantic_memory_digest = self.semantic_memory.export_digest(limit=32)
                        # fusion 调用已删除 - 不再需要
                        need_replan = True

                    if feedback_decision.action in ("replan", "reperceive_and_replan") or need_replan:
                        # 为规划阶段更新 feedback 提示
                        self._run_feedback_update(stage="planning", wm=wm, last_info=info)
                        # 重新规划时，不需要手动传 feedback_hint，
                        # 因为 _plan_with_vlm 会自动从 perception + planning 两个阶段
                        # 聚合最新的反馈提示
                        current_plan = self._plan_with_vlm(
                            img_path, instruction, wm=wm,
                        )

                        logger.info(
                            f"[MetaFlat] New plan after {feedback_decision.action}: "
                            f"n_actions={len(current_plan.action_ids)}; "
                            f"first_actions={current_plan.action_ids[:5]}"
                        )
                        wm.plans.append(current_plan)
                        plan_pointer = 0
                        # 反馈触发的新 plan：后续统计只看这个时刻之后的表现
                        self._last_plan_log_index = len(getattr(self.env, "episode_log", []) or [])
                        recent_invalid_flags = []
                        last_task_progress = info.get("task_progress", last_task_progress)
                        if len(current_plan.action_ids) == 0:
                            empty_plan_flag = 1
                            logger.info(
                                "[MetaFlat] Empty plan after replan, stop episode."
                            )
                            break

                    # ★★ 重点：已经 reperceive / replan 了，这一轮不再执行旧 action，直接进入下一轮 while
                    continue

            # 如果任务已经完成或达到 max_invalid / max_episode_steps，会在 env 内设置 done
            if done:
                logger.info("[MetaFlat] Env terminated episode (done=True).")
                break

            plan_pointer += 1

        # ===== Episode 结束，整理统计 =====
        elapsed = time.time() - episode_start_time
        env_step = info.get("env_step", self.env._current_step)
        env_step = int(env_step) if isinstance(env_step, (int, float)) else self.env._current_step

        reward_mean = float(sum(rewards) / len(rewards)) if len(rewards) > 0 else 0.0
        task_success = float(info.get("task_success", 0.0))
        task_progress = float(info.get("task_progress", 0.0))
        subgoal_reward = float(info.get("subgoal_reward", 0.0))
        num_invalid_ratio = num_invalid_actions / env_step if env_step > 0 else 0.0

        logger.info(
            "[MetaFlat] Episode end stats: "
            f"steps={env_step}, reward_mean={reward_mean:.3f}, task_success={task_success:.3f}, "
            f"task_progress={task_progress:.3f}, subgoal_reward={subgoal_reward:.3f}, "
            f"num_invalid={num_invalid_actions}, invalid_ratio={num_invalid_ratio:.3f}, "
            f"empty_plan={empty_plan_flag}"
        )

        try:
            # 🔥🔥🔥 新增：Episode结束时调用LLM总结整个episode
            logger.info("[MetaFlat] Generating LLM episode summary...")
            
            # 构建完整动作历史
            action_history = [
                {
                    "action_desc": step.action_desc,
                    "reward": step.reward,
                    "invalid": step.invalid,
                    "env_feedback": step.env_feedback,
                }
                for step in wm.execution_trace
            ]
            
            # 最终状态
            final_state = {
                "episode": self.env._current_episode_num,
                "steps": env_step,
                "is_holding": self.env.is_holding,
                "task_success": task_success,
                "task_progress": task_progress,
            }
            
            # 调用LLM生成episode总结
            llm_summary = self.perception.summarize_episode(
                instruction=instruction,
                action_history=action_history,
                final_state=final_state,
                task_success=task_success,
                task_progress=task_progress,
            )
            
            logger.info(
                f"[MetaFlat] LLM episode summary generated: "
                f"{len(llm_summary.get('learned_patterns', []))} patterns, "
                f"{len(llm_summary.get('spatial_knowledge', []))} spatial knowledge"
            )
            
            # 🔥 将LLM总结传入semantic_memory，先更新WM，再巩固到LTM
            wm.semantic_memory_report = self.semantic_memory.on_episode_end(
                llm_summary=llm_summary
            )
            
            # 保存LLM总结到WorkingMemory供后续分析
            wm.llm_episode_summary = llm_summary
            
        except Exception as e:
            logger.warning(f"[MetaFlat] Episode LLM summary or consolidation failed: {e}")
            # 即使LLM总结失败，仍然执行基本的WM→LTM巩固
            try:
                wm.semantic_memory_report = self.semantic_memory.on_episode_end()
            except Exception as e2:
                logger.warning(f"[MetaFlat] Fallback semantic memory consolidation also failed: {e2}")

        summary = EpisodeSummary(
            instruction=instruction,
            eval_set=self.env.eval_set if hasattr(self.env, "eval_set") else "",
            task_success=task_success,
            task_progress=task_progress,
            subgoal_reward=subgoal_reward,
            num_steps=env_step,
            planner_steps=self.planner.planner_steps,
            planner_output_error=self.planner.output_json_error,
            num_invalid_actions=num_invalid_actions,
            num_invalid_action_ratio=num_invalid_ratio,
            reward_mean=reward_mean,
            episode_elapsed_seconds=elapsed,
            empty_plan=empty_plan_flag,
            working_memory=wm,
        )

        # 同时返回一个和你原来 evaluator 兼容的 episode_info 字典
        episode_info = dict(
            instruction=instruction,
            reward=reward_mean,
            task_success=task_success,
            task_progress=task_progress,
            subgoal_reward=subgoal_reward,
            num_steps=env_step,
            planner_steps=self.planner.planner_steps,
            planner_output_error=self.planner.output_json_error,
            num_invalid_actions=num_invalid_actions,
            num_invalid_action_ratio=num_invalid_ratio,
            episode_elapsed_seconds=elapsed,
            empty_plan=empty_plan_flag,
        )
        return summary, episode_info
    
    def _apply_feedback_adjustment(self, stage: str, prompt_hint: str, suggestions: List[str]):
        """
        stage: 'perception' 或 'planning'
        prompt_hint: LLM 生成的一段总的提示文字
        suggestions: 若干 bullet 级别的建议
        """
        if stage not in self.feedback_adjustments:
            self.feedback_adjustments[stage] = []

        self.feedback_adjustments[stage].append(
            {
                "prompt_hint": prompt_hint or "",
                "suggestions": [s for s in suggestions if s],
                "timestamp": time.time(),
            }
        )

    def _get_feedback_hint(self, stage: str) -> str:
        """
        把最近一次的 feedback 建议变成一个小段落，拼进感知 / 规划的 prompt。
        """
        records = self.feedback_adjustments.get(stage, [])
        if not records:
            return ""

        last = records[-1]
        parts: List[str] = []
        if last.get("prompt_hint"):
            parts.append(last["prompt_hint"])
        for s in last.get("suggestions", []):
            parts.append(s)

        if not parts:
            return ""

        return "[Feedback hints]\n" + "\n".join(f"- {p}" for p in parts)
    
    def _compose_perception_feedback_hint(self) -> str:
        """
        感知阶段用的 feedback hint：
        - 只拿最近一次 perception 阶段的反馈。
        """
        return self._get_feedback_hint("perception")

    def _compose_planning_feedback_hint(self) -> str:
        """
        规划阶段用的 feedback hint：
        - 先包含最近一次 perception 的反馈（环境/物体理解的纠正）
        - 再包含最近一次 planning 的反馈（动作序列层面的纠正）
        这样当「重新感知」发生后，新的 perception hint 会自动进入后续所有规划。
        当「重新规划」发生后，新的 planning hint 也会自动作用到后续所有规划。
        """
        parts: List[str] = []

        # 让感知层面的总结也能影响后续规划
        h_perc = self._get_feedback_hint("perception")
        if h_perc:
            parts.append(h_perc)

        # 再叠加规划层面的调整建议
        h_plan = self._get_feedback_hint("planning")
        if h_plan:
            parts.append(h_plan)

        if not parts:
            return ""

        # 简单拼接即可，哪怕里面有多个 [Feedback hints] 也没关系
        return "\n\n".join(parts)

    
    def _run_feedback_update(self, stage: str, wm: WorkingMemory, last_info: Dict[str, Any]):
        """Synthesize episode experience into stage-specific feedback hints.

        Data contract:
            - WorkingMemory: instructions, perceptions, latest plan, execution trace (metacognition tap point #5).
            - ``last_info``: 原始 env 指标，确保最近一次反馈包含即时奖励/进度。
            - ``self.env.episode_log`` + 保存的图片：提供多模态证据供 LLM 诊断。

        Workflow:
            1. Slice episode_log 到“上一次规划之后”，统计 invalid / progress 概览。
            2. 采样最近若干张图像并转成 data URLs。
            3. 组合文本上下文（WorkingMemory 摘要 + last_info），组装成 LLM 输入。
            4. 调用 feedback LLM，解析 ``prompt_hint`` + ``adjustment_suggestions``，写入 ``feedback_adjustments``。

        Args:
            stage (str): ``perception`` 或 ``planning``，决定提示侧重点。
            wm (WorkingMemory): 当前 episode 的共享记忆，用于生成 context。
            last_info (Dict[str, Any]): 最近一步 env info。

        Notes:
            - 这是闭环元认知的核心，触发后下一次感知/规划 prompt 将自动携带最新 hints。
            - 图片/日志裁剪逻辑保证 LLM token 成本可控，同时覆盖最新 10~15 步。
        """
        # ---------- 1. 构造 episode_log 的文字摘要 ----------
        episode_log_all = getattr(self.env, "episode_log", []) or []
        episode_id = getattr(self.env, "_current_episode_num", 0)

        # 只取“上一次规划之后”的片段
        start_idx = getattr(self, "_last_plan_log_index", 0)
        episode_log = episode_log_all[start_idx:]

        total_steps = len(episode_log)
        invalid_count = 0
        progress_values = []
        segment_step_indices: List[int] = []

        log_lines: List[str] = []
        log_lines.append(
            f"Current episode id: {episode_id}, "
            f"total steps so far in episode: {len(episode_log_all)}, "
            f"steps considered since last plan: {total_steps}"
        )

        for i, rec in enumerate(episode_log):
            # i 是片段内的索引，global_idx 只是 debug 用
            global_idx = start_idx + i
            step_idx = rec.get("env_step", rec.get("step", global_idx))
            segment_step_indices.append(step_idx)

            action_id = rec.get("action_id", {})
            action_description = rec.get("action_description", {})
            reward = rec.get("subgoal_reward", rec.get("r", None))
            env_feedback = rec.get("env_feedback", {})
            task_progress = rec.get("task_progress", None)
            was_prev_action_invalid = rec.get(
                "was_prev_action_invalid", not rec.get("invalid", False)
            )
            task_success = rec.get("task_success", None)
            last_action_success = rec.get("last_action_success", None)

            if was_prev_action_invalid:
                invalid_count += 1
            if task_progress is not None:
                progress_values.append(float(task_progress))

            log_lines.append(
                f"- step={step_idx}, action_id={action_id}, desc='{action_description}', "
                f"reward={reward}, task_success={task_success}, task_progress={task_progress}, "
                f"env_feedback={env_feedback}, was_prev_action_invalid={was_prev_action_invalid}, "
                f"last_action_success={last_action_success}"
            )

        invalid_ratio = (
            float(invalid_count) / total_steps if total_steps > 0 else 0.0
        )
        avg_progress = (
            sum(progress_values) / len(progress_values)
            if progress_values
            else 0.0
        )

        log_lines.append(
            f"Overall (since last plan) invalid_count={invalid_count}, "
            f"invalid_ratio≈{invalid_ratio:.3f}, "
            f"avg_task_progress≈{avg_progress:.3f}"
        )

        # 为了避免 prompt 太长，只保留最后 max_detailed_steps 步的详细信息 
        max_detailed_steps = 15
        if total_steps > max_detailed_steps:
            kept = ["(Older steps since last plan omitted)"] + log_lines[-max_detailed_steps:]
            log_lines = log_lines[:2] + kept  # 前两行是全局统计 + 片段统计

        episode_log_summary = "\n".join(log_lines)


        # ---------- 2. 收集 episode 的图片（每一步一张） ----------
        # 为了 token / 带宽，通常只给最近几步的图片
        max_images = 10
        image_contents: List[Dict[str, Any]] = []

        # 图片目录：log_path/images/episode_{episode_id}
        images_folder = os.path.join(
            self.env.log_path, "images", f"episode_{episode_id}"
        )

        if os.path.isdir(images_folder):
            # 假设文件名：episode_{episode_id}_step_{step}.png
            # 我们按 step 排序，然后取最后 max_images 张
            all_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
            # 简单按文件名排序（通常 step 越大文件名越靠后）
            all_files = sorted(all_files)

            selected_files = all_files[-max_images:]
            for fname in selected_files:
                img_path = os.path.join(images_folder, fname)
                try:
                    data_url = local_image_to_data_url(image_path=img_path)
                except Exception as e:
                    logger.warning(
                        f"[MetaFlat] Failed to load image {img_path} for feedback update: {e}"
                    )
                    continue

                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    }
                )

        # ---------- 3. WorkingMemory + last_info 的额外摘要 ----------
        wm_text = summarize_recent_trace(wm, max_steps=5)
        last_info_text = json.dumps(last_info, ensure_ascii=False)

        # ---------- 4. 组装发给 feedback LLM 的消息 ----------
        actions_feedback_block = ""
        actions_raw = getattr(self.planner, "available_action_str", None)

        if actions_raw:
            # 兼容几种常见格式：list / tuple / 已经拼好的 str
            if isinstance(actions_raw, (list, tuple)):
                actions_str = "\n".join(f"- {a}" for a in actions_raw)
            else:
                actions_str = str(actions_raw)

            actions_feedback_block = (
                "\n\nHere is the full list of available low-level actions the robot can take "
                "in this dataset (each line is one action template):\n"
                f"{actions_str}\n"
                "When you propose hints or adjustments, refer to these actions explicitly where relevant, "
                "e.g., by suggesting concrete sequences like 'navigate to the table 2' → 'look around' → "
                "'pick up the pear' → 'navigate to the sofa' → 'place at the sofa'.\n"
            )

    # 针对不同阶段给一点额外说明
        if stage == "perception":
            stage_specific = (
                "Focus on summarizing what the environment looks like across steps "
                "(room layout, key furniture, containers, obstacles), "
                "what objects are present (including pears / oranges / containers), "
                "Point out if the camera keeps staring at the same spot or room, and suggest new rooms or viewpoints to scan when progress stalls. "
                "Then explain what the perception module should explicitly check or re-check "
                "in the next observation\n"
            )
        elif stage == "planning":
            stage_specific = (
                "Focus on analyzing the quality of the action sequence so far: "
                "which actions were repeated, which ones tended to be invalid, "
                "and how to change the next high-level action plan. "
                "If you cannot find the target object, navigate to a different area or room and scan from a new viewpoint, instead of revisiting the same locations."

            )
        else:
            stage_specific = (
                "Focus on explaining what went wrong so far and how the agent should "
                "adjust its behaviour in the next steps.\n"
            )

        system_prompt = (
            "You are a feedback coach for an embodied robot agent in a household "
            "rearrangement environment.\n\n"
            "Given:\n"
            "1) The full history of steps in the current episode so far (rewards, invalid actions, progress).\n"
            "2) Images from several recent steps.\n"
            "3) A short summary from the agent's WorkingMemory.\n"
            "4) The latest env info.\n\n"
            f"Your job is to generate knowledge and hints to improve the next {stage} prompt, "
            "so that the agent can perceive the scene more accurately or plan actions better.\n\n"
            "1. **DO NOT repeat successful but unproductive actions**:\n"
            "   - If 'navigate to X' was VALID (invalid=False) but subsequent 'pick up Y' FAILED,\n"
            "     it means Y is NOT at location X.\n"
            "   - DON'T suggest 'navigate to X again'. Instead suggest exploring OTHER locations.\n"
            "2. **DO NOT repeat invalid actions**:\n"
            "   - If an action was marked as INVALID, DO NOT suggest that same action.\n"
            "3. **Diagnose root cause accurately**:\n"
            "   - Navigation success + Pick failure → Object not at that location\n"
            "   - Navigation failure → Location not reachable or doesn't exist\n"
            "WM-FIRST: If WM lists multiple locations for the target, prioritize WM and try each location in turn."
            "In particular, your output MUST:\n"
            "- Summarize the environment and objects based on the images and logs\n"
            "- Explain what seems to be going wrong in the behaviour so far\n"
            "- Diagnose WHY actions failed (not just THAT they failed)\n"
            "- Provide concrete advice tailored to the next stage "
            f"({stage}), not just generic tips.\n\n"
            + stage_specific
            + actions_feedback_block +  # ★ 把动作列表插进来
            "Return ONLY a JSON object with the following structure:\n"
            "{\n"
            '  \"prompt_hint\": string,  // 3-6 sentences. First summarize the scene and objects across steps; '
            'then analyze mistakes and diagnose root cause; then tell the agent what to explicitly check / adjust in the next prompt.\n'
            '  \"adjustment_suggestions\": [string]  // 3-8 bullet suggestions; each is a short imperative sentence. '
            'AVOID repeating actions that already proved unproductive.\n'
            "}\n"
            "Do not wrap the JSON in markdown. Do not include any other top-level keys.\n"
            "The hints should be concrete and should reference observations from the images and episode_log, "
            "as well as the concrete low-level actions available to the agent, "
            "not just vague high-level advice."
        )

        # user 内容：若干 image_url + 文本说明
        user_contents: List[Dict[str, Any]] = []

        # 先把图片塞进去（每张图后面加一点说明文字，方便 LLM 对号入座）
        if image_contents:
            user_contents.extend(image_contents)
            user_contents.append(
                {
                    "type": "text",
                    "text": "Above are recent observation images from the current episode.",
                }
            )

        # 再加文字上下文（episode_log + WM + last_info）
        user_text_block = (
            "Textual context of the current episode so far:\n\n"
            "=== Episode log (all steps so far, possibly truncated) ===\n"
            f"{episode_log_summary}\n\n"
            "=== WorkingMemory short summary ===\n"
            f"{wm_text}\n\n"
            "=== Latest env info (reward, progress, etc.) ===\n"
            f"{last_info_text}\n\n"
            "Please analyze why the current behaviour may be ineffective or stuck, "
            "and propose hints that can be injected into the next prompt for this stage "
            f"({stage})."
        )
        user_contents.append({"type": "text", "text": user_text_block})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_contents,
            },
        ]

        # ---------- 5. 调 feedback LLM，解析 JSON ----------
        try:
            raw = self.feedback_llm.respond_freeform(
                messages, temperature=0.0, max_tokens=512
            )
            logger.info(f"[MetaFlat] Feedback raw output at stage={stage}:\n{raw}\n")
            try:
                fixed = fix_json(raw)
            except Exception:
                fixed = raw
            parsed = json.loads(fixed)
        except Exception as e:
            logger.warning(f"[MetaFlat] feedback update failed at stage={stage}: {e}")
            return

        prompt_hint = str(parsed.get("prompt_hint", "")).strip()
        suggestions = parsed.get("adjustment_suggestions", []) or []
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        self._apply_feedback_adjustment(
            stage, prompt_hint, [str(s) for s in suggestions]
        )