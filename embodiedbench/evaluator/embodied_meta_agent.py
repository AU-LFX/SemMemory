# embodied_meta_agent_flat.py
"""
    感知 → Memory-Web Fusion → TaskModel + MetaState（问题求解）→ ActionSequencePlanner → Executor → Memory
"""

from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
from embodiedbench.main import logger


# ===================== 通用接口 & 工具 =====================

class LLMClient(Protocol):
    """你自己的 LLM 封装，需要实现 request() 方法。"""
    def request(self, prompt: Any) -> str:
        ...


class MemoryStore(Protocol):
    """长期记忆接口（可选实现）。"""
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...

    def store_episode(self, episode_summary: "EpisodeSummary") -> None:
        ...

    def store_meta_reflection(self, reflection: Dict[str, Any]) -> None:
        ...


class WebRetriever(Protocol):
    """Web 检索接口（可选实现）。"""
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...


def _safe_first_json(text: str) -> Dict[str, Any]:
    """
    从 LLM 输出中尽量抽出第一段合法 JSON。
    """
    text = text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        snippet = text[start:end]
        return json.loads(snippet)
    except Exception:
        logger.warning("[MetaAgentFlat] Failed to parse JSON, fallback to empty dict.")
        return {}


# ===================== 核心数据结构 =====================

@dataclass
class PerceptionOutput:
    """感知模块输出：从图像 + 历史抽象出来的环境语义。"""
    scene_summary: str
    detected_objects: List[Dict[str, Any]]      # [{"name": "tomato", "state": "dirty", "position": ...}, ...]
    state_changes: List[str]                    # ["tomato_now_wet", "sink_water_running"]
    subgoals_proposal: List[str]                # 候选子目标（纯辅助）
    environment_snapshot: Dict[str, Any]        # 用于之后状态对比的结构化 snapshot


@dataclass
class FusionContext:
    """Memory-Web Fusion 的综合上下文，用于 Planner。"""
    perception: PerceptionOutput
    episodic_hints: List[str]
    semantic_knowledge: List[str]
    household_preferences: List[str]
    risk_constraints: List[str]
    fused_summary: str                          # 给 LLM 看的综合摘要（字符串）


@dataclass
class PlannedAction:
    """大模型输出的扁平动作计划中的一条动作。"""
    index: int
    action_id: int
    description: str
    rationale: str


@dataclass
class ActionPlan:
    """扁平动作计划：一串动作 + 全局推理。"""
    actions: List[PlannedAction]
    global_rationale: str


@dataclass
class ExecutionStepTrace:
    """执行日志的单步记录。"""
    action_index: int             # 在计划序列中的索引
    action_id: Any                # 实际执行的 action_id
    action_desc: str
    reasoning: str
    env_info: Dict[str, Any]
    invalid: bool


@dataclass
class EpisodeSummary:
    """整条 episode 的汇总结果。"""
    task_id: str
    instruction: str
    eval_set: str
    success: bool
    final_task_success: float
    final_task_progress: float
    num_steps: int
    num_invalid_actions: int
    num_invalid_ratio: float
    elapsed_seconds: float
    action_plan: ActionPlan
    execution_trace: List[ExecutionStepTrace]


# ===================== Problem-Solving 层的数据结构 =====================

@dataclass
class TaskModel:
    """
    问题求解视角下的任务模型：
    - 针对家庭任务：目标状态 / 关键物体 / 约束 / 难点 / 未决问题。
    """
    goal_description: str
    required_end_states: List[str]
    key_objects: List[str]
    constraints: List[str]
    assumptions: List[str]
    open_questions: List[str]
    difficulty_factors: List[str]


@dataclass
class MetaCognitiveState:
    """元认知状态：当前对任务理解 & 策略的自我评估。"""
    confidence_goal_understanding: float
    confidence_plan_feasibility: float
    perceived_difficulty: float
    recent_invalid_action_ratio: float
    progress_velocity: float
    last_decision: str = ""


@dataclass
class MetaDecision:
    """元认知 Controller 对下一步的控制决策。"""
    action: str                  # "continue" / "reperceive" / "replan" / "reperceive_and_remodel" / "abort"
    reason: str
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemory:
    """
    工作记忆：当前 episode 的“思维草稿纸”。

    贯穿：
    - 感知输出
    - Fusion 上下文
    - TaskModel
    - 扁平 ActionPlan
    - 执行轨迹
    - 元认知状态
    """
    instruction: str
    perception_history: List[PerceptionOutput] = field(default_factory=list)
    fusion_history: List[FusionContext] = field(default_factory=list)
    plans: List[ActionPlan] = field(default_factory=list)
    execution_trace: List[ExecutionStepTrace] = field(default_factory=list)

    task_model: Optional[TaskModel] = None
    meta_state: Optional[MetaCognitiveState] = None

    def latest_perception(self) -> Optional[PerceptionOutput]:
        return self.perception_history[-1] if self.perception_history else None

    def latest_fusion(self) -> Optional[FusionContext]:
        return self.fusion_history[-1] if self.fusion_history else None

    def latest_plan(self) -> Optional[ActionPlan]:
        return self.plans[-1] if self.plans else None


# ===================== 1. 感知模块 =====================

class PerceptionModule:
    """
    多模态感知与情境抽象模块：
    - Perception Abstraction
    - State Changes
    - Summary & Subgoals
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def perceive(
        self,
        image_path: str,
        instruction: str,
        working_memory: WorkingMemory,
    ) -> PerceptionOutput:
        """
        输入：当前图像路径 + 指令 + 简要历史（可选）
        输出：语义场景摘要 + 关键物体 + 状态变化 + 子目标建议 + 状态 snapshot。
        """
        prev_summary = working_memory.latest_perception().scene_summary \
            if working_memory.latest_perception() else ""

        prompt = [
            "你是家庭服务机器人的感知抽象模块。",
            "根据用户指令和当前环境图像（系统已完成基础检测），",
            "请输出一个 JSON：",
            "{",
            '  "scene_summary": "...",',
            '  "detected_objects": [',
            '      {"name": "...", "state": "...", "position": "..."}',
            "  ],",
            '  "state_changes": ["..."],',
            '  "subgoals_proposal": ["..."],',
            '  "environment_snapshot": {',
            '      "objects": [...],',
            '      "relations": [...],',
            '      "important_flags": ["..."]',
            "  }",
            "}",
            "=== 用户指令 ===",
            instruction,
            "=== 过去感知摘要（如有） ===",
            prev_summary,
            "=== 当前图像信息（文字版占位，可在真实系统中用 vision encoder 替代） ===",
            f"[Image at path: {image_path}]",
        ]

        raw = self.llm.request("\n".join(prompt))
        data = _safe_first_json(raw)

        return PerceptionOutput(
            scene_summary=data.get("scene_summary", ""),
            detected_objects=data.get("detected_objects", []),
            state_changes=data.get("state_changes", []),
            subgoals_proposal=data.get("subgoals_proposal", []),
            environment_snapshot=data.get("environment_snapshot", {}),
        )


# ===================== 2. Memory-Web Fusion 模块 =====================

class MemoryWebFusionEngine:
    """
    Memory-Web Fusion Engine：
    - 从 MemoryStore 检索相关 episode / 偏好 / 空间结构
    - 从 WebRetriever 检索通用常识
    - 和感知结果融合成结构化 FusionContext
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        web_retriever: Optional[WebRetriever] = None,
    ):
        self.memory_store = memory_store
        self.web_retriever = web_retriever

    def fuse(self, perception: PerceptionOutput, instruction: str) -> FusionContext:
        episodic_hints: List[str] = []
        semantic_knowledge: List[str] = []
        household_preferences: List[str] = []
        risk_constraints: List[str] = []

        # ---- Memory 检索 ----
        if self.memory_store is not None:
            key_obj_names = [o.get("name", "") for o in perception.detected_objects]
            q = instruction + " | " + " ".join(key_obj_names)
            results = self.memory_store.search(q, top_k=5)
            for r in results:
                if "hint" in r:
                    episodic_hints.append(r["hint"])
                if "household_preference" in r:
                    household_preferences.append(r["household_preference"])
                if "risk" in r:
                    risk_constraints.append(r["risk"])

        # ---- Web 检索 ----
        if self.web_retriever is not None:
            query = f"如何在家庭中完成如下任务：{instruction}（涉及物体：{[o.get('name') for o in perception.detected_objects]}）"
            results = self.web_retriever.search(query, top_k=3)
            for r in results:
                if "snippet" in r:
                    semantic_knowledge.append(r["snippet"])

        fused_summary = json.dumps(
            {
                "instruction": instruction,
                "scene_summary": perception.scene_summary,
                "state_changes": perception.state_changes,
                "episodic_hints": episodic_hints,
                "semantic_knowledge": semantic_knowledge,
                "household_preferences": household_preferences,
                "risk_constraints": risk_constraints,
            },
            ensure_ascii=False,
        )

        return FusionContext(
            perception=perception,
            episodic_hints=episodic_hints,
            semantic_knowledge=semantic_knowledge,
            household_preferences=household_preferences,
            risk_constraints=risk_constraints,
            fused_summary=fused_summary,
        )


# ===================== 3. ActionSequence Planner（无层级） =====================

class ActionSequencePlanner:
    """
    Planner：不做层级，只做扁平动作序列的规划。
    直接输出一串 action_id（+ 每步的自然语言说明和局部理由）。
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan_actions(
        self,
        fusion_ctx: FusionContext,
        instruction: str,
        available_actions: List[str],
        max_actions: int = 40,
    ) -> ActionPlan:
        """
        输出扁平 action 序列（长度 <= max_actions）。
        available_actions: 来自 env.language_skill_set
        """
        actions_prompt_str = ""
        for i, a in enumerate(available_actions):
            actions_prompt_str += f"\n- action id {i}: {a}"

        prompt = [
            "你是家庭服务机器人系统中的扁平动作规划器。",
            "工作模式：直接输出一串可执行的 action_id 序列，不做层级结构。",
            "动作空间如下：",
            actions_prompt_str,
            "",
            "要求：",
            f"- 生成一个长度不超过 {max_actions} 的动作序列。",
            "- 每个 action_id 必须是上面动作空间中的合法整数索引。",
            "- 尽量短，但要保证能够完成任务。",
            "",
            "输出 JSON：",
            "{",
            '  "global_rationale": "...",',
            '  "action_sequence": [',
            '    {',
            '      "index": 0,',
            '      "action_id": 3,',
            '      "description": "自然语言描述该步要做什么",',
            '      "rationale": "简单说明该步的理由"',
            "    }",
            "  ]",
            "}",
            "",
            "=== 综合上下文（来自感知 + 记忆 + Web 常识） ===",
            fusion_ctx.fused_summary,
        ]

        raw = self.llm.request("\n".join(prompt))
        data = _safe_first_json(raw)

        plan_actions: List[PlannedAction] = []
        for i, a in enumerate(data.get("action_sequence", [])):
            idx = int(a.get("index", i))
            aid = int(a.get("action_id", 0))
            if aid < 0 or aid >= len(available_actions):
                # 丢弃非法动作
                logger.warning(f"[ActionSequencePlanner] Invalid action_id {aid}, skip.")
                continue
            plan_actions.append(
                PlannedAction(
                    index=idx,
                    action_id=aid,
                    description=a.get("description", available_actions[aid]),
                    rationale=a.get("rationale", ""),
                )
            )

        # Fallback：如果模型没按格式输出，至少给一个 no-op / 停止动作（需要你根据实际动作空间替换）
        if not plan_actions:
            logger.warning("[ActionSequencePlanner] Empty plan_actions, fallback to single no-op if exists.")
            fallback_id = 0
            if len(available_actions) > 0:
                plan_actions.append(
                    PlannedAction(
                        index=0,
                        action_id=fallback_id,
                        description=available_actions[fallback_id],
                        rationale="fallback_single_action",
                    )
                )

        return ActionPlan(
            actions=plan_actions,
            global_rationale=data.get("global_rationale", "Default flat action rationale."),
        )


# ===================== 4. Executor 模块（直接执行 action_id 序列） =====================

class ExecutorModule:
    """
    Executor：直接用 env.step(action_id, reasoning=...) 执行计划好的动作序列。
    """

    def __init__(self, env: EBHabEnv):
        self.env = env

    def execute_action(
        self,
        action_id: int,
        rationale: str,
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行单个 action_id。
        """
        obs, reward, done, info = self.env.step(action_id, reasoning=rationale)
        return obs, reward, done, info


# ===================== 5. 元认知 Controller =====================

class MetaCognitiveController:
    """
    元认知 Problem-Solving 控制器：
    - TaskModel 构建
    - MetaState 初始化
    - 执行过程中的 Monitoring & Control（是否重感知 / 重规划）
    - Episode 结束时的 Meta-level 反思
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # --- 建模 ---

    def build_task_model(
        self,
        instruction: str,
        perception_summary: str,
        fusion_context_summary: str,
    ) -> TaskModel:
        prompt = [
            "你是家庭服务机器人中的任务建模模块。",
            "请根据信息，抽象出本次任务的 TaskModel，输出 JSON：",
            "{",
            '  "goal_description": "...",',
            '  "required_end_states": ["..."],',
            '  "key_objects": ["..."],',
            '  "constraints": ["..."],',
            '  "assumptions": ["..."],',
            '  "open_questions": ["..."],',
            '  "difficulty_factors": ["..."]',
            "}",
            "=== 用户指令 ===",
            instruction,
            "=== 当前感知摘要 ===",
            perception_summary,
            "=== 记忆与常识摘要 ===",
            fusion_context_summary,
        ]
        raw = self.llm.request("\n".join(prompt))
        data = _safe_first_json(raw)

        return TaskModel(
            goal_description=data.get("goal_description", instruction),
            required_end_states=data.get("required_end_states", []),
            key_objects=data.get("key_objects", []),
            constraints=data.get("constraints", []),
            assumptions=data.get("assumptions", []),
            open_questions=data.get("open_questions", []),
            difficulty_factors=data.get("difficulty_factors", []),
        )

    # --- 初始化元认知状态 ---

    def init_meta_state(self, task_model: TaskModel) -> MetaCognitiveState:
        n_open = len(task_model.open_questions)
        confidence_goal = max(0.2, 1.0 - 0.1 * n_open)
        perceived_difficulty = min(1.0, 0.3 + 0.1 * len(task_model.difficulty_factors))
        return MetaCognitiveState(
            confidence_goal_understanding=confidence_goal,
            confidence_plan_feasibility=0.7,
            perceived_difficulty=perceived_difficulty,
            recent_invalid_action_ratio=0.0,
            progress_velocity=0.0,
            last_decision="init_flat",
        )

    # --- 执行过程中的监控与决策 ---

    def assess_step(
        self,
        meta_state: MetaCognitiveState,
        task_model: TaskModel,
        recent_info: Dict[str, Any],
    ) -> MetaDecision:
        invalid_ratio = float(recent_info.get("invalid_action_ratio", 0.0))
        progress_delta = float(recent_info.get("task_progress_delta", 0.0))
        unexpected_changes = recent_info.get("unexpected_state_changes", [])

        meta_state.recent_invalid_action_ratio = invalid_ratio
        meta_state.progress_velocity = progress_delta

        # 简单规则（之后可替换为 LLM 元策略）
        if invalid_ratio > 0.6 and progress_delta <= 0.0:
            return MetaDecision(
                action="replan",
                reason="High invalid action ratio and no progress (flat plan).",
            )

        if len(unexpected_changes) > 0:
            return MetaDecision(
                action="reperceive_and_remodel",
                reason=f"Unexpected state changes: {unexpected_changes}",
            )

        if progress_delta < 0.01 and invalid_ratio > 0.3:
            return MetaDecision(
                action="reperceive",
                reason="Slow progress with non-trivial invalid actions (flat plan).",
            )

        return MetaDecision(
            action="continue",
            reason="Progress acceptable and no major anomaly.",
        )

    # --- Episode 结束时的元反思 ---

    def summarize_episode(
        self,
        task_model: TaskModel,
        episode_summary: EpisodeSummary,
    ) -> Dict[str, Any]:
        reflection = {
            "task_id": episode_summary.task_id,
            "goal_description": task_model.goal_description,
            "required_end_states": task_model.required_end_states,
            "final_success": episode_summary.final_task_success,
            "final_progress": episode_summary.final_task_progress,
            "num_steps": episode_summary.num_steps,
            "num_invalid_ratio": episode_summary.num_invalid_ratio,
            "what_worked": [],
            "what_failed": [],
            "suggestions_for_next_time": [],
        }
        # 这里可以用 LLM 再做一次“经验总结”，目前先返回结构壳
        return reflection


# ===================== 6. 顶层 Agent：扁平计划版 =====================

class EmbodiedProblemSolvingAgentFlat:
    """
    扁平计划版 Agent：
        感知 → Memory-Web Fusion → TaskModel + MetaState → ActionSequencePlanner → Executor → Memory

    特点：
    - 不用 HierarchicalPlan 和 HighLevelStep
    - 大模型一次性给出 action_id 序列（ActionPlan）
    - 执行过程中用元认知触发「重感知 / 重规划」
    """

    def __init__(
        self,
        env: EBHabEnv,
        llm: LLMClient,
        memory_store: Optional[MemoryStore] = None,
        web_retriever: Optional[WebRetriever] = None,
    ):
        self.env = env

        self.perception = PerceptionModule(llm)
        self.fusion_engine = MemoryWebFusionEngine(memory_store, web_retriever)
        self.action_planner = ActionSequencePlanner(llm)
        self.executor = ExecutorModule(env)
        self.meta = MetaCognitiveController(llm)
        self.memory_store = memory_store

    # ---- 一个 episode 的完整 Problem-Solving 流程 ----

    def run_episode(self) -> EpisodeSummary:
        # 环境 reset
        obs = self.env.reset()
        img_path = self.env.save_image(obs)
        instruction = self.env.episode_language_instruction

        wm = WorkingMemory(instruction=instruction)

        # 1. 感知
        p_out = self.perception.perceive(
            image_path=img_path,
            instruction=instruction,
            working_memory=wm,
        )
        wm.perception_history.append(p_out)

        # 2. Memory-Web Fusion
        fusion_ctx = self.fusion_engine.fuse(p_out, instruction)
        wm.fusion_history.append(fusion_ctx)

        # 3. Problem-Solving：TaskModel + MetaState
        task_model = self.meta.build_task_model(
            instruction=instruction,
            perception_summary=p_out.scene_summary,
            fusion_context_summary=fusion_ctx.fused_summary,
        )
        wm.task_model = task_model
        wm.meta_state = self.meta.init_meta_state(task_model)

        # 4. Planner：一次性生成扁平动作序列
        available_actions = list(self.env.language_skill_set)
        plan = self.action_planner.plan_actions(
            fusion_ctx=fusion_ctx,
            instruction=instruction,
            available_actions=available_actions,
        )
        wm.plans.append(plan)

        # 5. 执行 + 元认知监控
        summary = self._execute_flat_plan_with_meta(
            plan=plan,
            instruction=instruction,
            wm=wm,
            first_obs=obs,
            first_img_path=img_path,
        )

        # 6. Memory：存储 episode & 元反思
        if self.memory_store is not None and wm.task_model is not None:
            self.memory_store.store_episode(summary)
            reflection = self.meta.summarize_episode(wm.task_model, summary)
            self.memory_store.store_meta_reflection(reflection)

        return summary

    # ---- 执行扁平动作序列（包含元认知闭环） ----

    def _execute_flat_plan_with_meta(
        self,
        plan: ActionPlan,
        instruction: str,
        wm: WorkingMemory,
        first_obs: Any,
        first_img_path: str,
    ) -> EpisodeSummary:
        obs = first_obs
        img_path = first_img_path

        execution_trace: List[ExecutionStepTrace] = []
        start_time = time.time()

        info: Dict[str, Any] = {
            "task_success": 0.0,
            "task_progress": 0.0,
            "env_step": 0,
            "last_action_success": 1.0,
        }

        recent_invalid_flags: List[bool] = []
        last_task_progress = 0.0

        current_plan = plan
        pointer = 0

        while pointer < len(current_plan.actions):
            pa = current_plan.actions[pointer]
            logger.info(f"[MetaAgentFlat] Execute action {pa.index} (plan idx {pointer}): {pa.description}")

            # --------- 元认知监控 & 决策 ----------
            if pointer > 0 and wm.meta_state is not None and wm.task_model is not None:
                window = min(10, len(recent_invalid_flags))
                invalid_ratio = (
                    sum(recent_invalid_flags[-window:]) / window
                    if window > 0 else 0.0
                )
                cur_progress = info.get("task_progress", 0.0)
                progress_delta = cur_progress - last_task_progress
                last_task_progress = cur_progress

                recent_info = {
                    "invalid_action_ratio": invalid_ratio,
                    "task_progress_delta": progress_delta,
                    "unexpected_state_changes": [],  # 可以用 environment_snapshot 做对比填这里
                }
                meta_decision = self.meta.assess_step(
                    wm.meta_state,
                    wm.task_model,
                    recent_info,
                )
                wm.meta_state.last_decision = meta_decision.action

                if meta_decision.action in ("reperceive", "reperceive_and_remodel"):
                    p_out = self.perception.perceive(
                        image_path=img_path,
                        instruction=instruction,
                        working_memory=wm,
                    )
                    wm.perception_history.append(p_out)
                    if meta_decision.action == "reperceive_and_remodel":
                        fusion_ctx = self.fusion_engine.fuse(p_out, instruction)
                        wm.fusion_history.append(fusion_ctx)
                        wm.task_model = self.meta.build_task_model(
                            instruction,
                            p_out.scene_summary,
                            fusion_ctx.fused_summary,
                        )
                        # 重规划新的扁平动作序列（从当前状态开始）
                        available_actions = list(self.env.language_skill_set)
                        new_plan = self.action_planner.plan_actions(
                            fusion_ctx=fusion_ctx,
                            instruction=instruction,
                            available_actions=available_actions,
                        )
                        wm.plans.append(new_plan)
                        current_plan = new_plan
                        pointer = 0
                        logger.info("[MetaAgentFlat] Replanned (flat) after reperceive+remodel.")
                        continue
                    else:
                        logger.info("[MetaAgentFlat] Reperceived environment; keep current flat plan.")

                elif meta_decision.action == "replan":
                    fusion_ctx = wm.latest_fusion()
                    if fusion_ctx is None:
                        # 如果没有历史，就用最新感知重新 fuse 一次
                        p_latest = wm.latest_perception()
                        if p_latest is None:
                            p_latest = p_out  # fallback
                        fusion_ctx = self.fusion_engine.fuse(p_latest, instruction)
                        wm.fusion_history.append(fusion_ctx)
                    available_actions = list(self.env.language_skill_set)
                    new_plan = self.action_planner.plan_actions(
                        fusion_ctx=fusion_ctx,
                        instruction=instruction,
                        available_actions=available_actions,
                    )
                    wm.plans.append(new_plan)
                    current_plan = new_plan
                    pointer = 0
                    logger.info("[MetaAgentFlat] Replanned (flat) due to high invalid ratio.")
                    continue

            # --------- 执行动作 ----------
            try:
                obs2, reward, done, info2 = self.executor.execute_action(
                    action_id=pa.action_id,
                    rationale=pa.rationale,
                )
            except Exception as e:
                logger.error(f"[MetaAgentFlat] execute_action error: {e}")
                time.sleep(1.0)
                # 记录为 invalid，然后尝试下一步
                invalid_flag = True
                recent_invalid_flags.append(invalid_flag)
                trace = ExecutionStepTrace(
                    action_index=pa.index,
                    action_id=pa.action_id,
                    action_desc=pa.description,
                    reasoning=f"execution_error: {e}",
                    env_info={"error": str(e)},
                    invalid=invalid_flag,
                )
                execution_trace.append(trace)
                pointer += 1
                continue

            obs = obs2
            info = info2
            img_path = self.env.save_image(obs)

            invalid_flag = not info.get("last_action_success", 1.0)
            recent_invalid_flags.append(invalid_flag)

            action_desc = pa.description or self.env.language_skill_set[pa.action_id]
            trace = ExecutionStepTrace(
                action_index=pa.index,
                action_id=pa.action_id,
                action_desc=action_desc,
                reasoning=pa.rationale,
                env_info=info,
                invalid=invalid_flag,
            )
            execution_trace.append(trace)

            # 终止条件
            if info.get("task_success", 0.0) > 0.5:
                logger.info("[MetaAgentFlat] global task success reached.")
                break
            if done:
                logger.info("[MetaAgentFlat] env terminated episode.")
                break
            if info.get("env_step", 0) >= getattr(self.env, "_max_episode_steps", 200):
                logger.warning("[MetaAgentFlat] max env steps reached.")
                break

            pointer += 1

        elapsed = time.time() - start_time
        total_steps = info.get("env_step", len(execution_trace))
        invalid_total = len([t for t in execution_trace if t.invalid])
        invalid_ratio = invalid_total / max(1, total_steps)

        summary = EpisodeSummary(
            task_id=str(uuid.uuid4()),
            instruction=wm.instruction,
            eval_set=self.env.eval_set,
            success=bool(info.get("task_success", 0.0) > 0.5),
            final_task_success=info.get("task_success", 0.0),
            final_task_progress=info.get("task_progress", 0.0),
            num_steps=total_steps,
            num_invalid_actions=invalid_total,
            num_invalid_ratio=invalid_ratio,
            elapsed_seconds=elapsed,
            action_plan=current_plan,
            execution_trace=execution_trace,
        )
        return summary


# ===================== 7. 和 EB_HabitatEvaluator 对接的示例 =====================

"""
在 eb_habitat_evaluator.py 里，你可以增加一个「元认知扁平 Agent」评估模式，大致这样：

from embodiedbench.evaluator.embodied_meta_agent_flat import EmbodiedProblemSolvingAgentFlat
from your_llm_client_impl import YourLLMClient
from your_memory_impl import YourMemoryStore
from your_web_retriever_impl import YourWebRetriever

...

class EB_HabitatEvaluator():
    def evaluate_with_meta_flat(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes(flat-meta)")
        llm_client = YourLLMClient(...)
        memory_store = YourMemoryStore(...)
        web_retriever = YourWebRetriever(...)

        meta_agent = EmbodiedProblemSolvingAgentFlat(
            env=self.env,
            llm=llm_client,
            memory_store=memory_store,
            web_retriever=web_retriever,
        )

        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"[MetaFlat] Evaluating episode {self.env._current_episode_num} ...")
            episode_summary = meta_agent.run_episode()

            # 你可以把 episode_summary 转成原来的 episode_info 结构存起来，或者直接 dump 成 JSON。
            # 比如：
            #   summary_dict = dataclasses.asdict(episode_summary)
            #   json.dump(summary_dict, open(..., "w"), ensure_ascii=False, indent=2)

            progress_bar.update()

"""


