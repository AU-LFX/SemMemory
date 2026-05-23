# meta_habitat_planner.py
import json
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Callable
from functools import wraps

from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.main import logger

# ============= 基本数据结构 =============

@dataclass
class HomeTaskBrief:
    """家庭任务建模结果：从自然语言 instruction 抽象出来的关键信息。"""
    raw_instruction: str
    # scene_id: str = ""
    room_type: str = ""
    objects_of_interest: List[str] = field(default_factory=list)
    high_level_goal: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)  # 如 max_steps, max_invalid_ratio 等


@dataclass
class HomeTaskSignature:
    """
    家庭任务表示：在 Home 场景中替代原来的 problem_signature，
    更偏环境各种信息 + 子目标规划。
    """
    task_id: str
    brief: HomeTaskBrief

    initial_view_summary: str = ""      # 第一帧图像 + 指令的整体概括
    symbolic_state: Dict[str, Any] = field(default_factory=dict)  # 如 room layout / 对象-关系
    subgoals: List[str] = field(default_factory=list)   # 高层子目标序列
    action_primitives: List[int] = field(default_factory=list)  # 建议使用的 action id 列表
    success_criteria: str = ""          # 如何判断完成（人类可读描述）


@dataclass
class EpisodeSummary:
    """单个 episode 执行后的总结，作为 Memory 写入 / 日志使用。"""
    task_id: str
    instruction: str
    eval_set: str
    success: bool
    final_task_success: float
    final_task_progress: float
    reward_mean: float
    num_steps: int
    num_invalid_actions: int
    num_invalid_ratio: float
    planner_steps: int
    planner_output_error: int
    elapsed_seconds: float
    strategy: str
    subgoals: List[str]
    action_trace: List[Dict[str, Any]]  # 每步记录 action_id / desc / info / reasoning 等


# ============= Memory 接口（你之后可以接入自己的 Memory 系统） =============

class HomeTaskMemory(Protocol):
    """
    面向 Habitat 家庭任务规划的 Memory 接口。
    你可以用自己的 Memory-Web Fusion Engine 来实现这些方法。
    """

    def retrieve_similar_episodes(
        self,
        brief: HomeTaskBrief,
        top_k: int = 5
    ) -> List[EpisodeSummary]:
        """根据任务摘要检索历史 episode（用于 case-based / 统计策略表现）。"""
        ...

    def save_episode(
        self,
        signature: HomeTaskSignature,
        summary: EpisodeSummary
    ) -> None:
        """把本次 episode 的结果写入长期记忆（episodic + semantic）。"""
        ...

    def save_failure_pattern(
        self,
        signature: HomeTaskSignature,
        summary: EpisodeSummary
    ) -> None:
        """可选：记录失败模式 / 常见错误，用于之后的元认知分析。"""
        ...


# ============= 轻量 LLM 接口 =============

class LLMLike(Protocol):
    """
    只要实现一个最小接口：request(messages: List[str]) -> str
    就可以作为元认知 / 高层规划用的 LLM。
    你可以用 RemoteModel 包一层，也可以用你 brainary 里的 LLM 直接丢进来。
    """
    def request(self, messages: List[str]) -> str:
        ...


# ============= 元认知装饰器（简化版） =============

def with_metacognition(component_name: str):
    """Decorator for metacognition monitoring on component execution.

    This decorator wraps component methods to automatically:
    1. Execute the component logic and capture its output
    2. Run ``self._metacognition_check`` to compare output with the planned expectation
    3. Apply adjustments via ``self._apply_metacognition_adjustment`` if severe deviations are detected
    4. Optionally re-run downstream components with the updated hints (handled by the caller)

    Args:
        component_name (str): Name of the component being monitored. Valid values are
            ``{"task_modeling", "task_representation", "strategy_selection"}`` for this planner.

    Usage:
        >>> @with_metacognition("task_modeling")
        ... def _task_modeling(self, instruction: str) -> HomeTaskBrief:
        ...     ...

    Notes:
        - The decorated method must belong to :class:`HomeMetaPlanner`
        - ``self.execution_plan`` must be initialized before invocation
        - Metacognition results are appended to ``self.metacognition_adjustments`` for later prompts
        - 为关键函数提供元认知监控，确保与计划期望保持一致
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            if not hasattr(self, "execution_plan") or self.execution_plan is None:
                # 还没做 planning，就不做元认知
                return result

            # 封装 actual_output（简单版）
            if isinstance(result, dict):
                actual_output = result
            elif isinstance(result, (list, tuple)):
                actual_output = {"result": str(result), "output_type": "sequence"}
            else:
                actual_output = {"result": str(result), "output_type": type(result).__name__}

            execution_context = {
                "component": component_name,
                "function_name": func.__name__,
            }

            assessment = self._metacognition_check(
                component_name=component_name,
                actual_output=actual_output,
                execution_context=execution_context,
            )

            if assessment.get("adjustment_needed"):
                suggestions = assessment.get("adjustment_suggestions", [])
                self._apply_metacognition_adjustment(component_name, suggestions)

            return result
        return wrapper
    return decorator


# ============= HomeMetaPlanner 主体 =============

class HomeMetaPlanner:
    """
    把「四阶段 + 元认知 + 记忆」适配到 EBHabEnv / VLMPlanner 的总控模块。

    使用方式（伪代码）：
        env = EBHabEnv(...)
        planner = VLMPlanner(...)
        meta_planner = HomeMetaPlanner(llm, env, planner, memory=your_memory_impl)

        summary = meta_planner.run_episode()
    """

    def __init__(
        self,
        llm: LLMLike,
        env: EBHabEnv,
        vlm_planner: VLMPlanner,
        memory: Optional[HomeTaskMemory] = None,
        require_signature_confirmation: bool = False,
    ):
        self.llm = llm
        self.env = env
        self.vlm_planner = vlm_planner
        self.memory = memory
        self.require_signature_confirmation = require_signature_confirmation

        # 元认知相关状态
        self.execution_plan: Optional[Dict[str, Any]] = None
        self.metacognition_adjustments: Dict[str, List[Dict[str, Any]]] = {}
        self.feedback: Dict[str, Any] = {}

    # ------------ 对外主入口 ------------

    def run_episode(self) -> EpisodeSummary:
        """
        完整跑一整个 episode：
        1) 读取 env 的 instruction
        2) planning（生成 execution_plan）
        3) task modeling & representation
        4) 策略选择
        5) 执行 & 反馈
        6) 写入 Memory，返回 EpisodeSummary
        """
        obs = self.env.reset()
        img_path = self.env.save_image(obs)
        instruction = self.env.episode_language_instruction
        # scene_id = self.env.current_episode()["scene_id"]

        logger.info(f"[Meta] New episode: {instruction}")

        # 生成任务 id，用于跟 memory 对齐
        self.task_id = str(uuid.uuid4())

        # 1) Pre-planning：对当前任务粗略做一个执行期望
        self.execution_plan = self._planning(instruction, img_path)

        # 2) Task modeling & representation
        self.task_brief: HomeTaskBrief = self._task_modeling(instruction)
        self.task_signature: HomeTaskSignature = self._task_representation(
            self.task_brief, first_image_path=img_path
        )

        # 3) Strategy selection
        strategy = self._strategy_selection(self.task_signature)
        self.feedback["strategy"] = strategy

        # 4) 执行回路（包装现有 evaluate() 的逻辑）
        episode_summary = self._execute_with_feedback(
            instruction, initial_obs=obs, first_img_path=img_path, strategy=strategy
        )

        # 5) 写入 Memory
        if self.memory is not None:
            self.memory.save_episode(self.task_signature, episode_summary)
            if not episode_summary.success:
                self.memory.save_failure_pattern(self.task_signature, episode_summary)

        return episode_summary

    # ------------ 阶段 1：Task Modeling ------------

    @with_metacognition("task_modeling")
    def _task_modeling(self, instruction: str) -> HomeTaskBrief:
        """Structure the raw instruction into a :class:`HomeTaskBrief`.

        Pipeline:
            1. Prompt the LLM to extract room type, key objects, goals, and constraints
            2. Parse the JSON response (fallback to defaults if parsing fails)
            3. Populate :class:`HomeTaskBrief` so downstream components share a normalized view

        Args:
            instruction (str): Natural-language task description from the environment.

        Returns:
            HomeTaskBrief: Canonical summary of the task with inferred context fields.

        Notes:
            - Output feeds both task representation and strategy selection
            - 元认知装饰器会监控该阶段是否成功解析出关键信息
        """
        messages = [
            "你是一个家庭任务建模助手。",
            "用户给出一条指令，请你抽取：房间类型、关键物体、任务目标和约束。",
            "只返回一个 JSON，对象形式：",
            "{",
            '  "room_type": "kitchen / bedroom / living room ...",',
            '  "objects_of_interest": ["...", "..."],',
            '  "high_level_goal": "一句话概括要达成什么状态",',
            '  "constraints": { "max_steps_hint": 50, "notes": "不要打翻液体" }',
            "}",
            "不要额外的文字说明。",
            "=== 指令 ===",
            instruction,
        ]
        raw = self.llm.request(messages)
        try:
            parsed = json.loads(_first_json_str(raw))
        except Exception:
            parsed = {}

        brief = HomeTaskBrief(
            raw_instruction=instruction,
            room_type=parsed.get("room_type", ""),
            objects_of_interest=parsed.get("objects_of_interest", []) or [],
            high_level_goal=parsed.get("high_level_goal", instruction),
            constraints=parsed.get("constraints", {}) or {},
        )
        return brief

    # ------------ 阶段 2：Task Representation ------------

    @with_metacognition("task_representation")
    def _task_representation(
        self,
        brief: HomeTaskBrief,
        first_image_path: str,
    ) -> HomeTaskSignature:
        """Convert the task brief + first-view context into a :class:`HomeTaskSignature`.

        Steps:
            1. Ask the LLM to reason about the initial observation and produce symbolic state fields
            2. Capture subgoals, action primitives, and success criteria for downstream planning
            3. Persist the signature with the current ``task_id`` for memory indexing

        Args:
            brief (HomeTaskBrief): Output from :meth:`_task_modeling`.
            first_image_path (str): Path to the saved first-person RGB image for optional referencing.

        Returns:
            HomeTaskSignature: Structured representation containing subgoals and symbolic state.

        Notes:
            - Provides the “contract” for strategy selection and evaluation
            - 若图像不可用亦可依赖语言推断，但需在 JSON 中保持一致字段
        """
        task_id = self.task_id

        # 构造提示：让 LLM 看第一帧图像（可选）+ 指令，给出子目标和符号状态
        # 这里为了代码简单，用 text-only，实际你可以把图像丢给 VLM。
        messages = [
            "你是一个家庭环境任务规划器，需要根据任务简述和环境信息构建结构化表示。",
            "请输出一个 JSON：",
            "{",
            '  "initial_view_summary": "...",',
            '  "symbolic_state": { "room_layout": "...", "important_objects": ["..."] },',
            '  "subgoals": ["...", "..."],',
            '  "action_primitives": [0, 1, 3],',
            '  "success_criteria": "..."',
            "}",
            "=== 任务简述 ===",
            f"room_type: {brief.room_type}",
            f"objects_of_interest: {brief.objects_of_interest}",
            f"high_level_goal: {brief.high_level_goal}",
            f"constraints: {brief.constraints}",
            "（注意：如果不了解动作空间，也可以仅根据语言做一个合理假设。）",
        ]
        raw = self.llm.request(messages)
        try:
            parsed = json.loads(_first_json_str(raw))
        except Exception:
            parsed = {}

        sig = HomeTaskSignature(
            task_id=task_id,
            brief=brief,
            initial_view_summary=parsed.get("initial_view_summary", ""),
            symbolic_state=parsed.get("symbolic_state", {}) or {},
            subgoals=parsed.get("subgoals", []) or [],
            action_primitives=parsed.get("action_primitives", []) or [],
            success_criteria=parsed.get("success_criteria", ""),
        )
        return sig

    # ------------ 阶段 3：策略选择 ------------

    @with_metacognition("strategy_selection")
    def _strategy_selection(self, sig: HomeTaskSignature) -> str:
        """Choose an execution strategy based on the current task signature.

        Evaluation logic:
            - Compare candidate strategies ``reactive``, ``plan_and_execute``, ``case_based``
            - If memory is available, prefer the strategy with the highest historical success rate
            - Fall back to heuristics (e.g., many subgoals → ``plan_and_execute``)

        Args:
            sig (HomeTaskSignature): Structured representation produced by :meth:`_task_representation`.

        Returns:
            str: Selected strategy label recorded in ``self.feedback['strategy']``.

        Notes:
            - Strategy choice steers feedback loops and determines how strictly plans are followed
            - 该函数受元认知监控，若选择理由与预期不符会触发提示修正
        """
        candidates = ["reactive", "plan_and_execute", "case_based"]
        chosen = "reactive"

        # 如果有记忆，看看是否有成功率更高的策略
        if self.memory is not None:
            sims = self.memory.retrieve_similar_episodes(sig.brief, top_k=5)
            if sims:
                # 简单按 success 比例选一个策略
                stats = {"reactive": [], "plan_and_execute": [], "case_based": []}
                for ep in sims:
                    strategy = ep.strategy if hasattr(ep, "strategy") else "reactive"
                    if strategy in stats:
                        stats[strategy].append(ep.final_task_success)

                avg_success = {
                    k: (sum(v) / len(v) if v else 0.0) for k, v in stats.items()
                }
                chosen = max(avg_success.items(), key=lambda x: x[1])[0]

        # 如果子目标很多，可以倾向 plan_and_execute
        if len(sig.subgoals) >= 4 and chosen == "reactive":
            chosen = "plan_and_execute"

        logger.info(f"[Meta] strategy selected: {chosen}")
        return chosen

    # ------------ 阶段 4：执行 + 反馈（包一层 evaluate） ------------

    def _execute_with_feedback(
        self,
        instruction: str,
        initial_obs: Any,
        first_img_path: str,
        strategy: str,
    ) -> EpisodeSummary:
        """Run the low-level control loop with logging, feedback, and summary generation.

        Responsibilities:
            1. Reset the VLM planner and iterate env steps until termination
            2. Record planner outputs, rewards, invalid counts, and env feedback
            3. Surface metrics for metacognition monitoring (progress vs. invalid ratio)
            4. Package the execution trace inside :class:`EpisodeSummary`

        Args:
            instruction (str): Current episode instruction passed to the planner.
            initial_obs (Any): Observation returned by ``env.reset``.
            first_img_path (str): Path of the first saved RGB frame.
            strategy (str): Strategy label chosen via :meth:`_strategy_selection`.

        Returns:
            EpisodeSummary: Aggregated statistics, trajectories, and metadata for the episode.

        Notes:
            - Mirrors ``EB_HabitatEvaluator.evaluate`` but adds metacognition hooks
            - 执行过程中的 action_trace 会写入 memory 以便后续 case-based 检索
        """
        self.vlm_planner.reset()
        done = False
        obs = initial_obs
        img_path = first_img_path

        episode_info = {
            "reward": [],
            "num_invalid_actions": 0,
            "action_trace": [],
        }

        start_time = time.time()

        while not done and self.env._current_episode_num < self.env.number_of_episodes:
            try:
                action, reasoning = self.vlm_planner.act(img_path, instruction)
                logger.info(f"[Meta] Planner Output Action: {action}")

                # 记录 planner 输出（无论是否有效）
                step_record = {
                    "proposed_action": action,
                    "reasoning": reasoning,
                }

                if action == -2:
                    # 空计划，提前结束
                    episode_info["empty_plan"] = 1
                    info = {
                        "task_success": episode_info.get("task_success", 0),
                        "task_progress": episode_info.get("task_progress", 0),
                        "subgoal_reward": episode_info.get("subgoal_reward", 0),
                        "env_step": self.env._current_step,
                    }
                    done = True
                elif action == -1:
                    # 无效动作
                    self.env._cur_invalid_actions += 1
                    episode_info["reward"].append(-1)
                    episode_info["num_invalid_actions"] += 1

                    info = {
                        "task_success": episode_info.get("task_success", 0),
                        "task_progress": episode_info.get("task_progress", 0),
                        "subgoal_reward": episode_info.get("subgoal_reward", 0),
                        "env_step": self.env._current_step,
                    }

                    step_record.update(
                        {
                            "executed": False,
                            "invalid": True,
                            "env_info": info,
                        }
                    )

                    # 元认知：连续无效动作太多就考虑 replanning / 调整 prompt
                    self._meta_monitor_progress(info, episode_info)

                    if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                        done = True
                else:
                    # 正常动作 / 多步动作
                    if isinstance(action, list):
                        for act_single in action[
                            : min(
                                self.env._max_episode_steps - self.env._current_step,
                                len(action),
                            )
                        ]:
                            obs, reward, done, info = self.env.step(
                                act_single, reasoning=reasoning
                            )
                            action_str = (
                                act_single
                                if isinstance(act_single, str)
                                else self.env.language_skill_set[act_single]
                            )
                            logger.debug(
                                f"[Meta] Executed action: {action_str}, Task success: {info['task_success']}"
                            )

                            self.vlm_planner.update_info(info)
                            img_path = self.env.save_image(obs)

                            episode_info["reward"].append(reward)
                            episode_info["num_invalid_actions"] += (
                                info["last_action_success"] == 0
                            )

                            trace_item = dict(step_record)
                            trace_item.update(
                                {
                                    "executed": True,
                                    "invalid": info["last_action_success"] == 0,
                                    "action_id": act_single,
                                    "action_desc": action_str,
                                    "env_info": info,
                                }
                            )
                            episode_info["action_trace"].append(trace_item)

                            # 元认知监控
                            self._meta_monitor_progress(info, episode_info)

                            if done or info["last_action_success"] == 0:
                                break
                    else:
                        obs, reward, done, info = self.env.step(
                            action, reasoning=reasoning
                        )
                        action_str = (
                            action
                            if isinstance(action, str)
                            else self.env.language_skill_set[action]
                        )
                        logger.debug(
                            f"[Meta] Executed action: {action_str}, Task success: {info['task_success']}"
                        )

                        self.vlm_planner.update_info(info)
                        img_path = self.env.save_image(obs)

                        episode_info["reward"].append(reward)
                        episode_info["num_invalid_actions"] += (
                            info["last_action_success"] == 0
                        )

                        trace_item = dict(step_record)
                        trace_item.update(
                            {
                                "executed": True,
                                "invalid": info["last_action_success"] == 0,
                                "action_id": action,
                                "action_desc": action_str,
                                "env_info": info,
                            }
                        )
                        episode_info["action_trace"].append(trace_item)

                        # 元认知监控
                        self._meta_monitor_progress(info, episode_info)

            except Exception as e:
                logger.error(f"[Meta] Exception during act/step: {e}")
                time.sleep(5)

        # episode 结束，组装 EpisodeSummary
        if "task_success" not in info:
            info["task_success"] = 0.0
            info["task_progress"] = 0.0
            info["subgoal_reward"] = 0.0
            info["env_step"] = self.env._current_step

        elapsed = info.get(
            "episode_elapsed_seconds", time.time() - self.env._episode_start_time
        )

        reward_mean = float(
            sum(episode_info["reward"]) / len(episode_info["reward"])
        ) if episode_info["reward"] else 0.0

        num_invalid = int(episode_info["num_invalid_actions"])
        num_steps = int(info["env_step"])
        num_invalid_ratio = float(num_invalid / num_steps) if num_steps > 0 else 0.0

        summary = EpisodeSummary(
            task_id=self.task_id,
            instruction=instruction,
            eval_set=self.env.eval_set,
            success=bool(info["task_success"] > 0.5),
            final_task_success=float(info["task_success"]),
            final_task_progress=float(info["task_progress"]),
            reward_mean=reward_mean,
            num_steps=num_steps,
            num_invalid_actions=num_invalid,
            num_invalid_ratio=num_invalid_ratio,
            planner_steps=self.vlm_planner.planner_steps,
            planner_output_error=self.vlm_planner.output_json_error,
            elapsed_seconds=elapsed,
            strategy=self.feedback.get("strategy", "unknown"),
            subgoals=self.task_signature.subgoals,
            action_trace=episode_info["action_trace"],
        )
        return summary

    # ------------ 元认知相关辅助：调整 prompt / 监控进度 ------------

    def _planning(self, instruction: str, first_img_path: str) -> Dict[str, Any]:
        """Draft the expectation blueprint used by metacognition checks.

        The plan captures what “good” output looks like for each component so that
        :meth:`_metacognition_check` can reason about deviations later.

        Args:
            instruction (str): Raw episode instruction (currently unused but kept for parity).
            first_img_path (str): Path to the first RGB frame (reserved for future use).

        Returns:
            Dict[str, Any]: Mapping from component name to expectation description.

        Notes:
            - Keep the structure lightweight; the monitor only needs qualitative targets
            - 期望可扩展，例如新增 strategy_case_based、low_level_control 等条目
        """
        return {
            "task_modeling": {
                "expected_output": "明确 room_type / objects_of_interest / high_level_goal / constraints",
            },
            "task_representation": {
                "expected_output": "包含初始视图总结、子目标序列和成功条件。",
            },
            "strategy_selection": {
                "expected_output": "在 reactive / plan_and_execute / case_based 中选一个理由合理的策略。",
            },
            "low_level_control": {
                "expected_output": "在 max_steps 内完成大部分子目标，无效动作比例 < 0.3。",
            },
        }

    def _metacognition_check(
        self,
        component_name: str,
        actual_output: Dict[str, Any],
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare the actual component output with expectations via LLM reasoning.

        Workflow:
            1. Pull the expected blueprint for ``component_name`` from ``self.execution_plan``
            2. Build a prompt containing expected vs. actual outputs plus execution context
            3. Ask the LLM to classify deviation severity and recommend adjustments

        Args:
            component_name (str): Identifier for the monitored stage.
            actual_output (Dict[str, Any]): Serialized output collected by the decorator.
            execution_context (Dict[str, Any]): Metadata (function name, etc.) for debugging.

        Returns:
            Dict[str, Any]: LLM judgement with ``deviation_detected``, ``adjustment_needed`` and suggestions.

        Notes:
            - If no expectation is defined, the function returns a no-op response
            - 该检查结果将驱动后续 prompt hint 的生成
        """
        expected = (self.execution_plan or {}).get(component_name, {})
        if not expected:
            return {
                "deviation_detected": False,
                "deviation_severity": "none",
                "adjustment_needed": False,
                "adjustment_suggestions": [],
            }

        messages = [
            f"你是元认知监控代理，现在检查组件 `{component_name}` 的执行是否符合预期。",
            "返回 JSON：",
            "{",
            '  "deviation_detected": true/false,',
            '  "deviation_severity": "none"|"minor"|"moderate"|"severe",',
            '  "adjustment_needed": true/false,',
            '  "adjustment_suggestions": ["...","..."]',
            "}",
            "=== 期望 ===",
            json.dumps(expected, ensure_ascii=False),
            "=== 实际输出 ===",
            json.dumps(actual_output, ensure_ascii=False),
            "=== 上下文 ===",
            json.dumps(execution_context, ensure_ascii=False),
        ]
        raw = self.llm.request(messages)
        try:
            parsed = json.loads(_first_json_str(raw))
        except Exception:
            parsed = {}

        parsed.setdefault("deviation_detected", False)
        parsed.setdefault("deviation_severity", "none")
        parsed.setdefault("adjustment_needed", False)
        parsed.setdefault("adjustment_suggestions", [])
        if not isinstance(parsed.get("adjustment_suggestions"), list):
            parsed["adjustment_suggestions"] = []

        return parsed

    def _apply_metacognition_adjustment(
        self, component_name: str, adjustment_suggestions: List[str]
    ) -> None:
        """
        把调整建议保存起来，后面在生成 prompt 时插入到文本中。
        """
        if not adjustment_suggestions:
            return

        if component_name not in self.metacognition_adjustments:
            self.metacognition_adjustments[component_name] = []

        for s in adjustment_suggestions:
            if not s:
                continue
            self.metacognition_adjustments[component_name].append(
                {
                    "suggestion": s,
                    "time": time.time(),
                }
            )
        logger.info(
            f"[Meta] Apply {len(adjustment_suggestions)} adjustments to {component_name}"
        )

    def _get_adjusted_prompt(
        self, component_name: str, base_messages: List[str]
    ) -> List[str]:
        """
        可选：如果你要在某个地方给 LLM 增强 prompt（比如调用 VLMPlanner 之前做高层 prompt），
        就通过这个函数注入 [Metacognition Adjustments]。
        """
        adj_list = self.metacognition_adjustments.get(component_name, [])
        if not adj_list:
            return base_messages

        suggestions = [a["suggestion"] for a in adj_list]
        block = "[Metacognition Adjustments]\n" + "\n".join(
            f"- {s}" for s in suggestions
        )

        return [base_messages[0], block] + base_messages[1:]

    def _meta_monitor_progress(self, info: Dict[str, Any], episode_info: Dict[str, Any]):
        """
        在执行阶段监控：
        - task_progress 是否长时间不变
        - 无效动作比例是否过高
        这里可以根据情况触发：
        - 修改 strategy 为 plan_and_execute / case_based
        - 给 VLMPlanner 的 system prompt 加 “少重复走路，多尝试 XXX” 之类提示
        当前先打 log，方便你后面扩展。
        """
        step = info.get("env_step", 0)
        if step <= 0:
            return

        rewards = episode_info.get("reward", [])
        num_invalid = episode_info.get("num_invalid_actions", 0)
        invalid_ratio = num_invalid / step if step > 0 else 0.0

        # 示例规则：无效动作比例超过 0.4 就提示一下
        if invalid_ratio > 0.4:
            logger.warning(
                f"[Meta] High invalid ratio {invalid_ratio:.2f}, consider replanning or adjusting prompt."
            )


# ============= 工具函数 =============

def _first_json_str(text: str) -> str:
    """
    从 LLM 输出里提取第一段 {...} JSON 字符串。
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return text[start : end + 1]
