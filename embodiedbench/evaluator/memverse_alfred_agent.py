import json
import os
import re
import time
import uuid
from datetime import datetime

import numpy as np
import requests

from embodiedbench.main import logger


class MemVerseClient:
    """HTTP client for MemVerse and ALFRED SemMemory endpoints."""

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.insert_url = f"{self.base_url}/insert"
        self.query_url = f"{self.base_url}/query"
        self.start_episode_url = f"{self.base_url}/alfred/start_episode"
        self.retrieve_long_term_url = f"{self.base_url}/alfred/retrieve_long_term"
        self.end_episode_url = f"{self.base_url}/alfred/end_episode"

    def _post_json(self, url, payload, timeout=120):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            logger.error(f"[MemVerse] Request failed with status {response.status_code}: {response.text}")
            return {"status": "error", "message": response.text}
        except Exception as e:
            logger.error(f"[MemVerse] Request exception: {e}")
            return {"status": "error", "message": str(e)}

    def start_episode(self, payload):
        logger.info(f"[MemVerse] Starting SemMemory episode: {payload.get('episode_id')}")
        return self._post_json(self.start_episode_url, payload)

    def retrieve_long_term(self, payload):
        return self._post_json(self.retrieve_long_term_url, payload)

    def end_episode(self, payload):
        logger.info(f"[MemVerse] Ending SemMemory episode: {payload.get('episode_id')}")
        return self._post_json(self.end_episode_url, payload, timeout=300)

    def query(self, text):
        try:
            logger.info(f"[MemVerse] Querying legacy memory for: {text}")
            response = requests.post(self.query_url, data={"query": text}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("final_answer", "")
            logger.error(f"[MemVerse] Query failed with status {response.status_code}")
            return ""
        except Exception as e:
            logger.error(f"[MemVerse] Query exception: {e}")
            return ""

    def insert(self, text, image_path=None):
        try:
            logger.info(f"[MemVerse] Inserting legacy memory: {text}")
            data = {"query": text}
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    response = requests.post(self.insert_url, data=data, files={"image": f}, timeout=120)
            else:
                response = requests.post(self.insert_url, data=data, timeout=120)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"[MemVerse] Insert exception: {e}")
            return False


class MemVerseAlfredAgent:
    """ALFRED agent wrapper with local working memory and SemMemory long-term memory."""

    def __init__(self, env, planner):
        self.env = env
        self.planner = planner
        self.memverse = MemVerseClient()
        self.working_memory = self._empty_working_memory()

    def _empty_working_memory(self):
        return {
            "latest_perception": "",
            "plan_history": [],
            "action_history": [],
            "environment_feedback": [],
            "current_status": {},
            "active_entities": {},
            "active_relationships": [],
        }

    def perceive(self, instruction, img_path):
        from embodiedbench.planner.planner_utils import fix_json, local_image_to_data_url

        try:
            data_url = local_image_to_data_url(img_path)
            prompt = (
                f"You are an Embodied AI agent receiving the task in an ALFRED indoor environment: '{instruction}'.\n"
                "1. Briefly describe the current visual scene.\n"
                "2. Identify key target objects, destinations, or methods that are missing or unclear.\n"
                "3. Return retrieval queries as a list of strings.\n"
                "Reply ONLY with JSON:\n"
                "{"
                "\"scene_description\":\"...\","
                "\"missing_information\":\"...\","
                "\"retrieval_queries\":[\"query 1\"]"
                "}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            raw_response = self.planner.model.respond(messages)
            parsed_json = json.loads(fix_json(raw_response))
            perception_summary = (
                f"{parsed_json.get('scene_description', '')} "
                f"{parsed_json.get('missing_information', '')}"
            ).strip()
            return perception_summary, parsed_json.get("retrieval_queries", [])
        except Exception as e:
            logger.error(f"[Perception] Failed: {e}")
            return "", []

    def _update_working_memory_insights(self, action_str, current_image_path):
        from embodiedbench.planner.planner_utils import fix_json, local_image_to_data_url

        try:
            data_url = local_image_to_data_url(current_image_path)
            prompt = (
                "You are the Working Memory Insight module of an Embodied AI.\n"
                f"You just executed: '{action_str}'. Observe the current scene carefully.\n"
                f"Previous Status: {json.dumps(self.working_memory.get('current_status', {}))}\n"
                f"Previous Entities: {json.dumps(self.working_memory.get('active_entities', {}))}\n"
                f"Previous Relationships: {json.dumps(self.working_memory.get('active_relationships', []))}\n\n"
                "Extract only new or changed working-memory facts. Reply ONLY with JSON:\n"
                "{\"latest_perception_update\":\"\","
                "\"current_status_updates\":{},"
                "\"active_entities_updates\":{},"
                "\"active_relationships_updates\":[]}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            raw_response = self.planner.model.respond(messages)
            parsed_json = json.loads(fix_json(raw_response))

            if parsed_json.get("latest_perception_update"):
                self.working_memory["latest_perception"] = parsed_json["latest_perception_update"]
            if isinstance(parsed_json.get("current_status_updates"), dict):
                self.working_memory["current_status"].update(parsed_json["current_status_updates"])
            if isinstance(parsed_json.get("active_entities_updates"), dict):
                self.working_memory["active_entities"].update(parsed_json["active_entities_updates"])
            if isinstance(parsed_json.get("active_relationships_updates"), list):
                self.working_memory["active_relationships"].extend(parsed_json["active_relationships_updates"])
            return parsed_json
        except Exception as e:
            logger.error(f"[WorkingMemory] Update failed: {e}")
            return None

    def _make_episode_id(self):
        scene = getattr(self.env, "scene_id", None) or getattr(self.env, "_scene_id", None) or "alfred"
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{scene}_{stamp}_{uuid.uuid4().hex[:8]}"

    def _infer_phase(self):
        actions = " ".join(self.working_memory.get("action_history", [])[-4:]).lower()
        last_feedback = self.working_memory.get("environment_feedback", [])[-1:] or [""]
        if last_feedback[0] == "Failed":
            return "failure_recovery"
        if any(word in actions for word in ["clean", "wash", "heat", "cool", "slice"]):
            return "prepare_object"
        if any(word in actions for word in ["pick", "pickup", "take"]):
            return "acquire_object"
        if any(word in actions for word in ["place", "put"]):
            return "place_object"
        if any(word in actions for word in ["open", "close", "toggle"]):
            return "toggle_or_operate"
        return "search_object"

    def _working_memory_context(self):
        wm_str = "\n\n### Working Memory\n"
        if self.working_memory["latest_perception"]:
            wm_str += f"- Latest Perception: {self.working_memory['latest_perception']}\n"
        if self.working_memory["current_status"]:
            wm_str += f"- Current Status: {self.working_memory['current_status']}\n"
        if self.working_memory["active_entities"]:
            wm_str += f"- Active Entities: {self.working_memory['active_entities']}\n"
        if self.working_memory["active_relationships"]:
            wm_str += f"- Active Relationships: {self.working_memory['active_relationships']}\n"
        wm_str += "- Recent Actions & Feedback:\n"
        for act, fb in zip(
            self.working_memory["action_history"][-5:],
            self.working_memory["environment_feedback"][-5:],
        ):
            wm_str += f"  * Action: {act} | Feedback: {fb}\n"
        wm_str += (
            "\n[CRITICAL]: Do not loop. If you include working-memory updates, append JSON exactly like:\n"
            "```json\n{\"current_status_updates\":{}, \"active_entities_updates\":{}, \"active_relationships_updates\":[]}\n```\n"
        )
        return wm_str

    def _retrieve_payload(self, episode_id, instruction, step):
        active_entities = self.working_memory.get("active_entities", {})
        visible_objects = list(active_entities.keys()) if isinstance(active_entities, dict) else []
        return {
            "episode_id": episode_id,
            "instruction": instruction,
            "step": step,
            "phase": self._infer_phase(),
            "current_state_summary": {
                "latest_perception": self.working_memory.get("latest_perception", ""),
                "visible_objects": visible_objects,
                "holding": self.working_memory.get("current_status", {}).get("holding"),
                "current_status": self.working_memory.get("current_status", {}),
                "active_entities": self.working_memory.get("active_entities", {}),
                "active_relationships": self.working_memory.get("active_relationships", []),
            },
            "recent_feedback": [
                {"action": a, "success": f == "Success"}
                for a, f in zip(
                    self.working_memory.get("action_history", [])[-5:],
                    self.working_memory.get("environment_feedback", [])[-5:],
                )
            ],
            "top_k": 8,
        }

    def _append_reasoning_updates(self, reasoning):
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", str(reasoning), re.DOTALL | re.IGNORECASE)
            if not json_match:
                return
            updates = json.loads(json_match.group(1))
            self.working_memory["current_status"].update(updates.get("current_status_updates", {}))
            self.working_memory["active_entities"].update(updates.get("active_entities_updates", {}))
            if updates.get("active_relationships_updates"):
                self.working_memory["active_relationships"].extend(updates["active_relationships_updates"])
        except Exception:
            return

    def _action_desc(self, action):
        if isinstance(action, str):
            return action
        try:
            return self.env.language_skill_set[action]
        except Exception:
            return str(action)

    def run_single_episode(self, start_obs, start_img_path, user_instruction):
        episode_info = {"reward": [], "num_invalid_actions": 0, "empty_plan": 0}
        img_path = start_img_path
        episode_id = self._make_episode_id()
        step_records = []
        history = []
        info = {}

        self.working_memory = self._empty_working_memory()

        perception_summary, _query_questions = self.perceive(user_instruction, img_path)
        self.working_memory["latest_perception"] = perception_summary

        self.memverse.start_episode(
            {
                "episode_id": episode_id,
                "instruction": user_instruction,
                "available_actions": list(getattr(self.env, "language_skill_set", [])),
                "metadata": {
                    "scene_id": getattr(self.env, "scene_id", None) or getattr(self.env, "_scene_id", None),
                    "task_type": getattr(self.env, "task_type", None) or getattr(self.env, "_task_type", None),
                },
            }
        )

        self.planner.reset()
        self.planner.set_actions(self.env.language_skill_set)

        done = False
        while not done:
            try:
                ltm_response = self.memverse.retrieve_long_term(
                    self._retrieve_payload(episode_id, user_instruction, self.env._current_step)
                )
                retrieved_memory = ltm_response.get("long_term_context", "") if isinstance(ltm_response, dict) else ""

                augmented_instruction = user_instruction
                if retrieved_memory:
                    augmented_instruction += f"\n\n{retrieved_memory}\n"
                augmented_instruction += self._working_memory_context()

                action, reasoning = self.planner.act(img_path, augmented_instruction)
                self.working_memory["plan_history"].append(str(reasoning))
                logger.info(f"Planner Output Action: {action}")
                self._append_reasoning_updates(reasoning)

                if action == -2:
                    episode_info["empty_plan"] = 1
                    info = {"task_success": 0, "env_step": self.env._current_step}
                    break

                if action == -1:
                    self.env._cur_invalid_actions += 1
                    episode_info["reward"].append(-1)
                    episode_info["num_invalid_actions"] += 1
                    self.working_memory["action_history"].append("invalid action")
                    self.working_memory["environment_feedback"].append("Failed")
                    history.append({"step": self.env._current_step, "action_desc": "invalid action", "success": False})
                    if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                        info = {"task_success": 0, "env_step": self.env._current_step}
                        break
                    continue

                actions_to_execute = action if isinstance(action, list) else [action]
                max_remaining = self.env._max_episode_steps - self.env._current_step
                for act_single in actions_to_execute[: min(max_remaining, len(actions_to_execute))]:
                    obs, reward, done, info = self.env.step(act_single, reasoning=reasoning)
                    act_str = self._action_desc(act_single)
                    img_path = self.env.save_image(obs)
                    action_success = bool(info.get("last_action_success", 0))

                    episode_info["reward"].append(reward)
                    episode_info["num_invalid_actions"] += int(not action_success)
                    self.working_memory["action_history"].append(act_str)
                    self.working_memory["environment_feedback"].append("Success" if action_success else "Failed")
                    history.append({"step": self.env._current_step, "action_desc": act_str, "success": action_success})

                    step_records.append(
                        {
                            "step": self.env._current_step,
                            "image_path": img_path,
                            "perception": {
                                "scene_description": self.working_memory.get("latest_perception", ""),
                                "visible_objects": list(self.working_memory.get("active_entities", {}).keys()),
                            },
                            "action": {
                                "action_id": act_single if isinstance(act_single, int) else None,
                                "action_desc": act_str,
                            },
                            "reasoning": str(reasoning),
                            "feedback": {
                                "last_action_success": int(action_success),
                                "env_feedback": info,
                                "reward": reward,
                            },
                            "working_memory_snapshot": {
                                "current_status": dict(self.working_memory.get("current_status", {})),
                                "active_entities": dict(self.working_memory.get("active_entities", {})),
                                "active_relationships": list(self.working_memory.get("active_relationships", [])),
                            },
                        }
                    )

                    self.planner.update_info(info)
                    if done or not action_success:
                        self._update_working_memory_insights(act_str, img_path)
                        break

            except Exception as e:
                logger.error(f"Error during step: {e}")
                time.sleep(5)

        task_success = info.get("task_success", 0)
        end_payload = {
            "episode_id": episode_id,
            "instruction": user_instruction,
            "result": {
                "task_success": task_success,
                "task_progress": info.get("task_progress", info.get("progress")),
                "num_steps": info.get("env_step", self.env._current_step),
                "num_invalid_actions": episode_info["num_invalid_actions"],
                "mean_reward": float(np.mean(episode_info["reward"])) if episode_info["reward"] else 0.0,
            },
            "history": history,
            "step_records": step_records,
            "final_working_memory": self.working_memory,
        }
        end_response = self.memverse.end_episode(end_payload)
        logger.info(f"[MemVerse] End episode response: {end_response}")

        episode_info["task_success"] = task_success
        episode_info["num_steps"] = info.get("env_step", self.env._current_step)
        if episode_info["reward"]:
            episode_info["reward"] = np.mean(episode_info["reward"])

        return episode_info
