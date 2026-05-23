import os
import time
import requests
import numpy as np
from embodiedbench.main import logger

class MemVerseClient:
    """
    Handles communication with the MemVerse memory system.
    """
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.insert_url = f"{self.base_url}/insert"
        self.query_url = f"{self.base_url}/query"

    def query(self, text):
        try:
            logger.info(f"[MemVerse] Querying memory for: {text}")
            response = requests.post(self.query_url, data={"query": text})
            if response.status_code == 200:
                data = response.json()
                memory = data.get("final_answer", "")
                logger.info(f"[MemVerse] Retrieved memory successfully.")
                return memory
            else:
                logger.error(f"[MemVerse] Query failed with status {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"[MemVerse] Query exception: {e}")
            return ""

    def insert(self, text, image_path=None):
        try:
            logger.info(f"[MemVerse] Inserting memory: {text}")
            data = {"query": text}
            
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files = {"image": f}
                    response = requests.post(self.insert_url, data=data, files=files)
            else:
                response = requests.post(self.insert_url, data=data)
                
            if response.status_code == 200:
                logger.info(f"[MemVerse] Insert successful.")
                return True
            else:
                logger.error(f"[MemVerse] Insert failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"[MemVerse] Insert exception: {e}")
            return False

class MemVerseAlfredAgent:
    """
    A clean agent that integrates MemVerse for Long-Term Memory and maintains a Working Memory for ALFRED environment.
    """
    def __init__(self, env, planner):
        self.env = env
        self.planner = planner
        self.memverse = MemVerseClient()
        self.working_memory = {
            "latest_perception": "",
            "plan_history": [],
            "action_history": [],
            "environment_feedback": [],
            "current_status": {},
            "active_entities": {},
            "active_relationships": []
        }

    def perceive(self, instruction, img_path):
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(img_path)
            prompt = (
                f"You are an Embodied AI agent receiving the task in an ALFRED indoor environment: '{instruction}'.\n"
                "1. Briefly describe the current visual scene.\n"
                "2. Critically identify what key target object, destination, or method related to this task is CURRENTLY UNCLEAR or missing from your view.\n"
                "3. Formulate one or more precise questions asking for this missing knowledge. Return them as a LIST of strings.\n"
                "Reply ONLY with a JSON dictionary in the following format:\n"
                "{\n"
                '  "scene_description": "...",\n'
                '  "missing_information": "...",\n'
                '  "retrieval_queries": ["query 1", "query 2"]\n'
                "}"
            )
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]
            raw_response = self.planner.model.respond(messages)
            
            from embodiedbench.planner.planner_utils import fix_json
            parsed_json = json.loads(fix_json(raw_response))
            
            perception_summary = f"{parsed_json.get('scene_description', '')} {parsed_json.get('missing_information', '')}"
            queries = parsed_json.get('retrieval_queries', [])
            return perception_summary, queries
        except Exception as e:
            logger.error(f"[Perception] Failed: {e}")
            return "", ""

    def reflect_and_summarize(self, instruction, result_str, history_str, img_path, working_memory):
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(img_path)
            prompt = (
                f"You are a memory consolidation module for an Embodied AI agent in ALFRED indoor scene.\n"
                f"Task: '{instruction}'\nResult: {result_str}\nExecution History:\n{history_str}\n"
                f"Agent's final Working Memory: {working_memory}\n"
                "Extract valuable long-term knowledge based on the final scene, history, and memory.\n"
                "1. Describe the final scene.\n"
                "2. Summarize what happened, success/failure reasons, and lessons learned.\n"
                "3. Identify wrong assumptions/mistakes.\n"
                "4. Extract declarative knowledge (room layouts, relationships).\n"
                "5. Extract procedural skill (action sequences, slicing/heating requirements etc).\n"
                "6. Provide a condensed lesson.\n"
                "Reply ONLY with a JSON dictionary matching this structure:\n"
                "{\"summary_and_cause\": \"...\", \"error_correction\": \"...\", \"declarative_knowledge\": \"...\", \"procedural_skill\": \"...\", \"refined_lesson\": \"...\"}"
            )
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]
            raw_response = self.planner.model.respond(messages)
            
            from embodiedbench.planner.planner_utils import fix_json
            parsed_json = json.loads(fix_json(raw_response))
            
            return (f"Task: {instruction}. Result: {result_str}. "
                    f"Summary: {parsed_json.get('summary_and_cause', '')} "
                    f"Correction: {parsed_json.get('error_correction', '')} "
                    f"Declarative: {parsed_json.get('declarative_knowledge', '')} "
                    f"Procedural: {parsed_json.get('procedural_skill', '')} "
                    f"Lesson: {parsed_json.get('refined_lesson', '')}")
            
        except Exception as e:
            logger.error(f"[Reflection] Failed: {e}")
            return f"Task ended: {instruction}. Result: {result_str}. {history_str}"

    def _update_working_memory_insights(self, action_str, current_image_path):
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(current_image_path)
            
            # Serialize necessary current WM to string to provide context
            current_status_str = json.dumps(self.working_memory.get('current_status', {}))
            current_entities_str = json.dumps(self.working_memory.get('active_entities', {}))
            current_rels_str = json.dumps(self.working_memory.get('active_relationships', []))
            
            prompt = (
                f"You are the Working Memory Insight module of an Embodied AI.\n"
                f"You just executed: '{action_str}'. Observe the current scene carefully.\n"
                f"Previous Status: {current_status_str}\n"
                f"Previous Entities: {current_entities_str}\n"
                f"Previous Relationships: {current_rels_str}\n\n"
                "Extract the following updates based on the visual observation and the action taken:\n"
                "1. 'latest_perception_update': Give a brief description of the current global visual scene.\n"
                "2. 'current_status_updates': Extract only NEW changes in status. Include BOTH robot states (e.g., holding something) AND object states/locations (e.g., fridge is open, apple is on table).\n"
                "3. 'active_entities_updates': Extract NEW entities found or updates to existing entities. Key should be the entity name, and value should be details like type, concept, and functional affordances.\n"
                "4. 'active_relationships_updates': Extract or update NEW relationships discovered (e.g., spatial relations like 'apple IN fridge', or semantic relations). Format as list of strings or dicts.\n\n"
                "Leave values empty if nothing new. Reply ONLY with a JSON dictionary formatted exactly like:\n"
                "{\"latest_perception_update\": \"\", \"current_status_updates\": {}, \"active_entities_updates\": {}, \"active_relationships_updates\": []}"
            )
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]
            raw_response = self.planner.model.respond(messages)
            from embodiedbench.planner.planner_utils import fix_json
            parsed_json = json.loads(fix_json(raw_response))
            
            if parsed_json.get('latest_perception_update'):
                self.working_memory['latest_perception'] = parsed_json['latest_perception_update']
            if 'current_status_updates' in parsed_json and isinstance(parsed_json['current_status_updates'], dict):
                self.working_memory['current_status'].update(parsed_json['current_status_updates'])
            if 'active_entities_updates' in parsed_json and isinstance(parsed_json['active_entities_updates'], dict):
                self.working_memory['active_entities'].update(parsed_json['active_entities_updates'])
            if 'active_relationships_updates' in parsed_json and isinstance(parsed_json['active_relationships_updates'], list) and parsed_json['active_relationships_updates']:
                self.working_memory['active_relationships'].extend(parsed_json['active_relationships_updates'])

            return parsed_json
        except Exception as e:
            return None

    def correct_retrieved_memory(self, instruction, retrieved_memory, working_memory):
        import json
        if not retrieved_memory or not retrieved_memory.strip():
            return
        try:
            prompt = (
                f"You are a memory correction module for an ALFRED Embodied AI.\n"
                f"Task: '{instruction}'\nPreviously Retrieved: {retrieved_memory}\nActual Execution Memory: {working_memory}\n"
                "Provide a concise list of corrections if the previous memory contradicts actual dynamics. "
                "Reply ONLY with JSON: {\"corrections\": [\"correction 1\"]}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            raw_response = self.planner.model.respond(messages)
            
            from embodiedbench.planner.planner_utils import fix_json
            parsed_json = json.loads(fix_json(raw_response))
            for c in parsed_json.get("corrections", []):
                if c.strip():
                    self.memverse.insert(f"Correction for previous memory: {c}")
        except Exception as e:
            logger.error(f"[Memory Correction] Failed: {e}")

    def run_single_episode(self, start_obs, start_img_path, user_instruction):
        """
        Adapted specifically to run one ALFRED episode simulation loop for MemVerse integration.
        Returns episode_info metric.
        Must be called from within the evaluator evaluate() loop.
        """
        episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
        img_path = start_img_path
        initial_img_path = img_path
        
        # Reset Working Memory for ALFRED
        self.working_memory = {
            "latest_perception": "",
            "plan_history": [],
            "action_history": [],
            "environment_feedback": [],
            "current_status": {},
            "active_entities": {},
            "active_relationships": []
        }
        
        # 1. Perception
        perception_summary, query_questions = self.perceive(user_instruction, img_path)
        self.working_memory["latest_perception"] = perception_summary

        # 2. Insert start State
        # self.memverse.insert(f"ALFRED Task started: {user_instruction}. Initial environment.", img_path)

        # 3. Retrieve initial Memory
        retrieved_memory = ""
        if query_questions and isinstance(query_questions, list):
            for q in query_questions:
                res = self.memverse.query(f"Task: {user_instruction}\nContext Query: {q}\n"
                                          "Provide specific procedural and factual ALFRED environment knowledge to solve this task.")
                if res: retrieved_memory += f"\n[Query: {q}]\n{res}\n"
        else:
            retrieved_memory = self.memverse.query(f"Task: {user_instruction}\nRetrieve procedural knowledge to solve this task in an ALFRED house.")
        
        logger.info(f"Retrieved Memory: {retrieved_memory}")

        self.planner.reset()
        self.planner.set_actions(self.env.language_skill_set) # Important for ALFRED env specific dynamics 
        
        done = False
        info = {}
        
        while not done:
            try:
                # Compile working memory context
                wm_str = "\n\n### 📝 YOUR WORKING MEMORY:\n"
                if self.working_memory["latest_perception"]: wm_str += f"- Latest Perception: {self.working_memory['latest_perception']}\n"
                if self.working_memory["current_status"]: wm_str += f"- Current Status: {self.working_memory['current_status']}\n"
                if self.working_memory["active_entities"]: wm_str += f"- Active Entities: {self.working_memory['active_entities']}\n"
                if self.working_memory["active_relationships"]: wm_str += f"- Active Relationships: {self.working_memory['active_relationships']}\n"
                wm_str += "- Recent Actions & Feedback:\n"
                for act, fb in zip(self.working_memory["action_history"][-5:], self.working_memory["environment_feedback"][-5:]):
                    wm_str += f"  * Action: {act} | Feedback: {fb}\n"
                
                wm_str += "\n[CRITICAL]: Do not loop. Update knowledge appending a JSON explicitly:\n```json\n{\"current_status_updates\":{}, \"active_entities_updates\":{}, \"active_relationships_updates\":[]}\n```\n"

                augmented_instruction = user_instruction
                if retrieved_memory:
                    augmented_instruction += f"\n\n### 🧠 YOUR LONG-TERM MEMORY:\n{retrieved_memory}\n"
                augmented_instruction += wm_str

                # Inference Action
                action, reasoning = self.planner.act(img_path, augmented_instruction)
                self.working_memory["plan_history"].append(str(reasoning))
                action = [35,129,79,133,155,156,129,39,133,34,127,35,159,79,133,39,35,129,38,143,133,144]
                logger.info(f"Planner Output Action: {action}")
                
                # Parse WM Updates
                import re, json
                try:
                    json_match = re.search(r'```json\s*\n(.*?)\n```', str(reasoning), re.DOTALL | re.IGNORECASE)
                    if json_match:
                        updates = json.loads(json_match.group(1))
                        self.working_memory['current_status'].update(updates.get('current_status_updates', {}))
                        self.working_memory['active_entities'].update(updates.get('active_entities_updates', {}))
                        if updates.get('active_relationships_updates'): 
                            self.working_memory['active_relationships'].extend(updates['active_relationships_updates'])
                except Exception: pass

                # Handle Invalid / Empty Actions
                if action == -2:
                    episode_info['empty_plan'] = 1
                    info = {'task_success': 0, 'env_step': self.env._current_step}
                    break
                    
                if action == -1:
                    self.env._cur_invalid_actions += 1
                    episode_info['reward'].append(-1)
                    episode_info['num_invalid_actions'] += 1
                    self.working_memory["action_history"].append('invalid action')
                    self.working_memory["environment_feedback"].append('Failed')
                    if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                        info = {'task_success': 0, 'env_step': self.env._current_step}
                        break
                    continue

                # Execute Action
                actions_to_execute = action if isinstance(action, list) else [action]
                for act_single in actions_to_execute[:min(self.env._max_episode_steps - self.env._current_step, len(actions_to_execute))]:
                    obs, reward, done, info = self.env.step(act_single, reasoning=reasoning)
                    act_str = act_single if isinstance(act_single, str) else self.env.language_skill_set[act_single]
                    img_path = self.env.save_image(obs)
                    
                    episode_info['reward'].append(reward)
                    episode_info['num_invalid_actions'] += (info.get('last_action_success', 0) == 0)
                    self.working_memory["action_history"].append(act_str)
                    self.working_memory["environment_feedback"].append('Success' if info.get('last_action_success', 0) else 'Failed')
                    
                    self.planner.update_info(info)
                    if done or not info.get('last_action_success', 1):
                        self._update_working_memory_insights(act_str, img_path)
                        break

            except Exception as e:
                logger.error(f"Error during step: {e}")
                time.sleep(5)
                
        # 4. Episode End Reflection & Insertion
        task_success = info.get('task_success', 0)
        history_str = "\n".join([f"- {a} [{f}]" for a, f in zip(self.working_memory['action_history'], self.working_memory['environment_feedback'])])
        end_message = self.reflect_and_summarize(user_instruction, "Success" if task_success else "Failed", history_str, img_path, self.working_memory)
        
        self.memverse.insert(f"ALFRED Task end: {end_message}.", initial_img_path)
        self.correct_retrieved_memory(user_instruction, retrieved_memory, self.working_memory)

        episode_info['task_success'] = task_success
        episode_info['num_steps'] = info.get("env_step", self.env._current_step)
        if episode_info['reward']: episode_info['reward'] = np.mean(episode_info['reward'])
        
        return episode_info
