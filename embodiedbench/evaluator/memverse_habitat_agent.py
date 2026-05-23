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
                # Extract the final answer or rag memory from the response
                # memory = data.get("rag_memory", data.get("final_answer", ""))
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


class MemVerseHabitatAgent:
    """
    A clean agent that integrates MemVerse for Long-Term Memory and maintains a Working Memory.
    It queries memory at the start, and inserts memory ONLY at the start and end of an episode.
    """
    def __init__(self, env, planner):
        self.env = env
        self.planner = planner
        self.memverse = MemVerseClient()
        self.working_memory = {
            "action_history": [],
            "semantic_mapping": {},
            "spatial_cache": {},
            "subgoal_stack": []
        }

    def perceive(self, instruction, img_path):
        """
        A standalone, lightweight perception module using the existing planner model.
        It observes the initial environment and generates a query about what is unclear or missing.
        """
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(img_path)
            prompt = (
                f"You are an Embodied AI agent receiving the task: '{instruction}'.\n"
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
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            
            logger.info("[Perception] Sending vision perception prompt...")
            # Using the planner's underlying model directly, bypassing VLMPlanner's JSON action format restriction
            raw_response = self.planner.model.respond(messages)
            logger.info(f"[Perception] Raw response:\n{raw_response}")
            
            # Clean and parse JSON
            from embodiedbench.planner.planner_utils import fix_json
            fixed_response = fix_json(raw_response)
            parsed_json = json.loads(fixed_response)
            
            perception_summary = f"{parsed_json.get('scene_description', '')} {parsed_json.get('missing_information', '')}"
            # Extract queries as a list
            queries = parsed_json.get('retrieval_queries', [])
            if 'retrieval_query' in parsed_json and not queries:  # Fallback just in case LLM outputs the old key
                old_q = parsed_json.get('retrieval_query', '')
                if old_q:
                    queries = [old_q]
            
            return perception_summary, queries
            
        except Exception as e:
            logger.error(f"[Perception] Failed to execute standalone perception: {e}")
            return "", ""

    def reflect_and_summarize(self, instruction, result_str, history_str, img_path, working_memory):
        """
        Uses the LLM to reflect on the episode's execution history, correct any cognitive errors,
        and extract refined, generalizable knowledge into a short summary for long-term memory.
        """
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(img_path)
            prompt = (
                f"You are a memory consolidation module for an Embodied AI agent.\n"
                f"Task: '{instruction}'\n"
                f"Result: {result_str}\n"
                f"Execution History:\n{history_str}\n\n"
                f"Agent's final Working Memory State (CRITICAL FOR LEARNING):\n"
                f"- Semantic Mapping: {working_memory.get('semantic_mapping', {})}\n"
                f"- Spatial Cache: {working_memory.get('spatial_cache', {})}\n\n"
                "Based on the final observation image, the execution history, and the working memory, your job is to extract valuable long-term knowledge.\n"
                "1. Briefly describe the final visual scene in the observation image.\n"
                "2. Summarize in detail what happened and the reasons for success or failure. Summarize the lessons learned from success or failure.\n"
                "3. Identify any wrong assumptions or mistakes made during execution and explicitly correct them (e.g., 'The red object is not an apple, it is a strawberry.').\n"
                "4. Identify the actual INITIAL location where the target object was successfully found and picked up (if applicable).\n"
                "5. Extract 'declarative_knowledge' from the semantic mappings (e.g., 'The receptacle named sink matches the physical basin'). Filter out specific dynamic spatial coordinates as the environment resets, but KEEP generalized regional knowledge.\n"
                "6. Extract 'procedural_skill' based on the action history (e.g., sequential rules or interaction habits that succeeded or failed).\n"
                "7. Provide a refined, highly condensed lesson that should be stored in long-term memory for future tasks. IMPORTANT: The environment resets after each task! Do NOT memorize the object's final placement destination as its location. Only memorize where it was INITIALLY found.\n"
                "Reply ONLY with a JSON dictionary in the following format:\n"
                "{\n"
                '  "final_scene_description": "...",\n'
                '  "summary_and_cause": "...",\n'
                '  "error_correction": "...",\n'
                '  "initial_object_location": "...",\n'
                '  "declarative_knowledge": "...",\n'
                '  "procedural_skill": "...",\n'
                '  "refined_lesson": "..."\n'
                "}"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            
            logger.info("[Reflection] Sending reflection and summary prompt...")
            raw_response = self.planner.model.respond(messages)
            logger.info(f"[Reflection] Raw response:\n{raw_response}")
            
            from embodiedbench.planner.planner_utils import fix_json
            fixed_response = fix_json(raw_response)
            parsed_json = json.loads(fixed_response)
            
            # Note: We purposely DO NOT include 'final_scene_description' in the final_summary 
            # to prevent polluting the long-term memory with reset-prone target destinations.
            final_summary = (
                f"Task: {instruction}. Result: {result_str}. "
                f"Summary: {parsed_json.get('summary_and_cause', '')} "
                f"Correction: {parsed_json.get('error_correction', '')} "
                f"Declarative Knowledge: {parsed_json.get('declarative_knowledge', '')} "
                f"Procedural Skill: {parsed_json.get('procedural_skill', '')} "
                f"Object Verified Initial Location: {parsed_json.get('initial_object_location', '')} "
                f"Lesson Learned: {parsed_json.get('refined_lesson', '')}"
            )
            return final_summary
            
        except Exception as e:
            logger.error(f"[Reflection] Failed to execute reflection: {e}")
            # Fallback to the raw history if reflection fails
            return f"Task ended: {instruction}. Result: {result_str}. {history_str}"

    def _update_working_memory_insights(self, action_str, current_image_path):
        """
        After an action is executed and the environment yields a new observation,
        call the VLM to explicitly extract new semantic or spatial insights from the current scene.
        This provides an independent cognitive pass solely focused on state tracking.
        """
        import json
        from embodiedbench.planner.planner_utils import local_image_to_data_url
        
        try:
            data_url = local_image_to_data_url(current_image_path)
            prompt = (
                f"You are the Working Memory Insight module of an Embodied AI.\n"
                f"You just executed: '{action_str}'\n"
                "Look at the current scene and briefly extract any NEW spatial discoveries, semantic insights, or object state changes.\n"
                "If there are no new meaningful insights, leave the values empty. Reply ONLY with a JSON dictionary formatted EXACTLY like this:\n"
                "{\n"
                '  "semantic_mapping_updates": {"vocabulary": "physical_mapping"},\n'
                '  "spatial_cache_updates": {"object_name": "location"},\n'
                '  "subgoal_stack_updates": ["current_subgoal"]\n'
                "}"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            
            raw_response = self.planner.model.respond(messages)
            from embodiedbench.planner.planner_utils import fix_json
            parsed_json = json.loads(fix_json(raw_response))
            
            if 'semantic_mapping_updates' in parsed_json and isinstance(parsed_json['semantic_mapping_updates'], dict):
                self.working_memory['semantic_mapping'].update(parsed_json['semantic_mapping_updates'])
            if 'spatial_cache_updates' in parsed_json and isinstance(parsed_json['spatial_cache_updates'], dict):
                self.working_memory['spatial_cache'].update(parsed_json['spatial_cache_updates'])
            if 'subgoal_stack_updates' in parsed_json and isinstance(parsed_json['subgoal_stack_updates'], list) and len(parsed_json['subgoal_stack_updates']) > 0:
                self.working_memory['subgoal_stack'] = parsed_json['subgoal_stack_updates']

            return parsed_json
            
        except Exception as e:
            logger.error(f"[WM] Failed to extract dynamic insights: {e}")
            return None

    def correct_retrieved_memory(self, instruction, retrieved_memory, working_memory):
        """
        Compare previously retrieved memory with the actual execution experience.
        If there are deviations or errors in the retrieved knowledge, generate corrections and insert them.
        """
        import json
        if not retrieved_memory or not retrieved_memory.strip():
            return

        try:
            prompt = (
                f"You are a memory correction module for an Embodied AI agent.\n"
                f"Task: '{instruction}'\n"
                f"Previously Retrieved Memory:\n{retrieved_memory}\n\n"
                f"Actual memory from current task:\n{working_memory}\n\n"
                "Compare the Previously Retrieved Memory with the Actual Experience. "
                "If the retrieved memory contains incorrect information (e.g., wrong locations, wrong object mappings) "
                "that contradicts the actual experience, identify these errors. "
                "Provide a list of concise correction statements. If no corrections are needed, return an empty list.\n"
                "Reply ONLY with a JSON dictionary in the following format:\n"
                "IMPORTANT: The environment resets after each task! Do NOT memorize the object's final placement destination as its location. Only memorize where it was INITIALLY found.\n"
                "{\n"
                '  "corrections": ["correction 1", "correction 2"]\n'
                "}"
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            logger.info("[Memory Correction] Sending memory correction prompt...")
            raw_response = self.planner.model.respond(messages)
            logger.info(f"[Memory Correction] Raw response:\n{raw_response}")
            
            from embodiedbench.planner.planner_utils import fix_json
            fixed_response = fix_json(raw_response)
            parsed_json = json.loads(fixed_response)
            
            corrections = parsed_json.get("corrections", [])
            for correction in corrections:
                if correction.strip():
                    self.memverse.insert(f"Correction for previous memory: {correction}")
                    logger.info(f"[Memory Correction] Inserted correction: {correction}")
                    
        except Exception as e:
            logger.error(f"[Memory Correction] Failed to execute memory correction: {e}")

    def run_single_episode(self):
        episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
        obs = self.env.reset()
        img_path = self.env.save_image(obs)
        initial_img_path = img_path  # Save the initial image path for potential use in memory insertion
        user_instruction = self.env.episode_language_instruction
        logger.info(f"Instruction: {user_instruction}")
        
        # Clear/Init Working Memory for the new episode
        self.working_memory = {
            "action_history": [],
            "semantic_mapping": {},
            "spatial_cache": {},
            "subgoal_stack": []
        }
        
        # ==========================================
        # 1. [MemVerse] Insert the initial state
        # ==========================================
        perception_summary, query_questions = self.perceive(user_instruction, img_path)

        # ==========================================
        # 2. [MemVerse] Insert the initial state
        # ==========================================
  
        self.memverse.insert(f"Task started: {user_instruction}. I am observing the initial environment.", img_path)

        # ==========================================
        # 3. [MemVerse] Query at the start of the task
        # ==========================================
        retrieved_memory = ""
        if query_questions and isinstance(query_questions, list) and len(query_questions) > 0:
            for q in query_questions:
                memory_query = (f"Task: {user_instruction}\nContext Query: {q}\n"
                    "--- RETRIEVAL INSTRUCTIONS ---\n"
                    "1. CONCRETE IDENTIFICATION: You must provide a specific answer based on the retrieved memory. If the query contains abstract placeholders (e.g., 'receptacle', 'object', 'location'), you MUST cross-reference spatial clues in your memory to resolve and return the SPECIFIC CONCRETE ENTITY NAME (e.g., 'sink', 'drawer', 'microwave', 'apple'). DO NOT echo back the abstract terms; explicitly state the exact concrete entity it refers to.\n" 
                    f"2. PROCEDURAL STRATEGY: Retrieve and summarize all information relevant to solving the Task: '{user_instruction}'. Focus on 'HOW TO SOLVE IT'—including successful action sequences, required tools, and lessons learned from past attempts."
                )
                res = self.memverse.query(memory_query)
                if res:
                    retrieved_memory += f"\n[Query: {q}]\n{res}\n"
        else:
            memory_query = (
                f"Task: {user_instruction}\n"
                "Retrieve and summarize all procedural knowledge, past experiences, and successful strategies related to 'HOW TO SOLVE' this specific task."
            )
            retrieved_memory = self.memverse.query(memory_query)

        logger.info(f"Retrieved Memory: {retrieved_memory}")

        self.planner.reset()
        done = False
        info = {}
        
        while not done:
            try:
                # Build Multi-dimensional Working Memory string
                wm_str = "\n\n### 📝 YOUR WORKING MEMORY:\n"
                if self.working_memory.get("semantic_mapping"):
                    wm_str += f"- **Semantic Mapping**: {self.working_memory['semantic_mapping']}\n"
                if self.working_memory.get("spatial_cache"):
                    wm_str += f"- **Spatial Cache**: {self.working_memory['spatial_cache']}\n"
                if self.working_memory.get("subgoal_stack"):
                    wm_str += f"- **Current Subgoal Stack**: {self.working_memory['subgoal_stack']}\n"
                
                wm_str += "- **Recent Action History**:\n"
                if not self.working_memory["action_history"]:
                    wm_str += "  (None yet)\n"
                else:
                    for step in self.working_memory["action_history"]:
                        wm_str += f"  * Action: {step['action']} | Result: {step['result']} | Progress: {step['progress']}\n"
                
                wm_str += (
                    "\n[CRITICAL ANTI-LOOP INSTRUCTION]\n"
                    "Review your 'Recent Action History'. If you notice that you are repeating the same action multiple times "
                    "without success or progress, DO NOT REPEAT IT AGAIN. You must rethink your strategy, explore a new location or strategy\n"
                )

                wm_str += (
                    "\n[INSTRUCTION FOR WORKING MEMORY UPDATE]\n"
                    "If your reasoning discovers new spatial locations, establishes new semantic alignments, "
                    "or updates subgoals, APPEND a JSON block formatted EXACTLY like this at the end of your reasoning:\n"
                    "```json\n"
                    "{\n"
                    '  "semantic_mapping_updates": {"receptacle": "sink"},\n'
                    '  "spatial_cache_updates": {"apple": "on counter"},\n'
                    '  "subgoal_stack_updates": ["explore kitchen"]\n'
                    "}\n"
                    "```\n"
                    "(Only include necessary keys. Omit if no updates.)\n"
                )

                # Augment instruction with LTM and WM
                augmented_instruction = user_instruction
                if retrieved_memory and retrieved_memory.strip():
                    augmented_instruction += f"\n\n### 🧠 YOUR LONG-TERM MEMORY:\n{retrieved_memory}\n(Use this past experience to guide your current plan. If it's not helpful, rely on your current observation.)"
                
                augmented_instruction += wm_str

                # Pass the augmented instruction (with memory) to the planner
                action, reasoning = self.planner.act(img_path, augmented_instruction)
                # action = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # for test 
                logger.info(f"Planner Output Action: {action}")
                
                # Parse WM updates from reasoning
                import re, json
                try:
                    json_match = re.search(r'```json\s*\n(.*?)\n```', str(reasoning), re.DOTALL | re.IGNORECASE)
                    if json_match:
                        updates = json.loads(json_match.group(1))
                        if 'semantic_mapping_updates' in updates:
                            self.working_memory['semantic_mapping'].update(updates['semantic_mapping_updates'])
                        if 'spatial_cache_updates' in updates:
                            self.working_memory['spatial_cache'].update(updates['spatial_cache_updates'])
                        if 'subgoal_stack_updates' in updates:
                            self.working_memory['subgoal_stack'] = updates['subgoal_stack_updates']
                except Exception as e:
                    logger.debug(f"[WM] Failed to parse WM updates from reasoning: {e}")

                if action == -2: # empty plan
                    episode_info['empty_plan'] = 1
                    self.env.episode_log.append({
                        'last_action_success': 0.0,
                        'action_id': -2,
                        'action_description': 'empty plan',
                        'reasoning': reasoning,
                    })
                    info = {
                        'task_success': episode_info.get('task_success', 0),
                        'task_progress': episode_info.get("task_progress", 0),
                        'subgoal_reward': episode_info.get("subgoal_reward", 0),
                        'env_step': self.env._current_step,
                    }
                    break 
                    
                if action == -1: # invalid action
                    self.env._cur_invalid_actions += 1
                    episode_info['reward'].append(-1)
                    episode_info['num_invalid_actions'] += 1
                    self.env.episode_log.append({
                        'last_action_success': 0.0,
                        'action_id': -1,
                        'action_description': 'invalid action',
                        'reasoning': reasoning,
                    })
                    info = {
                        'task_success': episode_info.get('task_success', 0),
                        'task_progress': episode_info.get("task_progress", 0),
                        'subgoal_reward': episode_info.get("subgoal_reward", 0),
                        'env_step': self.env._current_step,
                    }
                    
                    # Update Working Memory for invalid action
                    self.working_memory["action_history"].append({
                        'action': 'invalid action',
                        'result': 'Failed (Invalid)',
                        'progress': info.get('task_progress', 0)
                    })
                    
                    if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                        break
                    continue

                # Execute action(s)
                actions_to_execute = action if isinstance(action, list) else [action]
                
                for action_single in actions_to_execute[:min(self.env._max_episode_steps - self.env._current_step, len(actions_to_execute))]:
                    obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                    action_str = action_single if isinstance(action_single, str) else self.env.language_skill_set[action_single]
                    logger.info(f"Executed action: {action_str}, Task Progress: {info.get('task_progress', 0.0)}, Task Success: {info['task_success']}")
                    
                    # Update Working Memory
                    self.working_memory["action_history"].append({
                        'action': action_str,
                        'result': 'Success' if info.get('last_action_success', 0) else 'Failed',
                        'progress': info.get('task_progress', 0)
                    })
                    
                    self.planner.update_info(info)
                    img_path = self.env.save_image(obs)
                    episode_info['reward'].append(reward)
                    episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                    # self.memverse.insert(f"Last action: {action_str}. I am observing the environment.", info['image_path'])
                    if done:
                        break

                # ------------------------------------------
                # Update Working Memory Insights mid-cycle
                # Before the next planning step starts, observe the new state
                # ------------------------------------------
                extracted_insights = self._update_working_memory_insights(self.working_memory["action_history"][-1]['action'], img_path)
                if extracted_insights:
                    logger.info(f"[WM] Mid-cycle Dynamic Insight Extracted: {extracted_insights}")

            except Exception as e: 
                logger.error(f"Error during step: {e}")
                time.sleep(10)

        # ==========================================
        # 3. [MemVerse] Insert at the end of the task
        # ==========================================
        task_success = info.get('task_success', 0)
        result_str = "Success" if task_success else "Failed"
        steps_taken = info.get("env_step", self.env._current_step)
        
        # Build full execution history
        history_str = "\nExecution History:\n"
        if hasattr(self.env, "episode_log") and self.env.episode_log:
            for i, step_log in enumerate(self.env.episode_log):
                action_desc = step_log.get("action_description", "unknown action")
                success_flag = "Success" if step_log.get("last_action_success", 0) > 0 else "Failed"
                # progress is mostly useful if available in the log, though habitat might not always track it in episode_log
                history_str += f"- Step {i+1}: {action_desc} [{success_flag}]\n"
        else:
            history_str += "(No detailed execution log found.)\n"

        # Use the reflection module to summarize experience and correct cognitive mistakes
        logger.info("[MemVerse] Summarizing episode experience before insertion...")
        end_message = self.reflect_and_summarize(user_instruction, result_str, history_str, img_path, self.working_memory)
        
        self.memverse.insert(f"Task end: {end_message}. This is the initial environment.", initial_img_path)

        # ==========================================
        # 4. [MemVerse] Compare retrieved memory with actual experience and correct if necessary
        # ==========================================
        self.correct_retrieved_memory(user_instruction, retrieved_memory, self.working_memory)

        # Compile metrics
        episode_info['instruction'] = user_instruction
        episode_info['reward'] = np.mean(episode_info['reward']) if episode_info['reward'] else 0
        episode_info['task_success'] = task_success
        episode_info["task_progress"] = info.get('task_progress', 0)
        episode_info['subgoal_reward'] = info.get('subgoal_reward', 0)
        episode_info['num_steps'] = steps_taken
        episode_info['planner_steps'] = self.planner.planner_steps
        episode_info['planner_output_error'] = self.planner.output_json_error
        episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / steps_taken if steps_taken > 0 else 0
        episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

        return "Episode Completed", episode_info
