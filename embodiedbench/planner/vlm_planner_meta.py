import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str

    def get_task_relevant_actions(self, user_instruction, available_actions, wm_context=None):
        """Extract task-relevant actions to highlight in the prompt.
        
        Strategy:
        - Keep ALL: navigate to, place at, open the, close the actions
        - For pick up: ONLY keep objects mentioned in Task Understanding and WM
        """
        import re
        
        # 🔥 Step 1: Extract object names from Task Understanding and WM
        confirmed_objects = set()
        
        if wm_context:
            # Extract from "Task Understanding" - pattern: "X IS the Y" or "X IS Y"
            tu_pattern = r'(?:the\s+)?(.+?)\s+(?:IS|is)\s+(?:the\s+)?(\w+)'
            tu_matches = re.findall(tu_pattern, wm_context)
            for phrase, actual_obj in tu_matches:
                obj_clean = actual_obj.strip().lower()
                if len(obj_clean) > 2:
                    confirmed_objects.add(obj_clean)
            
            # Extract from WM nodes: [object_name]
            # 🔥 Support multi-word object names: [red plate], [large gray furniture]
            wm_node_pattern = r'\[\s*([^\]]+?)\s*\]\s*\('
            wm_nodes = re.findall(wm_node_pattern, wm_context)
            for obj in wm_nodes:
                obj_clean = obj.strip().lower()
                # Skip generic terms
                if len(obj_clean) > 2 and obj_clean not in ['object', 'type', 'attribute', 'location', 'property']:
                    confirmed_objects.add(obj_clean)
                    # Also add individual words for flexible matching
                    for word in obj_clean.split():
                        if len(word) > 2 and word not in ['the', 'a', 'an', 'of', 'with']:
                            confirmed_objects.add(word)
        
        # 🔥 Step 2: Filter actions
        relevant_actions = []
        
        for action_id, action_desc in enumerate(available_actions):
            action_lower = action_desc.lower()
            
            # Keep ALL navigation, placement, open, close actions
            if any(keyword in action_lower for keyword in ['navigate to', 'place at', 'open the', 'close the']):
                relevant_actions.append((action_id, action_desc))
                continue
            
            # For "pick up" actions: only keep if object is in confirmed_objects
            if 'pick up' in action_lower:
                if any(obj in action_lower for obj in confirmed_objects):
                    relevant_actions.append((action_id, action_desc))
                # Skip pick up actions for non-confirmed objects
        
        # 🔥 Step 3: Build highlighted prompt
        if len(relevant_actions) > 0 and len(relevant_actions) < len(available_actions):
            highlight_str = '\n\n### ⚠️ TASK-RELEVANT ACTIONS (Focus on these!)'
            if confirmed_objects:
                highlight_str += f'\n📌 Task objects: {", ".join(confirmed_objects)}'
            highlight_str += f'\n📌 Showing {len(relevant_actions)} relevant actions (out of {len(available_actions)} total):\n'
            
            # Group by action type
            pick_actions = [(aid, desc) for aid, desc in relevant_actions if 'pick up' in desc.lower()]
            nav_actions = [(aid, desc) for aid, desc in relevant_actions if 'navigate' in desc.lower()]
            place_actions = [(aid, desc) for aid, desc in relevant_actions if 'place' in desc.lower()]
            open_actions = [(aid, desc) for aid, desc in relevant_actions if 'open' in desc.lower()]
            close_actions = [(aid, desc) for aid, desc in relevant_actions if 'close' in desc.lower()]
            
            if pick_actions:
                highlight_str += '\n🔹 PICK UP (only task-relevant objects):'
                for action_id, action_desc in pick_actions:
                    highlight_str += f'\n   action id {action_id}: {action_desc}'
            
            if nav_actions:
                highlight_str += '\n🔹 NAVIGATE TO (all locations):'
                for action_id, action_desc in nav_actions[:15]:  # Show first 15
                    highlight_str += f'\n   action id {action_id}: {action_desc}'
                if len(nav_actions) > 15:
                    highlight_str += f'\n   ... and {len(nav_actions)-15} more'
            
            if place_actions:
                highlight_str += '\n🔹 PLACE AT (all locations):'
                for action_id, action_desc in place_actions[:10]:
                    highlight_str += f'\n   action id {action_id}: {action_desc}'
                if len(place_actions) > 10:
                    highlight_str += f'\n   ... and {len(place_actions)-10} more'
            
            if open_actions:
                highlight_str += '\n🔹 OPEN:'
                for action_id, action_desc in open_actions:
                    highlight_str += f'\n   action id {action_id}: {action_desc}'
            
            if close_actions:
                highlight_str += '\n🔹 CLOSE:'
                for action_id, action_desc in close_actions:
                    highlight_str += f'\n   action id {action_id}: {action_desc}'
            
            return highlight_str
        else:
            return ''

    def process_prompt(self, user_instruction, prev_act_feedback=[], wm_context=None, feedback_hints=None):
        user_instruction = user_instruction.rstrip('.')
        
        # 🔥 Get task-relevant actions and build filtered action string
        import re
        confirmed_objects = set()
        
        if wm_context:
            # Extract from "Task Understanding" - pattern: "X IS the Y" or "X IS Y"
            tu_pattern = r'(?:the\s+)?(.+?)\s+(?:IS|is)\s+(?:the\s+)?(\w+)'
            tu_matches = re.findall(tu_pattern, wm_context)
            for phrase, actual_obj in tu_matches:
                obj_clean = actual_obj.strip().lower()
                if len(obj_clean) > 2:
                    confirmed_objects.add(obj_clean)
            
            # Extract from WM nodes: [object_name]
            # 🔥 Support multi-word object names: [red plate], [large gray furniture]
            wm_node_pattern = r'\[\s*([^\]]+?)\s*\]\s*\('
            wm_nodes = re.findall(wm_node_pattern, wm_context)
            for obj in wm_nodes:
                obj_clean = obj.strip().lower()
                if len(obj_clean) > 2 and obj_clean not in ['object', 'type', 'attribute', 'location', 'property']:
                    confirmed_objects.add(obj_clean)
                    # Also add individual words for flexible matching
                    for word in obj_clean.split():
                        if len(word) > 2 and word not in ['the', 'a', 'an', 'of', 'with']:
                            confirmed_objects.add(word)
        
        # 🔥 Filter actions: keep ALL navigate/place/open/close, ONLY relevant pick up
        filtered_actions = []
        for action_id, action_desc in enumerate(self.actions):
            action_lower = action_desc.lower()
            
            # Keep ALL navigation, placement, open, close actions
            if any(keyword in action_lower for keyword in ['navigate to', 'place at', 'open the', 'close the']):
                filtered_actions.append((action_id, action_desc))
                continue
            
            # For "pick up" actions: only keep if object is in confirmed_objects
            if 'pick up' in action_lower:
                if confirmed_objects and any(obj in action_lower for obj in confirmed_objects):
                    filtered_actions.append((action_id, action_desc))
                elif not confirmed_objects:
                    # If no WM context, keep all pick up actions
                    filtered_actions.append((action_id, action_desc))
        
        # 🔥 Build filtered action string
        if filtered_actions and len(filtered_actions) < len(self.actions):
            filtered_action_str = ''
            for action_id, action_desc in filtered_actions:
                filtered_action_str += f'\naction id {action_id}: {action_desc}, '
            
            # Add note about filtering
            action_note = f'\n\n📌 Note: Showing {len(filtered_actions)} task-relevant actions'
            if confirmed_objects:
                action_note += f' (filtered for objects: {", ".join(confirmed_objects)})'
            action_note += f' out of {len(self.actions)} total actions.'
            
            # Use filtered actions
            actions_to_show = filtered_action_str.rstrip(', ')
            max_action_id = len(self.actions) - 1  # Still use full range for validation
        else:
            # No filtering, use all actions
            actions_to_show = self.available_action_str
            action_note = ''
            max_action_id = len(self.actions) - 1
        
        # 🔥 Build action selection guidance
        action_guidance = '''
### ⚠️ ACTION SELECTION RULES (CRITICAL):
1. You MUST select action IDs ONLY from the actions listed above.
2. **OBJECT NAME MATCHING IS CRITICAL**: When you need to "pick up X":
   - Search for action where description contains "pick up X"
   - Use EXACT object name
   
3. **Use Task Understanding**: If instruction says "small red object", check what it IS in Task Understanding.

4. **Double-check**: Verify the action description matches your intent before using the action ID.
'''
        
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(max_action_id, actions_to_show, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(max_action_id, actions_to_show, '')

            # Add filtering note
            if action_note:
                prompt += action_note
            
            # Add action guidance
            prompt += action_guidance

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            if self.language_only:
                prompt += f""" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {max_action_id}) from the available actions to excute.

⚠️ CRITICAL: When outputting action_id, you MUST:
1. First find the EXACT action description you want to execute in the available actions list above
2. Then carefully find the corresponding action id for that EXACT description
3. Double-check: Does "action id X" match your intended action description?
"""
            else:
                prompt += f""" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {max_action_id}) from the available actions to excute.

⚠️ CRITICAL: When outputting action_id, you MUST:
1. First find the EXACT action description you want to execute in the available actions list above
2. Then carefully find the corresponding action id for that EXACT description
3. Double-check: Does "action id X" match your intended action description?
"""
        
        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            # Add action guidance
            prompt += action_guidance

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {max_action_id}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {max_action_id}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(max_action_id, actions_to_show, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(max_action_id, actions_to_show, '')
            
            # Add filtering note
            if action_note:
                prompt += action_note
            
            # Add action guidance
            prompt += action_guidance
            
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {max_action_id}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {max_action_id}) from the available actions.'''
        return prompt
    

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image 
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            if self.multistep: # handle multiple images
                ind = int(image_path.split('step_')[-1].strip('.png'))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind +1):
                    temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, stop here')
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        print('found invlid action')
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
        return action

    
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out


    def act(self, observation, user_instruction, wm_context=None, feedback_hints=None):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(
            user_instruction, 
            prev_act_feedback=self.episode_act_feedback,
            wm_context=wm_context,
            feedback_hints=feedback_hints
        )
        # some models do not support json scheme, add style into prompt
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name or self.model_type == 'custom':
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])


        

