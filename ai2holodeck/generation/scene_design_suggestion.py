import os
import json
import re
import traceback
from copy import deepcopy
from langchain import PromptTemplate, OpenAI
from colorama import Fore

import ai2holodeck.generation.prompts as prompts
from ai2holodeck.constants import ABS_PATH_OF_HOLODECK


# Analyze model and give scene design suggestions
class DesignSuggestionGenerator:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        
        self.scene_design_template = PromptTemplate(
            input_variables=["dataset", "evaluation_data"],
            template=prompts.scene_design_prompt
        )

        with open(os.path.join(ABS_PATH_OF_HOLODECK, 'generation/evaluation_metric.json')) as fp:
            self.metrics = json.load(fp)

    def generate(self, evaluation_data: dict):
        metrics = deepcopy(self.metrics)

        filtered_metrics = []
        for metric in metrics['metrics']:
            metric_name = metric['name']
            metric_value = evaluation_data.get(metric_name)
            if metric_value is not None:
                metric['value'] = metric_value
                filtered_metrics.append(metric)
            else:
                print(f"{Fore.RED}scene_design_suggestion.py: "
                      f"DesignSuggestionGenerator.generate_design_suggestion: "
                      f"metric '{metric_name}' not in evalution_data, removed.{Fore.RESET}")
        
        metrics['metrics'] = filtered_metrics

        scene_design_prompt = self.scene_design_template.format(
            dataset="HM3D", evaluation_data=json.dumps(metrics, indent=4)
        )

        counter = 0
        while counter < 5:
            try:
                response = self.llm(scene_design_prompt)
                response = re.sub("```(?:json)?\n?|```", "", response).strip()

                resp_json = json.loads(response)
                resp_json = self._check_json(resp_json)       

                output = self._parse_response(resp_json)

                print(f'User: {scene_design_prompt}')
                print(f'{Fore.GREEN}AI: \n {response}{Fore.RESET}')

                break

            except Exception as e:
                print(f'{Fore.RED}scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator.generate_design_suggestion: '
                      f'bad response from LLM: {e}.{Fore.RESET}')
                traceback.print_exc()
                counter += 1
        
        if counter == 5:
            raise Exception("can't get valid response from LLM")
        
        return output
    
    def _check_json(self, js: dict):
        if 'Model Analysis' not in js:
            raise KeyError("missing required key 'Model Analysis'")
        
        if 'Scenes' not in js:
            raise KeyError("missing required key 'Scenes'")

        output = {
            'Model Analysis': js['Model Analysis'],
            'Scenes': []
        }
        
        for scene in js['Scenes']:
            if (
                not (
                    'Scene Type' in scene
                    and 'Floor Plan Design Suggestions' in scene
                    and 'Object Layout Design Suggestions' in scene
                )
            ):
                print(f'{Fore.RED}scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator._check_json: '
                      f'missing required key(s) in a scene design, removed.{Fore.RESET}')
                continue

            output['Scenes'].append(scene)
        
        if len(output['Scenes']) == 0:
            raise ValueError('no scene design suggestion in response json')
        
        return output

    def _parse_response(self, response_json):
        output = {
            'model_analysis': response_json['Model Analysis'],
            'scene_design': []
        }

        for scene in response_json['Scenes']:
            output['scene_design'].append({
                'scene_type': scene['Scene Type'],
                'floor_plan': scene['Floor Plan Design Suggestions'],
                'object_layout': scene['Object Layout Design Suggestions']
            })
        
        return output
