import os
import json
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

    def generate(self, evalutaion_data: dict):
        metrics = deepcopy(self.metrics)

        filtered_metrics = []
        for metric in metrics['metrics']:
            metric_name = metric['name']
            metric_value = evalutaion_data.get(metric_name)
            if metric_value is not None:
                metric['value'] = metric_value
                filtered_metrics.append(metric)
            else:
                print(f'scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator.generate_design_suggestion: '
                      f'metric {metric_name} not in evalution_data.')
        
        metrics['metrics'] = filtered_metrics

        scene_design_prompt = self.scene_design_template.format(
            dataset="HM3D", evaluation_data=json.dumps(metrics, indent=4)
        )

        counter = 0
        while counter < 5:
            try:
                response = self.llm(scene_design_prompt)

                resp_json = json.loads(response)
                resp_json = self._check_json(resp_json)       

                output = self._parse_response(resp_json)

                print(f'User: {scene_design_prompt}')
                print(f'{Fore.GREEN}AI: \n {response}{Fore.RESET}')

                break

            except Exception as e:
                print(f'scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator.generate_design_suggestion: '
                      f'bad response: {e}')
                counter += 1
        
        if counter == 5:
            raise Exception("can't get valid response json from LLM")
        
        return output
    
    def _check_json(self, js: dict):
        output = {
            'Model Analysis': js['Model Analysis'],
            'Scene': []
        }
        
        for scene in js['Scene']:
            if (
                not (
                    'Scene Type' in scene
                    and 'Floor Plan Design Suggestions' in scene
                    and 'Spatial Layout Design Suggestions' in scene
                )
            ):
                print(f'scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator._check_json: '
                      f'missing required key(s) in a scene design, removed.')
                continue

            output['Scene'].append(scene)
        
        if len(output['Scene']) == 0:
            raise ValueError('no scene design in response json')
        
        return output

    def _parse_response(self, response_json):
        output = {
            'model_analysis': response_json['Model Analysis'],
            'scene_design': []
        }

        for scene in response_json['Scene']:
            output['scene_design'].append({
                'scene_type': scene['Scene Type'],
                'floor_plan': scene['Floor Plan Design Suggestions'],
                'spatial_layout': scene['Spatial Layout Design Suggestions']
            })
        
        return output
