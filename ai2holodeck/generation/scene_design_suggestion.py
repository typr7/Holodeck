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
        self.output_template = {
            'model_analysis': None,
            'scene_design': [
                {
                    'scene_type': None,
                    'floor_plan': [],
                    'spatial_layout': [],
                }
            ]
        }

        self.llm = llm
        
        self.scene_design_template = PromptTemplate(
            input_variables=["dataset", "evaluation_data"],
            template=prompts.scene_design_prompt
        )

        with open(os.path.join(ABS_PATH_OF_HOLODECK, 'generation/evaluation_metric.json')) as fp:
            self.metrics = json.load(fp)

    def generate_design_suggestion(self, evalutaion_data: dict):
        metrics = deepcopy(self.metrics)
        for metric in metrics['metrics']:
            metric_name = metric['name']
            metric_value = evalutaion_data.get(metric_name)
            if metric_value is not None:
                metric['value'] = metric_value
            else:
                # TODO: if metric in not in evaluation_data, delete it
                print(f'scene_design_suggestion.py: '
                      f'DesignSuggestionGenerator.generate_design_suggestion:'
                      f'metric {metric_name} is not in evalution_data.')

        scene_design_prompt = self.scene_design_template.format(
            dataset="HM3D", 
        )