import os
import uuid
import json
import gzip
import shutil
import inspect
import traceback
import habitat_sim.nav

from typing import List, Dict

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from colorama import Fore


def create_object_info(
    position: List[float],
    object_id: int,
    object_name: str,
    object_category: str,
) -> dict:
    return {
        "position": position,
        "radius": None,
        "object_id": object_id,
        "object_name": object_name,
        "object_name_id": None,
        "object_category": object_category,
        "room_id": None,
        "room_name": None,
        "view_points": []
    }

def create_episode(
    episode_id: str,
    scene_id: str,
    scene_dataset_config: str,
    start_position: List[float],
    start_rotation: list[float],
    geodesic_distance: float,
    euclidean_distance: float,
    closest_goal_object_id: int,
    object_category: str
) -> dict:
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "scene_dataset_config": scene_dataset_config,
        "additional_obj_config_paths": [],
        "start_position": start_position,
        "start_rotation": start_rotation,
        "info": {
            "geodesic_distance": geodesic_distance,
            "euclidean_distance": euclidean_distance,
            "closest_goal_object_id": closest_goal_object_id
        },
        "goals": [],
        "start_room": None,
        "shortest_paths": None,
        "object_category": object_category
    }

def create_object_info_list(scene_json, scene_uuid) -> Dict[str, List]:
    hm3d_by_category = {
        "chair": [],
        "bed": [],
        "plant": [],
        "toilet": [],
        "tv_monitor": [],
        "sofa": []
    }

    invert_x = lambda pos: [-pos[0], pos[1], pos[2]]

    for i, obj in enumerate(scene_json['objects']):
        obj_id = obj['id']
        obj_cls = obj_id.split('-')[0]
        if obj_cls in hm3d_by_category.keys():
            # holodeck 生成的模型由 unity 插件 gltfast 导出，unity 使用左手坐标系，gltf 模型使用右手坐标系，gltfast 在导出时会将 x 坐标取反
            object_position = invert_x(obj['position'])
            object_name = f'{obj_cls}_{i}'
            object_info = create_object_info(object_position, i, object_name, obj_cls)

            hm3d_by_category[obj_cls].append(object_info)
        
    ret = dict()
    for cls, obj_info_list in hm3d_by_category.items():
        ret[f'{scene_uuid}.basis.glb_{cls}'] = obj_info_list
    
    return ret

def create_episode_list(scene_json, scene_id, scene_dataset_config_path) -> List:
    pass

def generate_glb_scene(scene_json, scene_path):
    controller = Controller(
        local_executable_path=os.environ["AI2THOR_LOCAL_BUILD_PATH"],
        agentMode="default",
        scene=scene_json,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=OBJATHOR_ASSETS_DIR,
            asset_symlink=True,
            verbose=True
        )
    )

    controller.step(
        dict(action="ExportSceneToGLB", export_path=scene_path, binary=True)
    )
    controller.stop()

def generate_scene_navmesh(scene_path, save_path):
    if not os.path.exists(scene_path):
        raise Exception(f'scene path {scene_path} not exist')

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.AgentConfiguration()

    sim_cfg = habitat_sim.Configuration(sim_cfg, agent_cfg)
    sim = habitat_sim.Simulator(sim_cfg)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    
    success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    if not success:
        raise Exception(f'failed to compute scene navmesh')

    if not sim.pathfinder.save_nav_mesh(save_path):
        raise Exception(f'failed to save scene navmesh')
    sim.close()

def generate_training_json(scene_json, training_json_path, scene_uuid, scene_id, scene_dataset_config_path):
    goals_by_category = create_object_info_list(scene_json)
    episode_list = create_episode_list(scene_json, scene_id, scene_dataset_config_path)

    training_json = {
        "goals_by_category": goals_by_category,
        "episodes": episode_list,
        "category_to_task_category_id": {
            "chair": 0,
            "bed": 1,
            "plant": 2,
            "toilet": 3,
            "tv_monitor": 4,
            "sofa": 5
        },
        "category_to_scene_annotation_category_id": {
            "chair": 0,
            "bed": 1,
            "plant": 2,
            "toilet": 3,
            "tv_monitor": 4,
            "sofa": 5
        }
    }

    with gzip.open(os.path.join(training_json_path, scene_uuid + '.json.gz'), 'w') as fp:
        json.dump(training_json, fp)

def generate_single_training_data(scene_json):
    training_data_base_path = os.environ["TRAINING_DATA_BASEPATH"]

    scene_uuid = uuid.uuid4().hex

    scene_id = os.path.join('hm3df', 'train', scene_uuid, scene_uuid + '.basis.glb')

    scene_dataset_path = os.path.join(training_data_base_path,
                                      'scene_datasets',
                                      'hm3df')
    scene_dir_path = os.path.join(scene_dataset_path,
                                  'train',
                                  scene_uuid)
    scene_path = os.path.join(scene_dir_path, scene_uuid + '.basis.glb')
    scene_navmesh_path = scene_path.removesuffix('.glb') + '.navmesh'
    training_json_path = os.path.join(training_data_base_path,
                                      'datasets', 'hm3df', 'train', 'content')

    scene_dataset_config_path = os.path.join('./data', 'scene_datasets', 'hm3df',
                                             'hm3df_annotated_basis.scene_dataset_config.json')
    
    os.makedirs(scene_dir_path, exist_ok=True)

    try:
        generate_glb_scene(scene_json, scene_path)

        generate_scene_navmesh(scene_path, scene_navmesh_path)

        generate_training_json(scene_json, training_json_path, scene_id, scene_dataset_config_path)

    except Exception as e:
        print(f'{Fore.RED}{__file__}: '
              f'{inspect.currentframe().f_code.co_name}: '
              f'failed to create training data: {e}.{Fore.RESET}')
        traceback.print_exc()

        shutil.rmtree(scene_dir_path)

if __name__ == "__main__":
    scene_json_path = "/home/wu/Documents/Holodeck/data/scenes/Apartment-2025-06-04-15-42-53-471178/Apartment.json"

    with open(scene_json_path, 'r') as fp:
        scene_json = json.load(fp)

    generate_glb_scene(scene_json)