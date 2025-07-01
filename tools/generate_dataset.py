import os
import uuid
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

_hm3d_categories = (
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa"
)

def create_object_info_list_from_scene(scene_json) -> List:
    object_info_list = list()

    invert_x = lambda pos: [-pos[0], pos[1], pos[2]]

    for i, obj in enumerate(scene_json['objects']):
        obj_id = obj['id']
        obj_cls = obj_id.split('-')[0]
        if obj_cls in _hm3d_categories:
            # holodeck 生成的模型由 unity 插件 gltfast 导出，unity 使用左手坐标系，gltf 模型使用右手坐标系，gltfast 在导出时会将 x 坐标取反
            object_position = invert_x(obj['position'])
            object_name = f'{obj_cls}_{i}'
            object_info = create_object_info(object_position, i, object_name, obj_cls)

            object_info_list.append(object_info)
        
    return object_info_list

def create_episode_list_from_scene(scene_json) -> List:
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

def create_training_data(scene_json):
    scene_uuid = uuid.uuid4().hex

    training_data_base_path = os.environ["TRAINING_DATA_BASEPATH"]
    scene_dataset_path = os.path.join(training_data_base_path,
                                      'scene_datasets',
                                      'hm3df')
    scene_dataset_config_path = os.path.join(scene_dataset_path,
                                             'hm3df_annotated_basis.scene_dataset_config.json')
    scene_dir_path = os.path.join(scene_dataset_path,
                                  'train',
                                  scene_uuid)
    scene_path = os.path.join(scene_dir_path,
                              scene_uuid + '.basis.glb')
    scene_navmesh_path = os.path.join(scene_dir_path,
                                      scene_uuid + '.basis.navmesh')
    
    os.makedirs(scene_dir_path, exist_ok=True)

    try:
        generate_glb_scene(scene_json, scene_path)
        generate_scene_navmesh(scene_path, scene_navmesh_path)

        # object info
        object_info_list = create_object_info_list_from_scene(scene_json)

        # episodes
        episode_list = create_episode_list_from_scene(scene_json)
    except Exception as e:
        print(f'{Fore.RED}{__file__}: '
              f'{inspect.currentframe().f_code.co_name}: '
              f'failed to create training data: {e}.{Fore.RESET}')
        traceback.print_exc()

        shutil.rmtree(scene_dir_path)