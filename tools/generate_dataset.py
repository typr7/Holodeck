import os
import uuid

from typing import List, Dict

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner


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

def invert_x(pos: List[float]) -> List[float]:
    return [-pos[0], pos[1], pos[2]]

def create_object_info_list_from_scene(scene_json) -> List:
    object_info_list = list()

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

    # controller for export scene
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

    # object info
    object_info_list = create_object_info_list_from_scene(scene_json)

    # episodes
    episode_list = create_episode_list_from_scene(scene_json)