import os
import uuid
import json
import math
import gzip
import shutil
import random
import signal
import inspect
import traceback
import statistics
import habitat_sim
import numpy as np

from typing import List, Dict, Tuple
from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from colorama import Fore
from magnum import Vector2, Vector3


class lance_constant:
    HABITAT_DATA_PATH = os.environ["TRAINING_DATA_BASEPATH"]

    DATASET_NAME = 'lance'

    DATASET_PATH = os.path.join(HABITAT_DATA_PATH,
                                'datasets',
                                'objectnav',
                                DATASET_NAME)

    DATASET_TRAINING_JSON_DIR_PATH = os.path.join(DATASET_PATH, 'train', 'content')

    SCENE_DATASET_PATH = os.path.join(HABITAT_DATA_PATH, 'scene_datasets', DATASET_NAME)

    SCENE_DATASET_SCENE_DIR_PATH = os.path.join(SCENE_DATASET_PATH, 'scene')

    SCENE_DATASET_CONFIG_PATH = os.path.join('./data', 'scene_datasets', DATASET_NAME,
                                             f'{DATASET_NAME}.scene_dataset_config.json')

    PROCTHOR_TRAIN_JSON = {
        "episodes": [],
        "category_to_task_category_id": {
            "chair": 0,
            "bed": 1,
            "house_plant": 2,
            "toilet": 3,
            "television": 4,
            "sofa": 5
        },
        "category_to_scene_annotation_category_id": {
            "chair": 0,
            "bed": 1,
            "house_plant": 2,
            "toilet": 3,
            "television": 4,
            "sofa": 5
        }
    }

    SCENE_DATASET_CONFIG_JSON = {
        "stages": {
            "paths": {
                ".glb": [
                    "scene/*/*.glb"
                ]
            }
        },
        "objects": {},
        "scene_instances": {}
    }

def remove_lance_dataset():
    if os.path.exists(lance_constant.DATASET_PATH):
        shutil.rmtree(lance_constant.DATASET_PATH)
    if os.path.exists(lance_constant.SCENE_DATASET_PATH):
        shutil.rmtree(lance_constant.SCENE_DATASET_PATH)

def init_lance_dataset(exists_ok: bool = True):
    if not os.path.exists(lance_constant.HABITAT_DATA_PATH):
        raise FileNotFoundError(f'habitat data path {lance_constant.HABITAT_DATA_PATH} not exist')
    
    lance_dataset_exists = os.path.exists(lance_constant.DATASET_PATH)
    if lance_dataset_exists and exists_ok:
        return

    remove_lance_dataset()
    try:
        os.makedirs(lance_constant.DATASET_TRAINING_JSON_DIR_PATH)
        os.makedirs(lance_constant.SCENE_DATASET_SCENE_DIR_PATH)

        with gzip.open(
            os.path.join(lance_constant.DATASET_PATH, 'train', 'train.json.gz'), 'wt'
        ) as fp:
            json.dump(lance_constant.PROCTHOR_TRAIN_JSON, fp)
        
        with open(
            os.path.join(lance_constant.SCENE_DATASET_PATH, 'lance.scene_dataset_config.json'), 'w'
        ) as fp:
            json.dump(lance_constant.SCENE_DATASET_CONFIG_JSON, fp, indent=4)

    except Exception as e:
        remove_lance_dataset()
        raise e

def create_object_info(
    position: List[float],
    radius: float,
    object_id: int,
    object_name: str,
    object_category: str,
    view_points: List[Dict]
) -> Dict:
    return {
        "position": position,
        "radius": radius,
        "object_id": object_id,
        "object_name": object_name,
        "object_name_id": None,
        "object_category": object_category,
        "room_id": None,
        "room_name": None,
        "view_points": view_points
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
) -> Dict:
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

def create_view_point(
    position: List[float],
    rotation: List[float]
) -> Dict:
    return {
        "agent_state": {
            "position": position,
            "rotation": rotation
        },
        "iou": -1.0
    }

def initialize_simulator(scene_path: str) -> Tuple[habitat_sim.Simulator, habitat_sim.Simulator]:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.AgentConfiguration()

    sim_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True

    agent_cfg = habitat_sim.AgentConfiguration()

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "ray_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    agent_cfg.sensor_specifications = [sensor_spec]

    sim_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    phy_sim = habitat_sim.Simulator(sim_cfg)

    return sim, phy_sim

def get_horizon_height(sim: habitat_sim.Simulator, sample_num):
    ys = list()
    for _ in range(sample_num):
        ys.append(sim.pathfinder.get_random_navigable_point()[1])
    return statistics.mode(ys)

def get_geodesic_distance(start: Vector3, end: Vector3, sim: habitat_sim.Simulator) -> float:
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end

    if not sim.pathfinder.find_path(path):
        return float('inf')
    
    return path.geodesic_distance

def sample_view_points(
    object_position: List[float],
    object_radius: float,
    room_center: List[float],
    sim: habitat_sim.Simulator,
    sampling_height: float = 0.88,
    sampling_distance_range: List[float] = [0.7, 2.0],
    max_sampling_num: int = 400
) -> List[Dict]:
    object_position = Vector3(object_position)
    default_front_vector = np.array([0., 0., -1.0], dtype=np.float32)

    island = sim.pathfinder.get_island(Vector3(room_center))

    candidates = list()
    CANDIDATE_NUM = 1000
    for _ in range(CANDIDATE_NUM):
        candidates.append(sim.pathfinder.get_random_navigable_point(island_index=island))

    euclidean_distance = lambda a, b: np.linalg.norm(np.array(a - b))
    horizon_distance = lambda a, b: np.linalg.norm(np.array([a[0], a[2]]) - np.array([b[0], b[2]]))

    view_points = list()
    i = 0
    for candidate in candidates:
        if not (sampling_distance_range[0]
                <= horizon_distance(object_position, candidate) - object_radius
                <= sampling_distance_range[1]):
            continue

        ray_start_point = candidate + Vector3(0., sampling_height, 0.)
        ray = habitat_sim.geo.Ray(ray_start_point, object_position - ray_start_point)
        hits_info = sim.cast_ray(ray)

        if not hits_info.has_hits():
            continue

        dist_to_hit = euclidean_distance(hits_info.hits[0].point, ray_start_point)
        dist_to_obj = euclidean_distance(object_position, ray_start_point)

        if abs(dist_to_obj - dist_to_hit) > object_radius:
            continue

        direction_vector = np.array(
            Vector3(object_position[0], 0., object_position[2]) - Vector3(candidate[0], 0., candidate[2])
        )
        quat = habitat_sim.utils.common.quat_from_two_vectors(
            default_front_vector, direction_vector
        )

        view_points.append(create_view_point(list(candidate), [quat.x, quat.y, quat.z, quat.w]))

        i += 1
        if i >= max_sampling_num:
            break

    return view_points

"""
    object_position = Vector3(object_position)
    default_front_vector = np.array([0., 0., -1.0], dtype=np.float32)

    view_points = list()

    euclidean_distance = lambda a, b: np.linalg.norm(np.array(a - b))
    horizon_distance = lambda a, b: np.linalg.norm(np.array([a[0], a[2]]) - np.array([b[0], b[2]]))

    i = 0
    failed_count = 0
    while i < max_sampling_num:
        distance = random.uniform(sampling_distance_range[0], sampling_distance_range[1]) + object_radius

        angle = random.uniform(0., 2 * math.pi)

        potential_vp_pos = (Vector3(object_position[0], 0., object_position[2])
                            + Vector3(distance * math.cos(angle), horizon_height, distance * math.sin(angle)))
        
        navigable_point = sim.pathfinder.snap_point(potential_vp_pos)

        if (
            (euclidean_distance(navigable_point, potential_vp_pos) > 0.3)
            or (abs(navigable_point[1] - potential_vp_pos[1]) > 0.1)
            or not (sampling_distance_range[0]
                    <= horizon_distance(navigable_point, object_position) - object_radius
                    <= sampling_distance_range[1])
        ):
            failed_count += 1
            if failed_count >= 200:
                break
            continue

        ray_start_point = navigable_point + Vector3(0., sampling_height, 0.)
        ray = habitat_sim.geo.Ray(ray_start_point, object_position - ray_start_point)
        hits_info = sim.cast_ray(ray)

        if not hits_info.has_hits():
            failed_count += 1
            if failed_count >= 200:
                break
            continue

        dist_to_hit = euclidean_distance(hits_info.hits[0].point, ray_start_point)
        dist_to_obj = euclidean_distance(object_position, ray_start_point)

        if abs(dist_to_obj - dist_to_hit) > object_radius:
            failed_count += 1
            if failed_count >= 200:
                break
            continue

        direction_vector = np.array(Vector3(object_position[0], 0., object_position[2])
                                    - Vector3(navigable_point[0], 0., navigable_point[2]))
        quat = habitat_sim.utils.common.quat_from_two_vectors(
            default_front_vector, direction_vector
        )

        view_points.append(create_view_point(list(navigable_point), [quat.x, quat.y, quat.z, quat.w]))

        i += 1
        failed_count = 0
    
    return view_points
"""

def get_object_radius(object_horizon_box):
    x_diff = abs(object_horizon_box[0][0] - object_horizon_box[2][0]) / 200.0
    y_diff = abs(object_horizon_box[0][1] - object_horizon_box[1][1]) / 200.0
    object_radius = math.sqrt(x_diff**2 + y_diff**2)
    return object_radius

def get_room_id2center(scene_json: Dict, horizon_height: float) -> Dict[str, List[float]]:
    ret = dict()
    for room_info in scene_json['rooms']:
        room_id = room_info['id']
        room_vertices = room_info['vertices']

        max_xz = [float('-inf'), float('-inf')]
        min_xz = [float('inf'), float('inf')]
        for vertex in room_vertices:
            max_xz[0] = max(max_xz[0], vertex[0])
            max_xz[1] = max(max_xz[1], vertex[1])
            min_xz[0] = min(min_xz[0], vertex[0])
            min_xz[1] = min(min_xz[1], vertex[1])
        ret[room_id] = [-((min_xz[0] + max_xz[0]) / 2.0), horizon_height, ((min_xz[1] + max_xz[1]) / 2.0)]
    return ret

def create_goals_by_category(scene_json: Dict,
                             scene_uuid: str,
                             phy_sim: habitat_sim.Simulator) -> Dict[str, List]:
    procthor_by_category = {
        "chair": [],
        "bed": [],
        "house_plant": [],
        "toilet": [],
        "television": [],
        "sofa": []
    }

    horizon_height = get_horizon_height(phy_sim, 2000)
    room_id2center = get_room_id2center(scene_json, horizon_height)

    invert_x = lambda pos: [-pos['x'], pos['y'], pos['z']]

    for i, obj in enumerate(scene_json['floor_objects'] + scene_json['wall_objects']):
        obj_name = obj['object_name']
        obj_cls = obj_name.split('-')[0]
        if obj_cls in procthor_by_category.keys():
            # holodeck 生成的模型由 unity 插件 gltfast 导出，unity 使用左手坐标系，gltf 模型使用右手坐标系，gltfast 在导出时会将 x 坐标取反
            object_position = invert_x(obj['position'])
            object_name = f'{obj_cls}_{i}'
            object_radius = get_object_radius(obj['vertices'])
            view_points = sample_view_points(object_position,
                                             object_radius,
                                             room_id2center[obj['roomId']],
                                             phy_sim)
            print(f'{__file__}: '
                  f'{inspect.currentframe().f_code.co_name}: '
                  f'sampled {len(view_points)} view points of {object_name}.')
            object_info = create_object_info(object_position, object_radius, i, object_name, obj_cls, view_points)

            procthor_by_category[obj_cls].append(object_info)
        
    goals_by_category = dict()
    for cls, obj_info_list in procthor_by_category.items():
        if len(obj_info_list) != 0:
            goals_by_category[f'{scene_uuid}_{cls}'] = obj_info_list
    
    return goals_by_category

def get_start2goal_geodesic_distance(
    start_position: Vector3,
    view_points: List[Dict],
    sim: habitat_sim.Simulator,
    min_geodesic_distance: float = 2.0,
    max_geodestc_distance: float = 15.0
) -> float:
    min_dist = float('inf')
    for view_point in view_points:
        view_position = Vector3(view_point['agent_state']['position'])
        dist = get_geodesic_distance(start_position, view_position, sim)

        if math.isinf(dist) or math.isnan(dist):
            continue

        if (min_geodesic_distance <= dist <= max_geodestc_distance and dist < min_dist):
            min_dist = dist
    
    return min_dist

def create_episode_list(
    goals_by_category: Dict[str, List],
    scene_id: str,
    scene_dataset_config_path: str,
    sim: habitat_sim.Simulator,
    goal_num: int,
    episode_co: int = 1000,
    min_geodesic_distance: float = 2.0,
    max_geodestc_distance: float = 15.0
) -> List[Dict]:
    goal_categories = list(goals_by_category.keys())

    episode_list = list()
    
    episode_num = goal_num * episode_co
    i = 0
    while i < episode_num:
        # get random navigable point
        start_position = sim.pathfinder.get_random_navigable_point()

        goal_category = random.choice(goal_categories)

        closest_goal_info = None
        min_dist = float('inf')

        for goal_info in goals_by_category[goal_category]:
            dist = get_start2goal_geodesic_distance(start_position,
                                                    goal_info['view_points'],
                                                    sim,
                                                    min_geodesic_distance,
                                                    max_geodestc_distance)
            if math.isinf(dist) or math.isnan(dist):
                continue

            if dist < min_dist:
                closest_goal_info = goal_info
                min_dist = dist
        
        if closest_goal_info is None:
            """
            print(f'{Fore.YELLOW}{__file__}: '
                  f'{inspect.currentframe().f_code.co_name}: '
                  f'invalid start position: {start_pos}, skipped.{Fore.RESET}')
            """
            continue

        random_yaw_angle = random.uniform(0., 2 * math.pi)

        start_rotation = [0., math.sin(random_yaw_angle / 2), 0., math.cos(random_yaw_angle / 2)]

        euclidean_distance = np.linalg.norm(np.array(
            start_position - Vector3(closest_goal_info['position'])
        )) - 0.8 * closest_goal_info['radius']

        episode = create_episode(
            episode_id=str(i),
            scene_id=scene_id,
            scene_dataset_config=scene_dataset_config_path,
            start_position=list(start_position),
            start_rotation=start_rotation,
            geodesic_distance=min_dist,
            euclidean_distance=euclidean_distance,
            closest_goal_object_id=closest_goal_info['object_id'],
            object_category=closest_goal_info['object_category']
        )

        episode_list.append(episode)
        i += 1
    
    return episode_list

def generate_glb_scene(scene_json: Dict, scene_path: str):
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
    controller.step(action="ExportSceneToGLB", export_path=scene_path, binary=True)
    controller.stop()

def generate_scene_navmesh(save_path: str,
                           sim: habitat_sim.Simulator,
                           phy_sim: habitat_sim.Simulator,
                           agent_height: float,
                           agent_radius: float):
    # physics-enabled simulator for generating view points
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = 5.0
    navmesh_settings.agent_radius = agent_radius

    while not phy_sim.recompute_navmesh(phy_sim.pathfinder, navmesh_settings):
        navmesh_settings.agent_height -= 0.05

    # normal simulator
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = agent_height
    navmesh_settings.agent_radius = agent_radius
    
    if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings):
        raise Exception(f'failed to compute scene navmesh')

    if not sim.pathfinder.save_nav_mesh(save_path):
        raise Exception(f'failed to save scene navmesh')

def generate_training_json(
    scene_json: Dict,
    training_json_dir_path: str,
    scene_uuid: str,
    scene_id: str,
    scene_dataset_config_path: str,
    sim: habitat_sim.Simulator,
    phy_sim: habitat_sim.Simulator
):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10 * 60)

    goals_by_category = create_goals_by_category(scene_json, scene_uuid, phy_sim)

    if len(goals_by_category) == 0:
        raise ValueError('no goal object')

    episode_list = create_episode_list(goals_by_category,
                                       scene_id,
                                       scene_dataset_config_path,
                                       sim,
                                       len(goals_by_category))
    
    if len(episode_list) == 0:
        raise ValueError('no episode generated')

    training_json = {
        "goals_by_category": goals_by_category,
        "episodes": episode_list,
        "category_to_task_category_id":
            lance_constant.PROCTHOR_TRAIN_JSON['category_to_task_category_id'],
        "category_to_scene_annotation_category_id":
            lance_constant.PROCTHOR_TRAIN_JSON['category_to_scene_annotation_category_id']
    }

    with gzip.open(os.path.join(training_json_dir_path, scene_uuid + '.json.gz'), 'wt') as fp:
        json.dump(training_json, fp)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """
    信号处理函数，在超时时引发 TimeoutException
    """
    raise TimeoutException

def generate_single_training_data(scene_json: Dict, scene_json_name: str):
    scene_uuid = uuid.uuid4().hex

    scene_id = os.path.join(lance_constant.DATASET_NAME, 'scene', scene_uuid, scene_uuid + '.glb')

    scene_dir_path = os.path.join(lance_constant.SCENE_DATASET_SCENE_DIR_PATH, scene_uuid)

    scene_path = os.path.join(scene_dir_path, scene_uuid + '.glb')

    scene_navmesh_path = scene_path.removesuffix('.glb') + '.navmesh'

    os.makedirs(scene_dir_path, exist_ok=True)

    print(f'{__file__}: '
          f'{inspect.currentframe().f_code.co_name}: '
          f'generating training data from scene: {scene_json_name}.')

    sim = None
    phy_sim = None

    try:
        generate_glb_scene(scene_json, scene_path)

        sim, phy_sim = initialize_simulator(scene_path)

        generate_scene_navmesh(scene_navmesh_path, sim, phy_sim, agent_height=0.88, agent_radius=0.18)

        generate_training_json(scene_json,
                               lance_constant.DATASET_TRAINING_JSON_DIR_PATH,
                               scene_uuid,
                               scene_id,
                               lance_constant.SCENE_DATASET_CONFIG_PATH,
                               sim,
                               phy_sim)
        
        sim.close()
        phy_sim.close()
    
    except TimeoutException as e:
        print(f'{Fore.RED}{__file__}: '
              f'{inspect.currentframe().f_code.co_name}: '
              f'time out when generating from scene: {scene_json_name}.{Fore.RESET}')
        traceback.print_exc()

        if sim is not None:
            sim.close()
        if phy_sim is not None:
            phy_sim.close()
        shutil.rmtree(scene_dir_path)

    except Exception as e:
        print(f'{Fore.RED}{__file__}: '
              f'{inspect.currentframe().f_code.co_name}: '
              f'failed to create training data: {e}.{Fore.RESET}')
        traceback.print_exc()

        if sim is not None:
            sim.close()
        if phy_sim is not None:
            phy_sim.close()
        shutil.rmtree(scene_dir_path)
    finally:
        signal.alarm(0)

def generate_training_dataset(scene_json_list: List[Dict]):
    init_lance_dataset()
    
    for scene_json in scene_json_list:
        generate_single_training_data(scene_json)

if __name__ == "__main__":
    scene_json_path_list = list()
    for folder_name in os.listdir('data/scenes'):
        for file_name in os.listdir(os.path.join('data/scenes', folder_name)):
            if file_name.endswith('.json'):
                scene_json_path_list.append(os.path.join('data/scenes', folder_name, file_name))
    
    used_scene_list = list()
    if os.path.exists('tools/used_scene.txt'):
        with open('tools/used_scene.txt', 'r') as fp:
            used_scene_list = fp.readlines()
            used_scene_list = [line[:-1] for line in used_scene_list]
    
    init_lance_dataset(False)
    for scene_json_path in scene_json_path_list:
        if scene_json_path not in used_scene_list:
            with open(scene_json_path, 'r') as fp:
                scene_json = json.load(fp)
            generate_single_training_data(scene_json, scene_json_path)
            with open('tools/used_scene.txt', 'a') as fp:
                fp.write(scene_json_path + '\n')