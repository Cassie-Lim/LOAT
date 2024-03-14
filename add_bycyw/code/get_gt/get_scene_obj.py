'''
从ai2thor里面读取各个场景里包含的物体，用于分析结果
https://ai2thor.allenai.org/ithor/documentation/objects/object-types
NOTE 已找到更好的方式，代码弃用
'''
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/alfred_utils")
from alfred_utils import gen
from alfred_utils.env.thor_env import ThorEnv
import json
from tqdm import tqdm
from add_bycyw.code.analyse_results.ana_detail_fail_reason import get_traj,get_scene_num


# from ai2thor.controller import Controller
# controller = Controller(scene='FloorPlan1')

# for obj in controller.last_event.metadata["objects"]:
#     print(obj["objectType"])
# ai2thor版本太老了，用不了
def get_scene_object_type(env,scene_name):
    '''
    得到一个场景中的object type
    '''
    env.reset(scene_name)
    object_type = []
    for obj in env.last_event.metadata['objects']:
        object_type.append(obj["objectType"])
    return object_type

def get_scene_name(detail_data):
    '''
    从detail_file里面获取场景名称
    '''
    scene_names = []
    for item in detail_data:
        if item["scene_name"] not in scene_names:
            scene_names.append(item["scene_name"])
    scene_names = sorted(scene_names)
    return scene_names


if __name__ == "__main__":
    env = ThorEnv()
    detail_file = "add_bycyw/results_exp_tests/prompter512_low_gtseg_gtdepth/results/logs/detail_fail_info.json"
    with open(detail_file,'r') as f:
        detail_data = json.load(f)
    scene_names = get_scene_name(detail_data)
    scenes_objects = {}
    for scene_name in tqdm(scene_names):
        object_type = get_scene_object_type(env,scene_name)
        scenes_objects[scene_name] = object_type
    data_path = "add_bycyw/data/scene_data/"
    file_name = "scene_object_type_tests_unseen.json"
    with open(data_path + file_name,'w') as f:
        json.dump(scenes_objects,f,indent=2)
    print("over")