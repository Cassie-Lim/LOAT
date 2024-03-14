'''
与alfred traj数据交互的一些公用函数
'''
import json
import pickle
from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, get_arguments, get_arguments_test

def get_traj(task,repeat_idx):
    '''
    读取traj_json
    task:任务类型
    repeat_idx:重复编号
    '''
    json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
    traj_data = json.load(open(json_dir))
    return traj_data

def get_task_type(traj):
    '''
    从轨迹信息中读取任务类型，traj:dict
    '''
    return traj['task_type']

def get_scene_name(traj):
    '''
    返回(THOR scene name)
    '''
    return traj['scene']['floor_plan']

def get_pddl_params(traj):
    '''
    读取任务参数
    '''
    pddl_params = traj['pddl_params']
    return pddl_params


def get_ed_subgoal(traj):
    '''
    读取专家示例expert demostration的plan
    '''
    list_of_actions = []
    plan = traj['plan']['high_pddl']
    for sub_goal in plan:
        action = sub_goal["discrete_action"]["action"]
        # if action == "GotoLocation":
        #     continue
        target = sub_goal["discrete_action"]["args"]
        list_of_actions.append([target,action])
    return list_of_actions

def get_high_language(traj):
    '''
    读取high-level语言
    '''
    high_language = traj['ann']['goal']
    return high_language

def get_low_languages(traj):
    '''
    读取low level language
    '''
    low_language = traj['ann']['instr']
    return low_language


# 与预测参数以及traj交互的一些函数
def read_test_dict(split_name, language_granularity, unseen):
    if "train" in split_name:
        split_name = "train"
        split_type = ""
    elif "val" in split_name:
        split_name = "val"
        split_type = "unseen" if unseen else "seen"
    elif "test" in split_name:
        split_name = "test"
        split_type = "unseen" if unseen else "seen"
    

    granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
    granularity = granularity_map[language_granularity]

    if split_name != "train":
        dict_fname = f"models/instructions_processed_LP/instruction2_params_{split_name}_{split_type}_{granularity}.p"
    else:
        dict_fname = f"models/instructions_processed_LP/instruction2_params_{split_name}_{granularity}.p"
    return pickle.load(open(dict_fname, "rb"))

def get_pred_pddl_params(test_dict,traj):
    '''
    获得预测的pddl参数
    lan_granularity:语言粒度，high，hign_low
    '''
    task_type, mrecep_target, object_target, parent_target, sliced = get_arguments_test(test_dict,traj)[1:]
    object_sliced = True if sliced else False
    pddl_params = {"mrecep_target":mrecep_target,"object_sliced":object_sliced,"object_target":object_target,"parent_target":parent_target,}
    return task_type, pddl_params
