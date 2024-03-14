'''
读取轨迹的任务类型，房间号，东西是否在容器里等详细信息
'''
from tqdm import tqdm
import json
import argparse

SPLIT_PATH = "alfred_data_small/splits/oct21.json"
OPENABLE_CLASS_LIST = ['fridge', 'cabinet', 'microwave', 'drawer', 'safe', 'box']
# NOTE 这里与constant文件不同，这里都是小写

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

def det_obj_recp(list_of_actions):
    '''
    判断物体是否在容器里 determin whether object in recep
    '''
    # 只要物体在第一次拿起的时候，上一个动作是open，那么它就在容器里
    picked_object = [] #已经拿起过的物体
    inrecp_obj = None
    for i,action in enumerate(list_of_actions):
        if action[1] == 'PickupObject':
            if action[0][0] not in picked_object:
                picked_object.append(action[0][0])
                if i>0 and list_of_actions[i-1][1]=="GotoLocation" and list_of_actions[i-1][0][0] in OPENABLE_CLASS_LIST:
                    inrecp_obj=action[0][0]
                    break
    return inrecp_obj

def read_detial_info(id_file,save_path):
    '''
    读取详细信息
    '''
    split = id_file.split("/")[-1].split("_")[-2:]
    split = "_".join(split).split(".")[0]
    id_data = json.load(open(id_file,'r'))
    split_data = json.load(open(SPLIT_PATH,'r'))[split]
    data = []
    for idx in tqdm(id_data):
        item = split_data[idx]
        repeat_idx, task_num = item["repeat_idx"],item["task"]
        traj = get_traj(task_num,repeat_idx)
        task_type = get_task_type(traj)
        scene_name = get_scene_name(traj)
        pddl_params = get_pddl_params(traj)
        plan = get_ed_subgoal(traj)
        obj_inrecp = det_obj_recp(plan)
        item_new = {
            "id":idx,
            "task_type": task_type,
            "scene_name": scene_name,
            "pddl_params":pddl_params,
            "plan":plan,
            "obj_inrecp":"" if obj_inrecp is None else obj_inrecp+" objrecep is not None"
        }
        data.append(item_new)
    # 保存文件
    file_name = id_file.split("/")[-1].split(".")[0] + "_detail.json"
    json.dump(data,open(save_path+file_name,'w'),indent=4)
    print(f'file has saved in {save_path+file_name}')

def debug(id_file,id = 8):
    split = id_file.split("/")[-1].split("_")[-2:]
    split = "_".join(split).split(".")[0]
    id_data = json.load(open(id_file,'r'))[id]
    id_data = [id_data]
    split_data = json.load(open(SPLIT_PATH,'r'))[split]
    data = []
    for idx in tqdm(id_data):
        item = split_data[idx]
        repeat_idx, task_num = item["repeat_idx"],item["task"]
        traj = get_traj(task_num,repeat_idx)
        task_type = get_task_type(traj)
        scene_name = get_scene_name(traj)
        pddl_params = get_pddl_params(traj)
        plan = get_ed_subgoal(traj)
        obj_inrecp = det_obj_recp(plan)
        item_new = {
            "id":idx,
            "task_type": task_type,
            "scene_name": scene_name,
            "pddl_params":pddl_params,
            "plan":plan,
            "obj_inrecp":"" if obj_inrecp is None else obj_inrecp+" objrecep is not None"
        }
        data.append(item_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_file',type=str,default=None)
    parser.add_argument('--save_path',type=str,default=None)

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = "/".join(args.id_file.split("/")[:-1])+"/"
        
    
    read_detial_info(args.id_file,args.save_path)
    # debug(args.id_file)



    



        
    


    



