'''
读取scene metedata, 获取物体位置关系，只用train数据，在valid数据上验证效果
add_bycyw/code/get_gt/get_obj_loc.py使用多线程有问题，本脚本主要解决这个问题
'''
import os
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/alfred_utils")
import json
from alfred_utils.env.thor_env import ThorEnv
import argparse
from add_bycyw.code.get_gt import constants
from tqdm import tqdm
import multiprocessing as mp
import pickle

SPLIT_PATH = "alfred_data_small/splits/oct21.json"

def get_object_dict(metadata):
    return {obj['objectId']: obj for obj in metadata['objects']}

def get_traj(task,repeat_idx):
    '''
    读取traj_json
    task:任务类型
    repeat_idx:重复编号
    '''
    json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
    traj_data = json.load(open(json_dir))
    return traj_data

def restore_scene_traj(traj_data,env):
    '''
    从轨迹数据里面恢复场景
    '''
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    # print("Performing reset via thor_env API")
    env.reset(scene_name)
    # print("Performing restore via thor_env API")
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    # initialize to start position
    init_dict = dict(traj_data['scene']['init_action'])
    event = env.step(init_dict)
    return scene_name

def get_obj_co_occur(env):
    # 有一点点不好，这个env变量传的太深了
    objects = env.last_event.metadata['objects']
    object_dict = get_object_dict(env.last_event.metadata)
    in_receptacle_ids = {}
    for obj in objects:
        obj_id = obj['objectId']
        if not isinstance(obj['parentReceptacles'], list):
            obj['parentReceptacles'] = [obj['parentReceptacles']]
        for parent in obj['parentReceptacles']:
            if parent is None:
                break

            if parent not in in_receptacle_ids:
                in_receptacle_ids[parent] = set()
            # 改动得和alfred代码有点不一样
            
            parent_obj = object_dict[parent]
            if parent_obj['objectType'] not in constants.RECEPTACLES:
                # Weird corner cases of things that aren't listed as receptacles
                continue
            # TODO: cleanup suffix fix?
            fix_basin = False
            if parent.startswith('Sink') and not parent.endswith('Basin'):
                fix_basin = True
                parent = parent + "|SinkBasin"
            elif parent.startswith('Bathtub') and not parent.endswith('Basin'):
                fix_basin = True
                parent = parent + "|BathtubBasin"

            if fix_basin:
                try:
                    in_receptacle_ids[parent].add(obj_id)
                except KeyError:
                    raise Exception('No object named %s in scene' % (parent))
            else: 
                in_receptacle_ids[parent].add(obj_id)           
    return in_receptacle_ids

def read_obj_occur(in_receptacle_ids):
    '''
    将in_receptacle_ids中的坐标信息去掉，并且将多个容器合并起来记录，如counterTop里面包含所有ConterTop里有的东西
    return :dict{"receptacle",{objects,frequency}}
    '''
    in_recep = {}
    for recep, objs in in_receptacle_ids.items():
        recep = recep.split('|')[0]
        if recep not in in_recep:
            in_recep[recep] = {}
        for obj in objs:
            obj = obj.split('|')[0]
            if obj not in in_recep[recep]:
                in_recep[recep][obj] = 0
            in_recep[recep][obj] += 1
    return in_recep

def get_metadata(env,type = "obj_co_occur"):
    '''
    获取元数据
    type:要获取的元数据的类型, obj_co_occur物体共现关系
    '''
    if type == "obj_co_occur":
        in_receptacle_ids = get_obj_co_occur(env)
        in_recep = read_obj_occur(in_receptacle_ids)
        return in_recep

def suminfo_scene(data,norm_freq=False):
    '''
    对每个场景汇总其信息，目前主要是汇总物体共现信息
    data: {"scene_name":[{"recp":{"obj":frequency}...}...}...]
    return {"scene_name: {"obj":{"recp",freq}}}
    fred:是否要将共现次数转化为共现频率，这会丢失一些信息，默认为false
    ''' 
    scene_data = {}
    for scene_name, scene_infos in data.items():
        scene_data[scene_name] = {}
        for scene_info in scene_infos:
            for recp, obj_freq in scene_info.items():
                for obj, freq in obj_freq.items():
                    if obj not in scene_data[scene_name]:
                        scene_data[scene_name][obj] = {}
                    if recp not in scene_data[scene_name][obj]:
                        scene_data[scene_name][obj][recp] = 0
                    scene_data[scene_name][obj][recp] += freq
        # 对freq进行归一化
        if norm_freq:
            for obj, recp_freq in scene_data[scene_name].items():
                scene_data[scene_name][obj] = {k: v / sum(recp_freq.values()) for k, v in
                                                recp_freq.items()}
    return scene_data

def norm_freq_scene(data:dict):
    '''
    对每个场景的物体共现信息进行归一化
    data：{scene_name:{obj:{recp:counts}}}
    return : {scene_name:{obj:{recp:freq}}}
    '''
    data_new = {}
    for scene_name, obj_locs in data.items():
        data_new[scene_name] = {}
        for obj, recp_freq in obj_locs.items():
            data_new[scene_name][obj] = {k: v / sum(recp_freq.values()) for k, v in
                                          recp_freq.items()}
    return data_new

def norm_freq(data):
    '''
    对所有场景的物体共现频率进行归一化
    data: {scene_name:{obj:{recp:counts}}}
    return:{obj:{recep:frequency}}
    '''
    occur_freq = {}
    for scene_name, obj_locs in data.items():
        for obj, recp_freq in obj_locs.items():
            if obj not in occur_freq:
                occur_freq[obj] = {}
            for recp, counts in recp_freq.items():
                if recp not in occur_freq[obj]:
                    occur_freq[obj][recp] = 0
                occur_freq[obj][recp] += counts
    # 归一化
    for obj, recp_freq in occur_freq.items():
        occur_freq[obj] = {k: v / sum(recp_freq.values()) for k, v in
                           recp_freq.items()}
    return occur_freq

def sort_data(data):
    '''
    将data中的obj的recp按照fredqucy排序，程序自动判断data中是否带场景信息
    '''
    if "FloorPlan17" in data:
        for scene, obj_locs in data.items():
            for obj, recp_freq in obj_locs.items():
                data[scene][obj] = {k: v for k, v in sorted(recp_freq.items(), key=lambda item:
                                                             item[1], reverse=True)}
    else:
        for obj, recp_freq in data.items():
            data[obj] = {k: v for k, v in sorted(recp_freq.items(), key=lambda item: item[1],reverse=True)}
    return data
    # 这样写代码好像会改变data本身，不过也不重要了

def debug(args):
    # split = args.split
    # save_path = args.save_path
    # import pickle
    # results = pickle.load(open("add_bycyw/data/obj_occur_4threads/train_results.pkl", "rb"))
    # # 将结果合并起来
    # obj_occur_scene = gather_data(results)
    # obj_cocount_scene = suminfo_scene(obj_occur_scene)
    # obj_cofreq_scene = norm_freq_scene(obj_cocount_scene)
    # obj_cofreq = norm_freq(obj_cocount_scene)
    # # 保存文件
    # name = split + "_objcocount_scene.json"
    # json.dump(obj_cocount_scene, open(save_path + name, "w"),indent=4)
    # print("save to {}".format(save_path + name))

    # name = split + "_objcofreq_scene.json"
    # json.dump(obj_cofreq_scene, open(save_path + name, "w"),indent=4)
    # print("save to {}".format(save_path + name))

    # name = split + "_objcofreq.json"
    # json.dump(obj_cofreq, open(save_path + name, "w"),indent=4)
    # print("save to {}".format(save_path + name))

    file = "add_bycyw/data/obj_occur_4threads/train_objcofreq_scene.json"
    data = json.load(open(file, "r"))
    sorted_data = sort_data(data)
    json.dump(sorted_data, open(file, "w"),indent=4)
    file = "add_bycyw/data/obj_occur_4threads/train_objcofreq.json"
    data = json.load(open(file, "r"))
    sorted_data = sort_data(data)
    json.dump(sorted_data, open(file, "w"),indent=4)

def get_occur_fromdata(split_data):
    env = ThorEnv()
    obj_occur_scene = {} #各个场景下的物体共现频率
    for task in tqdm(split_data):
        repeat_idx, task_type = task["repeat_idx"],task["task"]
        traj_data = get_traj(task=task_type, repeat_idx=repeat_idx)
        scene_name = restore_scene_traj(traj_data,env)
        in_recep = get_metadata(env,type = "obj_co_occur")
        # 这里写得不好，如果一开始获得的就是{"obj":{"recep"}},后面就不用这么麻烦地调过来
        if scene_name not in obj_occur_scene:
            obj_occur_scene[scene_name] = []
        obj_occur_scene[scene_name].append(in_recep)
    return obj_occur_scene

def main(args,split_data):
    split = args.split
    save_path = args.save_path
    obj_occur_scene = get_occur_fromdata(split_data)
    obj_cocount_scene = suminfo_scene(obj_occur_scene)
    obj_cofreq_scene = norm_freq_scene(obj_cocount_scene)
    obj_cofreq = norm_freq(obj_cocount_scene)
    # 保存文件
    name = split + "_objcocount_scene.json"
    json.dump(obj_cocount_scene, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))

    name = split + "_objcofreq_scene.json"
    json.dump(obj_cofreq_scene, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))

    name = split + "_objcofreq.json"
    json.dump(obj_cofreq, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))

def gather_data(results):
    '''
    将多线程各自的结果聚集起来
    results:[{scene_name:[{recp:{obj,counts;...}...},{}...]...}]
    return: {"scene_name":[{"recp":{"obj":frequency}...}...}...]
    只要把results外面那个[]去掉就可以
    '''
    data_new = {}
    for result in results:
        for scene_name,recp_objs in result.items():
            if scene_name not in data_new:
                data_new[scene_name] = recp_objs
            else:
                data_new[scene_name] +=recp_objs
    return data_new



def parallel_main(args,split_data):
    '''
    并行处理数据
    '''
    split = args.split
    num_processes = args.num_threads
    save_path = args.save_path

    # 创建进程池
    pool = mp.Pool(processes=num_processes)
    
    # 将数据列表拆分成多个子列表
    chunk_size = len(split_data) // num_processes
    data_chunks = [split_data[i:i+chunk_size] for i in range(0, len(split_data), chunk_size)]
    
    # 在进程池中并行处理数据，并获取返回值
    results = pool.map(get_occur_fromdata, data_chunks)
    
    # 关闭进程池
    pool.close()
    pool.join()

    # # 保存results
    with open(save_path + split + "_results.pkl", "wb") as f:
        pickle.dump(results, f)
    # 为了调试，使用json保存
    # with open(save_path + split + "_results.json", "w") as f:
    #     json.dump(results, f,indent=4)
    
    # 将结果合并起来
    obj_occur_scene = gather_data(results)
    obj_cocount_scene = suminfo_scene(obj_occur_scene)
    obj_cofreq_scene = norm_freq_scene(obj_cocount_scene)
    obj_cofreq = norm_freq(obj_cocount_scene)
    # 保存文件
    name = split + "_objcocount_scene.json"
    json.dump(obj_cocount_scene, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))

    name = split + "_objcofreq_scene.json"
    json.dump(obj_cofreq_scene, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))

    name = split + "_objcofreq.json"
    json.dump(obj_cofreq, open(save_path + name, "w"),indent=4)
    print("save to {}".format(save_path + name))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--save_path', type=str, default='add_bycyw/data/obj_occur_test_4threads/')
    parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
    parser.add_argument("-n", "--num_threads", type=int, default=0, help="number of processes for parallel mode")
    parser.add_argument("--debug", action='store_true', help="debug mode")
    parse_args = parser.parse_args()

    if os.path.exists(parse_args.save_path) == False:
        os.makedirs(parse_args.save_path)

    if parse_args.debug:
        debug(parse_args)
    else:
        # 加载数据
        split_data = json.load(open(SPLIT_PATH))
        split_data = split_data[parse_args.split]
        # 消除多次标注的影响，这里只选取r_idx=0的轨迹复现
        split_data = [item for item in split_data if item['repeat_idx'] == 0]
        if parse_args.in_parallel and parse_args.num_threads > 1:
            parallel_main(parse_args,split_data)
        else:
            main(parse_args,split_data)

    print("over")




