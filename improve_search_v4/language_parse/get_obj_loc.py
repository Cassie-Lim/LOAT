'''
从语言里面解析出target的可能location
1. 解析任务参数（暂时用prompter的结果）
2. 根据上一步的instrest object 匹配language里面的location
3. 给出一个寻找的顺序
4. 从专家轨迹中读取物体真实的位置
5. 对比给出的顺序和专家轨迹数据，评判效率
'''
import sys
# sys.path.append("/home/cyw/task_planning/prompter_v2/")
from improve_search_v4.language_parse.utils import *
import json
import os
import argparse
from tqdm import tqdm
import alfred_utils.gen.constants as constants
from improve_search_v4.language_parse.parse_loc_util import *

# TODO 这里的一些东西是根据真实的任务参数来计算的，之后还需要根据预测的参数来计算

SPLIT_PATH = "alfred_data_small/splits/oct21.json"

occ_file = "add_byme/data/obj_occur_4threads/train_objcofreq.json" #记录物体共现关系的文件
with open(occ_file, 'r') as f:
    occ_data = json.load(f)

large2indx = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5, 'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                    'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15, 'Safe': 16, 'Shelf': 17, 'SideTable': 18, 'SinkBasin': 19, 'Sofa': 20, 'StoveBurner': 21, 'TVStand': 22, 'Toilet': 23}

parse_dict_high_file = 'improve_search_v4/language_parse/data/valid_unseen_high_parse_dict.json'
with open(parse_dict_high_file, 'r') as f:
    parse_dict_high = json.load(f)

parse_dict_low_file = 'improve_search_v4/language_parse/data/valid_unseen_low_parse_dict.json'
with open(parse_dict_low_file, 'r') as f:
    parse_dict_low = json.load(f)

def get_locs_occ(target,occ_data):
    '''
    从物体共现频率中获得搜索序列
    '''
    if target in occ_data:
        search_list = [recp for recp in occ_data[target] if recp in large2indx]
    else:
        search_list = []
    return search_list

def get_loc_plan(target,plan):
    '''
    从plan中获得loc
    '''
    target = target.lower()
    for i,sub_goal in enumerate(plan):
        if target in sub_goal[0] and sub_goal[1]=='PickupObject' and plan[i-1][1]=='GotoLocation' and target not in plan[i-1][0]:
            # debug
            if noun_map(plan[i-1][0][0]) is None:
                print(f'plan location {plan[i-1][0][0]} cant map to alfred name')
            return noun_map(plan[i-1][0][0])
    return []

def sort_locs(lan_locs,occ_locs,drop_locs):
    '''
    对loc排序
    drop_locs:一般就是parent target,str
    '''
    # locs = list(set(lan_locs+occ_locs)- set(drop_locs))
    # NOTE 不要用上面那个通过set的形式，会打乱顺序
    locs = [loc for loc in lan_locs if loc!=drop_locs]
    # locs = [loc for loc in lan_locs]
    open_obj = []
    for item in occ_locs:
        if item not in locs and item != drop_locs:
        # if item not in locs:
            if item not in constants.OPENABLE_CLASS_LIST:
                locs.append(item)
            else:
                open_obj.append(item)
                # 把要开的东西放在最后
    for item in open_obj:
        locs.append(item)
    if drop_locs is not None and len(drop_locs) !=0:
        locs.append(drop_locs) #把要drop的放在最后
    return locs

def get_drop_locs(task_type,pddl_params):
    if task_type == 'pick_heat_then_place_in_recep':
        drop_loc = 'Microwave'
    elif (task_type == "pick_two_obj_and_place" or task_type == 'pick_and_place_simple') and not pddl_params['object_sliced']:
        drop_loc = pddl_params['parent_target']
    elif task_type == 'pick_cool_then_place_in_recep':
        drop_loc = 'Fridge'
    else:
        drop_loc = []
    return drop_loc

def get_search_loc_for_target(target,language_granularity,lan_feature,get_lan_locs_method,drop_parents,task_type,pddl_params,log = True):
    occ_locs = get_locs_occ(target,occ_data)
    if language_granularity == 'high':
        lan_locs = parse_loc_high(lan_feature,get_lan_locs_method,target,parse_dict_high)
    elif language_granularity == 'low':
        lan_locs = parse_loc_low(lan_feature,get_lan_locs_method,target,parse_dict_low)
    elif language_granularity == 'high_low':
        high_loc = parse_loc_high(lan_feature[0],get_lan_locs_method,target,parse_dict_high)
        low_loc = parse_loc_low(lan_feature[1],get_lan_locs_method,target,parse_dict_low)
        # 去除重复的
        lan_locs = []
        for item in high_loc+low_loc:
            if item not in lan_locs:
                lan_locs.append(item)
    else:
        lan_locs = []
    if drop_parents:
        # 应该去除掉的位置
        drop_locs = get_drop_locs(task_type,pddl_params)
    else:
        drop_locs = None
    search_locs = sort_locs(lan_locs,occ_locs,drop_locs)
    if log:
        loc_logs = {"target":target,"lan_locs":lan_locs,"occ_locs":occ_locs,"drop_locs":drop_locs,'search_locs':search_locs}
    else:
        loc_logs = {}
    return search_locs,loc_logs

def get_search_loc(traj,language_granularity,get_lan_locs_method=None,drop_parents=False, analyse= True):
    '''
    得到搜索位置
    language_granularity:要使用的语言粒度：high analyselow high_low high_low or None
    get_lan_locs_method:从语言特征里面得到位置信息的方法
    drop_parents:是否要从搜索序列里面去掉parent
    '''
    # 确认参数合法性
    if language_granularity is not None:
        assert get_lan_locs_method is not None, 'get_lan_locs_method must not be None when language_granularity is not None'
    if analyse:
        analyse_info = {}

    pddl_params = get_pddl_params(traj)
    task_type = get_task_type(traj)
    plan = get_ed_subgoal(traj)
    if language_granularity == "high":
        lan_feature = get_high_language(traj)
    elif language_granularity == 'low':
        lan_feature = get_low_languages(traj)
    elif language_granularity == 'high_low':
        high_feature = get_high_language(traj)
        low_feature = get_low_languages(traj)
        lan_feature = [high_feature, low_feature]
    else:
        lan_feature = ''
    # 对任务参数里面的物体，逐一获得搜索序列
    target = pddl_params["object_target"]
    search_locs,loc_logs = get_search_loc_for_target(target,language_granularity,lan_feature,get_lan_locs_method,drop_parents,task_type,pddl_params,analyse)
    if analyse:
        if language_granularity is not None:
            if language_granularity != 'high_low':
                analyse_info['lan_feature'] = rewrite_sentence(lan_feature,language_granularity)
            else:
                analyse_info['lan_feature'] = rewrite_sentence(lan_feature[0],'high')+"\n"+rewrite_sentence(lan_feature[1],'low')
        else:
            analyse_info['lan_feature'] = ''
                
        analyse_info['pddl_params'] = pddl_params
        plan_loc = get_loc_plan(target,plan)
        # debug
        # if plan_loc == []:
        #     print('debug')

        analyse_info['plan_loc']  = plan_loc
        if plan_loc in search_locs:
            search_index = search_locs.index(plan_loc)
        else:
            search_index = -1
        loc_logs['search_index'] = search_index
        # 计算开东西的次数
        open_times = 0
        for search_loc in search_locs:
            if search_loc == plan_loc:
                break
            if search_loc in constants.OPENABLE_CLASS_LIST:
                open_times += 1
        loc_logs['open_times'] = open_times
        analyse_info['search_info'] = [loc_logs]

    if pddl_params['mrecep_target'] != '':
        target = pddl_params['mrecep_target']
        search_locs,loc_logs = get_search_loc_for_target(target,lan_feature,get_lan_locs_method,drop_parents,task_type,pddl_params,analyse)
        if analyse:
            if plan_loc in search_locs:
                search_index = search_locs.index(plan_loc)
            else:
                search_index = -1
            loc_logs['search_index'] = search_index
            # 计算开东西的次数
            open_times = 0
            for search_loc in search_locs:
                if search_loc == plan_loc:
                    break
                if search_loc in constants.OPENABLE_CLASS_LIST:
                    open_times += 1
            loc_logs['open_times'] = open_times
            analyse_info['search_info'].append(loc_logs)
    return search_locs ,analyse_info

def get_objloc_lan(split,language_granularity,get_lan_locs_method='loose',drop_parents = True,final_results=True,analyse=True,file_name='search'):
    '''
    get object location from language
    '''
    with open(SPLIT_PATH,'r') as f:
        split_data = json.load(f)[split]
    log_info = []
    results = []
    # for idx,item in tqdm(enumerate(split_data[57:58])):
    for idx,item in tqdm(enumerate(split_data)):
        repeat_idx, task_num = item["repeat_idx"],item["task"]
        traj = get_traj(task_num,repeat_idx)
        search_locs ,analyse_info = get_search_loc(traj,language_granularity,get_lan_locs_method=get_lan_locs_method,drop_parents=drop_parents, analyse=analyse)
        if not args.just_save_lan:
            info = {'idx':idx,'analyse_info':analyse_info}
            log_info.append(info)
        elif not final_results:
            analyse_info = {'lan_feature':analyse_info['lan_feature'],"search_info":[{'target':analyse_info["search_info"][i]['target'],'lan_locs':analyse_info["search_info"][i]['lan_locs']} for i in range(len(analyse_info["search_info"]))]}
            info = {'idx':idx,'analyse_info':analyse_info} 
            log_info.append(info)
        if final_results:
            search_locs = {}
            for item in analyse_info["search_info"]:
                search_locs[item['target']] = item['search_locs']
            info = {'idx':idx,'search_locs':search_locs} 
            results.append(info)
    # 保存信息
    if analyse:
        other_string = 'drop' if drop_parents else 'nodrop'
        other_string += 'just_lan' if args.just_save_lan else ''
        lan_str = language_granularity if language_granularity is not None else ''
        result_name = file_name +"_"+split+"_"+lan_str+"_"+get_lan_locs_method +"_"+ other_string+".json"
        if not final_results:
            with open(os.path.join("improve_search_v4/language_parse/results_debug",result_name),'w') as f:
                json.dump(log_info, f,indent=2)
                print(f'save file to {os.path.join("improve_search_v4/language_parse/results_debug",result_name)}')
        else:
            with open(os.path.join("improve_search_v4/language_parse/results",result_name),'w') as f:
                json.dump(results, f,indent=2)
                print(f'save file to {os.path.join("improve_search_v4/language_parse/results",result_name)}')
        if not args.just_save_lan:
            # 统计各个目标，搜索index的差异
            index_count,open_count = count_index(log_info)
            top_3 = sum(list(index_count.values())[:3])
            total_count = sum(index_count.values())
            print(index_count,top_3,total_count,open_count)


def count_index(log_info):
    '''
    计算各个index的数量，及占比
    '''
    index_count = {i:0 for i in range(15)}
    open_count = {i:0 for i in range(4)}
    for item in log_info:
        search_info = item['analyse_info']['search_info']
        # search_info = item['search_info']
        for info in search_info:
            search_index = info['search_index']
            if search_index < 14 and search_index != -1:
                index_count[search_index] += 1
            else:
                index_count[14] += 1
            if search_index <= 5: #只记录前五个开了东西的，后面的话，即使不开，已基本找不到了
                # 记录open times
                open_times = info['open_times']
                if open_times < 4:
                    open_count[open_times] += 1
                else:
                    open_count[3] += 1 #所有超过三次的都归为4这一类
    return index_count,open_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type=str,default=None)
    parser.add_argument('--language',type=str,default=None,choices=['high','low','high_low',None])
    parser.add_argument('--match_method',type=str,default=None,choices=['loose','strict','strict_loose','parse'])
    parser.add_argument('--file_name',type = str,help="the name of results file to save",default = None)
    parser.add_argument('--just_save_lan',action= 'store_true')
    parser.add_argument('--drop_parents',action= 'store_true')
    parser.add_argument('--final_results',action= 'store_true') #最终结果

    args = parser.parse_args()
    get_objloc_lan(args.split,args.language,args.match_method,args.drop_parents,args.final_results,True,args.file_name)
    # get_objloc_lan(split,language_granularity,get_lan_locs_method='loose',drop_parents = True,analyse=True,file_name='search')

    # file = "improve_search_v4/language_parse/results/search_info_valid_unseen_high_combine.json"
    # with open(file, 'r') as f:
    #     log_info = json.load(f)
    # index_times = count_index(log_info)
    # top_3_count = sum(list(index_times.values())[0:3])
    # print(index_times,top_3_count)

    

