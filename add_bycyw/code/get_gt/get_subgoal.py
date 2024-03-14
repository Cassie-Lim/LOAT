'''
使用prompter代码生成语言指导
1. 读取轨迹
2. 根据轨迹读取参数
3. 根据参数套用模板生成high_level action
4. 保存为json格式，保存轨迹的必要信息
'''
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")

from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, get_arguments, get_arguments_test
import json
from arguments import get_args
import os
import pickle
from tqdm import tqdm

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

def get_task_argument(args,data_path):
    '''
    批量获得任务参数
    '''
    split_file = "alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[args.eval_split]
    task_argument_list = []
    split_name = args.eval_split
    unseen = 'unseen' in args.eval_split
    high_test_dict = read_test_dict(split_name,"high",unseen)
    high_low_test_dict = read_test_dict(split_name,"high_low",unseen)
    for i, r_and_task in tqdm(enumerate(r_and_tasks)):
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        #  json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))

        high_level_lang = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        low_level_lang = traj_data['turk_annotations']['anns'][repeat_idx]['high_descs']

        high_arg = get_arguments_test(high_test_dict,traj_data)[1:]
        low_arg = get_arguments_test(high_low_test_dict,traj_data)[1:]

        if high_arg == low_arg:
            high_eq_low = True
        else:
            high_eq_low = False
        
        if repeat_idx == 0:
            unique = 1
        else:
            unique = -1
            for j in range(1,repeat_idx+1):
                if low_arg == task_argument_list[i-j]['low_arg']:
                    unique = task_argument_list[i-j]['unique']
            if unique == -1:
                unique = task_argument_list[i-j]["unique"]+1
        
        # 记录
        temp = {"id":i,'task_description':high_level_lang,"step by step instruction":low_level_lang,"high_arg":high_arg,'low_arg':low_arg,"high arg equall low arg": high_eq_low,"unique":unique}
        task_argument_list.append(temp)

    # 保存文件
    file_name = "task_arg_"+args.eval_split+".json"
    json.dump(task_argument_list,open(data_path+"/"+file_name,"w"),indent=4)           


def get_subgoal_from_prompter(args,data_path,gt_subgoal = False):
    '''
        按照prompter的方式生成subgoal二元组，用来判断各个失败任务是在哪一步失败，如果指定了gt_subgoal，则使用pddl给出的subgoal
        data_path:文件要保存的位置
    '''
    split_file = "alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[args.eval_split]
    subgoal_tuples_list = []
    
    # 如果是test文件,形成prompter生成的文件
    test_dict = read_test_dict(
        args.eval_split, args.language_granularity, 'unseen' in args.eval_split)
    # 这个主要用来生成test文件的任务类型和任务参数
        

    for i, r_and_task in enumerate(r_and_tasks):
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        #  json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))

        high_level_lang = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        # print(f"high level languge is {high_level_lang}")
        
        if not gt_subgoal:
            # test_dict里面没有train数据，可能还需要读取ground truth 任务类型和任务参数啥的
            # 搁置
            list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(
                traj_data, test_dict, args.nonsliced)
            
        else:
            if "test" in args.eval_split:
                
                print(f"high level languge is {high_level_lang}")
                list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(
                    traj_data, test_dict, args.nonsliced)
            else:
                # 如果是valid or train使用PDDL的文件
                list_of_actions = []
                plan = traj_data['plan']['high_pddl']
                for sub_goal in plan:
                    action = sub_goal["discrete_action"]["action"]
                    # if action == "GotoLocation":
                    #     continue
                    target = sub_goal["discrete_action"]["args"]
                    list_of_actions.append([target,action])
        # print(f"list subgoal is {list_of_actions}")
        subgoal_tuples_list.append({"high level instruction":high_level_lang,"subgoal list": list_of_actions})
        if gt_subgoal:
            source = "PDDLsubgoal"
        else:
            source = "Prompter"
        data_file = data_path + f"subgoal_tuple_prompter_{args.eval_split}_{source}.json"
    json.dump(subgoal_tuples_list, open(data_file,'w'),indent=4)


def compare_task_argument(args,data_save_path):
    '''
    按照轨迹在split中的顺序，逐一对比真实的参数和预测的参数的不同
    '''
    split_file = "alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[args.eval_split]
    not_matched_list = []
    matched_list = []

    test_dict = read_test_dict(
        args.eval_split, args.language_granularity, 'unseen' in args.eval_split)
        # 这个主要用来生成test文件的任务类型和任务参数

    for i, r_and_task in enumerate(r_and_tasks):
        error_mode = ""
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))
        language_goal_instr, task_type_gt, mrecep_target_gt, object_target_gt, parent_target_gt, sliced_gt = \
            get_arguments(traj_data)
        
        instruction, task_type_pred, mrecep_target_pred, object_target_pred, parent_target_pred, sliced_pred = \
            get_arguments_test(test_dict,traj_data)

        if task_type_gt != task_type_pred:
            error_mode +="task_type_error"
        if mrecep_target_gt != mrecep_target_pred:
            error_mode += "mrecep_target_error"
        if object_target_gt != object_target_pred:
            error_mode+="object_target_error"
        if parent_target_gt != parent_target_pred:
            error_mode+="parent_target_error"
        if sliced_gt != sliced_pred:
            error_mode+="sliced_error"

        if not len(error_mode) == 0:
            row_argument = {"task_type":task_type_gt,"mrecep_target":mrecep_target_gt,"object_target":object_target_gt,"parent_target":parent_target_gt,"sliced":sliced_gt}
            pred_argument = {"task_type":task_type_pred,"mrecep_target":mrecep_target_pred,"object_target":object_target_pred,"parent_target":parent_target_pred,"sliced":sliced_pred}
            not_matched_list.append({"id":i,"language_goal_instr":language_goal_instr,"error_mode":error_mode,"row argument":row_argument,"pred argument":pred_argument})
        
        else:
            row_argument = {"task_type":task_type_gt,"mrecep_target":mrecep_target_gt,"object_target":object_target_gt,"parent_target":parent_target_gt,"sliced":sliced_gt}
            pred_argument = {"task_type":task_type_pred,"mrecep_target":mrecep_target_pred,"object_target":object_target_pred,"parent_target":parent_target_pred,"sliced":sliced_pred}
            matched_list.append({"id":i,"language_goal_instr":language_goal_instr,"row argument":row_argument,"pred argument":pred_argument})
        
    # save json file
    if not len(not_matched_list) == 0:
        json_path = os.path.join(data_save_path, f"{args.eval_split}_not_matched_list.json")
        json.dump(not_matched_list,open(json_path,"w"),indent=4)

    json_path = os.path.join(data_save_path, f"{args.eval_split}_matched_list.json")
    json.dump(matched_list,open(json_path,"w"),indent=4)


def get_gt_argument(args,data_save_path):
    '''
    读取轨迹文件，获取真实的任务参数
    '''
    data = []
    split_file = "/home/cyw/task_planning/prompter/alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[args.eval_split]
    for i, r_and_task in enumerate(r_and_tasks):
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))
        language_goal_instr, task_type_gt, mrecep_target_gt, object_target_gt, parent_target_gt, sliced_gt = \
            get_arguments(traj_data)
        gt_argument = {"task_type":task_type_gt,"mrecep_target":mrecep_target_gt,"object_target":object_target_gt,"parent_target":parent_target_gt,"sliced":sliced_gt}
        data.append({"id":i,"language_goal_instr":language_goal_instr,"row argument":gt_argument})
    json_path = os.path.join(data_save_path, f"{args.eval_split}_gt_argument.json")
    json.dump(data,open(json_path,"w"),indent=4)

def get_goal_description(eval_split,data_save_path):
    '''
    获取每个任务的description
    '''
    data = []
    split_file = "alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[eval_split]
    for i, r_and_task in enumerate(r_and_tasks):
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))
        try:
            r_idx = traj_data['repeat_idx']
        except:
            r_idx = 0
        language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        data.append({"id":i,"language_goal_instr":language_goal_instr})
    json_path = os.path.join(data_save_path, f"{eval_split}_id_instruction.json")
    json.dump(data,open(json_path,"w"),indent=4)


# 对比一下llama二元组计算的任务参数和真实任务参数的不同，按照prompter生成读取轨迹的方式测试一下
def test_get_list_of_highlevel_actions_from_plan(args):
    '''
    相应的参数在跑代码的时候设置
    '''
    # 读取相关文件
    if args.run_idx_file is None:
        episode_nos = range(args.from_idx,args.to_idx)
    else:
        episode_nos = json.load(open(args.run_idx_file, 'r'))
    split_file = "alfred_data_small/splits/oct21.json"
    r_and_tasks = json.load(open(split_file,'r'))[args.eval_split]
    test_dict = read_test_dict(
        args.eval_split, args.language_granularity, 'unseen' in args.eval_split)
    data_path = "add_byme/data/compare_parameter/"
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)
    # plan = json.load(open(data_path+args.subgoal_file,'r'))
    plan = json.load(open(args.subgoal_file,'r'))

    not_matched_list = []
    cant_found_info = []
    
    for id,ep_num in enumerate(episode_nos):
        message = ""
        r_and_task = r_and_tasks[ep_num]
        repeat_idx, task = r_and_task["repeat_idx"],r_and_task["task"]
        json_dir = 'alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))

        ## 得到prompter原本的参数
        task_type = get_arguments_test(test_dict, traj_data)[1]
        sliced_prompter = get_arguments_test(test_dict, traj_data)[-1]
        list_of_actions_prompter, categories_in_inst_prompter, second_object_prompter, caution_pointers_prompter = get_list_of_highlevel_actions(
            traj_data,test_dict, args.nonsliced)

        # 按照顺序读取吧，很难匹配
        goal_from_llama = plan[id]["high_level_instructions"]

        # 对比其和goal from traj是不是一样
        # 去除标点符号
        goal_from_llama = goal_from_llama.replace("'","")
        # 去除多余的空格
        goal_from_llama = " ".join(goal_from_llama.split())

        goal_description = " ".join(traj_data['ann']["goal"][:-1])
        # 去除标点符号
        goal_description = goal_description.replace(",", "").replace(".", "").replace("'","")
        # 去除多余的空格
        goal_description = " ".join(goal_description.split())
        # 不分大小写的匹配
        if not goal_description.lower() == goal_from_llama.lower():
            cant_found_info.append({"goal_description_traj":goal_description,"goal_description_llama":\
                goal_from_llama, "episode number":ep_num})
        # plan_idx = plan["high_level_instructions"].index(goal_description)
        list_of_actions_llama = plan[id]["plan"]
        list_of_actions_llama, categories_in_inst_llama, second_object_llama, caution_pointers_llama,sliced_llama = get_list_of_highlevel_actions_from_plan(list_of_actions_llama,args.nonsliced)
        if set(categories_in_inst_llama) != set(categories_in_inst_prompter):
            message +="categories_in_inst unmatched\b"
        if list_of_actions_llama != list_of_actions_prompter:
            message +="list_of_actions unmatched\b"
        if second_object_llama != second_object_prompter:
            message +="second_object unmatched\b"
        if sliced_llama != sliced_prompter:
            message +="sliced unmatched\b"
        if caution_pointers_llama != caution_pointers_prompter:
            message +="caution_pointers unmatched\b"
        if message != "":
            parameter_old = {"list_of_action_prompter":list_of_actions_prompter,"categories_in_inst_prompter":categories_in_inst_prompter,"second_object_prompter":second_object_prompter,"caution_pointers_prompter":caution_pointers_prompter,"sliced_prompter":sliced_prompter}
            parameter_new = {"list_of_action_llama":list_of_actions_llama,"categories_in_inst_llama":categories_in_inst_llama,"second_object_llama":second_object_llama,"caution_pointers_llama":caution_pointers_llama,"sliced_llama":sliced_llama}
            parameter_dict = {"epnum":ep_num, "goal description":goal_description,"messege":message,"parameter_old":parameter_old,"parameter_new":parameter_new}
            not_matched_list.append(parameter_dict)
    # 保存文件
    json.dump(not_matched_list,open(data_path+"compare_"+args.subgoal_file.split("/")[-1].split(".")[0] + "_prompter.json","w"),indent=4)
    json.dump(cant_found_info,open(data_path+"cant_found_log_"+args.subgoal_file.split("/")[-1].split(".")[0] + "_llama.json","w"),indent=4)


            




if __name__ == "__main__":
    args = get_args()

    # 按照prompter的形式得到子目标，通过指定gt_subgoal可以从轨迹中提取子目标
    gt_subgoal = False #是否对train和valid读取ground truth subgoal
    data_path = "add_bycyw/data/language_process/"
    get_task_argument(args,data_path)

    # data_save_path  = "/raid/cyw/Prompter_results/clean_data/compare_gt_bert"
    # data_save_path  = "/raid/cyw/Prompter_results/"

    # # 比较真实的任务参数和预测的任务参数
    # # test文件没有真实的任务参数
    # if not os.path.exists(data_save_path):
    #     os.mkdir(data_save_path)
    # compare_task_argument(args,data_save_path)

    # # 获取真实的任务参数，保存在文件里，供其它函数使用
    # if not os.path.exists(data_save_path):
    #     os.mkdir(data_save_path)
    # get_gt_argument(args,data_save_path)


    # # 比较llama二元组和真实二元组，以及一些自己计算的参数
    # test_get_list_of_highlevel_actions_from_plan(args)

    # 从任务轨迹里提取goal description和id号，提供给llama输出
    # get_goal_description("tests_seen",data_save_path)

    


