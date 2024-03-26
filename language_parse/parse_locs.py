'''
批量解析物体及其位置
匹配模板：（NP (NN (PP （IN (NN)))即
1. 找到最大的名词短语
2. 找到这个名词短语里面的NN
3. 找到这个名词短语里面的PP
4. 找到这个PP里面的IN
5. 找到这个pp里面的NN
'''
import nltk
from nltk.tree import Tree
import argparse
from tqdm import tqdm
import os
import json
import re
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")
from improve_search_v4.language_parse.utils import *
import alfred_utils.gen.constants as constants
from improve_search_v4.language_parse.parse_loc_util import *


SPLIT_PATH = "alfred_data_small/splits/oct21.json"
with open(SPLIT_PATH,'r') as f:
    data = json.load(f)

DATA_PATH = "improve_search_v4/language_parse/data"

def get_parse_locs(args):
    split_data = data[args.split]
    parse_dict_high_file = f'improve_search_v4/language_parse/data/{args.split}_high_parse_dict.json'
    with open(parse_dict_high_file, 'r') as f:
        parse_dict_high = json.load(f)

    language_granularity = args.language_granularity
    if language_granularity == 'low' or language_granularity == "high_low":
        parse_dict_low_file = f'improve_search_v4/language_parse/data/{args.split}_low_parse_dict.json'
        with open(parse_dict_low_file, 'r') as f:
            parse_dict_low = json.load(f)
    logs = []
    for idx,item in tqdm(enumerate(split_data)):
        repeat_idx, task_num = item["repeat_idx"],item["task"]
        traj = get_traj(task_num,repeat_idx)
        # ************debug
        if idx == 395:
            print("debug")
        # ***********
        if language_granularity == "high":
            lan_feature = get_high_language(traj)
        elif language_granularity == 'low':
            lan_feature = get_low_languages(traj)
        elif language_granularity == 'high_low':
            high_feature = get_high_language(traj)
            low_feature = get_low_languages(traj)
            lan_feature = [high_feature, low_feature]
        if language_granularity == 'high':
            loc_info = get_loc_info(parse_dict_high, lan_feature,'high')
        elif language_granularity == 'low':
            loc_info = get_loc_info(parse_dict_low, lan_feature,'low')
        elif language_granularity == 'high_low':
            loc_info_high = get_loc_info(parse_dict_high, lan_feature[0],'high')
            loc_info_low = get_loc_info(parse_dict_low, lan_feature[1],'low')
            loc_info = union_loc_infos([loc_info_high,loc_info_low])
        
        if language_granularity != 'high_low':
            lan_feature_log = rewrite_sentence(lan_feature,language_granularity)
        else:
            lan_feature_log = rewrite_sentence(lan_feature[0],'high')+"\n"+rewrite_sentence(lan_feature[1],'low')

        item = {'idx':idx,'lan_feature':lan_feature_log,'loc_info':loc_info}
        logs.append(item)
        # if len(logs[0]["loc_info"]["Bowl"])>3:
        #     print("debug")
    # 保存文件
    file_name = "parse_results_"+args.split+"_"+ language_granularity+".json"
    with open(os.path.join(DATA_PATH,file_name), 'w') as f:
        json.dump(logs, f,indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type = str, default = "valid_unseen")
    parser.add_argument('--language_granularity',type = str,default = None)
    args = parser.parse_args()
    get_parse_locs(args)



