'''
读取语言特征，形成单独的txt文件，供stanford parser进行句法分析
'''
import argparse
import json
import os
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")
from improve_search_v4.language_parse.utils import *
from improve_search_v4.language_parse.parse_loc_util import rewrite_sentence
from tqdm import tqdm
import re

SPLIT_PATH = "alfred_data_small/splits/oct21.json"
DATA_PATH = "improve_search_v4/language_parse/data"

def get_language(split,language):
    with open(SPLIT_PATH,'r') as f:
        split_data = json.load(f)[split]
    content = []
    for item in tqdm(split_data):
        repeat_idx, task_num = item["repeat_idx"],item["task"]
        traj = get_traj(task_num,repeat_idx)
        if language == "high":
            lan_feature = get_high_language(traj)
        else:
            lan_feature = get_low_languages(traj)
        lan_feature = rewrite_sentence(lan_feature,language)
        content.append(lan_feature)
    save_name = "lan_"+split+"_"+language+".txt" #NOTE 这里取了一个别名，避免覆盖掉之前的文件
    with open(os.path.join(DATA_PATH,save_name),'w') as f:
        for sentence in content:
            f.write(sentence)
            f.write("\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type=str,default=None)
    parser.add_argument('--language',type=str,default=None)
    args = parser.parse_args()
    get_language(args.split,args.language)
