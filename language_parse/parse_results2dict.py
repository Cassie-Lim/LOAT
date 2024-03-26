'''
将解析结果重写为字典
'''
import argparse
import json
import sys
sys.path.append('/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/')
from improve_search_v4.language_parse.utils import *
import os
from tqdm import tqdm
from nltk.tree import Tree
from improve_search_v4.language_parse.parse_loc_util import rewrite_parse_result_sentence

def main(args):
    with open(args.parse_txt_file,'r') as f:
        parse_results = f.readlines()
    parse_results_dict = {}
    for parse_string in tqdm(parse_results):
        parse_tree = Tree.fromstring(parse_string)
        parse_sentence = rewrite_parse_result_sentence(parse_tree)
        parse_results_dict[parse_sentence] = parse_string
    file_path = args.parse_txt_file.split('/')[:-1]
    file_path = '/'.join(file_path)
    file_name = args.parse_txt_file.split('/')[-1].split(".")[0]
    file_name = file_name + "_dict.json"
    with open(os.path.join(file_path, file_name), 'w') as f:
        json.dump(parse_results_dict, f,indent=2)
    print(f'save file in {os.path.join(file_path, file_name)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_txt_file',type=str,default= None)
    args = parser.parse_args()
    main(args)

    

