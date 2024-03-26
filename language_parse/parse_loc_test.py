'''
从句法分析的结果中解析出物体位置
'''
import nltk
from nltk.tree import Tree
import json
from tqdm import tqdm

import sys
# sys.path.append('/home/cyw/task_planning/prompter_v2/')
import alfred_utils.gen.constants as constants
from improve_search_v4.language_parse.parse_loc_util import *

SPLIT_PATH = "alfred_data_small/splits/oct21.json"
# other_name_map = {'chair':"ArmChair", 'bathtub':'BathtubBasin','table':['CoffeeTable','DiningTable','SideTable'],'sink':'SinkBasin','stove':'StoveBurner','tv':'TVStand','closet':'Cabinet','counter':'CounterTop','soap':'SoapBar','phone':'CellPhone','salt':'SaltShaker','shaker':['SaltShaker','PepperShaker'],'lamp':['FloorLamp','DeskLamp'],'cell':'CellPhone','computer':'Laptop','coffeemaker':'CoffeeMachine','cupboard':'Cabinet','bottle':['Glassbottle','SoapBottle'],'refrigerator':'Fridge','disc':"CD",'keys':'KeyChain','shelving':"Shelf",'trashbin':'GarbageCan','garbagebin':'GarbageCan','remote':'RemoteControl','couch':'Sofa'}

# # debug
# # 不用记录的未匹配到的名词
# drop_noun = ['front','left','corner','room','right','end','kitchen','kitchenisland','side','top','the']
# noun_unmatch = []

debug_data_path = "improve_search_v4/language_parse/debug_data/"
# get_noun_phrase_log_file = debug_data_path + "get_noun_phrase_log.txt"
# pp_log_file = debug_data_path + "pp_log.txt"
np_log_file = debug_data_path + "np_log.txt"
# sbar_log_file = debug_data_path + "sbar_log.txt"
# vp_log_file = debug_data_path + "vp_log.txt"
# noun_map_log_file = debug_data_path + "noun_map_log.json"

# parse_file = "/home/cyw/task_planning/prompter_v2/improve_search_v4/language_parse/data/valid_unseen_low1_parse_dict.json"
parse_file = 'improve_search_v4/language_parse/data/valid_unseen_high_parse_dict.json'
with open(parse_file, 'r') as f:
    parse_string_s = json.load(f)
parse_results = {}
parse_tree_s_keys = list(parse_string_s.keys())
for i,parse_tree_string in tqdm(enumerate(parse_string_s.values())):
    # if i== 98:
    if parse_tree_s_keys[i] == 'place a chilled tomato in a microwave':
        print('debug')

    # 调用封装好的函数
    loc_info = get_loc_info_parse_string(parse_tree_string)
    # 按照readme.md的几种模板匹配
    parse_results[parse_tree_s_keys[i]] = loc_info

# 保存结果
parse_results_file = debug_data_path+"parse_results_high.json"
with open(parse_results_file, "w") as f:
    json.dump(parse_results, f,indent=2)



