# import sys
# sys.path.append('../')
# from alfred_utils.gen import constants
# from ai2thor.controller import Controller
# import json
# # from alfred_utils.env.thor_env_code import ThorEnvCode
# gt_recept = {obj:set() for obj in constants.map_all_objects}
# env = Controller()

# for scene_num in constants.SCENE_NUMBERS:
#     scene_name = 'FloorPlan%d' % scene_num
#     event = env.reset(scene_name)
#     for obj in event.metadata["objects"]:
#         obj_type = obj['objectType']
#         if obj_type in constants.map_all_objects and obj['parentReceptacles'] is not None:
#             for recept in obj['parentReceptacles']:
#                 recept = recept.split("|")[0]
#                 if recept == 'Sink':
#                     recept = 'SinkBasin'
#                 if recept == 'Bathtub':
#                     recept = 'BathtubBasin'

#                 if recept in constants.map_save_large_objects:
#                     gt_recept[obj_type].add(recept)
# for obj in gt_recept.keys():
#     gt_recept[obj] = list(gt_recept[obj])                
# with open("gt_recept.json", 'w', newline='\n') as f:
#  	f.write(json.dumps(gt_recept, indent=4))

import sys
sys.path.append('../')
from alfred_utils.gen import constants
from ai2thor.controller import Controller
import json
# from alfred_utils.env.thor_env_code import ThorEnvCode
gt_recept = {obj:{} for obj in constants.map_all_objects}
seen_objs_cnt = {obj:0 for obj in set(constants.map_save_large_objects+constants.map_all_objects)}
env = Controller()
env.start()
for scene_num in constants.TRAIN_SCENE_NUMBERS:
    scene_name = 'FloorPlan%d' % scene_num
    event = env.reset(scene_name)
    for obj in event.metadata["objects"]:
        obj_type = obj['objectType']
        if obj_type == 'Sink':
            obj_type = 'SinkBasin'
        if obj_type == 'Bathtub':
            obj_type = 'BathtubBasin'
        if obj_type in seen_objs_cnt.keys():
            seen_objs_cnt[obj_type] += 1
        if obj_type in constants.map_all_objects and obj['parentReceptacles'] is not None:
            for recept in obj['parentReceptacles']:
                recept = recept.split("|")[0]
                if recept == 'Sink':
                    recept = 'SinkBasin'
                if recept == 'Bathtub':
                    recept = 'BathtubBasin'

                if recept in constants.map_save_large_objects:
                    if recept in gt_recept[obj_type].keys():
                        gt_recept[obj_type][recept] += 1
                    else:
                        gt_recept[obj_type][recept] = 1
              
with open("gt_recept_train_cnt_orig2.json", 'w', newline='\n') as f:
 	f.write(json.dumps(gt_recept, indent=4))

with open("gt_seen_train_cnt_orig2.json", 'w', newline='\n') as f:
 	f.write(json.dumps(seen_objs_cnt, indent=4))