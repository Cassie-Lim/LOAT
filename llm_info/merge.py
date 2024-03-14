import json
import sys
sys.path.append('../')
from alfred_utils.gen import constants
nearby_attribute = json.load(open("llm_nearby.json", "r"))
gt_recept = json.load(open("gt_recept.json", "r"))
full_nearby = {}
for obj in constants.map_all_objects:
    full_nearby[obj] = list(set(nearby_attribute[obj])|set(gt_recept[obj]))
    print("*"*40)
    print(obj)
    print(set(nearby_attribute[obj])-set(gt_recept[obj]))
    print(set(gt_recept[obj])-set(nearby_attribute[obj]))

with open("llm_nearby_full.json","w") as f:
    f.write(json.dumps(full_nearby, indent=4))