import json
import sys
sys.path.append('../')
from alfred_utils.gen import constants
collide_attribute = json.load(open("llm_collide.json", "r"))
contain_attribute = json.load(open("llm_container.json", "r"))
nearby_attribute = json.load(open("llm_nearby_full.json", "r"))
object_attribute = {}
for o in constants.map_all_objects:
    object_attribute[o] = {}
    if o in collide_attribute["collide_safe_objects"]:
        object_attribute[o]["collide_safe"] = True
    else:
        object_attribute[o]["collide_safe"] = False
    object_attribute[o]["nearby_objects"] = sorted(list(set(nearby_attribute[o]).intersection(set(constants.OBJECTS))))
    object_attribute[o]["containers"] = []

for c in constants.OPENABLE_CLASS_SET - {'Box'}:
    for o in contain_attribute[c]:
        if o in constants.map_all_objects:
            object_attribute[o]["containers"].append(c)

with open("llm_attr.json","w") as f:
    f.write(json.dumps(object_attribute, indent=4))    