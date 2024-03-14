#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:49:38 2021

@author: soyeonmin
"""
from code import InteractiveConsole
import pickle
from re import T
import alfred_utils.gen.constants as constants
import string


exclude = set(string.punctuation)
task_type_dict = {2: 'pick_and_place_simple',
 5: 'look_at_obj_in_light',
 1: 'pick_and_place_with_movable_recep',
 3: 'pick_two_obj_and_place',
 6: 'pick_clean_then_place_in_recep',
 4: 'pick_heat_then_place_in_recep',
 0: 'pick_cool_then_place_in_recep'}


def read_test_dict(test, language_granularity, unseen):
    split_name = "test" if test else "val"
    split_type = "unseen" if unseen else "seen"

    granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
    granularity = granularity_map[language_granularity]

    dict_fname = f"models/instructions_processed_LP/instruction2_params_{split_name}_{split_type}_{granularity}.p"
    return pickle.load(open(dict_fname, "rb"))


def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1

def none_or_str(string):
    if string == '':
        return None
    else:
        return string


def cleanInstruction(instruction):
    instruction = instruction.lower()
    instruction = ''.join(ch for ch in instruction if ch not in exclude)
    return instruction


def get_arguments_test(test_dict, traj_data):
    r_idx = traj_data['repeat_idx']
    high_level_lang = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    low_level_lang = traj_data['turk_annotations']['anns'][r_idx]['high_descs']

    instructions = [high_level_lang] + low_level_lang
    instructions = [cleanInstruction(instruction) for instruction in instructions]
    instruction = '[SEP]'.join(instructions)

    task_type, mrecep_target, object_target, parent_target, sliced = \
        test_dict[instruction]['task_type'],  test_dict[instruction]['mrecep_target'], test_dict[instruction]['object_target'], test_dict[instruction]['parent_target'],\
             test_dict[instruction]['sliced']

    if isinstance(task_type, int):
        task_type = task_type_dict[task_type]
    return instruction, task_type, mrecep_target, object_target, parent_target, sliced 


def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['repeat_idx']
    except:
        r_idx = 0
    language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])
    object_target = none_or_str(traj_data['pddl_params']['object_target'])
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])
    #toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])
    
    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced


def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions

def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions)-1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer+1][0]:
            returned = True
            #target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer+1][1] == "CloseObject":
            returned = True
            #target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer-1][0]
        #Faucet
        elif list_of_actions[previous_pointer+1][0] == "Faucet" and list_of_actions[previous_pointer+1][1] in ["ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        #Pick up after faucet 
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        # # ***********************
        # # put object, pick up object, 并且东西相同
        # elif list_of_actions[previous_pointer][1] == "PutObject" and list_of_actions[previous_pointer+1][1] == "PickupObject" and (list_of_actions[previous_pointer][0]==list_of_actions[previous_pointer+1][0]):
        #     returned = True
        #     target_instance = list_of_actions[0][0]
        # # **********************
    return returned, target_instance


def get_list_of_highlevel_actions(traj_data, test_dict=None, args_nonsliced=False):
    language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments_test(test_dict, traj_data)

    #obj_target = 'Tomato'
    #mrecep_target = "Plate"
    if parent_target == "Sink":
        parent_target = "SinkBasin"
    if parent_target == "Bathtub":
        parent_target = "BathtubBasin"
    
    #Change to this after the sliced happens
    if args_nonsliced:
        if sliced == 1:
            obj_target = obj_target +'Sliced'
        #Map sliced as the same place in the map, but like "|SinkBasin" look at the objectid

    
    categories_in_inst = []
    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    #obj_target = "Tomato"

    if sliced == 1:
       list_of_highlevel_actions.append(("Knife", "PickupObject"))
       list_of_highlevel_actions.append((obj_target, "SliceObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions.append(("SinkBasin", "PutObject"))
       categories_in_inst.append(obj_target)
       
    if sliced:
        obj_target = obj_target +'Sliced'

    
    if task_type == 'pick_cool_then_place_in_recep': #0 in new_labels 
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target("Fridge", "PutObject", list_of_highlevel_actions)
       list_of_highlevel_actions.append(("Fridge", "OpenObject"))
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       list_of_highlevel_actions.append(("Fridge", "CloseObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
       categories_in_inst.append(obj_target)
       categories_in_inst.append("Fridge")
       categories_in_inst.append(parent_target)
       
    elif task_type == 'pick_and_place_with_movable_recep': #1 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(mrecep_target, "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append((mrecep_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(mrecep_target)
        categories_in_inst.append(parent_target)
    
    elif task_type == 'pick_and_place_simple':#2 in new_labels 
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        #list_of_highlevel_actions.append((parent_target, "PutObject"))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
    
    elif task_type == 'pick_heat_then_place_in_recep': #4 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target("Microwave", "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOn" ))
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOff" ))
        list_of_highlevel_actions.append(("Microwave", "OpenObject"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        list_of_highlevel_actions.append(("Microwave", "CloseObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("Microwave")
        categories_in_inst.append(parent_target)
        
    elif task_type == 'pick_two_obj_and_place': #3 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        if parent_target in constants.OPENABLE_CLASS_LIST:
            second_object = [False] * 4
        else:
            second_object = [False] * 2
        if sliced:
            second_object = second_object + [False] * 3
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        #caution_pointers.append(len(list_of_highlevel_actions))
        second_object.append(True)
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        second_object.append(False)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
        
    elif task_type == 'look_at_obj_in_light': #5 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        #if toggle_target == "DeskLamp":
        #    print("Original toggle target was DeskLamp")
        toggle_target = "FloorLamp"
        list_of_highlevel_actions.append((toggle_target, "ToggleObjectOn" ))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(toggle_target)
        
    elif task_type == 'pick_clean_then_place_in_recep': #6 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append(("SinkBasin", "PutObject"))  #Sink or SinkBasin? 
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOff"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("SinkBasin")
        categories_in_inst.append("Faucet")
        categories_in_inst.append(parent_target)
    else:
        raise Exception("Task type not one of 0, 1, 2, 3, 4, 5, 6!")

    if sliced == 1:
       if not(parent_target == "SinkBasin"):
            categories_in_inst.append("SinkBasin")
    
    #return [(goal_category, interaction), (goal_category, interaction), ...]
    print("instruction goal is ", language_goal)
    #list_of_highlevel_actions = [ ('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject')]
    #list_of_highlevel_actions = [('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject'), ('Microwave', 'ToggleObjectOn'), ('Microwave', 'ToggleObjectOff'), ('Microwave', 'OpenObject'), ('Apple', 'PickupObject'), ('Microwave', 'CloseObject'), ('Fridge', 'OpenObject'), ('Fridge', 'PutObject'), ('Fridge', 'CloseObject')]
    #categories_in_inst = ['Microwave', 'Fridge']
    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers


# *******************
def get_list_of_highlevel_actions_from_plan(plan, args_nonsliced=False):
    '''
    从plan的二元组中提取categories_in_inst, second_object, caution_pointers参数
    '''
    categories_in_inst = [] #集合
    sliced = 0
    caution_pointers = []
    pick_two_object = False
    # pickupObjects = []
    second_object = []
    list_of_highlevel_actions = []
    open_target = [] #第一次打开某种东西，只有第一次打开的时候需要caution
    slice_object = ""
    # if task_type == 'look_at_obj_in_light':
    #     categories_in_inst.add("FloorLamp")
    for i, subgoal in enumerate(plan):
        is_second_pickup = False
        target_object = subgoal[0]
        interaction = subgoal[1]
        if "another" in target_object or "Another" in target_object:
            is_second_pickup = True
            target_object = target_object.replace("another", "").strip()
            target_object = target_object.replace("Another", "").strip()
            pick_two_object = True
            slice_object = ""
            # 拿起第二个东西的时候要caution
            caution_pointers.append(i)

        if target_object == "Sink":
            target_object = "SinkBasin"
        if target_object == "Bathtub":
            target_object = "BathtubBasin"
        # if target_object == "DeskLamp":
        #     target_object = "FloorLamp"
        if "Lamp" in target_object or "lamp" in target_object:
            categories_in_inst += constants.lamp_object
        if "cup" in target_object or "mug" in target_object or "Cup" in target_object or "Mug" in target_object:
            categories_in_inst += constants.cup_object
        if "bottle" in target_object or "Bottle" in target_object:
            categories_in_inst += constants.bottle_object
        if "GlassBottle" in target_object or "Glassbottle" in target_object:
            categories_in_inst += ["Cup"]

        
        if not args_nonsliced:
            # 将sliced加上
            if target_object == slice_object and sliced == 1:
                target_object = target_object + "Sliced"

            # 将slice去掉
            # if "Sliced" in target_object:
            #     target_object = target_object.replace("Sliced", "")

        if interaction == "SliceObject":
            sliced = 1
            slice_object = target_object
        
        categories_in_inst.append(target_object)

        # 原本prompter的caution_pointer的设定：
        # 打开冰箱，放东西在容器里面，打开微波炉
        # OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']

        # if interaction == "OpenObject"  and target_object not in open_target:
        #     caution_pointers.append(i)
        #     open_target.append(target_object)
        # if interaction == "CloseObject" and target_object in open_target:
        #     open_target.remove(target_object)

        # 如果上一步是拿起东西，而这一步是open东西，那么需要caution
        if interaction == "OpenObject" and (plan[i-1][1]=="PickupObject" or target_object not in open_target):
            caution_pointers.append(i)
            open_target.append(target_object)

        if interaction == "PutObject" and target_object not in open_target:
            # 如果是需要放进openable target，那么在开的时候已经caution
            caution_pointers.append(i)
            # 放东西在任意位置都要caution不太合理，应该是放在小容器里面要caution
            # 可能put的时候要找一个空位，需要caution
        



        second_object.append(is_second_pickup)
        list_of_highlevel_actions.append(tuple((target_object,interaction)))
    categories_in_inst = set(categories_in_inst)
    categories_in_inst = list(categories_in_inst)
    if not pick_two_object:
        second_object = []
    
    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers,sliced
# *******************

# ***********************
def get_newplan(list_of_actions,second_object,caution_pointers,target:str,index:int,type_set:str):
    '''
    重新设置plan
    type: replace_target or open4search or close_open
    index:要更改的index
    return: list_of_actions_new,second_object_new,caution_pointers_new
    '''
    # TODO main reset之前先把要找的东西记下来 
    if type_set == 'replace_target':
        to_replace = list_of_actions[index][0]
        for idx in range(index,len(list_of_actions)):
            if list_of_actions[idx][0] == to_replace:
                list_of_actions[idx][0] = target
        list_of_actions_new = list_of_actions
        second_object_new = second_object
        caution_pointers_new = caution_pointers
    elif type_set == 'open4search':
        list_of_actions_new = []
        second_object_new = []
        caution_pointers_new = []
        second = len(second_object) != 0
        close_has_append = False
        to_search_target = list_of_actions[index][0] #要找的东西
        print(f"origin second object is {second_object}, origin caution pointer is {caution_pointers}")
        for i in range(len(list_of_actions)):
            if i < index:
                # TODO 其实如果只关心后面的动作的话，前面的可以不要
                list_of_actions_new.append(list_of_actions[i])
                if second and i<len(second_object):
                    second_object_new.append(second_object[i])
            if i == index:
                list_of_actions_new.append((target,"OpenObject"))
                list_of_actions_new.append(list_of_actions[index])
                if second and i<len(second_object):
                    second_object_new.append(False)
                    second_object_new.append(second_object[i])
                if ((index+1>=len(list_of_actions)) or (list_of_actions[index+1][0]!=to_search_target)) and not close_has_append:
                    list_of_actions_new.append((target,"CloseObject")) #应该放在pick up动作之后,不应该是pick up之后，应该是对这个对象的所有操作完成之后:下一步的操作对象不等于要找的东西
                    if second:
                        second_object_new.append(False) #随着close动作添加
                    close_has_append = True
            if i > index:
                list_of_actions_new.append(list_of_actions[i])
                if second and i<len(second_object):
                    second_object_new.append(second_object[i])
                if ((i+1>=len(list_of_actions)) or (list_of_actions[i+1][0]!=to_search_target)) and not close_has_append:
                    list_of_actions_new.append((target,"CloseObject")) #应该放在pick up动作之后
                    if second and i<len(second_object):
                        second_object_new.append(False) #随着close动作添加
                    close_has_append = True
        # 处理caution_pointers
        for caution_idx in caution_pointers:
            if caution_idx < index:
                caution_pointers_new.append(caution_idx)
            elif caution_idx == index:
                caution_pointers_new.append(caution_idx+1)
            elif caution_idx > index:
                caution_pointers_new.append(caution_idx+2)
        caution_pointers_new.append(index) #原来的动作换成了open recep应该caution
        caution_pointers_new.sort()
        print(f"second objct changed as {second_object_new}, caution_pointers changed as \
            {caution_pointers_new}")
    elif type_set == 'close_open':
        # 将pickup object 和 close object对调,close不一定在index之后，应该把pickup遇到的第一个close提前，其它的顺序往后
        print(f"origin second object is {second_object}, origin caution pointer is {caution_pointers}")
        list_of_actions_new = []
        second_object_new = []
        caution_pointers_new = []
        second = len(second_object) != 0
        to_search_target = list_of_actions[index][0]
        for i in range(index,len(list_of_actions)-1):
            if (i+2>=len(list_of_actions) or list_of_actions[i+2][0]!= to_search_target) and list_of_actions[i+1][1]=='CloseObject' and list_of_actions[i+1][0]!= to_search_target:
                close_index = i+1 #报错close_index没有指定
                break
        for i in range(len(list_of_actions)):
            if i < index:
                list_of_actions_new.append(list_of_actions[i])
                if second and i<len(second_object):
                    second_object_new.append(second_object[i])
            elif i == index:
                list_of_actions_new.append(list_of_actions[close_index])
                list_of_actions_new.append(list_of_actions[i])
                if second and i<len(second_object):
                    second_object_new.append(False)
                    second_object_new.append(second_object[i])
            elif i > index and i != close_index:
                list_of_actions_new.append(list_of_actions[i])
                if second and i<len(second_object):
                    second_object_new.append(second_object[i])
        # 处理caution pointer
        for idx in caution_pointers:
            if idx == index:
                caution_pointers_new.append(index+1)
            else:
                caution_pointers_new.append(idx)
        caution_pointers_new.sort()
        print(f"second objct changed as {second_object_new}, caution_pointers changed as \
        {caution_pointers_new}")
    else:
        raise ValueError("type_set is not in the list")
    print(f"list of actions new is {list_of_actions_new}")
    return list_of_actions_new, second_object_new, caution_pointers_new
# *********************

