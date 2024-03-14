#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:49:38 2021

@author: soyeonmin
"""
import sys
sys.path.append("/raid/cyw/task_planning/prompter_cyw/prompter/")
import json
import pickle
import alfred_utils.gen.constants as constants
import string
import os
import progressbar
import pprint

exclude = set(string.punctuation)
task_type_dict = {2: 'pick_and_place_simple',
                  5: 'look_at_obj_in_light',
                  1: 'pick_and_place_with_movable_recep',
                  3: 'pick_two_obj_and_place',
                  6: 'pick_clean_then_place_in_recep',
                  4: 'pick_heat_then_place_in_recep',
                  0: 'pick_cool_then_place_in_recep'}

def read_test_dict(test, appended, unseen):
    if test:
        if appended:
            if unseen:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_test_unseen_appended.p", "rb"))
            else:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_test_seen_appended.p", "rb"))
        else:
            if unseen:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_test_unseen_916_noappended.p", "rb"))
            else:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_test_seen_916_noappended.p", "rb"))
    else:
        if appended:
            if unseen:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_val_unseen_appended.p", "rb"))
            else:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_val_seen_appended.p", "rb"))
        else:
            if unseen:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_val_unseen_916_noappended.p", "rb"))
            else:
                return pickle.load(
                    open("models/instructions_processed_LP/instruction2_params_val_seen_916_noappended.p", "rb"))

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
    # toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])

    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced

def add_target_instuct(target, target_action, obj_target, list_of_actions):
    k = 0
    list_of_actions.append("Find a " + target.lower())
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not (a == 'Box')]:
        k = 1
        list_of_actions.append("Open the " + target.lower())
    if target in [a for a in ["fridge", "cabinet", "microwave", "drawer", "safe", "box"] if not (a == 'box')]:
        k = 1
        list_of_actions.append("Open the " + target.lower())
    list_of_actions.append(target_action + "the " + obj_target.lower()+ " on " + target.lower())
    if k == 1 :
        list_of_actions.append("Close the " + target.lower())
    return list_of_actions

def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions

def add_target_instuct2(target, target_action, obj_target, list_of_actions):
    k = 0
    list_of_actions.append("Find the " + target.lower())
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not (a == 'Box')]:
        k = 1
        list_of_actions.append("Open the " + target.lower())
    if target in [a for a in ["fridge", "cabinet", "microwave", "drawer", "safe", "box"] if not (a == 'box')]:
        k = 1
        list_of_actions.append("Open the " + target.lower())
    list_of_actions.append(target_action + "the " + obj_target.lower()+ " on " + target.lower())
    if k == 1 :
        list_of_actions.append("Close the " + target.lower())
    return list_of_actions


def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions) - 1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer + 1][0]:
            returned = True
            # target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        # Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer + 1][
            1] == "PickupObject":
            returned = True
            # target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                # target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        # Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer + 1][
            1] == "CloseObject":
            returned = True
            # target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer - 1][0]
        # Faucet
        elif list_of_actions[previous_pointer + 1][0] == "Faucet" and list_of_actions[previous_pointer + 1][1] in [
            "ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        # Pick up after faucet
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer + 1][
            1] == "PickupObject":
            returned = True
            # target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                # target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
    return returned, target_instance


def exchange(string):
    new_str = ""
    if len(string) > 2:
        new_str += string[0].lower()
        for i in range(1, len(string)):
            if string[i].isupper():
                new_str += " "
            new_str += string[i].lower()
    else:
        new_str += string.lower()
    return new_str

# use plate to make language instruction
def get_text_of_highlevel_actions(traj_data, prompt):
    language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments(traj_data)

    if parent_target != None:
        parent_target = exchange(parent_target)
    if obj_target != None:
        obj_target = exchange(obj_target)
    if mrecep_target != None:
        mrecep_target = exchange(mrecep_target)
    # Change to this after the sliced happens
    categories_in_inst = []
    text_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    if "ed " in prompt : return text_of_highlevel_actions, task_type, prompt
    if sliced == 1:
        prompt.replace(" and chill a knife", "")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("put the knife, yellow apple slice and pan together on the black table", "put the yellow apple slice and pan together on the black table")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("the knife and slice of", "slice of")
        prompt.replace("a knife and slice of", "slice of")
        prompt.replace(" knife and slice of", " slice of")
        prompt.replace("place knife in sink, ", "")
        text_of_highlevel_actions.append("Find a knife")
        text_of_highlevel_actions.append("Pick up the knife")
        text_of_highlevel_actions.append("Find a "+obj_target.lower())
        text_of_highlevel_actions.append("Slice the "+obj_target.lower())
        caution_pointers.append(len(text_of_highlevel_actions))
        categories_in_inst.append(obj_target)

    if sliced:
        obj_target = 'sliced '+ obj_target

    if task_type == 'pick_cool_then_place_in_recep':  # 'pick_with_mrecep_then_cool_and_place_in_recep'if sliced
        if ("slice" in obj_target.lower()) and (("cool" in prompt) or ("chill" in prompt) or ("cold" in prompt) or ("refrigerat" in prompt) or ("fridge" in prompt)) and (("slice" in prompt) or ("piece" in prompt) or ("cut" in prompt)):
            text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Fridge", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Open the fridge")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the fridge")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_cool_and_place_in_recep'
            if "fridge" in prompt: prompt = prompt.split("fridge",1)[0] + "fridge with plate"+ prompt.split("fridge",1)[1]
            elif "refrigerator" in prompt: prompt = prompt.split("refrigerator", 1)[0] + "refrigerator with plate" + prompt.split("refrigerator", 1)[1]
            else: prompt = prompt + " with plate"

    elif task_type == 'pick_and_place_with_movable_recep':  # 'pick_two_obj_and_place_with_movable_recep'
        if "slice" not in obj_target.lower():
            if " a " + obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2]=="f" and obj_target[-1]=="e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a "+obj_target.lower(), " two "+obj_targets.lower()+"s")
                text_of_highlevel_actions.append("Find a " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target, text_of_highlevel_actions)
                text_of_highlevel_actions.append( "Find an another " + obj_target.lower())# modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target, text_of_highlevel_actions) #does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", mrecep_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'
            elif " an " + obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" an " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                text_of_highlevel_actions.append("Find an " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target, text_of_highlevel_actions)
                text_of_highlevel_actions.append( "Find an another " + obj_target.lower())# modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target, text_of_highlevel_actions) #does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", mrecep_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'

    elif task_type == 'pick_heat_then_place_in_recep':  # 'pick_with_mrecep_then_heat_and_place_in_recep'if sliced
        if "slice" in obj_target.lower() and (("heat" in specific_task) or ("warm" in specific_task) or ("cook" in specific_task) or ("microwave" in specific_task) or (
                            "hot" in specific_task) or ("dry" in specific_task)) and (("slice" in prompt) or ("piece" in prompt) or ("cut" in prompt)):
            text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Microwave", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Toggle microwave on")
            text_of_highlevel_actions.append("Toggle microwave off")
            text_of_highlevel_actions.append("Open the microwave")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the microwave")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_heat_and_place_in_recep'
            if " microwave " in prompt: prompt = prompt.split("microwave",1)[0] + "microwave with plate"+ prompt.split("microwave",1)[1]
            else: prompt = prompt + " with plate"

    elif task_type == 'pick_two_obj_and_place':  # 'pick_three_obj_and_place'
        if "slice" not in obj_target.lower():
            if " two " + obj_target.lower() in prompt:
                prompt = prompt.replace(" two " + obj_target.lower(), " three " + obj_target.lower())
                text_of_highlevel_actions.append("Find a " + obj_target.lower() )
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find a third " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_three_obj_and_place'

    elif task_type == 'pick_clean_then_place_in_recep':  # pick_two_clean_then_place_in_recep
        if "slice" not in obj_target.lower():
            if " a " +obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                text_of_highlevel_actions.append("Find a " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up a " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up a " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                task_type = 'pick_two_clean_then_place_in_recep'
            if " an " + obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" an " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                text_of_highlevel_actions.append("Find an " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append(
                "Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                task_type = 'pick_two_clean_then_place_in_recep'

    # print("instruction goal is ", language_goal)
    return text_of_highlevel_actions, task_type, prompt

def get_text3_of_highlevel_actions(traj_data, prompt):
    language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments(traj_data)
    if parent_target != None:
        parent_target = exchange(parent_target)
    if obj_target != None:
        obj_target = exchange(obj_target)
    if mrecep_target != None:
        mrecep_target = exchange(mrecep_target)
    # Change to this after the sliced happens

    categories_in_inst = []
    text_of_highlevel_actions = []
    if sliced:
        prompt.replace(" and chill a knife", "")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("put the knife, yellow apple slice and pan together on the black table", "put the yellow apple slice and pan together on the black table")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("the knife and slice of", "slice of")
        prompt.replace("a knife and slice of", "slice of")
        prompt.replace(" knife and slice of", " slice of")
        prompt.replace("place knife in sink, ", "")
        obj_target = 'sliced '+ obj_target
        env_d = "The "+obj_target+" is in hand" + ", "
    # else :
    #      env_d = "A "+obj_target+" is in hand" + ", "
    #     text_of_highlevel_actions.append("Pick up the "+obj_target.lower())

    if task_type == 'pick_cool_then_place_in_recep':  # 0 in new_labels
        if ("slice" in obj_target.lower()) and (("cool" in prompt) or ("chill" in prompt) or ("cold" in prompt) or ("refrigerat" in prompt) or ("fridge" in prompt)) and ("slice " not in prompt) and ("cut" not in prompt):
            if ("sliced" not in prompt) and ("piece" not in prompt) and ("slice of " not in prompt): return text_of_highlevel_actions, task_type, prompt
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Fridge", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Open the fridge")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the fridge")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_cool_and_place_in_recep'
            if "fridge" in prompt: prompt = prompt.split("fridge",1)[0] + "fridge with plate"+ prompt.split("fridge",1)[1]
            elif "refrigerator" in prompt: prompt = prompt.split("refrigerator", 1)[0] + "refrigerator with plate" + prompt.split("refrigerator", 1)[1]
            else: prompt = prompt + " with plate"
            prompt = env_d + prompt

    elif task_type == 'pick_and_place_with_movable_recep':  # 1 in new_labels
        if "slice" not in obj_target.lower():
            if " a " +obj_target.lower() in prompt:
                if "ed " in prompt: return text_of_highlevel_actions, task_type, prompt
                env_d = "A " + obj_target + " is in hand" + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                prompt = env_d + prompt
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target,
                                                               text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower())  # modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target,
                                                               text_of_highlevel_actions)  # does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", mrecep_target.lower(),
                                                               text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'
            elif " an " + obj_target.lower() in prompt:
                if "ed " in prompt: return text_of_highlevel_actions, task_type, prompt
                env_d = "An " + obj_target + " is in hand" + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" an " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                prompt = env_d + prompt
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target,
                                                               text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower())  # modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target,
                                                               text_of_highlevel_actions)  # does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", mrecep_target.lower(),
                                                               text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'

    elif task_type == 'pick_heat_then_place_in_recep':  # 4 in new_labels
        if "slice" in obj_target.lower() and (("slice" in prompt) or ("piece" in prompt) or ("cut" in prompt)):
            if ("sliced" not in prompt) and ("piece" not in prompt) and ("slice of " not in prompt): return text_of_highlevel_actions, task_type, prompt
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Microwave", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Toggle microwave on")
            text_of_highlevel_actions.append("Toggle microwave off")
            text_of_highlevel_actions.append("Open the microwave")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the microwave")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_heat_and_place_in_recep'
            if " microwave " in prompt: prompt = prompt.split("microwave",1)[0] + "microwave with plate"+ prompt.split("microwave",1)[1]
            else: prompt = prompt + " with plate"
            prompt = env_d + prompt

    elif task_type == 'pick_two_obj_and_place':  # 3 in new_labels
        if "slice" not in obj_target.lower():
            if " two " + obj_target in prompt:
                env_d = "A " + obj_target + " is in hand" + ", "
                prompt = prompt.replace(" two "+ obj_target, " three "+ obj_target)
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find a third " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_three_obj_and_place'
                prompt = env_d + prompt

    elif task_type == 'pick_clean_then_place_in_recep':  # 6 in new_labels
        if "slice" not in obj_target.lower():
            if " a " +obj_target.lower() in prompt:
                if "ed " in prompt: return text_of_highlevel_actions, task_type, prompt
                env_d = "A " + obj_target + " is in hand" + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up a "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up a " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(),text_of_highlevel_actions)
                task_type = 'pick_two_clean_then_place_in_recep'
                prompt = env_d + prompt

            if " an " + obj_target.lower() in prompt:
                if "ed " in prompt: return text_of_highlevel_actions, task_type, prompt
                env_d = "An " + obj_target + " is in hand" + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" an " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(),text_of_highlevel_actions)
                task_type = 'pick_two_clean_then_place_in_recep'
                prompt = env_d + prompt

    # print("instruction goal is ", language_goal)
    return text_of_highlevel_actions, task_type, prompt

def get_text2_of_highlevel_actions(traj_data, prompt):
    language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments(traj_data)
    if parent_target != None:
        parent_target = exchange(parent_target)
    if obj_target != None:
        obj_target = exchange(obj_target)
    if mrecep_target != None:
        mrecep_target = exchange(mrecep_target)
    # Change to this after the sliced happens

    categories_in_inst = []
    text_of_highlevel_actions = []
    if "ed " in prompt: return text_of_highlevel_actions, task_type, prompt
    if sliced == 1:
        prompt.replace(" and chill a knife", "")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("put the knife, yellow apple slice and pan together on the black table", "put the yellow apple slice and pan together on the black table")
        prompt.replace("a knife in the refrigerator, and ", "")
        prompt.replace("the knife and slice of", "slice of")
        prompt.replace("a knife and slice of", "slice of")
        prompt.replace(" knife and slice of", " slice of")
        prompt.replace("place knife in sink, ", "")
        env_d = "A knife in hand" + ", "
        text_of_highlevel_actions.append("Find a "+obj_target.lower())
        text_of_highlevel_actions.append("Slice the "+obj_target.lower())

    if sliced:
        obj_target = 'sliced '+ obj_target

    if task_type == 'pick_cool_then_place_in_recep':  # 0 in new_labels
        if ("slice" in obj_target.lower()) and (("cool" in prompt) or ("chill" in prompt) or ("cold" in prompt) or ("refrigerat" in prompt) or ("fridge" in prompt)) and (("slice" in prompt) or ("piece" in prompt) or ("cut" in prompt)):
            text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Fridge", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Open the fridge")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the fridge")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_cool_and_place_in_recep'
            if "fridge" in prompt: prompt = prompt.split("fridge",1)[0] + "fridge with plate"+ prompt.split("fridge",1)[1]
            elif "refrigerator" in prompt: prompt = prompt.split("refrigerator", 1)[0] + "refrigerator with plate" + prompt.split("refrigerator", 1)[1]
            else: prompt = prompt + " with plate"
            prompt = env_d + prompt

    elif task_type == 'pick_and_place_with_movable_recep':  # 1 in new_labels
        if "slice" not in obj_target.lower():
            if " a " +obj_target.lower() in prompt:
                env_d = "I see a " + obj_target.lower() + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                prompt = env_d + prompt
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target, text_of_highlevel_actions)
                text_of_highlevel_actions.append( "Find an another " + obj_target.lower())# modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target, text_of_highlevel_actions) #does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", mrecep_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'
            elif " an " + obj_target.lower() in prompt:
                env_d = "I see an " + obj_target.lower() + ", "
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" an " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                prompt = env_d + prompt
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(mrecep_target, "Put ", obj_target, text_of_highlevel_actions)
                text_of_highlevel_actions.append( "Find an another " + obj_target.lower())# modify two
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(mrecep_target, "Put ", obj_target, text_of_highlevel_actions) #does know this means plate with a object
                text_of_highlevel_actions.append("Pick up the " + mrecep_target)
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", mrecep_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_two_obj_and_place_with_movable_recep'


    elif task_type == 'pick_heat_then_place_in_recep':  # 4 in new_labels
        if "slice" in obj_target.lower() and (("slice" in prompt) or ("piece" in prompt) or ("cut" in prompt)):
            text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
            text_of_highlevel_actions = add_target_instuct("Plate", "Put ", obj_target, text_of_highlevel_actions)
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions = add_target_instuct("Microwave", "Put ", "plate", text_of_highlevel_actions)
            text_of_highlevel_actions.append("Toggle microwave on")
            text_of_highlevel_actions.append("Toggle microwave off")
            text_of_highlevel_actions.append("Open the microwave")
            text_of_highlevel_actions.append("Pick up the plate")
            text_of_highlevel_actions.append("Close the microwave")
            text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", "plate", text_of_highlevel_actions)
            task_type = 'pick_with_mrecep_then_heat_and_place_in_recep'
            if " microwave " in prompt: prompt = prompt.split("microwave",1)[0] + "microwave with plate"+ prompt.split("microwave",1)[1]
            else: prompt = prompt + " with plate"
            prompt = env_d + prompt

    elif task_type == 'pick_two_obj_and_place':  # 3 in new_labels
        if "slice" not in obj_target.lower():
            if " two " + obj_target.lower() in prompt:
                prompt = prompt.replace(" two "+ obj_target.lower(), " three "+ obj_target.lower())
                env_d = "I see a " + obj_target.lower() + ", "
                prompt = env_d + prompt
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find a third " + obj_target.lower())# modify two?
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                task_type = 'pick_three_obj_and_place'

    elif task_type == 'pick_clean_then_place_in_recep':  # 6 in new_labels
        if "slice" not in obj_target.lower():
            if " a "+obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                env_d = "I see a " + obj_target.lower() + ", "
                prompt = env_d + prompt
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up a "+ obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(), text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up a " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2(parent_target, "Put ", obj_target.lower(),text_of_highlevel_actions)

                task_type = 'pick_two_clean_then_place_in_recep'

            if " an " + obj_target.lower() in prompt:
                obj_targets = obj_target
                if obj_target[-2] == "f" and obj_target[-1] == "e":
                    obj_targets = obj_targets[:-2]+"v"+obj_targets[-1]
                prompt = prompt.replace(" a " + obj_target.lower(), " two " + obj_targets.lower() + "s")
                env_d = "I see an " + obj_target.lower() + ", "
                prompt = env_d + prompt
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)
                text_of_highlevel_actions.append(
                "Find an another " + obj_target.lower() if "slice" not in obj_target.lower() else "Find the " + obj_target.lower())
                text_of_highlevel_actions.append("Pick up the " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct2("sink basin", "Put ", obj_target.lower(),text_of_highlevel_actions)  # Sink or SinkBasin?
                text_of_highlevel_actions.append("Toggle faucet on")
                text_of_highlevel_actions.append("Toggle faucet off")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                text_of_highlevel_actions.append("Find the sink basin")
                text_of_highlevel_actions.append("Pick up an " + obj_target.lower())
                text_of_highlevel_actions = add_target_instuct(parent_target, "Put ", obj_target.lower(),
                                                               text_of_highlevel_actions)
                task_type = 'pick_two_clean_then_place_in_recep'


    # return [(goal_category, interaction), (goal_category, interaction), ...]
    # print("instruction goal is ", language_goal)
    return text_of_highlevel_actions, task_type, prompt

with open('alfred_data_small/splits/oct21.json') as f:
    splits = json.load(f)
    pprint.pprint({k: len(v) for k, v in splits.items()})  # k键名，v键值-列表

for k, d in splits.items():
    print('Preprocessing {}'.format(k))
    train_mode = 'test' not in k  # 非测试集，train_mode true
    json_content = []
    count = 0
    if train_mode:
        for id,task in progressbar.progressbar(enumerate(d)):  # 列表中的元素（字典）诶个赋值
        # load json file
            json_path = os.path.join('alfred_data_all/json_2.1.0', k, task['task'], 'traj_data.json')
            with open(json_path) as f:
                ex = json.load(f)  # 加载为字典
                j = task["repeat_idx"]
                pas = 0
                for n in ex['turk_annotations']['anns'][j]['votes']:
                    if n == 0: pas = 1
                if pas == 1: continue
                for i in range(3):
                    specific_task = ex['turk_annotations']['anns'][j]['task_desc'].lower()
                    if specific_task[-1] == "." or specific_task[-2] == ".":
                        specific_task = specific_task.replace(".", "")
                    if "how to" not in specific_task:prompt = ( "how " if specific_task[0].lower() == "t" and specific_task[1].lower() == "o" else "how to ") + specific_task
                    if ("vegetable" in specific_task) or (" all " in specific_task) : continue
                    if i == 0:text_of_highlevel_actions, task_type, prompt= get_text_of_highlevel_actions(ex, prompt)
                    if i == 1:text_of_highlevel_actions, task_type, prompt= get_text2_of_highlevel_actions(ex, prompt)
                    if i == 2:text_of_highlevel_actions, task_type, prompt= get_text3_of_highlevel_actions(ex, prompt)
                    if (task_type != 'pick_with_mrecep_then_cool_and_place_in_recep') and (task_type != 'pick_two_obj_and_place_with_movable_recep') and (task_type != 'pick_with_mrecep_then_heat_and_place_in_recep') and(task_type != 'pick_three_obj_and_place') and (task_type != 'pick_two_clean_then_place_in_recep'): continue
                    task_id = ex['task_id']
                    dict0 = {}
                    # dict0['task_id'] = task_id
                    dict0['task_type'] = task_type
                    dict0['prompt'] = prompt +"?"
                    # dict0['list_of_highlevel_actions'] = list_of_highlevel_actions
                    dict0['text_of_highlevel_actions'] = text_of_highlevel_actions

                    ann = ex['turk_annotations']['anns'][j]['high_descs']
                    ann_id = ex['turk_annotations']['anns'][j]['assignment_id']
                    # dict = {}
                    # dict['ann'] = ann
                    # dict['ann_id'] = ann_id
                    # dict0['ann_dict'] = dict
                    dict0["base_id"] = id
                    json_content.append(dict0)
                    count += 1
        with open("/raid/cyw/Prompter_results/clean_data/data_generate_bycode/5_10_5newtype_%s.json" % k, 'w', encoding='utf-8') as f:
            # json.dump(json_content, f, ensure_ascii=False)
            json.dump(json_content, f, sort_keys=False, indent=4)
            print(str(count) + " instructions in 5_10_5newtype_%s.json" % k)
    else:
        continue