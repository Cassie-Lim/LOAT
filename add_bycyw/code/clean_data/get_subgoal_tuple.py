from models.instructions_processed_LP.ALFRED_task_helper import add_target
import alfred_utils.gen.constants as constants

def get_list_of_highlevel_actions_from_para(language_goal, task_type, mrecep_target, obj_target, parent_target, sliced, args_nonsliced=False):
    # language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments_test(test_dict, traj_data)

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