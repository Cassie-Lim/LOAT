import alfred_utils.gen.constants as constants 

def get_drop_locs(task_type,pddl_params):
    if task_type == 'pick_heat_then_place_in_recep':
        drop_loc = 'Microwave'
    elif (task_type == "pick_two_obj_and_place" or task_type == 'pick_and_place_simple') and not pddl_params['object_sliced']:
        drop_loc = pddl_params['parent_target']
    elif task_type == 'pick_cool_then_place_in_recep':
        drop_loc = 'Fridge'
    else:
        drop_loc = []
    return drop_loc

def sort_locs(lan_locs,occ_locs,drop_locs):
    # locs = list(set(lan_locs+occ_locs)- set(drop_locs))
    locs = [loc for loc in lan_locs if loc!=drop_locs]
    # locs = [loc for loc in lan_locs]
    open_obj = []
    for item in occ_locs:
        if item not in locs and item != drop_locs:
        # if item not in locs:
            if item not in constants.OPENABLE_CLASS_LIST:
                locs.append(item)
            else:
                open_obj.append(item)
    for item in open_obj:
        locs.append(item)
    if drop_locs is not None and len(drop_locs) !=0:
        locs.append(drop_locs) 
    return locs

def get_locs_occ(target,occ_data,large2indx):
    if target in occ_data:
        search_list = [recp for recp in occ_data[target] if recp in large2indx]
    else:
        search_list = []
    return search_list

def get_search_loc_for_target(target,occ_data,lan_locs_dict,drop_locs,large2indx):
    occ_locs = get_locs_occ(target,occ_data,large2indx)
    if target in lan_locs_dict:
        lan_locs = lan_locs_dict[target]
    else:
        lan_locs = []
    search_locs = sort_locs(lan_locs,occ_locs,drop_locs)
    return search_locs
