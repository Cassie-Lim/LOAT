'''
得到搜索位置的一些函数
'''
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
    '''
    对loc排序
    drop_locs:一般就是parent target,str
    '''
    # locs = list(set(lan_locs+occ_locs)- set(drop_locs))
    # NOTE 不要用上面那个通过set的形式，会打乱顺序
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
                # 把要开的东西放在最后
    for item in open_obj:
        locs.append(item)
    if drop_locs is not None and len(drop_locs) !=0:
        locs.append(drop_locs) #把要drop的放在最后
    return locs

def get_locs_occ(target,occ_data,large2indx):
    '''
    从物体共现频率中获得搜索序列
    '''
    if target in occ_data:
        search_list = [recp for recp in occ_data[target] if recp in large2indx]
    else:
        search_list = []
    return search_list

def get_search_loc_for_target(target,occ_data,lan_locs_dict,drop_locs,large2indx):
    # NOTE 注意看看这里当没有任何信息的时候怎么样
    occ_locs = get_locs_occ(target,occ_data,large2indx)
    if target in lan_locs_dict:
        lan_locs = lan_locs_dict[target]
    else:
        lan_locs = []
    search_locs = sort_locs(lan_locs,occ_locs,drop_locs)
    return search_locs
