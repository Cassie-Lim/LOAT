def all_satisfy(fn, list_to_deal):
    for element in list_to_deal:
        if not fn(element):
            return False
    return True

def any_satisfy(fn, list_to_deal):
    '''
    fn
    '''
    for element in list_to_deal:
        if fn(element):
            return True
    return False

def clip_value(x,min_value,max_value):
    return max(min_value,min(x,max_value))

def set_dict2value(dict_name:dict,value_to_set):
    for key in dict_name.keys():
        dict_name[key] = value_to_set
    return dict_name

