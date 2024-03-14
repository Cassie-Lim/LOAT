'''
一些辅助计算的函数
'''

# 判断是否list中所有元素都满足方程
def all_satisfy(fn, list_to_deal):
    '''
    fn:一个判断函数，返回true or false
    '''
    for element in list_to_deal:
        if not fn(element):
            return False
    return True

# 判断list中是否有任意元素满足方程
def any_satisfy(fn, list_to_deal):
    '''
    fn
    '''
    for element in list_to_deal:
        if fn(element):
            return True
    return False

def clip_value(x,min_value,max_value):
    '''
    x: 需要裁剪的数
    min: 最小值
    max: 最大值
    '''
    return max(min_value,min(x,max_value))

def set_dict2value(dict_name:dict,value_to_set):
    '''
    将一个字典的值全部设置为value
    '''
    for key in dict_name.keys():
        dict_name[key] = value_to_set
    return dict_name

