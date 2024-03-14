'''
杂项文件，放一些辅助计算的函数
'''

def get_opensearch_times(list_of_actions,index):
    '''
    得到为了寻找当前目标，共open了多少个东西
    '''
    action = list_of_actions[index][1]
    if action == 'OpenObject' or action == "CloseObject":
        return 0
    else:
        # 让i从index倒着往回走
        opensearch_count = 0
        for i in range(index-1,0,-1):#这里只会取到1
            if list_of_actions[i][1] == 'CloseObject' and list_of_actions[i-1][1] == 'OpenObject':
                opensearch_count = opensearch_count + 1
            elif list_of_actions[i][1] != 'OpenObject' and list_of_actions[i][1] != 'CloseObject':
                # 是其它正常的动作
                break
        return opensearch_count

def clip_value(x,min_value,max_value):
    '''
    x: 需要裁剪的数
    min: 最小值
    max: 最大值
    '''
    return max(min_value,min(x,max_value))


if __name__ == '__main__':
    list_of_actions = [('ButterKnife', 'PickupObject'),('Cabinet', 'OpenObject'),('Cabinet', 'CloseObject'),('Cabinet', 'OpenObject'),('Cabinet', 'CloseObject'),('Cabinet', 'OpenObject'),('Cup', 'PutObject'),('Cabinet', 'CloseObject')]
    for i in range(8):
        opensearch_count = get_opensearch_times(list_of_actions,i)
        print(opensearch_count)