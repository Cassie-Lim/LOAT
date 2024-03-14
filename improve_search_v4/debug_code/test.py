'''
测试文件，主要用来调试函数
'''
import sys
# sys.path.append("/home/cyw/task_planning/prompter_v2/")
from models.instructions_processed_LP.ALFRED_task_helper import get_newplan

if __name__=="__main__":
    # list_of_actions = [('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject'), ('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject')]
    # second_objects = [False, False, False, False, True, False]
    # # 这个second_object长度和list_of_actions不一样
    # caution_pointers = [1, 4, 5]
    target = 'Cabinet'
    # index = 0
    # type_set = 'open4search'
    # list_of_actions = [('Cabinet', 'OpenObject'), ('PepperShaker', 'PickupObject'), ('Cabinet', 'CloseObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject'), ('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject')]
    # second_objects = [False, False, False, False, False, False, True, False]
    # caution_pointers = [0, 3, 6, 7]
    # index = 1
    # type_set = 'close_open'

    # list_of_actions = [('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject'), ('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject')]
    # second_objects = [False, False, False, False, True, False]
    # caution_pointers = [1, 4, 5]
    # index = 1
    # type_set = 'open4search'

    # list_of_actions = [('PepperShaker', 'PickupObject'), ('Cabinet', 'OpenObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject'), ('Cabinet', 'CloseObject'), ('PepperShaker', 'PickupObject'), ('Drawer', 'OpenObject'), ('Drawer', 'PutObject'), ('Drawer', 'CloseObject')]
    # second_objects = [False, False, False, False, False, False, True, False]
    # caution_pointers = [1, 2, 6, 7]
    # index = 2
    # type_set = 'close_open'

    # list_of_actions = [("vase","PickupObject"),('safe','OpenObject'),('safe','PutObject'),('safe','CloseObject')]
    # second_objects = []
    # caution_pointers = [1]
    # index = 1
    # type_set = 'open4search'

    list_of_actions = [('Fork', 'PickupObject'), ('Cup', 'PutObject'), ('Cup', 'PickupObject'), ('CounterTop', 'PutObject')]
    second_objects = []
    caution_pointers = [1]
    index = 1
    type_set = 'open4search'

    list_of_actions_new,second_objects_new,caution_pointer_new = \
                get_newplan(list_of_actions,second_objects,caution_pointers,target,index,type_set)
    print(list_of_actions_new)
    print(second_objects_new)
    print(caution_pointer_new)