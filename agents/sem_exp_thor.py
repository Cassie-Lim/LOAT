import os, sys
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')

import pickle, json
import copy
import string

import torch
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import skimage.morphology
import cv2
from PIL import Image

from envs.utils.fmm_planner import planNextMove
import envs.utils.pose as pu
import alfred_utils.gen.constants as constants
from alfred_utils.env.thor_env_code import ThorEnvCode
# from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, determine_consecutive_interx, get_arguments, get_arguments_test, read_test_dict
# *****************
# *****************
from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, determine_consecutive_interx, get_arguments, get_arguments_test, read_test_dict,get_list_of_highlevel_actions_from_plan
# *******************
# *******************
from models.segmentation.segmentation_helper import SemgnetationHelper
#from models.depth.depth_helper import DepthHelper
import utils.control_helper as CH

import envs.utils.depth_utils as du
import envs.utils.rotation_utils as ru

# **************************
from add_bycyw.code.auxiliary_calculate import all_satisfy,any_satisfy,clip_value,set_dict2value
import clip


from transformers import DPTForDepthEstimation, AutoImageProcessor
# ******************

sit_object = constants.sit_object
table_object = constants.table_object
cup_object = constants.cup_object
bottle_object = constants.bottle_object
lamp_object = constants.lamp_object
drawer_object = constants.drawer_object

'''
done:
这个文件的ep num还稍有不对，之后还需要再修改
之后和v100上代码合并的时候，一定要对比一下，以免一些小细节忘了
'''

class Sem_Exp_Env_Agent_Thor(ThorEnvCode):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, scene_names, rank):
        self.fails_cur = 0

        self.args = args
        self.seed = self.args.seed
        # episode_no = self.args.from_idx + rank
        #  *********************
        if self.args.run_idx_file is None:
            episode_no = self.args.from_idx + rank
        else:
            idx = json.load(open(self.args.run_idx_file, 'r'))
            episode_no = idx[self.args.from_idx+rank]
        # 这里其实不影响，它只用作随机数
        # *********************
        self.local_rng = np.random.RandomState(self.seed + episode_no)

        # super().__init__(args, rank)

        super().__init__(args, rank, player_screen_height=512, player_screen_width=512)



        # initialize transform for RGB observations
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        

        # initializations for planning:
        self.selem = skimage.morphology.square(self.args.obstacle_selem)
        self.flattened = pickle.load(open("miscellaneous/flattened.p", "rb"))
        
        self.last_three_sidesteps = [None]*3
        self.picked_up = False
        # ****************
        self.pick_up_obj = None
        # ****************
        self.picked_up_mask = None
        self.sliced_mask = None
        self.sliced_pose = None
        
        self.transfer_cat = {'ButterKnife': 'Knife', "Knife":"ButterKnife"}
        
        self.scene_names = scene_names
        self.scene_pointer = 0
        
        self.obs = None
        self.steps = 0
        
        self.action_5_count = 0
        self.goal_visualize = None
        self.prev_goal = None

        self.reached_goal = False
        
        self.test_dict = read_test_dict(
            self.args.test, self.args.language_granularity, 'unseen' in self.args.eval_split)

        #Segmentation
        # if self.args.use_yolo:
        #     from add_byme.code.segmentation_helper_yolo import SemgnetationHelper_YOLO
        #     self.seg = SemgnetationHelper_YOLO(self)
        # else:
        #     self.seg = SemgnetationHelper(self)
        self.seg = SemgnetationHelper(self)


        self.do_log = self.args.debug_local


        #Depth
        # Changed by Trisoil
        self.depth_img_processor = AutoImageProcessor.from_pretrained('models/models_ckpt/dpt-dinov2-base-nyu')
        
        self.dino_depth = DPTForDepthEstimation.from_pretrained('models/models_ckpt/dpt-dinov2-base-nyu')
        self.dino_depth.load_state_dict(torch.load('models/models_ckpt/base_epoch_36.pth'))
        self.dino_depth.cuda(self.args.depth_gpu)
        # self.dino_depth =  torch.nn.DataParallel(self.dino_depth, device_ids=[0, 1, 2, 3])
        print('Finish loading dino depth model')

        #Depth
        # # if replan
        # self.replan = True
        # self.SubTask_fail_num = 0
        # self.SubTask_pointer = 0
        # Clip
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.args.use_replan:
            self.clip_gpu =  torch.device("cuda:" + str(args.which_gpu) if args.cuda else "cpu")
            self.clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.clip_gpu)
            self.clip_tokenizer = clip.tokenize
            print('use replan')

        # 加载plan数据
        if self.args.subgoal_file is not None:
            self.plan_data = json.load(open(self.args.subgoal_file,'r'))



    
    def load_traj(self, scene_name):
        json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
        traj_data = json.load(open(json_dir))
        return traj_data

    
    def load_initial_scene(self):
        self.side_step_order = 0
        self.rotate_before_side_step_count = 0
        self.fails_cur = 0
        self.put_rgb_mask = None
        self.pointer = 0

        self.prev_rgb = None
        self.prev_depth = None
        self.prev_seg = None

        self.steps_taken = 0
        self.goal_name = None
        self.steps = 0
        self.last_err = ""
        self.prev_number_action = None
        self.move_until_visible_order = 0
        self.consecutive_steps = False
        self.cat_equate_dict = {} #map "key" category to "value" category
        self.rotate_aftersidestep = None
        self.errs = []
        self.logs = []

        exclude = set(string.punctuation)

        self.broken_grid = []
        self.where_block = []
        self.remove_connections_to = None

        self.reached_goal = False
        
        self.action_5_count = 0
        self.prev_goal = None
        
        self.last_three_sidesteps = [None]*3
        self.picked_up = False
        # ****************
        self.pick_up_obj = None
        # *****************


        self.learned_depth_frame = None

        self.picked_up_mask = None
        self.sliced_mask = None
        self.sliced_pose = None
        
        self.holding_mask = None

        # ****************
        self.first_goal = False #是否要先走到指定的目标点，再交互
        self.open_object = False #是否开了某个东西，如果开了，可能之前记录的visited会失效，并且在这个东西关上之前，不应该走动太远
        # ***************

        # episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank   
        # #  *********************
        # if self.args.run_idx_file is None:
        #     episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
        # else:
        #     idx = json.load(open(self.args.run_idx_file, 'r'))
        #     episode_no = idx[self.scene_pointer* self.args.num_processes + self.rank]
        # 这里跑到最后的时候会报错:index out of range
        # # ********************
        # self.local_rng = np.random.RandomState(self.seed + episode_no)

        # **********************
        text = None
        # ******************
        try:
            #  *********************
            if self.args.run_idx_file is None:
                episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
            else:
                idx = json.load(open(self.args.run_idx_file, 'r'))
                episode_no = idx[self.args.from_idx+self.scene_pointer* self.args.num_processes + self.rank]
                # 注意修改到这里会不会导致程序非法结束？
                print(f"episode number is {episode_no}")
            # ********************
            self.local_rng = np.random.RandomState(self.seed + episode_no)

            traj_data = self.load_traj(self.scene_names[self.scene_pointer]); r_idx = self.scene_names[self.scene_pointer]['repeat_idx']
            # ;表示句子终止，为啥要这样写？！
            self.traj_data = traj_data; self.r_idx = r_idx

            # self.picture_folder_name = "pictures/" + self.args.eval_split + "/"+ self.args.dn + "/" + str(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank) + "/"
            # 这里的picture name，如果指定了id,就不准确
            self.picture_folder_name = "pictures/" + self.args.eval_split + "/"+ self.args.dn + "/" + str(episode_no) + "/"
            if self.args.save_pictures and not (episode_no in self.args.skip_indices):
                # ******************
                if os.path.exists(self.picture_folder_name):
                    import shutil
                    shutil.rmtree(self.picture_folder_name)
                # ***********
                os.makedirs(self.picture_folder_name)
                os.makedirs(self.picture_folder_name + "/fmm_dist")
                os.makedirs(self.picture_folder_name + "/obstacles_pre_dilation")
                os.makedirs(self.picture_folder_name + "/Sem")
                os.makedirs(self.picture_folder_name + "/Sem_Map")
                os.makedirs(self.picture_folder_name + "/Sem_Map_Target")
                os.makedirs(self.picture_folder_name + "/rgb")
                os.makedirs(self.picture_folder_name + "/depth")
                os.makedirs(self.picture_folder_name + "/depth_thresholded")
                # *************
                os.makedirs(self.picture_folder_name + "/Sem_Map_new")
                os.makedirs(self.picture_folder_name + "/mask")
                # ****************

            task_type = get_arguments_test(self.test_dict, traj_data)[1]
            # 这里task_type主要用来进行一些物体名的微调以及加载sene，而加载sene，主要用来规定任务类型，可能是用来计算成功率？
            sliced = get_arguments_test(self.test_dict, traj_data)[-1]
            list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(
                traj_data, self.test_dict, self.args.nonsliced)                      
            # 以上读取任务参数

            # **********************
            # 从轨迹里面读取goal description，为了与plan里面的对比，方便debug
            goal_description = " ".join(traj_data['ann']["goal"][:-1])
            # 去除标点符号
            goal_description = goal_description.replace(",", "").replace(".", "").replace("'","")
            # 去除多余的空格
            goal_description = " ".join(goal_description.split())
            # 不分大小写的匹配
            # *******************

            # # ********************
            if self.args.subgoal_file is not None:
            # 用plan的数据重新计算以上变量
                list_of_actions_old, categories_in_inst_old, second_object_old, caution_pointers_old,sliced_old = list_of_actions, categories_in_inst, second_object, caution_pointers, sliced
                # plan_id = [i, for i, item in enumerate(self.plan_data) if item['id'] == episode_no]
                list_of_actions = self.plan_data[self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank]['plan']
                goal_name = self.plan_data[self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank]['high_level_instructions']

                # 判断goal_name和traj里面的goal_name匹不匹配，如果不匹配记录信息，方便调试
                # 去除标点符号
                goal_name = goal_name.replace("'","")
                # 去除多余的空格
                goal_name = " ".join(goal_name.split())

                if not goal_description.lower() == goal_name.lower():
                    message = {"goal_description_traj":goal_description,"goal_description_llama":\
                        goal_name, "episode number":episode_no}
                    json.dump(message, open(f"add_byme/data/goal_description_not_match_{self.args.dn}", 'a'))

                # if self.args.run_idx_file is None:
                #     # 将from_id转换为goal再匹配到useful_plan里的plan
                #     sene_instruction_data = json.load(open("/home/cyw/task_planning/prompter/add_byme/data/sene_instruction.json",'r'))["high_level_instructions"]
                #     # 记录了各个轨迹的goal name和对应的轨迹的路径
                #     goal_name = sene_instruction_data[episode_no]
                #     idx_in_usefulplan = goal.index(goal_name)
                #     list_of_actions_orig = plan[idx_in_usefulplan]
                # 现在的文件是完整的goal plan， 用不到这个

                # 注意这个plan文件的排列顺序需要和id的排列顺序一致
                # list_of_actions_orig = plan[self.scene_pointer* self.args.num_processes + self.rank+1]
                # list_of_actions = [i for i in list_of_actions_orig if i[1] != "FindObject"]
                if len(list_of_actions) != 0:
                    list_of_actions, categories_in_inst, second_object, caution_pointers,sliced = get_list_of_highlevel_actions_from_plan(list_of_actions,self.args.nonsliced)
                else:
                    text = "attention: the plan is []"
                    list_of_actions = list_of_actions_old
                if self.args.record_replan:
                    if list_of_actions!=list_of_actions_old or set(categories_in_inst) !=set(categories_in_inst_old) or second_object!=second_object_old or caution_pointers!= caution_pointers_old or sliced != sliced_old:
                        text = f"goal is {goal_name} \n the number of this episode is {episode_no} \n the params has change,old params is:\n {list_of_actions_old,categories_in_inst_old,second_object_old,caution_pointers_old}\nthe new params is:\n {list_of_actions,categories_in_inst,second_object,caution_pointers}"
                    if text is not None:
                        print(text)
                            # print("the params has change,old params is:")
                            # print(list_of_actions_old,categories_in_inst_old,second_object_old,caution_pointers_old)
                            # print("the new params is:")
                            # print(list_of_actions,categories_in_inst,second_object,caution_pointers)
                        with open(f"{self.args.result_file}logs/plan_com_{self.args.from_idx}_{self.args.to_idx}.txt",'a') as f:
                            f.write(text+"\n")
                # # # ***************

            # ******************
            if self.args.record_replan:
                if text is None:
                    text = f"goal is {goal_description} \n the number of this episode is {episode_no} \n the params is:\n {list_of_actions,categories_in_inst,second_object,caution_pointers}"
                    print(text)
                # 向文本文件中追加text
                with open(f"{self.args.result_file}logs/replan_{self.args.from_idx}_{self.args.to_idx}.txt",'a') as f:
                    f.write(text+"\n")
            # *****************

            self.sliced = sliced
            self.caution_pointers = caution_pointers
            if self.args.no_caution_pointers:
                self.caution_pointers = []
            self.print_log("list of actions is ", list_of_actions)
            self.task_type = task_type
            self.second_object = second_object

            self.reset_total_cat_new(categories_in_inst)
            obs, info = self.setup_scene(traj_data, task_type, r_idx, self.args) 
            goal_name = list_of_actions[0][0]
            info = self.reset_goal(True, goal_name, None)

            # ai2thor ver. 2.1.0 only allows for 90 degrees left/right turns
            # lr_turn = 90
            # num_turns = 360 // lr_turn
            curr_angle = self.camera_horizon
            # self.lookaround_seq = [f"LookUp_{curr_angle}"] + [f"RotateLeft_{lr_turn}"] * (360 // lr_turn) + [f"LookDown_{curr_angle}"]
            # self.lookaround_seq = [f"RotateLeft_{lr_turn}"] * num_turns + [f"LookUp_{curr_angle}"] + [f"RotateLeft_{lr_turn}"] * num_turns + [f"LookDown_{curr_angle}"]
            self.lookaround_seq = [f"LookUp_{curr_angle}"] + [f"RotateLeft_90"] * 3 + [f"LookDown_{curr_angle}"] + [f"RotateRight_90"] * 2
            # self.lookaround_seq = [f"RotateLeft_{lr_turn}"] * (360 // lr_turn)

            if "no_lookaround" in self.args.mlm_options:
                self.lookaround_seq = []

            self.lookaround_counter = 0

            self.target_search_sequence = list()
            self.centering_actions = list()
            self.centering_history = list()
            self.force_move = False

            if task_type == 'look_at_obj_in_light':
                # look_at_obj_in_light原本代码里如果是look_at_obj类型，一定会有FloorLamp物体，但是LLM给出的plan不一定有FlooLamp，甚至可能类别都不一样，因此这里可能报错，在读取catagory里面修改了
                self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp']
                self.cat_equate_dict['DeskLamp'] = 'FloorLamp' #if DeskLamp is found, consider it as FloorLamp
            # 上面几行是promtper原本就有的代码

            # if self.args.subgoal_file is None:
            #     if task_type == 'look_at_obj_in_light':
            #         self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp']
            #         self.cat_equate_dict['DeskLamp'] = 'FloorLamp' #if DeskLamp is found, consider it as FloorLamp   
            #         #还是不行，万一有子目标文件，并且原本的任务类型还是desklamp呢  

            if sliced:
                self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife']
                self.cat_equate_dict['ButterKnife'] = 'Knife' #if ButterKnife is found, consider it as Knife

            # actions_dict = {'task_type': task_type, 'list_of_actions': list_of_actions, 'second_object': second_object, 'total_cat2idx': self.total_cat2idx, 'sliced':self.sliced}
            # ****************
            # add caution pointers
            actions_dict = {'task_type': task_type, 'list_of_actions': list_of_actions, 'second_object': second_object, 'total_cat2idx': self.total_cat2idx, 'sliced':self.sliced,'caution_pointers':caution_pointers}
            # ****************
            self.print_log('total cat2idx is ', self.total_cat2idx)

            self.actions_dict = actions_dict
            
            # ********************
            # 记录机器人当前视角下一共寻找了多少次新目标
            self.num_newgoal_curhorizon = {45.0:0,0.0:0,-45.0:0}
            # 设置当前explored area缩减的大小
            self.explored_area_reduce_size = self.args.explored_selem
            # 该变量伴随整个轨迹
            # 允许的最大的可以请求下一个目标的次数
            self.max_next_goal_request = self.args.max_next_goal_request
            # **************
        except:
            import traceback
            traceback.print_exc()
            print("the code above not exacute all!")
            self.print_log("Scene pointers exceeded the number of all scenes, for env rank", self.rank)
            obs = np.zeros(self.obs.shape)
            info = self.info
            actions_dict = None

        self.seg.update_agent(self)
            
        return obs, info, actions_dict
    
    def load_next_scene(self, load):
        if load == True:
            self.scene_pointer += 1
            obs, info, actions_dict = self.load_initial_scene()
            return obs, info, actions_dict
        
        return self.obs, self.info, self.actions_dict
     
    def update_last_three_sidesteps(self, new_sidestep):
        self.last_three_sidesteps = self.last_three_sidesteps[:2]
        self.last_three_sidesteps = [new_sidestep] + self.last_three_sidesteps

    def lookDiscretizer(self, action, mask=None, do_postprocess=True):
        # LookDown/LookUp should be performed in increments of 15 degrees, according to https://github.com/askforalfred/alfred/issues/87
        # so discretize the actions by 15 degrees
        act, angle = self.splitAction(action)
        for _ in range(angle // 15 - 1):
            obs, rew, done, info, success, a, target_instance, err, api_action = self.va_interact_new(
                f"{act}_{15}", mask, False)
            if not success:
                # abort if a look action fails
                if do_postprocess:
                    obs, seg_print = self.preprocess_obs_success(success, obs)
                return obs, rew, done, info, success, a, target_instance, err, api_action

        return self.va_interact_new(f"{act}_{15}", mask, do_postprocess)

    def splitAction(self, action):
        tokens = action.split("_")

        if len(tokens) == 1:
            return (tokens[0], None)

        act, num = tokens
        num = int(float(num))
        return act, num

    def va_interact_new(self, action, mask=None, do_postprocess=True):
        if ("Look" in action) and (self.splitAction(action)[-1] > 15):
            return self.lookDiscretizer(action, mask, do_postprocess)

        if "Angle" in action:
            return self.set_back_to_angle(self.splitAction(action)[-1])

        self.print_log(f"action taken in step {self.steps_taken}: {action}")
        self.last_action_ogn = action

        obs, rew, done, info, success, a, target_instance, err, api_action = \
                                super().va_interact(action, mask)

        if self.args.save_pictures:
            cv2.imwrite(
                self.picture_folder_name + "rgb/"+ "rgb_" + str(self.steps_taken) + ".png",
                obs[:3, :, :].transpose((1, 2, 0)))
            # ************
            if mask is not None:
                cv2.imwrite(
                self.picture_folder_name + "mask/"+ "mask_" + str(self.steps_taken) + ".png",
                mask*255)
            # ************

        if not(success):
            self.fails_cur +=1

        if self.args.approx_last_action_success:
            success = CH._get_approximate_success(self.prev_rgb, self.event.frame, action)

        self.last_success = success

        #Use api action just for leaderboard submission purposes, as in https://github.com/askforalfred/alfred/blob/master/models/eval/leaderboard.py#L101
        self.actions = CH._append_to_actseq(success, self.actions, api_action)
        self.seg.update_agent(self)

        if do_postprocess:
            obs, seg_print = self.preprocess_obs_success(success, obs)

        return obs, rew, done, info, success, a, target_instance, err, api_action

    def set_back_to_angle(self, angle_arg):
        delta_angle = self.camera_horizon - angle_arg
        direction = "Up" if delta_angle >= 0 else "Down"
        delta_angle = abs(int(np.round(delta_angle / 15)) * 15)  # round to the nearest multiple of 15
        action = f"Look{direction}_{delta_angle}"

        return self.va_interact_new(action)

    def reset_goal(self, truefalse, goal_name, consecutive_interaction):
        # ***************add by me********
        print(f"reset goal, current goal is {goal_name}")
        # *******************
        if self.args.ignore_sliced:
            goal_name = goal_name.replace('Sliced', '')

        if truefalse == True:
            self.goal_name = goal_name
            if "Sliced" in goal_name :
                self.cur_goal_sliced = self.total_cat2idx[goal_name.replace('Sliced', '')]
            else:
                self.cur_goal_sliced = None
            self.goal_idx = self.total_cat2idx[goal_name]
            self.info['goal_cat_id'] = self.goal_idx
            self.info['goal_name'] = self.goal_name

            self.prev_number_action = None
            self.where_block = []

            self.search_end = False

            self.cur_goal_sem_seg_threshold_small = self.args.sem_seg_threshold_small
            self.cur_goal_sem_seg_threshold_large = self.args.sem_seg_threshold_large
            
            if abs(int(self.camera_horizon)  - 45) >5 and consecutive_interaction is None:
                obs, rew, done, info, success, _, target_instance, err, _ = self.set_back_to_angle(45)
            
            self.info['view_angle'] = self.camera_horizon

            # ******************
            # self.num_newgoal_curhorizon = 0
            # self.num_newgoal_curhorizon = {"45":0,"0":0,"-45":0}
            self.num_newgoal_curhorizon = {45.0:0,0.0:0,-45.0:0}

            self.interaction_fail_pose = [] #记录下交互失败的位置
            # *************
        
        return self.info
        
    def reset_total_cat_new(self, categories_in_inst):
        total_cat2idx = {}

        total_cat2idx["Knife"] =  len(total_cat2idx)
        total_cat2idx["SinkBasin"] =  len(total_cat2idx)
        if self.args.sem_policy_type != "none":
            for obj in constants.map_save_large_objects:
                if not(obj == "SinkBasin"):
                    total_cat2idx[obj] = len(total_cat2idx)
        # 这里长度25
        # total_cat2idx["Cup"] = len(total_cat2idx)
        # total_cat2idx["Mug"] = len(total_cat2idx)
        # total_cat2idx["GlassBottle"] = len(total_cat2idx)
        # total_cat2idx["WineBottle"] = len(total_cat2idx)
        # total_cat2idx["SprayBottle"] = len(total_cat2idx)

        # total_cat2idx["SaltShaker"] = len(total_cat2idx)
        # total_cat2idx["PepperShaker"] = len(total_cat2idx)
        # total_cat2idx["SoapBottle"] = len(total_cat2idx)


        start_idx = len(total_cat2idx)  # 1 for "fake"
        start_idx += 4 *self.rank
        cat_counter = 0
        # assert len(categories_in_inst) <=6
        #Keep total_cat2idx just for 
        for v in categories_in_inst:
            if not(v in total_cat2idx):
                total_cat2idx[v] = start_idx+ cat_counter
                cat_counter +=1 
        
        total_cat2idx["None"] = 1 + 1 + 5 * self.args.num_processes-1
        if self.args.sem_policy_type != "none":
            # total_cat2idx["None"] = total_cat2idx["None"] + 23
            total_cat2idx["None"] = total_cat2idx["None"] + 30
        self.total_cat2idx = total_cat2idx
        self.goal_idx2cat = {v:k for k, v in self.total_cat2idx.items()}
        print("self.goal_idx2cat is ", self.goal_idx2cat)
        self.cat_list = categories_in_inst
        self.args.num_sem_categories = 1 + 1 + 1 + 5 * self.args.num_processes 
        if self.args.sem_policy_type != "none":
            # self.args.num_sem_categories = self.args.num_sem_categories + 23
            self.args.num_sem_categories = self.args.num_sem_categories + 30
        print(f"self.args.num_sem_categories are {self.args.num_sem_categories}")

    def setup_scene(self, traj_data, task_type, r_idx, args, reward_type='dense'):
        args = self.args

        obs, info = super().setup_scene(traj_data,task_type, r_idx, args, reward_type)
        obs, seg_print = self._preprocess_obs(obs)

        self.obs_shape = obs.shape
        self.obs = obs

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.col_width = 5
        self.curr_loc = [args.map_size_cm/100.0/2.0,
                         args.map_size_cm/100.0/2.0, 0.]
        self.last_action_ogn = None
        self.seg_print = seg_print

        return obs, info

    def getWorldCoord3DObs(self, depth_map):
        hei, wid = depth_map.shape[:2]
        cam_mat = du.get_camera_matrix(wid, hei, self.args.hfov)

        pcloud = du.get_point_cloud_from_z(depth_map, cam_mat)
        agent_view = du.transform_camera_view(pcloud, self.args.camera_height, -self.camera_horizon)

        scale = 100.0 / self.args.map_resolution
        shift_loc = [self.curr_loc[0] * scale, self.curr_loc[1] * scale, np.deg2rad(self.curr_loc[2])]
        global_view = du.transform_pose(agent_view * scale, shift_loc)

        return agent_view, global_view

    def is_visible_from_mask(self, mask, visibility_threshold=1):
        # use bottom 25 percentile for a more robust depth estimation
        # use distance horizontal to ground (and not distance from the camera) to check if it is visible
        if mask is None or np.sum(mask) == 0:
            return None, False

        depth_map = self.learned_depth_frame if self.args.learned_visibility else self.event.depth_frame / 1000

        # xyz is agent centric, units in meters
        # +x axis: right, +y axis: away from the agent, +z axis: up
        xyz, global_xyz = self.getWorldCoord3DObs(depth_map)

        mask_coords = np.where(mask)
        xyz_roi = xyz[mask_coords[0], mask_coords[1], :]
        global_xyz_roi = global_xyz[mask_coords[0], mask_coords[1], :]
        dists = np.sqrt(xyz_roi[:, 0] ** 2 + xyz_roi[:, 1] ** 2)
        gnd_dist = np.percentile(dists, 25)
        rep_indx = np.argmin(np.abs(dists - gnd_dist))

        reachable = gnd_dist <= visibility_threshold
        self.print_log(f"object is {gnd_dist}m away, reachable: {reachable}")
        return [int(coord) for coord in global_xyz_roi[rep_indx]], reachable

    def is_visible_from_mask_depth(self, mask, visibility_threshold=1):
        # for ablation studies
        # use bottom 25 percentile for a more robust depth estimation
        # use depth to check if it is visible
        if mask is None or np.sum(mask) == 0:
            return None, False

        depth_map = self.learned_depth_frame if self.args.learned_visibility else self.event.depth_frame / 1000

        # xyz is agent centric, units in meters
        # +x axis: right, +y axis: away from the agent, +z axis: up
        _, global_xyz = self.getWorldCoord3DObs(depth_map)

        mask_coords = np.where(mask)
        global_xyz_roi = global_xyz[mask_coords[0], mask_coords[1], :]

        dists = depth_map[mask_coords[0], mask_coords[1]]
        gnd_dist = np.percentile(dists, 25)
        rep_indx = np.argmin(np.abs(dists - gnd_dist))

        reachable = gnd_dist <= visibility_threshold
        self.print_log(f"object is {gnd_dist}m away, reachable: {reachable}")
        return [int(coord) for coord in global_xyz_roi[rep_indx]], reachable

    def preprocess_obs_success(self, success, obs):
        obs, seg_print = self._preprocess_obs(obs) #= obs, seg_print
        self.obs = obs
        self.seg_print = seg_print
        return obs, seg_print

    def genPickedUpMask(self, rgb, prev_rgb, original_seg):
        if original_seg is None:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), np.uint8)

        # determine the location and shape of the picked up object by
        # checking which pixels have changed from the previous frame
        diff_thresh = 5
        bgr_s = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1]
        oldbgr_s = cv2.cvtColor(prev_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1]
        diff_mask = cv2.absdiff(bgr_s, oldbgr_s) > diff_thresh
        # For higher resolution
        diff_mask = cv2.resize(diff_mask.astype(np.float32), (300, 300))
        diff_mask = diff_mask[0]

        # diff_mask contains the location of the picked-up object, before and after the pickup action
        # so remove the mask before pickup
        # TODO: this fails for pixels that belong to the picked up object, before and after picking up
        original_seg = cv2.dilate(original_seg, np.ones((5, 5)))
        cleaned_mask = np.logical_and(diff_mask, np.logical_not(original_seg))
        cleaned_mask = cv2.dilate(cleaned_mask.astype(np.uint8), np.ones((5, 5)))

        return cleaned_mask

    def goBackToLastSlicedLoc(self, interaction):
        if "no_slice_replay" in self.args.mlm_options:
            return False, False

        pickup_sliced = (interaction == "PickupObject") and ("Sliced" in self.goal_idx2cat[self.goal_idx])
        on_path_back = pickup_sliced and (self.sliced_pose is not None)
        ready_to_slice = pickup_sliced and (not on_path_back) and (self.sliced_mask is not None)
        return on_path_back, ready_to_slice

    def consecutive_interaction(self, interaction, interaction_mask):
        # **************
        old_interaction_mask = interaction_mask
        # *************
        # if interaction == "PutObject" and self.last_action_ogn == "OpenObject":
        if interaction == "PutObject" and self.last_action_ogn == "OpenObject" and self.goal_name in constants.OPENABLE_CLASS_LIST: #需要还是放在原来开的东西里面才用之前开的mask
            interaction_mask = self.open_mask
        elif interaction == "CloseObject":
            interaction_mask = self.open_mask

        # Changed by Trisoil
        # have_used = False
        # if interaction_mask is None:
        #     interaction_mask = self.put_rgb_mask
        #     have_used = True
        obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
            interaction, interaction_mask, do_postprocess=False)

        if interaction == "PickupObject":
            if not success:

                interaction_mask = self.put_rgb_mask
                # # obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
                #     interaction, interaction_mask, do_postprocess=False)
                # ************
                # 如果刚开始拿的话，是不会有mask的，这里加一个判断条件
                if interaction_mask is not None:
                    obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
                    interaction, interaction_mask, do_postprocess=False)

                # NOTE 拿起东西的时候没有成功，直接使用put mask，这样过于简单粗暴，会造成动作浪费;另外，之后需要加上鑫垚的用depth取mask的


            if success:
                self.picked_up = True
                self.picked_up_mask = self.genPickedUpMask(
                    self.event.frame, self.prev_rgb, interaction_mask)
                # ********************
                self.pick_up_obj = self.goal_name
                
                # Changed by Trisoil
                self.holding_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame, self.learned_depth_frame, self.dino_depth_pred(self.event.frame).squeeze())
                
                # 尝试去除拿起来的物体，使其不遮挡视线，调试用
                # *****************

        elif interaction == "PutObject" and success:
            self.picked_up = False

            # Changed by Trisoil
            self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame, self.learned_depth_frame, self.dino_depth_pred(self.event.frame).squeeze())
            # self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame)
            self.picked_up_mask = None
            self.holding_mask = None
    
        # ****************
        # 如果上一步放了东西的话，之前的open mask可能会失效，因此用分割给的mask NOTE 有可能这时候已经走动了，给的是另一个mask，效果需要测试
        elif interaction == "CloseObject" and not success:
            interaction_mask = old_interaction_mask
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(
        interaction, interaction_mask, do_postprocess=False)
        # *******************


        # update the depth information after picked_up-related information are updated
        obs, seg_print = self.preprocess_obs_success(success, obs)

        # store the current interact mask in anticipation to use it in the future
        if success:
            if interaction == "OpenObject":
                self.open_mask = copy.deepcopy(interaction_mask)
            elif interaction == "SliceObject":
                self.sliced_mask = copy.deepcopy(interaction_mask)
                self.sliced_pose = [self.curr_loc_grid[0], self.curr_loc_grid[1], self.curr_loc[2]]

                # self.curr_loc[2]这里记录的应该是角度
            # *********************
            # 开了大物体，会阻挡视线，之前的visited 就不能用了，并且开东西之后，在关上之前，不应该走动太远
            if interaction == "OpenObject":
                self.open_object =  True
            if interaction == "CloseObject":
                self.open_object = False
            # *******************
        self.info = info

        # **************
        # 如果交互失败，记录下当前交互失败的位置
        if self.args.drop_interaction_fail_loc and not success:
            # self.interaction_fail_pose.append([self.curr_loc_grid[0], self.curr_loc_grid[1], self.curr_loc[2]])
            # 按照这种机器人可能就站在一个位置，但是角度和之前的不一样
            # NOTE 有些时候交互失败不是因为位置不好，可能是mask给的不好，还需要实验验证
            self.interaction_fail_pose.append([self.curr_loc_grid[0], self.curr_loc_grid[1]])
            # debug用 
            print(f"remember interaction fail loc: {[self.curr_loc_grid[0], self.curr_loc_grid[1], self.curr_loc[2]]}")
        # *************

        return obs, rew, done, info, success, err

    def which_direction(self, interaction_mask):
        if interaction_mask is None:
            return 150
        widths = np.where(interaction_mask !=0)[1]
        center = np.mean(widths)
        return center

    def isTrapped(self, traversible, planning_window, area_thresh=400):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            traversible.astype(np.uint8), connectivity=4)

        [gx1, gx2, gy1, gy2] = planning_window
        agent_loc = self.meter2coord(self.curr_loc[1], self.curr_loc[0], gy1, gx1, traversible.shape)
        roi_label = labels[agent_loc[0], agent_loc[1]]
        area = stats[roi_label][4]
        trapped = area < area_thresh
        if trapped:
            self.print_log(f"Current area is too small (area: {area}), I think I'm trapped!")
        return trapped

    def get_traversible_new(self, grid, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        obstacles = grid[y1:y2, x1:x2]

        # add collision map
        collision_map = self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1

        # dilate the obstacles so that the agent doesn't get too close to the obstacles
        # but do not dilate the obstacles added via collision_map
        obstacles = skimage.morphology.binary_dilation(obstacles, self.selem)

        obstacles[collision_map] = 1

        # remove visited path from the obstacles
        # obstacles[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        # *****************
        # NOTE 如果是开了某个容器，之前走过的位置这时候就可能不能再走了
        if not self.open_object:
            obstacles[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        # ***************
        traversible = np.logical_not(obstacles)

        # check if the agent is trapped
        # if the agent is trapped, only use the collision_map as the obstacle map


        if self.isTrapped(traversible, planning_window):
            obstacles = np.zeros_like(obstacles)
            obstacles[collision_map] = 1
            traversible = np.logical_not(obstacles)

        return traversible

    def get_traversible(self, grid, planning_window):
        if "new_obstacle_fn" in self.args.mlm_options:
            return self.get_traversible_new(grid, planning_window)
        # get_traversible_new 和 get_traversible内容一样

        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        obstacles = grid[y1:y2, x1:x2]

        # add collision map
        collision_map = self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1

        # dilate the obstacles so that the agent doesn't get too close to the obstacles
        # but do not dilate the obstacles added via collision_map
        obstacles = skimage.morphology.binary_dilation(obstacles, self.selem)
        obstacles[collision_map] = 1

        # remove visited path from the obstacles
        obstacles[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = np.logical_not(obstacles)

        # check if the agent is trapped
        # if the agent is trapped, only use the collision_map as the obstacle map
        if self.isTrapped(traversible, planning_window):
            obstacles = np.zeros_like(obstacles)
            obstacles[collision_map] = 1
            traversible = np.logical_not(obstacles)

        return traversible

    def getGoalDirection(self, planner_inputs):
        agent_x, agent_y, curr_angle, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        agent_x = int(agent_x * 100.0/self.args.map_resolution - gx1)
        agent_y = int(agent_y * 100.0/self.args.map_resolution - gy1)

        ys, xs = np.where(planner_inputs["goal"])

        distances = (xs - agent_x) ** 2 + (ys - agent_y) ** 2
        mindx = np.argmin(distances)
        target_x, target_y = xs[mindx], ys[mindx]

        dx = target_x - agent_x
        dy = target_y - agent_y

        if abs(dx) > abs(dy):
            if dx > 0:
                target_angle = 0
            else:
                target_angle = 180
        else:
            if dy > 0:
                target_angle = 90
            else:
                target_angle = 270

        delta_angle = (target_angle - curr_angle) % 360
        return delta_angle

    def arrivedAtSubGoal(self, planner_inputs, goal_free_and_not_shifted):
        # TODO goal_free_and_not_shifted什么意思
        self.print_log("Arrived at subgoal")

        if goal_free_and_not_shifted or ("lookaroundAtSubgoal" in self.args.mlm_options):
            self.print_log("Looking in all 4 directions")
            return ["RotateLeft_90"] * 3 + ["Angle_0"] + ["RotateLeft_90"] * 3 + ["Angle_45", "Done"]

        self.print_log("Looking only in 1 direction")
        actions = list()

        cur_hor = np.round(self.camera_horizon, 4)
        if abs(cur_hor-45) > 5:
            actions.append("Angle_45")

        # figure out which direction the agent must turn to
        delta_angle = self.getGoalDirection(planner_inputs)
        if delta_angle == 0:  # no need to turn
            pass
        elif delta_angle == 90:
            actions += ["RotateLeft_90"]
        elif delta_angle == 270:
            actions += ["RotateRight_90"]
        elif delta_angle == 180:
            # rotate left twice to do a 180 degrees turn
            actions += ["RotateLeft_90", "RotateLeft_90"]
        
        # TODO 找到一个正确的角度后,确定是否要openobject

        # the agent is looking at the correct direction, so look up and down
        # NOTE: last item in actions needs to be "Done", since that is how getNextAction() knows
        # if search sequence failed
        actions += ["LookUp_0", "Angle_0", "Angle_45", "Done"]

        return actions

    def getSceneNum(self):
        return self.scene_names[self.scene_pointer]["scene_num"]

    def centerAgentNone(self, planner_inputs, interaction_mask):
        # no agent centering, for ablation study
        self.print_log("agent centered")
        return True, False

    def centerAgentSimple(self, planner_inputs, interaction_mask):
        # 2D agent centering in FILM, for ablation study

        # figure out which direction to move and generate a sequence of actions to center the agent
        self.print_log("check centered")

        self.centering_actions = list()
        wd = self.which_direction(interaction_mask)

        width = interaction_mask.shape[1]
        margin = 65
        if wd > (width - margin):
            self.print_log(f"stepping to right, wd={wd}")
            turn_angle = -90
            self.centering_actions = ["RotateRight_90", "MoveAhead_25", "RotateLeft_90"]
            self.centering_history.append('R')

        elif wd < margin:
            self.print_log(f"stepping to left, wd={wd}")
            turn_angle = 90
            self.centering_actions = ["RotateLeft_90", "MoveAhead_25", "RotateRight_90"]
            self.centering_history.append('L')

        else:
            self.print_log("agent centered")
            return True, False

        # check for collisions
        grid = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = [int(start_y * 100.0/self.args.map_resolution - gx1),
                 int(start_x * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        traversible = self.get_traversible(grid, planning_window)

        turn_angle = start_o + turn_angle
        movable = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, start, turn_angle)
        if not movable:
            # do not include MoveAhead, since the agent will collide
            self.centering_actions = [self.centering_actions[0]]

        # check for repeated left and right movement
        # assumption here is that the agent should always move in one direction
        repeating_steps = len(set(self.centering_history)) > 1
        if repeating_steps:
            self.print_log("adjust movement repetition detected")
            self.centering_history = list()
            self.centering_actions = list()

            self.print_log(f"aborting centering sequence")
        abort_centering = repeating_steps

        return False, abort_centering

    def centerAgent(self, planner_inputs, interaction_mask):
        # returns True if agent is centered, else it sets centering_actions for next actions

        def searchTargetPt(depth_map, camera_matrix, target_pt, search_range=2):
            '''
            在target_pt附近找一个有可信深度预测的点
            '''
            # target_pt should be ordered (y, x)

            # search for a coordinate near target_pt with a reliable depth prediction
            reliable_pt = None
            h, w = depth_map.shape[:2]
            pty, ptx = target_pt
            for i in range(search_range):
                lower_y, upper_y = max(0, pty - i), min(h, pty + i + 1)
                lower_x, upper_x = max(0, ptx - i), min(w, ptx + i + 1)
                ys, xs = np.where(depth_map[lower_y:upper_y, lower_x:upper_x])
                if len(xs) > 0:
                    reliable_pt = (lower_y + ys[0], lower_x + xs[0])
                    break

            if reliable_pt is None:
                return None

            xyz = du.get_point_cloud_from_z(depth_map, camera_matrix)
            return xyz[reliable_pt[0], reliable_pt[1], :]

        # figure out which direction to move and generate a sequence of actions to center the agent
        self.print_log("check centered")

        self.centering_actions = list()
        wd = self.which_direction(interaction_mask)

        # rotation matrix and translation vector are defined from the current coordinate system,
        # where +x is right, +y is the direction in which the agent is facing, and +z is up, in accordance with depth_utils
        Rmat = np.eye(3)

        # left and right commands for agent are with respect to the world coordinate, but
        # the agent could be looking up/downward, making the camera coordinate different from the world coordinate,
        # so correct it
        x_axis = np.asarray([1, 0, 0])
        z_axis = np.asarray([0, 0, 1])
        head_angle = -self.camera_horizon  # camera_horizon is defined as negative rotation around the x axis of the camera coordinate
        correction_R = ru.get_r_matrix(x_axis, -head_angle / 180 * np.pi)  # undo rotation by @head_angle
        world_z_vec = correction_R @ z_axis

        width = interaction_mask.shape[1]
        if wd > (width - self.center_margin):
            self.print_log(f"stepping to right, wd={wd}")
            turn_angle = -90
            self.centering_actions = ["RotateRight_90", "MoveAhead_25", "RotateLeft_90"]
            self.centering_history.append('R')

            Rmat_tmp = ru.get_r_matrix(world_z_vec, np.pi / 2)
            tvec = np.asarray([-25, 0, 0])  # depth values in obs is in centimeters

        elif wd < self.center_margin:
            self.print_log(f"stepping to left, wd={wd}")
            turn_angle = 90
            self.centering_actions = ["RotateLeft_90", "MoveAhead_25", "RotateRight_90"]
            self.centering_history.append('L')

            Rmat_tmp = ru.get_r_matrix(world_z_vec, -np.pi / 2)
            tvec = np.asarray([25, 0, 0])

        else:
            self.print_log("agent centered")
            return True, False
            # 不需要往任何方向移动的时候返回True False?第二个变量是abord_centering

        # check for collisions
        grid = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = [int(start_y * 100.0/self.args.map_resolution - gx1),
                 int(start_x * 100.0/self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        traversible = self.get_traversible(grid, planning_window)

        turn_angle = start_o + turn_angle
        movable = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, start, turn_angle)
        if not movable:
            # do not include MoveAhead, since the agent will collide
            self.centering_actions = [self.centering_actions[0]]
            Rmat = Rmat_tmp
            tvec = np.zeros_like(tvec)

        # check for repeated left and right movement
        # assumption here is that the agent should always move in one direction
        repeating_steps = len(set(self.centering_history)) > 1
        if repeating_steps:
            self.print_log("adjust movement repetition detected")

        # calculate where the target coordinate is, after all actions in centering_actions are executed
        # 1. calculate the target coordinate and find its 3D coordinate
        imask_h, imask_w = interaction_mask.shape[:2]
        ratio = self.args.frame_height / interaction_mask.shape[0]
        curr_target_pt = np.asarray([np.average(indices) * ratio for indices in np.where(interaction_mask)], dtype=int)

        # search for coordinates near curr_target_pt with a reliable depth prediction
        cam_mat = du.get_camera_matrix(self.args.frame_width, self.args.frame_height, self.args.hfov)
        curr_target_pt_3d = searchTargetPt(self.obs[3, :, :], cam_mat, curr_target_pt, search_range=3)
        if curr_target_pt_3d is None:
            proj_pt = (-1, -1)
        else:
            # 2. project the 3D coordinate to future camera coordinate
            proj_pt = du.projectPoints(curr_target_pt_3d, Rmat, tvec, cam_mat.array, self.args.frame_height) / ratio

        self.print_log(f"expects target at {curr_target_pt / ratio} to move to {proj_pt} after adjustment")
        self.centering_actions += ["Done", proj_pt]

        # abort if the expected coordinate is out of visual bounds
        target_oob = (proj_pt[0] < 0) or (proj_pt[1] < 0) or (proj_pt[0] >= imask_h) or (proj_pt[1] >= imask_w)
        if target_oob:
            self.print_log("tracking target out of bounds")

        abort_centering = repeating_steps | target_oob #重复的动作，或者target out of boundary
        if abort_centering:
            self.centering_history = list()
            self.centering_actions = list()
            abort_centering = True

            self.center_margin = max(0, self.center_margin - 10)
            self.print_log(f"aborting centering sequence, margin reduced to {self.center_margin}")

        return False, abort_centering

    def interactionSequence(self, planner_inputs, interaction_mask):
        # interact with the target
        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        interaction = list_of_actions[pointer][1]

        obs, rew, done, info, success, err = self.consecutive_interaction(
            interaction, interaction_mask)
        self.force_move = not success

        return obs, rew, done, info, success, err

    def getNextAction(self, planner_inputs, during_lookaround, target_offset, target_xyz, force_slice_pickup):
        next_goal = False
        reached_goal = False

        # lookaround sequence
        if during_lookaround:
            self.print_log("action chosen: lookaround")
            # initial lookaround sequence
            action = self.lookaround_seq[self.lookaround_counter]
            self.lookaround_counter += 1
            next_goal = len(self.lookaround_seq) == self.lookaround_counter

        # if sliced mask is available and now want to pick up the sliced object,
        # go back to the location in which the agent sliced last time
        elif force_slice_pickup:
            self.print_log("action chosen: force_slice_pickup, overriding goal")
            action, _, goal_free_and_not_shifted = self._plan(
                planner_inputs, target_offset, self.sliced_pose)
                # TODO 之后调试一下这里是不是没有调整到合适的朝角

            # upon arrival, try to pick up the sliced object
            if action in ["ReachedSubgoal", "<<stop>>"]:
                action = "SliceObjectFromMemory"

        # # can see the target, but not close enough
        # elif target_xyz is not None:
        #     self.print_log("action chosen: target_visible")
        #     mod_goal = np.zeros_like(planner_inputs["goal"])
        #     mod_goal[target_xyz[1], target_xyz[0]] = 1
        #     action, next_subgoal = self._plan(planner_inputs, target_offset, mod_goal, False)

        # arrived at the subgoal, search sequence
        elif len(self.target_search_sequence) != 0:
            self.print_log("action chosen: target search")
            action = self.target_search_sequence.pop(0)
            reached_goal = True
            if self.target_search_sequence[0] == "Done":
                next_goal = True
                self.target_search_sequence = list()

        # # check to make sure that the agent is looking at 45 deg below horizon
        # elif self.camera_horizon != 45 :
        #     self.print_log("action chosen: correct agent angle to 45")
        #     action = "Angle_45"

        # **********************
        # check to make sure that the agent is looking at 45 deg below horizon
        elif self.camera_horizon != 45 and not self.args.change_altitude :
            self.print_log("action chosen: correct agent angle to 45")
            action = "Angle_45"
        # *****************

        else:
            self.print_log("action chosen: _plan")
            goal_coord = np.where(planner_inputs['goal'])
            goal_coord = [goal_coord[0][0], goal_coord[1][0]]






            action, next_goal, goal_free_and_not_shifted = self._plan(
                planner_inputs, target_offset, goal_coord)
            
            # ************************
            if next_goal:
                print("plan request for next goal")
            # ********************

        # found the target object and arrived at the location
        if action in ["ReachedSubgoal", "<<stop>>"]:
            self.target_search_sequence = self.arrivedAtSubGoal(planner_inputs, goal_free_and_not_shifted)
            action = self.target_search_sequence.pop(0)
            reached_goal = True
        # look around 完毕，搜索完毕，走到目的地会请求next_goal
        return action, next_goal, reached_goal

    def check4interactability(self, target_coord, visibility_threshold):
        if self.args.use_sem_seg:
            interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(
                self.goal_idx2cat[self.goal_idx], target_coord, self.args.ignore_sliced)
        else:
            interaction_mask = self.seg.get_instance_mask_from_obj_type(
                self.goal_idx2cat[self.goal_idx], target_coord)

        if "visibility_use_depth_distance" in self.args.mlm_options:
            target_loc, reachable = self.is_visible_from_mask_depth(interaction_mask, visibility_threshold)
        else:
            target_loc, reachable = self.is_visible_from_mask(interaction_mask, visibility_threshold)
            #is_visible_from_mask代码中：reachable = gnd_dist <= visibility_threshold， visibility_threshold = 1

        # guess the interaction mask and force interaction if the target tracked during centering sequence is lost
        target_lost = (target_coord is not None) and ((interaction_mask is None) or (not reachable))
        if target_lost:
            self.print_log(f"target expected at {target_coord} is lost, force interaction")
            interaction_mask = np.zeros((300, 300), dtype=np.float64)
            interaction_mask[int(target_coord[0]), int(target_coord[1])] = 1
            reachable = True

        return target_loc, reachable, interaction_mask

    def isNewGoal(self, curr_goal):
        is_same_goal = np.array_equal(self.prev_goal, curr_goal)
        self.prev_goal = curr_goal
        return not is_same_goal

    def interactionProcess(self, interaction_fn, needs_centering, planner_inputs, interaction_mask):
        action = None
        obs, rew, done, info, goal_success, err = None, None, None, None, False, None
        abort_centering = False
        if needs_centering:
            centering_fns = {"local_adjustment": self.centerAgent, "simple": self.centerAgentSimple, "none": self.centerAgentNone}
            centering_fn = centering_fns[self.args.centering_strategy]
            done_centering, abort_centering = centering_fn(planner_inputs, interaction_mask)

        if (not needs_centering) or done_centering:#如果判断done centering，直接就交互
            obs, rew, done, info, goal_success, err = interaction_fn()
        elif not abort_centering:
            action = self.centering_actions.pop(0)

        if not abort_centering:
            self.target_search_sequence = list()
        # NOTE 如果不终止centering，target_search_sequence置为空？不应该是如果终止centering, 设置为空吗
        # 这里的target_search_sequence和centering_actions不一样，target_search_sequence是到达目标点应该执行什么动作

        return obs, rew, done, info, goal_success, err, abort_centering, action
    
    def need_change_angel(self,goal_spotted):
        '''
        是否应该改变仰角
        记录：在当前仰角下请求了好几次新目标，还没找到，就更换倾角
        '''
        # 先测试最简单版本
        if self.num_newgoal_curhorizon[self.camera_horizon] > self.max_next_goal_request and not goal_spotted:
            return True
        else:
            return False


    def can_lookup(self,explore_area):
        '''
        能不能向上看: 当机器人在探索过的区域边缘的时候，不能向上看，设置一个参数，什么才叫边缘
        agent_pos 调用 self.curr_loc_grid 就好
        '''
        # 将explored_area缩小一圈
        explored_map = explore_area.copy()
        # 做出原本的explored图并保存
        if self.args.save_exp:
            import matplotlib.pyplot as plt
            if not os.path.exists(self.args.save_exp_dir):
                os.makedirs(self.args.save_exp_dir)
            plt.imshow(explored_map)
            plt.show()
            plt.imsave(self.args.save_exp_dir + "origin_explored_area_"+str(self.steps) + ".jpg",explored_map)
        # *******************************
        selem = skimage.morphology.square(self.explored_area_reduce_size)
        explored_map = 1-skimage.morphology.binary_dilation((1-explored_map)!=0, selem)
        # 做出更改后的explored图并保存
        if self.args.save_exp:
            import matplotlib.pyplot as plt
            if not os.path.exists(self.args.save_exp_dir):
                os.makedirs(self.args.save_exp_dir)
            plt.imshow(explored_map)
            plt.show()
            plt.imsave(self.args.save_exp_dir + "reduced_explored_area_"+str(self.steps) + ".jpg",explored_map)
        # *******************************
        # 如果机器人在
        if explored_map[self.curr_loc_grid[1],self.curr_loc_grid[0]] == 0 and self.camera_horizon > -45:
            # 已经在探索区域边缘，不能抬头
            # TODO 这里应该是标反了，在地图中应该是先0，后1，如果调用cv2.line，由于其坐标系有点不同，需要先1，后2
            return False
        else:
            return True
        
    def new_angel(self,goal_spotted,explored_area):
        '''
        计算是否要换新仰角，以及新仰角应该是多少,如果返回None，则说明不能或者不需要更换仰角
        有一处代码是将仰角置为-45的，注意一下
        '''
        new_horizon_angle = None
        if self.need_change_angel(goal_spotted):
            # 获取当前的仰角
            cur_horizon_angel = self.camera_horizon
            # 如果仰角是45度，并且可以抬头
            # self.camera_horizon可能有哪些取值？
            if cur_horizon_angel > 0 and self.can_lookup(explored_area):
                new_horizon_angle = 0
            elif cur_horizon_angel < 0 and cur_horizon_angel > -45:
                new_horizon_angle = -45
        return new_horizon_angle
        # 目前只是设置了最简单的0度角的，其它之后再尝试
        # 记得如果改变了仰角，记得把self.num_newgoal_curhorizon设置为0，这个已经在plan变量里面修改了


    def convertManualInput(self, code):
        ctrl_map = {'a': "RotateLeft_90", 'w': "MoveAhead_25", 'd': "RotateRight_90", 'u': 'LookUp_15', 'n': 'LookDown_15'}
        if code in ctrl_map:
            return ctrl_map[code], None

        if len(code.split(',')) != 3:
            return "LookUp_0", None

        action, coordx, coordy = code.split(',')
        interaction_mask = np.zeros((300, 300), dtype=np.float64)
        interaction_mask[int(coordy), int(coordx)] = 1

        return action, interaction_mask

    def specialManualInputs(self, code):
        codes = code.split(',')

        if (len(codes) == 2) and (codes[0] == "thresh"):
            thresh = float(codes[1])
            self.args.sem_seg_threshold_small = thresh
            self.args.sem_seg_threshold_large = thresh
        elif (len(codes) == 1) and (codes[0] == "show_all"):
            self.args.override_sem_seg_thresh = True
        else:
            return False

        return True

    def manualControl(self, code):
        if self.specialManualInputs(code):
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new("LookUp_0")

        else:
            action, mask = self.convertManualInput(code)
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(action, mask)

        #  *********************
        if self.args.run_idx_file is None:
            episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
        else:
            idx = json.load(open(self.args.run_idx_file, 'r'))
            episode_no = idx[self.args.from_idx+self.scene_pointer* self.args.num_processes + self.rank]
            # 注意修改到这里会不会导致程序非法结束？
            print(f"episode number is {episode_no}")
        # ********************
        next_step_dict = {
            'keep_consecutive': False, 'view_angle': self.camera_horizon,
            'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken,
            'broken_grid':self.broken_grid,
            'actseq':{(episode_no, self.traj_data['task_id']): self.actions[:1000]}, 
            'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'next_goal': False,
            'delete_lamp': False, 'fails_cur': self.fails_cur}

        return obs, rew, done, info, False, next_step_dict

    def plan_act_and_preprocess(self, planner_inputs, goal_spotted):
        # ****************************
        explored_map = planner_inputs["exp_pred"]
        # # 保存exp图层
        # if self.args.save_exp:
        #     import matplotlib.pyplot as plt
        #     if not os.path.exists(self.args.save_exp_dir):
        #         os.makedirs(self.args.save_exp_dir)
        #     # pickle.dump(planner_inputs["exp_pred"], open(self.args.save_exp_dir + "explored_area_" +str(self.steps) + '.pkl',
        #     #                                  'wb'))
        #     # 直接作图
        #     # 1代表探索过的区域
        #     plt.imshow(explored_map)
        #     plt.show()
        #     plt.imsave(self.args.save_exp_dir + str(self.steps) + ".jpg",explored_map)
        # # *******************************
        self.fails_cur = 0
        self.pointer = planner_inputs['list_of_actions_pointer']
        self.seg.cp_goal = planner_inputs['list_of_actions'][self.pointer][0]
        # TODO self.seg.cp_goal是什么
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) matrix denoting goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, and evaluation metric info
        """
        def updateVisited(pose_pred):
            curr_x, curr_y, curr_o, gx1, gx2, gy1, gy2 = pose_pred

            self.last_loc = self.curr_loc
            self.curr_loc = [curr_x, curr_y, curr_o]

            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            self.curr_loc_grid = self.meter2coord(
                self.curr_loc[1], self.curr_loc[0], gy1, gx1, self.visited.shape)
            prev = self.meter2coord(self.last_loc[1], self.last_loc[0], gy1, gx1, self.visited.shape)

            self.visited[gx1:gx2, gy1:gy2] = cv2.line(
                self.visited[gx1:gx2, gy1:gy2], (prev[1], prev[0]),
                (self.curr_loc_grid[1], self.curr_loc_grid[0]), 1, 1)

        self.steps += 1
        sdroate_direction = None
        next_goal = False
        abort_centering = False
        goal_success = False
        keep_consecutive = False
        action = None

        # *******add by me ************
        if self.steps == 1:
            print(f"sub goal lists is {planner_inputs['list_of_actions']}")

        # ****************

        updateVisited(planner_inputs['pose_pred'])

        # *************************
        if self.camera_horizon not in self.num_newgoal_curhorizon.keys():
            self.num_newgoal_curhorizon[self.camera_horizon] = 0
        # ***********************

        if self.isNewGoal(planner_inputs["goal"]):
            self.action_5_count = 0
            self.reached_goal = False
            self.print_log("newly goal set")
            self.center_margin = 65
            # *************add by me*****
            print("new goal set")
            # if self.camera_horizon not in self.num_newgoal_curhorizon.keys():
            #     self.num_newgoal_curhorizon[self.camera_horizon] = 1
            #     # 这种写法只有在设置了新目标的时候更新，如果产生了一个新的仰角，但是没有设置新目标，就不会更新
            #     # 而之后调用是每一步都调用
            # else:
            self.num_newgoal_curhorizon[self.camera_horizon] += 1
            # 这里主要是为了防止出现一些不在字典里的camera_horizon，之后如果确定camera_horizon 的取值可以去掉
            # ***********
        
        # 判断是否需要缩小explored area map size
        largerthan_max_next_goal_request = lambda x: x > self.max_next_goal_request
        largerthan_threetime_newgoal_request = lambda x: x > 3*self.max_next_goal_request
        # 比三倍最大限制还要大
        if any_satisfy(largerthan_max_next_goal_request, self.num_newgoal_curhorizon.values()):
            self.explored_area_reduce_size -= 2
            self.explored_area_reduce_size = clip_value(self.explored_area_reduce_size, 1, 11)
            # 2,11这两个参数可能还需要调整
            self.print_log(f"reduce explored area reduce size to {self.explored_area_reduce_size}")

        # 判断是否需要重规划
        if self.args.use_replan:
            if self.args.change_altitude:
                replan = (all_satisfy(largerthan_max_next_goal_request,self.num_newgoal_curhorizon.values()) or any_satisfy(largerthan_threetime_newgoal_request,self.num_newgoal_curhorizon.values())) and not goal_spotted
            else:
                replan = sum(self.num_newgoal_curhorizon.values())> self.args.max_next_goal_request and not goal_spotted
            # replan = True
            if replan:

                print("request for replan --------------------------")
                scene_obj_list,planner_inputs,has_replan = self.replan_subtask(planner_inputs)
                # TODO 把这个replan改一下
                if not has_replan:
                    # 满足replan的执行条件：到达某个子目标好几次了，但是没有真正地replan成功
                    # 看作已经replan,避免频繁地调用replan函数
                    print("replan false")
                    self.num_newgoal_curhorizon = set_dict2value(self.num_newgoal_curhorizon, 0.0)
                # 这个变量后续返回给main函数
                # 这里师兄写的直接在sem函数里面replan了
                # 重置目标后，需要将self.num_newgoal_curhorizon也重置
                # self.num_newgoal_curhorizon = set_dict2value(self.num_newgoal_curhorizon, 0.0)
                # replan的函数自动会reset goal，这里不用改
                # planner_inputs['list_of_actions']
                # 将replan记录下来
                if has_replan and self.args.record_replan:
                    text = f"replan: {planner_inputs['list_of_actions']}"
                    print(text)
                    with open(f"{self.args.result_file}logs/replan_{self.args.from_idx}_{self.args.to_idx}.txt",'a') as f:
                            f.write(text+"\n")

        

        if planner_inputs["wait"]:
            self.last_action_ogn = None
            self.info["sensor_pose"] = [0., 0., 0.]
            self.rotate_aftersidestep =  None
            # return np.zeros(self.obs.shape), 0., False, self.info, False, {
            #     'view_angle': self.camera_horizon,  'picked_up': self.picked_up,
            #     'steps_taken': self.steps_taken, 'broken_grid': self.broken_grid,
            #     'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['task_id']): self.actions[:1000]},
            #     'errs': self.errs, 'logs':self.logs, 'current_goal_sliced':self.cur_goal_sliced,
            #     'next_goal': next_goal, 'delete_lamp': False, 'fails_cur': 0}
            # ******************
            #  *********************
            if self.args.run_idx_file is None:
                episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
            else:
                idx = json.load(open(self.args.run_idx_file, 'r'))
                episode_no = idx[self.args.from_idx+self.scene_pointer* self.args.num_processes + self.rank]
                # 注意修改到这里会不会导致程序非法结束？
                print(f"episode number is {episode_no}")
            # ********************
            return np.zeros(self.obs.shape), 0., False, self.info, False, {
                'view_angle': self.camera_horizon,  'picked_up': self.picked_up,
                'steps_taken': self.steps_taken, 'broken_grid': self.broken_grid,
                'actseq':{(episode_no, self.traj_data['task_id']): self.actions[:1000]},
                'errs': self.errs, 'logs':self.logs, 'current_goal_sliced':self.cur_goal_sliced,
                'next_goal': next_goal, 'delete_lamp': False, 'fails_cur': 0,'none_interaction_mask':False},planner_inputs['list_of_actions']
            # 没有interaction_mask的时候会删除东西，但正常等待的时候应该不做任何事情，所以，这里none_interaction_mask设为False
            # *******************

        # check for collision
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        self.collisionHandling(gx1, gy1)

        self._visualize(planner_inputs)

        # manual control by the user
        if planner_inputs["manual_step"] is not None:
            # ************add by me ******
            print("manul step")

            # ********************
            return self.manualControl(planner_inputs["manual_step"]),planner_inputs['list_of_actions']

        # check if the target is within interactable distance,
        # but do not execute interaction during the initial lookaround sequence
        during_lookaround = self.lookaround_counter < len(self.lookaround_seq)

        # if the next action is in caution_pointers, adjust agent location to center the goal
        needs_centering = planner_inputs['list_of_actions_pointer'] in self.caution_pointers

        target_coord = None
        if (len(self.centering_actions) > 0) and (self.centering_actions[0] == "Done"):
            target_coord = self.centering_actions[-1] # centering_action done后面是point
            self.centering_actions = list()
            # *************** add by me *********
            # print("center done")
            # ****************************

        # AI2THOR agent can only see objects that are within 1.5m from itself
        # so set 1.45m to account for errors in depth estimation
        visibility_threshold = 1.45

        target_xyz, interactable, interaction_mask = self.check4interactability(target_coord, visibility_threshold)
        # NOTE interactable记录能不能够得到，如果够不到，往前一步不就行了？这里有可能不太准，因为mask也不太准
        # 需要centering done才interactable, 但是centering 需要interact来计算？这里target_coord is None也有可能是interactable, 主要是判断interaction_mask
        # execute_interaction = interactable & (not during_lookaround)
        # ******************
        none_interaction_mask = True if interaction_mask is None else False #在拿两个东西的时候应该删去
        # NOTE 在main函数里修改了：如果发现了interaction mask，就不删除地图中的元素
        # **********************
        in_fail_loc = False
        for loc in self.interaction_fail_pose:
            if self.curr_loc_grid[0] == loc[0] and self.curr_loc_grid[1] == loc[1]:
                in_fail_loc = True
        execute_interaction = interactable & (not during_lookaround) & (not self.first_goal) and not in_fail_loc #NOTE 这里条件已经改变


        # if interaction_mask is None:
        #     print("the interaction mask is None")
        # elif not interactable:
        #     print("interaction mask is not None, but can't interact")
        # else:
        #     print('interaction mask is not None, and interactable')

        # if sliced mask is available and now want to pick up the sliced object,
        # do not interact with object along the path
        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        interaction = list_of_actions[pointer][1]
        force_slice_pickup, _ = self.goBackToLastSlicedLoc(interaction)
        if force_slice_pickup:
            self.force_move = True

        # an interaction happens only when goal is spotted AND the agent is near enough to the goal
        # i.e. do not finish tasks during/along the path if it is a found goal
        interact_ok = (not self.force_move) and ((not needs_centering) or (not goal_spotted) or self.reached_goal)
        # *************add by me **************
        if goal_spotted:
            print("found goal")
        # **************************
        if self.args.change_altitude:
            new_horizon_angle = self.new_angel(goal_spotted,explored_map)
        else:
            new_horizon_angle = None
        # ************************

        do_move = False
        is_centering = len(self.centering_actions) > 0
        if is_centering:
            self.print_log("centering_history", self.centering_history)

            high_priority_act = self.centering_actions.pop(0)
            obs, rew, done, info, success, err, _ = self.execAction(high_priority_act)
            self.print_log("action chosen: high priority")
            # # ********** add by me *******************
            # print(f"centering, action chosen is {high_priority_act}")
            # # ******************

        # run consecutive actions (ex. open -> close -> turn on -> turn off a microwave)
        elif interact_ok and planner_inputs['consecutive_interaction'] != None:
            self.print_log("action chosen: consecutive interaction")

            interaction_fn = lambda: self.consecutive_interaction(
                planner_inputs['consecutive_interaction'], interaction_mask)

            obs, rew, done, info, goal_success, err, abort_centering, action = self.interactionProcess(
                interaction_fn, needs_centering, planner_inputs, interaction_mask)

            # # ************add by me ***********
            # print("consecutive interaction done")
            # # *****************

        # run a one-step interaction
        elif interact_ok and planner_inputs['consecutive_interaction'] == None and execute_interaction:
            self.print_log("action chosen: single interaction")

            interaction_fn = lambda: self.interactionSequence(planner_inputs, interaction_mask)

            obs, rew, done, info, goal_success, err, abort_centering, action = self.interactionProcess(
                interaction_fn, needs_centering, planner_inputs, interaction_mask)

        elif new_horizon_angle is not None:
            action = f"Angle_{new_horizon_angle}"
            self.print_log(f"change horizon angle to {new_horizon_angle}")

        else:
            do_move = True

        if do_move or abort_centering:
            self.force_move = False
            goal_xyz = target_xyz if goal_spotted else None

            target_offset = (int(self.args.target_offset_interaction * 100.0/self.args.map_resolution)
                             if (interaction == "OpenObject") else 0) #NOTE 只有开东西的时候有偏执，其它都没有
            action, next_goal, self.reached_goal = self.getNextAction(
                planner_inputs, during_lookaround, target_offset, goal_xyz, force_slice_pickup)
            # ********************
            if self.reached_goal:
                self.first_goal = False 
            # **********************

            # ******************
            if next_goal:
                print("next goal")
                if self.args.drop_interaction_fail_loc and not interactable:
                    self.interaction_fail_pose.append([self.curr_loc_grid[0], self.curr_loc_grid[1]])
            # **************

        if action is not None:
            obs, rew, done, info, _, err, goal_success = self.execAction(action)

        will_center = len(self.centering_actions) > 0
        if (not is_centering) and (not will_center):
            self.centering_history = list()

        delete_lamp = (self.goal_name == 'FloorLamp') and (self.action_received == "ToggleObjectOn")
        if self.args.no_delete_lamp:
            delete_lamp = False

        # request next goal when non-movement action fails
        act, _ = self.splitAction(self.last_action_ogn)
        interaction_failed = (act not in ["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp"]) and (not self.last_success)
        # ***********
        if not next_goal and interaction_failed:
            print("request next goal because interaction fail")
        # **********
        # next_goal |= interaction_failed

        # # request next goal when an interaction succeeds
        # next_goal |= goal_success
        # ***********
        next_goal |= interaction_failed and not self.open_object

        # request next goal when an interaction succeeds
        next_goal |= goal_success and not self.open_object
        # next_goal = next_goal and not self.open_object
        if self.open_object:
            print("some object opened and not be closed")
            print(f"next goal is {next_goal}")

        if next_goal:
            print("request next goal")


        self.rotate_aftersidestep = sdroate_direction
        if self.args.run_idx_file is None:
            episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
        else:
            idx = json.load(open(self.args.run_idx_file, 'r'))
            episode_no = idx[self.args.from_idx+self.scene_pointer* self.args.num_processes + self.rank]
            print(f"episode number is {episode_no}")
        # ********************
        next_step_dict = {
        'keep_consecutive': keep_consecutive, 'view_angle': self.camera_horizon,
        'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken,
        'broken_grid':self.broken_grid,
        'actseq':{(episode_no, self.traj_data['task_id']): self.actions[:1000]},
        'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'next_goal': next_goal,
        'delete_lamp': delete_lamp, 'fails_cur': self.fails_cur,'none_interaction_mask':none_interaction_mask}
        # ******************

        if err != "":
            self.print_log(f"step: {self.steps_taken} err is {err}")
            # *******************
            self.errs.append(err)
            # *******************

        self.last_err = err 
        self.info = info

        list_of_actions = planner_inputs['list_of_actions']
        pointer = planner_inputs['list_of_actions_pointer']
        if goal_success and (pointer + 1 < len(list_of_actions)):
            self.print_log("pointer increased goal name ", list_of_actions[pointer+1])

        return obs, rew, done, info, goal_success, next_step_dict, planner_inputs['list_of_actions']

    def collisionHandling_new(self, gx1, gy1):
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        collision = (abs(x1 - x2) < self.args.collision_threshold) and (abs(y1 - y2) < self.args.collision_threshold)
        moved_ahead = self.last_action_ogn == "MoveAhead_25"

        if moved_ahead and collision:
            y, x = self.curr_loc_grid

            step_sz = self.args.step_size
            robot_radius = self.args.obstacle_selem // 2
            shift_sz = robot_radius + 1
            width = step_sz
            height = self.args.obstacle_selem

            rad = np.deg2rad(t1)
            rmat = np.asarray([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            for i in range(height):
                for j in range(shift_sz, shift_sz + width):
                    delta = np.asarray([j, i - self.args.collision_obstacle_length // 2])
                    dx, dy = rmat @ delta
                    [r, c] = pu.threshold_poses([int(y + dy), int(x + dx)], self.collision_map.shape)
                    self.collision_map[r, c] = 1
            
            # *********************
            self.explored_area_reduce_size += 2
            self.explored_area_reduce_size = clip_value(self.explored_area_reduce_size, 1, 11)
            if self.camera_horizon != 45:
                action = "Angle_45"
                obs, rew, done, info, _, err, goal_success = self.execAction(action)
                print("collision happened during look up, set horizon angel to 45")
            # *************************

    def collisionHandling(self, gx1, gy1):
        if "new_obstacle_fn" in self.args.mlm_options:
            return self.collisionHandling_new(gx1, gy1)

        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        collision = (abs(x1 - x2) < self.args.collision_threshold) and (abs(y1 - y2) < self.args.collision_threshold)
        moved_ahead = self.last_action_ogn == "MoveAhead_25"

        if moved_ahead and collision:
            y, x = self.curr_loc_grid

            width = 3
            rad = np.deg2rad(t1)
            rmat = np.asarray([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            for i in range(self.args.collision_obstacle_length):
                for j in range(1, width+1):
                    delta = np.asarray([j, i - self.args.collision_obstacle_length // 2])
                    dx, dy = rmat @ delta
                    [r, c] = pu.threshold_poses([int(y + dy), int(x + dx)], self.collision_map.shape)
                    self.collision_map[r, c] = 1
            # *********************
            self.explored_area_reduce_size += 2
            self.explored_area_reduce_size = clip_value(self.explored_area_reduce_size, 1, 11)
            if self.camera_horizon != 45:
                action = "Angle_45"
                obs, rew, done, info, _, err, goal_success = self.execAction(action)
                print("collision happened during look up, set horizon angel to 45")
            # *************************

    def execAction(self, action):
        goal_success = False
        if action == "SliceObjectFromMemory":
            obs, rew, done, info, success, err = self.consecutive_interaction(
                "PickupObject", self.sliced_mask)
            self.sliced_mask = None
            self.sliced_pose = None
            goal_success = success
        else:
            obs, rew, done, info, success, _, _, err, _ = self.va_interact_new(action)

        self.print_log("obj type for mask is :", self.goal_idx2cat[self.goal_idx])

        return obs, rew, done, info, success, err, goal_success

    def meter2coord(self, y, x, miny, minx, map_shape):
        r, c = y, x
        start = [int(r * 100.0/self.args.map_resolution - minx),
                 int(c * 100.0/self.args.map_resolution - miny)]
        return pu.threshold_poses(start, map_shape)

    def _plan(self, planner_inputs, target_offset, goal_coord, measure_offset_from_edge=True):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        next_goal = False

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        start = self.curr_loc_grid
        start = [start[0], start[1], start_o]

        self.goal_visualize = np.zeros_like(planner_inputs["goal"])
        self.goal_visualize[goal_coord[0], goal_coord[1]] = 1
        nextAction, stop, next_goal, goal_free_and_not_shifted = self._get_stg(
            map_pred, start, goal_coord, planning_window, target_offset, measure_offset_from_edge)

        if nextAction is None:
            action = "LookUp_0"
        elif stop and planner_inputs["found_goal"]:
            action = "<<stop>>"
        elif stop:
            if self.action_5_count < 1:
                action = "ReachedSubgoal"
                self.action_5_count +=1
            else:
                next_goal = True
                action = "LookUp_0"
        else:
            action = nextAction

        return action, next_goal, goal_free_and_not_shifted

    def _get_stg(self, grid, start, goal_coord, planning_window, target_offset, measure_offset_from_edge=False):
        traversible = self.get_traversible(grid, planning_window)

        nextAction, new_goal, stop, next_goal, goal_free_and_not_shifted = planNextMove(
            traversible, self.args.step_size, start, goal_coord, target_offset, measure_offset_from_edge,self.interaction_fail_pose)


        # nextAction, new_goal, stop, next_goal, goal_free_and_not_shifted = planNextMove(
        #     traversible, self.args.step_size, start, goal_coord, target_offset,measure_offset_from_edge)

        if self.args.save_pictures:
            viz = np.repeat(255 - traversible[:, :, np.newaxis] * 255, 3, axis=2)

            h, w = viz.shape[:2]
            viz[max(0, goal_coord[0]-2):min(h-1, goal_coord[0]+3), max(0, goal_coord[1]-2):min(w-1, goal_coord[1]+3), :] = [255, 0, 0]
            viz[max(0, new_goal[0]-2):min(h-1, new_goal[0]+3), max(0, new_goal[1]-2):min(w-1, new_goal[1]+3), :] = [0, 0, 255]
            # ***********
            viz[max(0, start[0]-2):min(h-1, start[0]+3), max(0, start[1]-2):min(w-1, start[1]+3), :] = [0, 255, 0]
            # ***********
            cv2.imwrite(self.picture_folder_name +"fmm_dist/"+ "fmm_dist_" + str(self.steps_taken) + ".png", viz)

            # save the obstacle images pre dilation for debugging purposes
            obst = grid.astype(np.uint8) * 255
            cv2.imwrite(self.picture_folder_name +"obstacles_pre_dilation/"+ "obst_" + str(self.steps_taken) + ".png", obst)

        return nextAction, stop, next_goal, goal_free_and_not_shifted

    def depth_pred_later(self, sem_seg_pred):
        rgb = cv2.cvtColor(self.event.frame.copy(), cv2.COLOR_RGB2BGR)#shape (h, w, 3)
        # For higher resolution
        rgb = cv2.resize(rgb, (300, 300))
        rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255

        use_model_0 = abs(self.camera_horizon % 360) <= 5

        if use_model_0:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float())
            include_mask_prop = self.args.valts_trustworthy_obj_prop0
        else:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())
            include_mask_prop = self.args.valts_trustworthy_obj_prop

        depth = pred_depth.get_trustworthy_depth(
            max_conf_int_width_prop=self.args.valts_trustworthy_prop, include_mask=sem_seg_pred,
            include_mask_prop=include_mask_prop) #default is 1.0
        depth = depth.squeeze().detach().cpu().numpy()

        self.learned_depth_frame = pred_depth.mle().detach().cpu().numpy()[0, 0, :, :]
        # self.learned_depth_frame = pred_depth.expectation().detach().cpu().numpy()[0, 0, :, :]

        if self.args.save_pictures:
            depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
            depth_pre_mask = (pred_depth.mle()).detach().cpu().numpy()[0, 0, :, :]
            cv2.imwrite(depth_imgname % "depth", depth_pre_mask * 100)
        del pred_depth

        depth = np.expand_dims(depth, 2)
        return depth
    
    # Add by Trisoil
    def dino_depth_pred(self, rgb):
        inputs = self.depth_img_processor(images=rgb, return_tensors='pt')
        inputs.to(self.args.depth_gpu)
        
        with torch.no_grad():
            outputs = self.dino_depth(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=[300, 300],
            mode="bicubic",
            align_corners=False,
        )

        output = prediction.squeeze().cpu().numpy()
        self.learned_depth_frame = output
        
        output = np.expand_dims(output, axis=2)
        
        # show depth
        # cv2.imshow('dpt', (output * 255 / np.max(output)).astype("uint8"))
        # cv2.waitKey(1)
        
        return output

    def _preprocess_obs(self, obs):
        # make semantic segmentation and depth predictions

        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]

        # rgb = cv2.resize(rgb, (300, 300))

        sem_seg_pred = self.seg.get_sem_pred(rgb.astype(np.uint8)) #(300, 300, num_cat)

        # Changed by Trisoil
        if self.args.use_learned_depth:
            # rgb = cv2.resize(rgb, (300, 300))
            # include_mask = np.sum(sem_seg_pred, axis=2).astype(bool).astype(float)
            # include_mask = np.expand_dims(np.expand_dims(include_mask, 0), 0)
            # include_mask = torch.tensor(include_mask).to(self.depth_gpu)

            # depth = self.depth_pred_later(include_mask)
            depth = self.dino_depth_pred(rgb)
        else:
            depth = obs[:, :, 3:4]
            if depth.shape!=(300,300,1):
                depth = cv2.resize(depth,(300,300),interpolation=cv2.INTER_NEAREST)
                depth =np.expand_dims(depth,axis=2)
            if self.args.save_pictures:
                depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
                cv2.imwrite(depth_imgname % "depth", depth * 100)

        rgb = np.asarray(self.res(rgb.astype(np.uint8)))

        depth = self._preprocess_depth_new(depth, self.holding_mask)

        # depth = self._preprocess_depth(depth)
        # cv2.imwrite(f'debug_image/depth_pro_{self.steps_taken}.png',depth)
        
        if self.args.save_pictures:
            depth_imgname = os.path.join(self.picture_folder_name, '%s', "depth_" + str(self.steps_taken) + ".png")
            cv2.imwrite(depth_imgname % "depth_thresholded", depth)

        ds = args.env_frame_width // args.frame_width # Downscaling factor
        if ds != 1:
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis = 2).transpose(2, 0, 1)

        return state, sem_seg_pred

    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0] * 100

        if self.picked_up:
            mask_err_below = depth < 50
            depth[np.logical_or(self.picked_up_mask, mask_err_below)] = 10000.0

        return depth

    # *************************
    def _preprocess_depth_new(self, depth, target_mask, cal_optical_flow=False):
        depth = depth[:, :, 0] * 100
        if target_mask is None:
            return depth

        if self.picked_up:
            goal_idx = self.total_cat2idx[self.pick_up_obj]
            # Changed by Trisoil
            # target_mask = sem_seg_pred[:,:,goal_idx]
            target_mask = skimage.morphology.dilation(target_mask,skimage.morphology.square(10))
            if cal_optical_flow:
                hsv = np.zeros_like(self.prev_rgb)
                prvs = cv2.cvtColor(self.prev_rgb,cv2.COLOR_BGR2GRAY)
                curr = cv2.cvtColor(self.event.frame,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                flow_gray = cv2.cvtColor(flow_bgr,cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(f'debug_image/flow_bgr_{self.steps_taken}.png',flow_bgr)
                # cv2.imwrite(f'debug_image/flow_grey_{self.steps_taken}.png',flow_gray)

            depth[np.where(target_mask)] = 10000.0
            # depth[np.where(self.picked_up_mask)] = depth.max() #取最大和取10000都会撞，似乎这里不是关键
            # cv2.imwrite(f'debug_image/depth_me_{self.steps_taken}.png',depth)
        return depth
    # ************************
    def colorImage(self, sem_map, color_palette):
        semantic_img = Image.new("P", (sem_map.shape[1],
                                       sem_map.shape[0]))

        semantic_img.putpalette(color_palette)
        #semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
        semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img = np.asarray(semantic_img)
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGBA2BGR)

        return semantic_img

    def _visualize(self, inputs):
        args = self.args

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
                inputs['pose_pred']

        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        if self.steps <=1:
            goal = inputs['goal']
        else:
            goal = self.goal_visualize
        sem_map = inputs['sem_map_pred'].copy()

        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        sem_map += 5

        if self.args.ground_truth_segmentation:
            no_cat_mask = sem_map == 5 + args.num_sem_categories -1
        else:
            no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        map_mask = np.logical_or(map_mask, self.collision_map==1)
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        curr_mask = np.zeros(vis_mask.shape)
        selem = skimage.morphology.disk(2)
        curr_mask[start[0], start[1]] = 1
        curr_mask = 1 - skimage.morphology.binary_dilation(
            curr_mask, selem) != True
        curr_mask = curr_mask ==1
        sem_map[curr_mask] = 3

        if goal is not None:
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1
            sem_map[goal_mask] = 4

        #color_palette = d3_40_colors_rgb.flatten()
        color_palette2 = [1.0, 1.0, 1.0,
                0.6, 0.6, 0.6,
                0.95, 0.95, 0.95,
                0.96, 0.36, 0.26,
                0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
                0.9400000000000001, 0.7818, 0.66,
                0.9400000000000001, 0.8868, 0.66,
                0.8882000000000001, 0.9400000000000001, 0.66,
                0.7832000000000001, 0.9400000000000001, 0.66,
                0.6782000000000001, 0.9400000000000001, 0.66,
                0.66, 0.9400000000000001, 0.7468000000000001,
                0.66, 0.9400000000000001, 0.9018000000000001,
                0.66, 0.9232, 0.9400000000000001,
                0.66, 0.8182, 0.9400000000000001,
                0.66, 0.7132, 0.9400000000000001,
                0.7117999999999999, 0.66, 0.9400000000000001,
                0.8168, 0.66, 0.9400000000000001,
                0.9218, 0.66, 0.9400000000000001,
                0.9400000000000001, 0.66, 0.9031999999999998,
                0.9400000000000001, 0.66, 0.748199999999999]
        
        color_palette2 += self.flattened.tolist()
        
        color_palette2 = [int(x*255.) for x in color_palette2]
        color_palette = color_palette2

        semantic_img = self.colorImage(sem_map, color_palette)

        # arrow_sz = 20
        # h, w = semantic_img.shape[:2]
        # pt1 = (w - arrow_sz * 2, h - arrow_sz * 2)
        # arry, arrx = CH._which_direction(start_o)
        # pt2 = (pt1[0] + arrx * arrow_sz, pt1[1] + arry * arrow_sz)
        # cv2.arrowedLine(
        #     semantic_img, pt1, pt2, color=(0, 0, 0, 255), thickness=1,
        #     line_type=cv2.LINE_AA, tipLength=0.2)

        if self.args.visualize:
            cv2.imshow("Sem Map", semantic_img)
            cv2.waitKey(1)

        if self.args.save_pictures:
            cv2.imwrite(self.picture_folder_name + "Sem_Map_new/"+ "Sem_Map_" + str(self.steps_taken) + ".png", semantic_img)

    def evaluate(self):
        goal_satisfied = self.get_goal_satisfied()
        if goal_satisfied:
            success = True
        else:
            success = False
            
        pcs = self.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(self.traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(self.steps_taken))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(self.steps_taken))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight
        
        goal_instr = self.traj_data['turk_annotations']['anns'][self.r_idx]['task_desc']
        sliced = get_arguments(self.traj_data)[-1]
        
        if self.args.run_idx_file is None:
            episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank
        else:
            idx = json.load(open(self.args.run_idx_file, 'r'))
            episode_no = idx[self.scene_pointer* self.args.num_processes + self.rank] 
        # log success/fails
        log_entry = {'trial': self.traj_data['task_id'],
                     #'scene_num': self.traj_data['scene']['scene_num'],
                     'type': self.traj_data['task_type'],
                     'repeat_idx': int(self.r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'sliced':sliced,
                     'episode_no':  episode_no,
                     'steps_taken': self.steps_taken}
                     #'reward': float(reward)}
        
        return log_entry, success

    #*****************************
    def calculate_similarity(self, target_word, scene_obj_list):
        # Using clip to calculate the similarity
        target_input = self.clip_tokenizer([target_word]).to(device=self.depth_gpu)
        with torch.no_grad():
            target_embedding = self.clip_model.encode_text(target_input).float()
        
        similarity_scores = []
        for word in scene_obj_list:
            word_tokens = self.clip_tokenizer([word]).to(device=self.depth_gpu)
            with torch.no_grad():
                word_embedding = self.clip_model.encode_text(word_tokens).float()
            similarity_score = torch.cosine_similarity(target_embedding, word_embedding).item()
            similarity_scores.append(similarity_score)
        similar_words = [word for _, word in sorted(zip(similarity_scores, scene_obj_list), reverse=True)[:1]]
        return similar_words[0]

    def replan_subtask(self, planner_inputs):
        obj_num = len(self.goal_idx2cat)
        scene_obj_list = []
        sem_map_pred = planner_inputs['sem_map_pred']
        for index_obj in range(obj_num-1):
            if index_obj in sem_map_pred:
                scene_obj_list.append(self.goal_idx2cat[index_obj])

        target_num = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])[0]

        if target_num not in scene_obj_list:
            if target_num == "Glassbottle"  and  "Cup" in scene_obj_list:
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = "Cup"
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, "Cup", None)
                return scene_obj_list, planner_inputs,True
            if (target_num in sit_object) and any(element in scene_obj_list for element in sit_object):
                similar_words = self.calculate_similarity(target_num, list(set(sit_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            elif (target_num in cup_object) and any(element in scene_obj_list for element in cup_object):
                similar_words = self.calculate_similarity(target_num, list(set(cup_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            elif (target_num in table_object) and any(element in scene_obj_list for element in table_object):
                similar_words = self.calculate_similarity(target_num, list(set(table_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            elif (target_num in bottle_object) and any(element in scene_obj_list for element in bottle_object):
                similar_words = self.calculate_similarity(target_num, list(set(bottle_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            elif (target_num in lamp_object) and any(element in scene_obj_list for element in lamp_object):
                similar_words = self.calculate_similarity(target_num, list(set(lamp_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            elif (target_num in drawer_object) and any(element in scene_obj_list for element in drawer_object):
                similar_words = self.calculate_similarity(target_num, list(set(drawer_object) & set(scene_obj_list)))
                temp = list(planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']])
                temp[0] = similar_words
                planner_inputs['list_of_actions'][planner_inputs['list_of_actions_pointer']] = tuple(temp)
                self.info = self.reset_goal(True, temp[0], None)
            else:
                return scene_obj_list, planner_inputs,False
        return scene_obj_list, planner_inputs,True


    # ***********************
    def reset_plan(self,second_object,caution_pointers,reset_type='open4search'):
        '''
        重新设置plan,其实agent没有记录plan是什么，这里是设置plan相关的变量
        type: open4search or origin
        '''
        self.second_object = second_object
        self.caution_pointers = caution_pointers
        # if reset_type=='open4search':
        if reset_type=='open4search' and not self.open_object:
            self.first_goal = True #先到目标那里
 