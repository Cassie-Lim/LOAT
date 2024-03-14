# *******************add by me *********
import sys
sys.path.append("/home/cyw/task_planning/prompter_v3/prompter-alfred-YOLO-no_replan_v1/")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" #NOTE
# *********************************

import pickle
# from models.semantic_policy.sem_map_model import UNetMulti, MLM
# **************
# 使用自己的搜索
from improve_search_v4.semantic_policy.sem_map_model import UNetMulti,MLM,SEQ_SEARCH
# ***********
import alfred_utils.gen.constants as constants
from models.instructions_processed_LP.ALFRED_task_helper import determine_consecutive_interx,get_newplan
# from models.sem_mapping import Semantic_Mapping
# *****************
from improve_search_v4.sem_mapping import Semantic_Mapping
# 使用自己的语义映射，语义图构建
# ***************
# import envs.utils.pose as pu
# from envs import make_vec_envs
# ****************
import improve_search_v4.envs.utils.pose as pu
from improve_search_v4.envs import make_vec_envs
import json
from improve_search_v4.misc import get_opensearch_times
from improve_search_v4.misc import clip_value
# *************
from arguments import get_args
from datetime import datetime
from collections import defaultdict
import skimage.morphology
import math
import numpy as np
import torch.nn as nn
import torch
import cv2
import os
import sys
import matplotlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == 'darwin':
    matplotlib.use("tkagg")

'''
NOTE: 此版本已经加上搜索的log
main函数 prompter-alfred-YOLO-no_replan似乎没有什么改变，这里完全采用seq search v4的代码
'''
def into_grid(ori_grid, grid_size):
    if ori_grid.shape[0] == grid_size:
        return ori_grid

    one_cell_size = math.ceil(240 / grid_size)

    ori_grid = ori_grid.unsqueeze(0).unsqueeze(0)

    m = nn.AvgPool2d(one_cell_size, stride=one_cell_size)
    avg_pooled = m(ori_grid)[0, 0, :, :]
    return_grid = (avg_pooled > 0).float()

    return return_grid


def getCurrImgCoord(planner_pose_input, map_resolution):
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_pose_input
    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
    r, c = start_y, start_x
    yx = [int(r * 100.0/map_resolution - gy1),
          int(c * 100.0/map_resolution - gx1)]
    return yx


def getNeighborMap(ori_map, dil_sz):
    goal_neighbor = ori_map.copy()
    goal_neighbor = skimage.morphology.binary_dilation(
        goal_neighbor, skimage.morphology.square(dil_sz))
    return goal_neighbor


def searchArgmax(conv_sz, score_map, mask=None):
    # fast 2D convolution
    kernel = np.ones(conv_sz)
    conv_1d = lambda m: np.convolve(m, kernel, mode='same')
    ver_sum = np.apply_along_axis(conv_1d, axis=0, arr=score_map)
    conved = np.apply_along_axis(conv_1d, axis=1, arr=ver_sum)

    conved_masked = conved if (mask is None) else conved * mask

    return np.unravel_index(conved_masked.argmax(), conved_masked.shape)
    # 为什么要卷积一下,卷积可以帮助选取的点在中心？TODO 可视化一下这个过程


def main():
    args = get_args()
    dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    args.dn = dn
    if args.set_dn != "":
        args.dn = args.set_dn
        dn = args.set_dn
    print("dn is ", dn)

    # *****************
    # 对文件名进行进一步处理
    # --use_replan \
    # --record_replan \
    # --max_next_goal_request ${MAX_NEXT_GOAL_REQUEST} \
    # --change_altitude \
    # --explored_selem 4 \
    # --subgoal_file ${SUBGOAL_FILE} \
    text = ""
    if args.use_replan:
        text += "_replan"
    if args.change_altitude:
        text += "_changealt"
    if args.subgoal_file is not None:
        text += "_subgoal_" + args.subgoal_file.split("/")[-1].split(".")[0]
    if args.run_idx_file is not None and args.subgoal_file is None:
        text += "_runidx_" + args.run_idx_file.split("/")[-1].split(".")[0]
    if args.use_yolo:
        text += "_yolo"
    if args.language_granularity == "high_low":
        text += "_low"
    if not args.use_sem_seg:
        text += "_gtseg"
    if not args.use_learned_depth:
        text += '_gtdepth'
    if args.sem_policy_type == 'seq':
        text += '_seqsearch'
    if args.drop_interaction_fail_loc:
        text += "_dropFailLoc"
    dn += text
    args.set_dn = dn
    args.dn = dn
    print("new dn is ", dn) #NOTE 之后加上seq search

    # ***********************
    # /raid/cyw/Prompter/tests_unseen_fail_found_object_object not in scene.json
    # data_path = "./"
    if args.result_file is not None:
        data_path = args.result_file
    else:
        data_path = "./"
        args.result_file = data_path
    args.result_file = args.result_file + dn + "/"
    data_path = args.result_file

     # 保存arg，方便后续查看
    if not os.path.exists(data_path+"args"):
        os.makedirs(data_path+"args")
    json.dump(vars(args), open(data_path +"args/"+ dn + "_args" + ".json", "w"), indent=4, sort_keys=True)

    # 保存log
    # 由于有些文件是用apend的形式给出，如果这些文件存在，先删除
    # with open(f"add_byme/logs/plan_transfer_compare_{self.args.set_dn}.txt",'a')
    # with open(f"add_byme/logs/replan_log_{self.args.set_dn}.txt",'a')
    if os.path.exists(data_path+"logs/plan_transfer_compare_" + dn + ".txt"):
        os.remove(data_path+"logs/plan_transfer_compare_" + dn + ".txt")
    if os.path.exists(data_path+"logs/replan_log_" + dn + ".txt"):
        os.remove(data_path+"logs/replan_log_" + dn + ".txt")
    if not os.path.exists(data_path+"logs"):
        os.mkdir(data_path+"logs")
    # 多线程log会非常混乱
    

    if not os.path.exists(data_path + "results/logs"):
        os.makedirs(data_path +"results/logs")
    if not os.path.exists(data_path +"results/leaderboard"):
        os.makedirs(data_path +"results/leaderboard")
    if not os.path.exists(data_path +"results/successes"):
        os.makedirs(data_path +"results/successes")
    if not os.path.exists(data_path +"results/fails"):
        os.makedirs(data_path +"results/fails")
    if not os.path.exists(data_path +"results/analyze_recs"):
        os.makedirs(data_path +"results/analyze_recs")
    # *********************

    # if not os.path.exists("results/logs"):
    #     os.makedirs("results/logs")
    # if not os.path.exists("results/leaderboard"):
    #     os.makedirs("results/leaderboard")
    # if not os.path.exists("results/successes"):
    #     os.makedirs("results/successes")
    # if not os.path.exists("results/fails"):是的
    #     os.makedirs("results/fails")
    # if not os.path.exists("results/analyze_recs"):
    #     os.makedirs("results/analyze_recs")

    completed_episodes = []

    # 读取一些外部文件
    skip_indices = {}
    if args.exclude_list != "":
        if args.exclude_list[-2:] == ".p":
            skip_indices = pickle.load(open(args.exclude_list, 'rb'))
            skip_indices = {int(s): 1 for s in skip_indices}
        else:
            skip_indices = [a for a in args.exclude_list.split(',')]
            skip_indices = {int(s): 1 for s in skip_indices}
    args.skip_indices = skip_indices

    if args.run_idx_file is not None:
        idx_torun = json.load(open(args.run_idx_file, 'r'))


    actseqs = []
    all_completed = [False] * args.num_processes
    successes = []
    failures = []
    analyze_recs = []
    traj_number = [0] * args.num_processes
    num_scenes = args.num_processes


    # local_rngs = [np.random.RandomState(args.seed + args.from_idx + e) for e in range(args.num_processes)]
    # 计算更为准确的id，保证每次复现结果一样
    # *********************
    if args.run_idx_file is not None:
        local_rngs = [np.random.RandomState(args.seed + idx_torun[args.from_idx + e]) for e in range(args.num_processes)]
    else:
        local_rngs = [np.random.RandomState(args.seed + args.from_idx + e) for e in range(args.num_processes)]
    # **********

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    large_objects2idx = {obj: i for i, obj in enumerate(
        constants.map_save_large_objects)}
    all_objects2idx = {o: i for i, o in enumerate(constants.map_all_objects)}
    softmax = nn.Softmax(dim=1)

    # Logging and loss variables
    num_episodes = [0] * args.num_processes
    # for e in range(args.from_idx, args.to_idx):
    #     remainder = e % args.num_processes
    #     num_episodes[remainder] += 1
    
    # ************************
    if args.run_idx_file is not None:
        for e in idx_torun:
            remainder = e % args.num_processes
            num_episodes[remainder] += 1
    else:
        for e in range(args.from_idx, args.to_idx):
            remainder = e % args.num_processes
            num_episodes[remainder] += 1
    # ************************




    device = args.device = torch.device(
        "cuda:" + args.which_gpu if args.cuda else "cpu")
    if args.sem_policy_type == "mlm":
        Unet_model = MLM(
            (240, 240), (args.grid_sz, args.grid_sz), f"models/semantic_policy/{args.mlm_fname}.csv",
            options=args.mlm_options,device = device).to(device=device)
        if "mixed_search" in args.mlm_options:
            Unet_model_equal = MLM(
                (240, 240), (args.grid_sz, args.grid_sz),
                f"models/semantic_policy/mlmscore_equal.csv",
                options=args.mlm_options).to(device=device)

    elif args.sem_policy_type == "cnn":
        assert args.grid_sz == 8, "grid size should be 8 when sem_policy_type is 'film'"
        Unet_model = UNetMulti(
            (240, 240), num_sem_categories=24).to(device=device)
        sd = torch.load(
            'models/semantic_policy/new_best_model.pt', map_location=device)
        Unet_model.load_state_dict(sd)
        del sd
    
    # ******************
    elif args.sem_policy_type == "seq":
        lan_gran = args.language_granularity
        # __init__(self, input_shape, output_shape, occur_fname, device,options=list()):
        Unet_model = SEQ_SEARCH((240, 240), (args.grid_sz, args.grid_sz), args.objcofreq_file,lan_gran=lan_gran,
            device = device,options=args.seq_options,split=args.eval_split).to(device=device)
        if args.run_idx_file is not None:
            number_of_this_episode = idx_torun[args.from_idx]
        else:
            number_of_this_episode = args.from_idx
        Unet_model.reset_model(number_of_this_episode)
        # TODO: 之后把option换成自己的options
        # NOTE explored不在option里,所以它会反复找
        # NOTE 这里只能单线程跑了
    # *******************

    finished = np.zeros((args.num_processes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    fails = [0] * num_scenes
    prev_cns = [None] * num_scenes

    obs, infos, actions_dicts = envs.load_initial_scene()
    second_objects = []
    list_of_actions_s = []
    task_types = []
    whether_sliced_s = []
    # ************
    caution_pointer_s = []
    # TODO 是否要记忆当前要找的东西？
    # ***************
    for e in range(args.num_processes):
        second_object = actions_dicts[e]['second_object']
        list_of_actions = actions_dicts[e]['list_of_actions']
        task_type = actions_dicts[e]['task_type']
        sliced = actions_dicts[e]['sliced']
        # ****************
        caution_pointer = actions_dicts[e]['caution_pointers']
        # ****************
        second_objects.append(second_object)
        list_of_actions_s.append(list_of_actions)
        task_types.append(task_type)
        whether_sliced_s.append(sliced)
        # *************
        caution_pointer_s.append(caution_pointer)
        # ************

    task_finish = [False] * args.num_processes
    first_steps = [True] * args.num_processes
    num_steps_so_far = [0] * args.num_processes
    load_goal_pointers = [0] * args.num_processes
    list_of_actions_pointer_s = [0] * args.num_processes
    goal_spotted_s = [False] * args.num_processes
    list_of_actions_pointer_s = [0] * args.num_processes
    goal_logs = [[] for i in range(args.num_processes)]
    goal_cat_before_second_objects = [None] * args.num_processes
    subgoal_counter_s = [0] * args.num_processes
    found_subgoal_coordinates = [None] * args.num_processes
    # **************
    open4searches = [None] * args.num_processes #记录语义搜索模块，给出的搜索目标，如果这个目标已经open，则在名字后加上opened
    open_next_target_search = [False] * args.num_processes #开下一个同样的东西去搜索，如开下一个柜门
    new_plan_s = [False] * args.num_processes #记录plan是否已经修改，如果已经修改，则应该reset_goal
    drop_search_list_s = [None]*args.num_processes
    # TODO 加一个新的标识符，标记是否要replan
    reset_plan_type_s = [None] * args.num_processes
    if args.sem_policy_type == "seq":
        search_log_s = [[] for _ in range(args.num_processes)]
    # ****************


    do_not_update_cat_s = [None] * args.num_processes
    wheres_delete_s = [np.zeros((240, 240))] * args.num_processes
    sem_search_searched_s = [np.zeros((240, 240))] * args.num_processes

    args.num_sem_categories = 1 + 1 + 1 + 5 * args.num_processes
    if args.sem_policy_type != "none":
        # args.num_sem_categories = args.num_sem_categories + 23
        args.num_sem_categories = args.num_sem_categories + 30
    obs = torch.tensor(obs).to(device)

    torch.set_grad_enabled(False)

    # Initialize map variables
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5-: Semantic categories, as defined in sem_exp_thor.total_cat2idx
    # i.e. 'Knife': 0, 'SinkBasin': 1, 'ArmChair': 2, 'BathtubBasin': 3, 'Bed': 4, 'Cabinet': 5, 'Cart': 6, 'CoffeeMachine': 7, 'CoffeeTable': 8, 'CounterTop': 9, 'Desk': 10, 'DiningTable': 11, 'Drawer': 12, 'Dresser': 13, 'Fridge': 14, 'GarbageCan': 15, 'Microwave': 16, 'Ottoman': 17, 'Safe': 18, 'Shelf': 19, 'SideTable': 20, 'Sofa': 21, 'StoveBurner': 22, 'TVStand': 23, 'Toilet': 24, 'CellPhone': 25, 'FloorLamp': 26, 'None': 29
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
        int(full_h / args.global_downscaling)

    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]
                :lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # slam
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()
    sem_map_module.set_view_angles([45] * args.num_processes)

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    # _, local_map, _, local_pose = \
    #     sem_map_module(obs, poses, local_map, local_pose)

    # **********************
    _, local_map, _, local_pose,last_explore = \
        sem_map_module(obs, poses, local_map, local_pose) #last_explore是新加的，表示上一步的探索地图
    # ************************

    # Compute Global policy input
    locs = local_pose.cpu().numpy()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

    # For now
    global_goals = []
    for e in range(num_scenes):
        c1 = local_rngs[e].choice(local_w)
        c2 = local_rngs[e].choice(local_h)
        global_goals.append((c1, c2))

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]
        p_input['found_goal'] = 0
        p_input['wait'] = finished[e]
        p_input['list_of_actions'] = list_of_actions_s[e]
        p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
        p_input['consecutive_interaction'] = None
        p_input['consecutive_target'] = None
        p_input['manual_step'] = None
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                :].argmax(0).cpu().numpy()

    # obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(
    #     planner_inputs, goal_spotted_s)
    # **************************
    obs, rew, done, infos, goal_success_s, next_step_dict_s, palnner_list_action = envs.plan_act_and_preprocess(
        planner_inputs, goal_spotted_s)
    for e in range(num_scenes):
        list_of_actions_s[e] = palnner_list_action[e]
        # TODO 可能在这里处理replan
    # **************************
    goal_success_s = list(goal_success_s)
    view_angles = []
    for e in range(num_scenes):
        next_step_dict = next_step_dict_s[e]
        view_angle = next_step_dict['view_angle']
        view_angles.append(view_angle)

        fails[e] += next_step_dict['fails_cur']

    sem_map_module.set_view_angles(view_angles)

    consecutive_interaction_s, target_instance_s = [None]*num_scenes, [None]*num_scenes
    for e in range(num_scenes):
        num_steps_so_far[e] = next_step_dict_s[e]['steps_taken']
        first_steps[e] = False
        if goal_success_s[e]:
            if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) - 1:
                all_completed[e] = True
            else:
                subgoal_counter_s[e] = 0
                found_subgoal_coordinates[e] = None
                list_of_actions_pointer_s[e] += 1
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                reset_goal_true_false = [False] * num_scenes
                reset_goal_true_false[e] = True

                # If consecutive interactions,
                returned, target_instance_s[e] = determine_consecutive_interx(
                    list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                if returned:
                    consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]

                infos = envs.reset_goal(
                    reset_goal_true_false, goal_name, consecutive_interaction_s)
                    # 这里goal name 只传进去了一个东西，是默认每次只reset一个场景吗？多线程写的不太好

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for _ in range(args.num_training_frames//args.num_processes):
        skip_save_pic = task_finish.copy()
        # Reinitialize variables when episode ends
        for e, x in enumerate(task_finish):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)#这两个变量应该是成功率的统计信息
                success_per_category[infos[e]['goal_name']].append(success)
                traj_number[e] += 1
                init_map_and_pose_for_env(e)

                if not(finished[e]):
                    # load next episode for env
                    # number_of_this_episode = args.from_idx + \
                    #     traj_number[e] * num_scenes + e
                    # *********************
                    if args.run_idx_file is not None:
                        try:
                            number_of_this_episode = idx_torun[args.from_idx +traj_number[e] * num_scenes + e]
                        except:
                            print("idx_torun is not run out")
                    else:
                        number_of_this_episode = args.from_idx + \
                        traj_number[e] * num_scenes + e
                    # **********
                    print("steps taken for episode# ",  number_of_this_episode -
                          num_scenes, " is ", next_step_dict_s[e]['steps_taken'])
                    # 当前记录的还是上一轮的信息，因此要减一下？
                    completed_episodes.append(number_of_this_episode)
                    pickle.dump(
                        completed_episodes,
                        # open(f"results/completed_episodes_{args.eval_split}{args.from_idx}_to_{args.to_idx}_{dn}.p", 'wb'))
                        # 所以这里是不是搞错了？
                        open(f"{data_path}results/completed_episodes_{args.eval_split}{args.from_idx}_to_{args.to_idx}_{dn}.p", 'wb'))
                    # if args.leaderboard and args.test:
                    #     if args.test_seen:
                    #         add_str = "seen"
                    #     else:
                    #         add_str = "unseen"
                    #     # pickle.dump(actseqs, open(
                    #     #     f"results/leaderboard/actseqs_test_{add_str}_{dn}_{args.from_idx}_to_{args.to_idx}.p", "wb"))
                    #     pickle.dump(actseqs, open(
                    #         f"{data_path}results/leaderboard/actseqs_test_{add_str}_{dn}_{args.from_idx}_to_{args.to_idx}.p", "wb"))

                    # NOTE 为了还原动作序列，在valid里面也保存动作
                    if args.leaderboard:
                        pickle.dump(actseqs, open(
                            f"{data_path}results/leaderboard/actseqs_{args.split}_{dn}_{args.from_idx}_to_{args.to_idx}.p", "wb"))
                    load = [False] * args.num_processes
                    load[e] = True
                    do_not_update_cat_s[e] = None
                    wheres_delete_s[e] = np.zeros((240, 240))
                    sem_search_searched_s[e] = np.zeros((240, 240))
                    obs, infos, actions_dicts = envs.load_next_scene(load)
                    local_rngs[e] = np.random.RandomState(args.seed + number_of_this_episode)#之前初始化的时候有一个，为什么这里还有一个
                    view_angles[e] = 45
                    sem_map_module.set_view_angles(view_angles)
                    if actions_dicts[e] is None:
                        finished[e] = True
                    else:
                        second_objects[e] = actions_dicts[e]['second_object']
                        print("second object is ", second_objects[e])
                        list_of_actions_s[e] = actions_dicts[e]['list_of_actions']
                        task_types[e] = actions_dicts[e]['task_type']
                        whether_sliced_s[e] = actions_dicts[e]['sliced']
                        # ****************
                        caution_pointer_s[e] = actions_dicts[e]['caution_pointers']
                        # *************

                        task_finish[e] = False
                        num_steps_so_far[e] = 0
                        list_of_actions_pointer_s[e] = 0
                        goal_spotted_s[e] = False
                        found_goal[e] = 0
                        subgoal_counter_s[e] = 0
                        found_subgoal_coordinates[e] = None
                        first_steps[e] = True

                        all_completed[e] = False
                        goal_success_s[e] = False

                        obs = torch.tensor(obs).to(device)
                        fails[e] = 0
                        goal_logs[e] = []
                        goal_cat_before_second_objects[e] = None

                        # **************
                        open4searches[e] = None
                        open_next_target_search[e] = False
                        if args.sem_policy_type=="seq":
                            Unet_model.reset_model(number_of_this_episode)
                        drop_search_list_s[e] = []
                        new_plan_s[e] = False
                        reset_plan_type_s[e] = None
                        # ****************
                        # 一些关于地图的变量，如：full_map, loacal_map,full_pose, local_pose,origins,lmb?
                        # initial_map_and_pose
                        # 这些变量上面的initial_map_and_pose已经重置了
                        # slam set_view_angle?一开始的时候确实都是低着头,上面已经设置了
                        # 初始化global_goals
                        c1 = local_rngs[e].choice(local_w)
                        c2 = local_rngs[e].choice(local_h)
                        global_goals[e] = (c1,c2)
                        # 因为接下来就使用next_step_dict_s来选择目标，所以应该初始化一下next_step_dict_s
                        # 要重置整个next_step_dict比较难，重置后面用到的那几个量
                        # next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp'],next_step_dict_s[e]['none_interaction_mask']
                        next_step_dict_s[e]['next_goal']=False
                        next_step_dict_s[e]['delete_lamp']=False
                        next_step_dict_s[e]['none_interaction_mask']=False
                        if args.sem_policy_type == "seq":
                            search_log_s[e] = []        


        if sum(finished) == args.num_processes:
            print("all finished")
            # if args.leaderboard and args.test:
            #     if args.test_seen:
            #         add_str = "seen"
            #     else:
            #         add_str = "unseen"
            #     # pickle.dump(actseqs, open(
            #     #     "results/leaderboard/actseqs_test_" + add_str + "_" + dn + ".p", "wb"))
            #     pickle.dump(actseqs, open(
            #         f"{data_path}results/leaderboard/actseqs_test_" + add_str + "_" + dn + ".p", "wb"))
            if args.leaderboard:
                pickle.dump(actseqs, open(
                    f"{data_path}results/leaderboard/actseqs_{args.split}_{dn}_{args.from_idx}_to_{args.to_idx}.p", "wb"))            
            break

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        # _, local_map, _, local_pose = sem_map_module(
        #     obs, poses, local_map, local_pose, build_maps=True, no_update=False)

         # **********************
        _, local_map, _, local_pose,last_explore = \
            sem_map_module(obs, poses, local_map, local_pose,build_maps=True, no_update=False) 
        # ************************

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        for e in range(num_scenes):
            new_goal_needed = args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp']) #prompter原本设置
            # NOTE 刚加载第二条场景的时候，还没有运行过plan_act,按理说next_step_dict_s应该是没有的，所以，用的是上一步的东西？
            # new_goal_needed = args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp']) and not subgoal_counter_s[e]==0
            # args.delete_from_map_after_move_until_visible 在代码里面已经设置为True
            # 这里虽然写delete from map但是没看见它删除目标啊？
            # 确实不是在slam中被删除，而是在这里被删除了
            # 这里很难判断无法交互究竟是因为语义图多了一个点，还是交互姿势不准确，保留prompter原本设计比较好
            if len(second_objects[e])!=0 and list_of_actions_pointer_s[e]<len(second_objects[e]) and second_objects[e][list_of_actions_pointer_s[e]]:
                delete_from_map = True
            else:
                # delete_from_map = (next_step_dict_s[e]['none_interaction_mask'] and not subgoal_counter_s[e]==0)
                delete_from_map = not subgoal_counter_s[e]==0
            # delete_from_map = delete_from_map and new_goal_needed #这一行代码主要是为了调试
            # 如果这一步要拿的东西是第二个东西,把上一步的goal map附近的点删除,否则,如果是到目标点,遇到空掩码,就把上一步goal删除
            # if new_goal_needed:
            # *******************
            # # NOTE: 这里的条件已经被改变
            # none_interaction_mask =  next_step_dict_s[e]['none_interaction_mask']
            if new_goal_needed:
            # if delete_from_map:
            # ***********************
                # ep_num = args.from_idx + traj_number[e] * num_scenes + e

                # *********************
                if args.run_idx_file is not None:
                    ep_num = idx_torun[args.from_idx + traj_number[e] * num_scenes + e] #NOTE:这里如果设置了from id!=0就会出错，再详细检查一遍
                else:
                    ep_num = args.from_idx + \
                    traj_number[e] * num_scenes + e
                # **********
                
                if delete_from_map: 
                    # search failed, so delete regions neigboring the previous goal
                    # TODO: use disk dilation and connected entity for this part?
                    goal_neighbor_gl = getNeighborMap(goal_maps[e], dil_sz=args.goal_search_del_size)
                    wheres_delete_s[e][np.where(goal_neighbor_gl == 1)] = 1 #记录了语义图应该删除哪些东西
                    # 这个变量将会在这一子目标里一直使用，所以，一旦删除了，将会永远找不到
                    # TODO 之后使用HLSM那样的更新地图的方式？
                    print("delete last goal")
                    # 如果从地图删除了,搜索图里也应该赋值
                    goal_neighbor_ss = getNeighborMap(goal_maps[e], dil_sz=args.sem_search_del_size)
                    sem_search_searched_s[e][np.where(goal_neighbor_ss == 1)] = 1 #记录了搜索模块应该删除哪些东西, 应该在reset goal的时候重置, reset goal的时候重置了，但是plan_act立马会请求下一个goal，结果在这里又删除了
                    # NOTE 这个是累计的吗，删着删着，就全删完了，确实是累计的，步数累计太多会出现：目标在语义图里，但是已经被删除完了，也无法调用搜索模块，陷入死循环
                elif not subgoal_counter_s[e]==0:
                    # 但是如果没有从地图删除,并且当前是搜索的第一步,就不应该给搜索过的图赋值
                    goal_neighbor_ss = getNeighborMap(goal_maps[e], dil_sz=args.sem_search_del_size)
                    sem_search_searched_s[e][np.where(goal_neighbor_ss == 1)] = 1

            cn = infos[e]['goal_cat_id'] + 4
            wheres = np.where(wheres_delete_s[e])
            if args.save_pictures:
                image_vis = local_map[e, cn, :, :].clone()
            local_map[e, cn, :, :][wheres] = 0.0

            if args.save_pictures:
                # *********************
                if args.run_idx_file is not None:
                    ep_num = idx_torun[args.from_idx + traj_number[e] * num_scenes + e] 
                else:
                    ep_num = args.from_idx + \
                    traj_number[e] * num_scenes + e
                steps_taken = next_step_dict_s[e]['steps_taken']
                print(steps_taken)
                # **********
                image_vis[wheres] = 0.5
                image_vis = torch.cat((image_vis.unsqueeze(0),local_map[e, cn, :, :].unsqueeze(0)))
                # example:
                # self.plotSample(
                #             image_vis.cpu(), os.path.join(
                #                 out_dname, "large_search_info", f"{steps_taken}.png"),
                #             plot_type=None, names=[self.go_target,'explored_large','prob_scores'], wrap_sz=3)
                pics_dname = os.path.join("pictures", args.eval_split, args.dn, str(ep_num))

                Unet_model.plotSample(image_vis.cpu(),os.path.join(pics_dname
                                , "sem_map_deleted", f"{steps_taken}.png"),plot_type=None, names=[infos[e]['goal_name'],'after_delete'], wrap_sz=2)
            # NOTE 这里开的是橱柜，删的却是cup那一层的，只要请求下一个目标，就删除东西，这合理吗？
        
        # ***********************
        # NOTE 改变了代码顺序，先判断是否发现目标，然后再决定是否调用搜索模块；其实没什么差别
        # 从这一行开始，到1096行都在完成给出目标点这个事情
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        # for e in range(num_scenes):
        #     goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        for e in range(num_scenes):
            # 保存图像
            cn = infos[e]['goal_cat_id'] + 4 #这里的goal通过infos获得,而后面的搜索通过planner 获得,NOTE 查看两者的统一
            # 因为这里刚加载轨迹的时候还没有list_of_action?不，刚加载环境的时候也设置了
            prev_cns[e] = cn #这个变量好像没啥用
            cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
            # ep_num = args.from_idx + traj_number[e] * num_scenes + e
            # *********************
            if args.run_idx_file is not None:
                ep_num = idx_torun[args.from_idx + traj_number[e] * num_scenes + e]#TODO 核对一遍所有的轨迹号计算
            else:
                ep_num = args.from_idx + \
                traj_number[e] * num_scenes + e
            # **********
            if (not finished[e]) and args.save_pictures and (not skip_save_pic[e]):
                pics_dname = os.path.join("pictures", args.eval_split, args.dn, str(ep_num))
                target_pic_dname = os.path.join(pics_dname, "Sem_Map_Target")
                os.makedirs(target_pic_dname, exist_ok=True)
                steps_taken = next_step_dict_s[e]['steps_taken']
                cv2.imwrite(os.path.join(target_pic_dname, f"Sem_Map_Target_{steps_taken}.png"), cat_semantic_map * 255)#目标没有在语义图中的时候就会全部是黑色

            # 判断是否发现目标
            if cat_semantic_map.sum() != 0.:
                new_goal_needed = args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp'])
                if new_goal_needed or (found_subgoal_coordinates[e] is None):
                    cat_semantic_scores = np.zeros_like(cat_semantic_map)
                    cat_semantic_scores[cat_semantic_map > 0] = 1.

                    # delete coordinates in which the search failed
                    # NOTE 这里有一个bug，找到目标也会申请next goal，这时把目标删除不合理,已经修改了删除条件，但是这里会重复给出删除目标。其实local map已经删除过一遍，这边又删除一次？
                    # delete_coords = np.where(wheres_delete_s[e])
                    # 原本的代码这样写不太合理吧？之前都已经在lacal_map删除过一次了，现在又删除一遍？
                    # *************
                    delete_coords = np.where(sem_search_searched_s[e]) #不在语义图里面删除,但是在下一次给目标的时候删除, NOTE 这样有可能会重复操作
                    # **************
                    cat_semantic_scores[delete_coords] = 0

                    # TODO: might be better if cat_semantic_scores is eroded first, so that the goal won't be at edge of the receptacle region
                    wheres_y, wheres_x = np.where(cat_semantic_scores)#这里不太对吧，已经全都置为1了搜索最大的？
                    if len(wheres_x) > 0:
                        # go to the location where the taget is observed the most
                        target_y, target_x = searchArgmax(
                            args.goal_search_del_size, cat_semantic_scores)
                        found_subgoal_coordinates[e] = (target_y, target_x)
                        # 通过删除上一步的goal, 来求max的方式不太合理，感觉还是sample的方式好一点？
                        # TODO 在目标附近采样几个点?试一试随机迁移一下？local_rngs[e].choice(len(np.where(mask)[0]))
                    # **************************
                    else: #语义图中是有的，但是因为探索多次，都没交互成功，结果就被删除了，找不到一个好的交互点，在原来的目标点上加上一些随机性
                        # 与上面几行代码的区别仅仅在于：没有删除之前探索过的区域
                        # 一种情况：那边是一面镜子，认错了，goal点在镜子里，应该要删除那个goal点的
                        cat_semantic_scores = np.zeros_like(cat_semantic_map)
                        cat_semantic_scores[cat_semantic_map > 0] = 1.
                        wheres_y, wheres_x = np.where(cat_semantic_scores)
                        if len(wheres_x) > 0:
                            # go to the location where the taget is observed the most
                            target_y, target_x = searchArgmax(
                                args.goal_search_del_size, cat_semantic_scores)
                            # TODO 这里可能不应该重新搜索max，应该直接在之前的基础上偏移，因为，如果有两个目标点的话，搜索max可能就直接到了下一个goal那里
                            # 整个过程没有看到镜子里的台灯，为什么searargmax的值会改变？
                            found_subgoal_coordinates[e] = (target_y, target_x)
                            print(f'arg mas goal is {found_subgoal_coordinates[e]}')
                        temp_array = np.array([-2,-1,0,1,2]) #不至于偏到机器人找不到的地方
                        # example: random_element = rng.choice(my_array)
                        y_shift = local_rngs[e].choice(temp_array)*5
                        x_shift = local_rngs[e].choice(temp_array)*5
                        target_y = clip_value(target_y + y_shift,0,239) 
                        target_x = clip_value(target_x + x_shift,0,239)
                        found_subgoal_coordinates[e] = (target_y, target_x)
                        print(f"goal shift, new goal is {found_subgoal_coordinates[e]}")
                    # ****************************
                    # TODO 这样加上随机性是否好还有待实验;另外，如果目标是被误识别了（例如识别到镜子里的物体），那么应该删除

                if found_subgoal_coordinates[e] is None:
                    if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                        found_goal[e] = 0
                        goal_spotted_s[e] = False
                else:
                    goal_maps[e] = np.zeros_like(cat_semantic_map)
                    goal_maps[e][found_subgoal_coordinates[e]] = 1
                    print(f"set goal to goal location, target is {(target_y,target_x)}")
                    found_goal[e] = 1
                    goal_spotted_s[e] = True
                    # 从已发现的目标中选一个最大的点作为目标，所以会反复开同一个柜门
                    # 另外 goal_maps只是一个点，为什么会导致微波炉从语义图中消失

            else:
                found_subgoal_coordinates[e] = None
                if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                    found_goal[e] = 0
                    goal_spotted_s[e] = False
            
            # TODO 可视化 cat_semantic_map,sem_search_searched_s[e],cat_semantic_scores,goal_maps[e]
            # 只有new goal need的时候才会来详细判断是否发现目标，并给出目标点


        # ************************

        # Semantic Policy
        for e in range(num_scenes):
            # ep_num = args.from_idx + traj_number[e] * num_scenes + e
            # *********************
            if args.run_idx_file is not None:
                ep_num = idx_torun[args.from_idx + traj_number[e] * num_scenes + e]
            else:
                ep_num = args.from_idx + \
                traj_number[e] * num_scenes + e
            # **********
            if open4searches[e] is not None and 'opened' in open4searches[e]:
                open_object = True
            else:
                open_object = False
            if next_step_dict_s[e]['next_goal'] and (not finished[e]) and not open_object:
            # 只有请求新目标的时候才调用搜索序列
            # # *****************
            # # NOTE 改变了条件,没发现目标的时候才调用搜索模块
            # if next_step_dict_s[e]['next_goal'] and (not finished[e]) and not goal_spotted_s[e]:
            # # # look around 完毕，搜索完毕，走到目的地会请求next_goal(这样好像不行，因为如果发现了目标，但是又要开东西，结果就导致陷入了死循环，sem_exp_thor请求新目标，而goal found没用，搜索序列也没用，一直给不了一个新目标 NOTE 不过，为什么会出现这么一个无效的点？)
            # # ********************
                subgoal_counter_s[e] += 1

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e] #好像每次加载环境的时候没有很好的初始化？
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                obstacles = np.rint(local_map[e][0].cpu().numpy())
                if obstacles[120, 120] == 0:
                    mask = np.zeros((240, 240))
                    connected_regions = skimage.morphology.label(1 - obstacles, connectivity=2)
                    # 计算没有障碍物的区域的连通图
                    connected_lab = connected_regions[120, 120]
                    mask[np.where(connected_regions == connected_lab)] = 1
                    # 找出和[120, 120]联通的区域
                    mask[np.where(skimage.morphology.binary_dilation(
                        obstacles, skimage.morphology.square(5)))] = 1
                    # 怎么障碍物附近也标为1?
                    # TODO 可视化每一步mask的改变
                else:
                    dilated = skimage.morphology.binary_dilation(
                        obstacles, skimage.morphology.square(5))
                    mask = skimage.morphology.convex_hull_image(
                        dilated).astype(float)
                mask_grid = into_grid(torch.tensor(mask), args.grid_sz)
                where_ones = len(torch.where(mask_grid)[0])
                mask_grid = mask_grid.numpy().flatten()

                if (args.sem_policy_type == "none") or (args.explore_prob == 1.0):
                    chosen_i = local_rngs[e].choice(len(np.where(mask)[0]))
                    x_240 = np.where(mask)[0][chosen_i]
                    y_240 = np.where(mask)[1][chosen_i]
                    global_goals[e] = [x_240, y_240]

                else:
                    # Just reconst the common map save objects
                    map_reconst = torch.zeros(
                        (4+len(large_objects2idx), 240, 240))
                    map_reconst[:4] = local_map[e][:4]
                    test_see = {}
                    map_reconst[4+large_objects2idx['SinkBasin']
                                ] = local_map[e][4+1]
                    test_see[1] = 'SinkBasin'

                    start_idx = 2
                    for cat, catid in large_objects2idx.items():
                        if not (cat == 'SinkBasin'):
                            map_reconst[4+large_objects2idx[cat]
                                        ] = local_map[e][4+start_idx]
                            test_see[start_idx] = cat
                            start_idx += 1
                    # 重构的地图中只包含大物体
                    if args.save_pictures and (not skip_save_pic[e]):
                        pics_dname = os.path.join(
                            "pictures", args.eval_split, args.dn, str(ep_num))
                    else:
                        # NOTE for debug; change this, later will install detectron2 in environment
                        pics_dname = None
                        # # ******************
                        # pics_dname = os.path.join(
                        #     "pictures", args.eval_split, args.dn, str(ep_num))
                        # os.makedirs(pics_dname, exist_ok=True)
                        # # *******************

                    steps_taken = next_step_dict_s[e]['steps_taken']
                    if (goal_name in all_objects2idx) or ("sem_search_all" in args.mlm_options):
                        if ("mixed_search" in args.mlm_options) and (subgoal_counter_s[e] > 5):
                            pred_probs = Unet_model_equal(map_reconst.unsqueeze(0).to(
                                device), target_name=goal_name, out_dname=pics_dname,
                                steps_taken=steps_taken, temperature=args.mlm_temperature)
                        # ************************
                        elif args.sem_policy_type == 'seq':
                            print("sequence search")
                            pred_probs,go_target = Unet_model(map_reconst.unsqueeze(0).to(
                                device), target_name=goal_name, out_dname=pics_dname,
                                steps_taken=steps_taken,last_explore=last_explore,drop_search_list=drop_search_list_s[e])
                            print(f"current search target is {go_target}")
                            search_log_s[e].append(f"steps_taken: {steps_taken},goal name is {goal_name},search target is {go_target}")
                            if go_target is not None and go_target in constants.OPENABLE_CLASS_LIST:
                                if open4searches[e] is None:
                                    open4searches[e] = go_target
                                elif 'closed' in open4searches[e]:
                                    if go_target == open4searches[e].replace('closed',''): #NOTE 这里或许应该缩进？不应该
                                        open_next_target_search[e] = True
                                    else:
                                        open_next_target_search[e] = False
                                    open4searches[e] = go_target
                            elif not goal_name == open4searches[e]:#为了打开东西而调用的搜索
                                open4searches[e] = None
                                open_next_target_search[e] = False
                        # TODO 之后可能加一个记忆物体状态的？这样判断感觉很容易出错，代码很复杂
                        # **********************
                        else:
                            sem_temperature = subgoal_counter_s[e] if ("temperature_annealing" in args.mlm_options) else args.mlm_temperature
                            pred_probs = Unet_model(map_reconst.unsqueeze(0).to(
                                device), target_name=goal_name, out_dname=pics_dname,
                                steps_taken=steps_taken, temperature=sem_temperature)

                        # TODO: integrate the contents of this if-statements to sem_map_model.py
                        # if isinstance(Unet_model, MLM) or args.sem_policy_type=="seq":
                        if args.sem_policy_type=="seq" or args.sem_policy_type=="mlm":
                            # do not search where we have already searched before
                            pred_probs = pred_probs.detach().cpu()
                            pred_probs *= into_grid(torch.tensor(1 - sem_search_searched_s[e]), args.grid_sz) #把上一次goal附近去掉，NOTE 在当前搜索模块下，是否有点多余，需要更好的可视化一下
                            pred_probs = pred_probs.numpy().flatten()
                        else:
                            pred_probs = pred_probs.view(73, -1)
                            pred_probs = softmax(pred_probs)
                            pred_probs = pred_probs.detach().cpu().numpy()
                            pred_probs = pred_probs[all_objects2idx[goal_name]]

                        pred_probs = (1-args.explore_prob) * pred_probs + \
                            args.explore_prob * mask_grid * \
                            1 / float(where_ones)

                    else:
                        pred_probs = mask_grid * 1 / float(where_ones)

                    # Now sample one index
                    pred_probs = pred_probs.astype('float64')
                    pred_probs = pred_probs.reshape(args.grid_sz ** 2)

                    # TODO: incorporate subgoal counter with argmax_prob?
                    argmax_prob = 1.0 if ("search_argmax_100" in args.mlm_options) else 0.5
                    if ("search_argmax" in args.mlm_options) and (local_rngs[e].rand() < argmax_prob):
                        pred_probs_2d = pred_probs.reshape((args.grid_sz, args.grid_sz))
                        max_x, max_y = searchArgmax(1, pred_probs_2d)

                        # center the obtained coordinates so that the sum of pred_probs is maximized
                        # for the square region of args.sem_search_del_size
                        del_sz = args.sem_search_del_size // 2
                        search_mask = np.zeros_like(pred_probs_2d)
                        search_mask[max(0, max_x - del_sz):min(240, max_x + del_sz + 1),
                                    max(0, max_y - del_sz):min(240, max_y + del_sz + 1)] = 1
                        chosen_cell_x, chosen_cell_y = searchArgmax(
                            args.sem_search_del_size, pred_probs_2d, mask=search_mask)

                        # chosen_cell_x, chosen_cell_y = searchArgmax(
                            # args.sem_search_del_size // 2, pred_probs.reshape((args.grid_sz, args.grid_sz)))
                    else:
                        pred_probs = pred_probs / np.sum(pred_probs)
                        chosen_cell = local_rngs[e].multinomial(1, pred_probs.tolist())
                        chosen_cell = np.where(chosen_cell)[0][0]
                        chosen_cell_x = int(chosen_cell / args.grid_sz)
                        chosen_cell_y = chosen_cell % args.grid_sz

                    # Sample among this mask
                    mask_new = np.zeros((240, 240))
                    shrink_sz = 240 // args.grid_sz
                    mask_new[chosen_cell_x*shrink_sz:chosen_cell_x*shrink_sz+shrink_sz,
                                chosen_cell_y*shrink_sz:chosen_cell_y*shrink_sz+shrink_sz] = 1
                    mask_new = mask_new * mask * (1 - sem_search_searched_s[e])
                    # TODO 可视化mask sem_search_searched_s mask_new

                    if np.sum(mask_new) == 0:
                        chosen_i = local_rngs[e].choice(len(np.where(mask)[0]))
                        x_240 = np.where(mask)[0][chosen_i]
                        y_240 = np.where(mask)[1][chosen_i]

                    else:
                        chosen_i = local_rngs[e].choice(
                            len(np.where(mask_new)[0]))
                        x_240 = np.where(mask_new)[0][chosen_i]
                        y_240 = np.where(mask_new)[1][chosen_i]

                    if args.save_pictures and (not skip_save_pic[e]):
                        os.makedirs(pics_dname, exist_ok=True)
                        with open(os.path.join(pics_dname, f"{ep_num}.txt"), "a") as f:
                            f.write(
                                f"{steps_taken},{goal_name},{chosen_cell_x},{chosen_cell_y},{x_240},{y_240},{subgoal_counter_s[e]}\n")
                        Unet_model.plotSample(
                            pred_probs.reshape(
                                (1, args.grid_sz, args.grid_sz)),
                            os.path.join(pics_dname, "goal_sem_pol", f"{steps_taken}.html"), names=[goal_name], wrap_sz=1,
                            zmax=0.01)

                    global_goals[e] = [x_240, y_240]

                # for e in range(num_scenes):
                #     goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1 #只有用了搜索模块,才将goal maps置为搜索模块的值
                #     print("use search results")
                #     # NOTE 这个写在这里不合理,因为给一个目标,不会一步就走到地方,所以这个变量需要持续多步,直到请求下一个目标为止
        
        for e in range(num_scenes):
            if not goal_spotted_s[e] or (open_next_target_search[e] and Unet_model.go_target is not None and Unet_model.go_target==open4searches[e]): #如果让开东西找目标，那么按照搜索模块给的目标来找，打开后就可以用真实的目标
                goal_maps[e] = np.zeros((local_w, local_h))
                if next_step_dict_s[e]['next_goal']:#只要请求搜索模块，这个条件是一定满足的，所以，第一次请求搜索模块后，给出的目标会被偏移，下一次才不偏移，之后才固定下来
                    temp_array = np.array([-2,-1,0,0,0,0,1,2]) #不至于偏到机器人找不到的地方 #NOTE 0应该多一点
                    # example: random_element = rng.choice(my_array)
                    w_shift = local_rngs[e].choice(temp_array)
                    h_shift = local_rngs[e].choice(temp_array)
                    goal_x = clip_value(global_goals[e][0]+w_shift,0,239)
                    goal_y = clip_value(global_goals[e][1]+h_shift,0,239)
                else:
                    goal_x = global_goals[e][0]
                    goal_y = global_goals[e][1]
                goal_maps[e][goal_x, goal_y] = 1
                print(f"use search results, the goal is {goal_x},{goal_y}")
                # TODO 如果不是请求下一个目标，就不飘移
            else:
                print('use really goal')
                
        # ************************************************
        # 在plan act and preprocess之前，如果检测到需要更换plan，先set new plan然后执行
        # 这一步执行的goal可能不有效，但是下一步将完全按照new plan执行
        for e in range(num_scenes):
            if open4searches[e] is not None and not 'opened' in open4searches[e] and not goal_spotted_s[e] and go_target==open4searches[e]:
                list_of_actions_s[e],second_objects[e],caution_pointer_s[e] = \
                get_newplan(list_of_actions_s[e],second_objects[e],caution_pointer_s[e],go_target,list_of_actions_pointer_s[e],'open4search')
                # 修改env环境的相关参数
                reset_plan_type_s[e]='open4search'
                envs.reset_plan(second_objects,caution_pointer_s,reset_plan_type_s)
                reset_plan_type_s[e] = None #重置完后将其置为原来的值
                consecutive_interaction_s[e] = None
                new_plan_s[e] = True
                steps_taken = next_step_dict_s[e]['steps_taken']
                search_log_s[e].append(f"steps_taken is {steps_taken}, open {go_target} for search, the new lists of actions are {list_of_actions_s[e]},and the new caution_pointer_s is {caution_pointer_s[e]}, the new second_objects is {second_objects[e]}")
            # elif open4searches[e] is not None and 'opened' in open4searches[e] and not goal_spotted_s[e]:
            elif open4searches[e] is not None and 'opened' in open4searches[e] and not goal_spotted_s[e]: # NOTE 注意对比前后是否有差别
            # next_step_dict_s[0]['none_interaction_mask']
            # elif open4searches[e] is not None and 'opened' in open4searches[e] and next_step_dict_s[e]['none_interaction_mask']: 
                # NOTE :v4版本新改进，goal_spotted是通过语义图来判断的，不太准确，因此改为了下面这一行;也不对，这里mask是上一步开东西的mask
                # 已经打开容器，但是没发现目标
                list_of_actions_s[e],second_objects[e],caution_pointer_s[e] = \
                get_newplan(list_of_actions_s[e],second_objects[e],caution_pointer_s[e],None,list_of_actions_pointer_s[e],'close_open')
                reset_plan_type_s[e]='close_open'
                envs.reset_plan(second_objects,caution_pointer_s,reset_plan_type_s) #之前忘了reset，竟然可以正常工作？一旦plan修改，一定要在evs里reset, envs的操作都是针对所有环境的操作，要把所有环境的参数传进去
                reset_plan_type_s[e]=None
                returned, target_instance_s[e] = determine_consecutive_interx(
                list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                if returned:
                    consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                    # NOTE 如果打开柜子，柜子旁边发现了目标，并且捡空了，就会跳出consecutive interaction? 就会走动，应该是，然后就关不上柜子了
                new_plan_s[e] = True
                steps_taken = next_step_dict_s[e]['steps_taken']
                search_log_s[e].append(f'steps_taken is {steps_taken}, the object not in recp, close it and new list of actions is {list_of_actions_s[e]}, and the new caution_pointer_s is {caution_pointer_s[e]}, the new second_objects is {second_objects[e]}')
            #NOTE 之后应该加上replan的

            # reset goal：这里和循环的reset goal不同在于，这里没有让pointer+1，consecutive_interaction_s放到前面计算，并且因为这里是sem给出信号，所以这里subgoal_counter_s[e]不置为0，这里改不改都不影响，因为再次pick up object的时候，伴随着reset goal, 它都会被置为0
            if new_plan_s[e]: 
                found_subgoal_coordinates[e] = None
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                reset_goal_true_false = [False] * num_scenes
                reset_goal_true_false[e] = True
                infos = envs.reset_goal(
                    reset_goal_true_false, goal_name, consecutive_interaction_s)
                goal_spotted_s[e] = False
                found_goal[e] = 0
                wheres_delete_s[e] = np.zeros((240, 240))
                sem_search_searched_s[e] = np.zeros((240, 240))
                new_plan_s[e] = False
        # ****************************************************

        manual_step = None
        if args.manual_control:
            manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, u: LookUp, n: LookDown)")

        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = finished[e]
            p_input['list_of_actions'] = list_of_actions_s[e]
            p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
            p_input['consecutive_interaction'] = consecutive_interaction_s[e]
            p_input['consecutive_target'] = target_instance_s[e]
            p_input['manual_step'] = manual_step
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()

            if first_steps[e]:
                p_input['consecutive_interaction'] = None
                p_input['consecutive_target'] = None

        # obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(
        #     planner_inputs, goal_spotted_s)
        # *****************
        obs, rew, done, infos, goal_success_s, next_step_dict_s, palnner_list_action = envs.plan_act_and_preprocess(
        planner_inputs, goal_spotted_s)
        # 如果沿用之前的replan方式的话，可能要重置要下list_of_actions
        for e in range(num_scenes):
            if open4searches[e] is not None:
                if goal_success_s[e]: #这里的goal success可能只是成功地往前走了一步,不是子动作成功了?
                    # 不是,一开始忘了加[e]写为goal_success_s了,结果if [false] ->条件判定为成立
                    # goal_success_s记录的是子目标是否成功,如果只是走了一步,返回值也是false
                    if list_of_actions_s[e][list_of_actions_pointer_s[e]][1]=="OpenObject":
                        # 成功地打开了东西，记录一下
                        open4searches[e] = open4searches[e] + "opened"
                    elif list_of_actions_s[e][list_of_actions_pointer_s[e]][1]=="CloseObject":
                        # 成功地关上了之前打开的东西，消除记录
                        open4searches[e] = open4searches[e].replace('opened','closed')
                # elif list_of_actions_s[e][list_of_actions_pointer_s[e]][1]=="OpenObject":
                #     # 打开东西失败了
                #     open_fail_s[e] = True 
                #     # 下一次会请求新目标,这里把open4searches置为0
                #     open4searches[e] = None
                # TODO 想要把开东西失败的反馈加入,但是这里有点难,sem_exp_thor并没有把交互失败的信息返回回来
            # if open4searches[e] is not None and goal_success_s[e]: 
            #     if 'opened' not in open4searches[e]:
            #         #成功地打开了东西，记录一下
            #         #TODO 这里加上opened,但是下一次搜索,一下就把opened 去掉了,之前的代码是怎么work的?
            #         open4searches[e] = open4searches[e] + "opened"
            #     elif 'opened' in open4searches[e]: # 这里之前写成了if，刚加上的opened结果一下就被消除了
            #         # 成功地关上了之前打开的东西，消除记录
            #         open4searches[e] = None
        # ******************
        goal_success_s = list(goal_success_s)
        view_angles = []

        for e, p_input in enumerate(planner_inputs):
            next_step_dict = next_step_dict_s[e]

            view_angle = next_step_dict['view_angle']
            view_angles.append(view_angle)

            num_steps_so_far[e] = next_step_dict['steps_taken']
            first_steps[e] = False #加载新场景后，走了一步，立马变为False

            fails[e] += next_step_dict['fails_cur']
            if args.leaderboard and fails[e] >= args.max_fails:
                print("Interact API failed %d times" % fails[e])
                task_finish[e] = True

            if not(args.no_pickup) and (args.map_mask_prop != 1 or args.no_pickup_update) and next_step_dict['picked_up'] and goal_success_s[e]:
                do_not_update_cat_s[e] = infos[e]['goal_cat_id']
            elif not(next_step_dict['picked_up']):
                do_not_update_cat_s[e] = None

        sem_map_module.set_view_angles(view_angles)

        for e, p_input in enumerate(planner_inputs):
            if p_input['wait'] == 1 or next_step_dict_s[e]['keep_consecutive']:
                pass
            else:
                consecutive_interaction_s[e], target_instance_s[e] = None, None

            if goal_success_s[e]:
                if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) - 1:
                    all_completed[e] = True
                else:
                    # subgoal_counter_s[e] = 0 #这个变量应该是记录当前子目标去了多少个goal pointer,如果按照之前的逻辑，是否replan就不用sem来决定
                    # *******************************
                    # 由于要开东西找物体的时候，经常reset goal到这subgoal_counter_s混乱，这里重新计算
                    subgoal_counter_s[e] = get_opensearch_times(list_of_actions_s[e],list_of_actions_pointer_s[e]+1)
                    # *****************************
                    found_subgoal_coordinates[e] = None
                    list_of_actions_pointer_s[e] += 1
                    goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                    reset_goal_true_false = [False] * num_scenes
                    reset_goal_true_false[e] = True

                    returned, target_instance_s[e] = determine_consecutive_interx(
                        list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                    if returned:
                        consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                        # 这里记录了要连续的交互动作
                    # 打开东西的下一个动作不要低头
                    if list_of_actions_s[e][list_of_actions_pointer_s[e]-1][1] == 'OpenObject':
                        not_look_down_s = consecutive_interaction_s.copy()
                        not_look_down_s[e] = True
                        infos = envs.reset_goal(
                        reset_goal_true_false, goal_name, not_look_down_s)
                    else:
                        infos = envs.reset_goal(
                            reset_goal_true_false, goal_name, consecutive_interaction_s)
                    # 这里也只能是一个一个的reset
                    # 这里的reset goal会将角度置为45度
                    goal_spotted_s[e] = False
                    found_goal[e] = 0
                    wheres_delete_s[e] = np.zeros((240, 240))
                    sem_search_searched_s[e] = np.zeros((240, 240))

                    # ***********************
                    # 如果当前要去找第二个东西，不去刚刚放的位置找
                    if len(second_objects[e])!=0 and list_of_actions_pointer_s[e]<len(second_objects[e]) and second_objects[e][list_of_actions_pointer_s[e]]:
                        drop_search_list_s[e]=list_of_actions_s[e][list_of_actions_pointer_s[e]-1][0]
                    # *********************

                    # open4searches[e] = None
                    # open_next_target_search[e] = False #正常的reset goal应该可以加，但是在因为搜索导致的reset goal不能加，NOTE 都不能加

        # ------------------------------------------------------------------
        # End episode and log
        for e in range(num_scenes):
            # number_of_this_episode = args.from_idx + \
            #     traj_number[e] * num_scenes + e
            # *********************
            if args.run_idx_file is not None:
                number_of_this_episode = idx_torun[args.from_idx+traj_number[e] * num_scenes + e]
            else:
                number_of_this_episode = args.from_idx + \
                traj_number[e] * num_scenes + e
            # **********
            if number_of_this_episode in skip_indices:
                task_finish[e] = True

        for e in range(num_scenes):
            if all_completed[e]:
                if not(finished[e]) and args.test:
                    print("This episode is probably Success!")
                task_finish[e] = True

        for e in range(num_scenes):
            if num_steps_so_far[e] >= args.max_episode_length and not(finished[e]):
                print("This outputted")
                task_finish[e] = True

        for e in range(num_scenes):
            # number_of_this_episode = args.from_idx + \
            #     traj_number[e] * num_scenes + e
            # *********************
            if args.run_idx_file is not None:
                number_of_this_episode = idx_torun[args.from_idx + traj_number[e] * num_scenes + e]
            else:
                number_of_this_episode = args.from_idx + \
                traj_number[e] * num_scenes + e
            # **********
            if task_finish[e] and not(finished[e]) and not(number_of_this_episode in skip_indices):
                # logname = "results/logs/log_" + args.eval_split + "_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".txt"
                logname = data_path + "results/logs/log_" + args.eval_split + "_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".txt"
                with open(logname, "a") as f:
                    # number_of_this_episode = args.from_idx + \
                    #     traj_number[e] * num_scenes + e
                    # *********************
                    if args.run_idx_file is not None:
                        number_of_this_episode = idx_torun[args.from_idx + traj_number[e] * num_scenes + e]
                    else:
                        number_of_this_episode = args.from_idx + \
                        traj_number[e] * num_scenes + e
                    # **********
                    f.write("\n")
                    f.write("===================================================\n")
                    f.write("episode # is " +
                            str(number_of_this_episode) + "\n")

                    for log in next_step_dict_s[e]['logs']:
                        f.write(log + "\n")
                        # 按理说所有的都会保存的，为什么之前跑的代码没有保存？
                        # 没有加上debug_local？

                    if all_completed[e]:
                        if not(finished[e]) and args.test:
                            f.write("This episode is probably Success!\n")

                    if not(args.test):
                        # success is  (True,), log_entry is ({..}, )
                        log_entry, success = envs.evaluate(e)
                        log_entry, success = log_entry[0], success[0]
                        print("success is ", success)
                        f.write("success is " + str(success) + "\n")
                        print("log entry is " + str(log_entry))
                        f.write("log entry is " + str(log_entry) + "\n")
                        if success:
                            successes.append(log_entry)
                        else:
                            failures.append(log_entry)

                        print("saving success and failures for episode # ",
                              number_of_this_episode, "and process number is", e)
                        # with open("results/successes/" + args.eval_split + "_successes_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as g:
                        with open(data_path+"results/successes/" + args.eval_split + "_successes_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as g:
                            pickle.dump(successes, g)
                        # with open("results/fails/" + args.eval_split + "_failures_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as h:
                        with open(data_path+"results/fails/" + args.eval_split + "_failures_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as h:
                            pickle.dump(failures, h)

                    else:
                        print("episode # ", number_of_this_episode,
                              "ended and process number is", e)

                    # if args.leaderboard and args.test:
                    #     actseq = next_step_dict_s[e]['actseq']
                    #     actseqs.append(actseq)
                    if args.leaderboard:
                        actseq = next_step_dict_s[e]['actseq']
                        actseqs.append(actseq)
                
                if args.sem_policy_type=='seq':
                    search_log_name = data_path + "results/logs/search_log_" + args.eval_split + "_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".txt"
                    # NOTE 应该尽量不要写为append的形式，多次实验的话，会造成混乱
                    with open(search_log_name, 'a') as f:
                        f.write("\n")
                        f.write("===================================================\n")
                        f.write("episode # is " +
                                str(number_of_this_episode) + "\n")
                        for log in search_log_s[e]:
                            f.write(log + "\n")



                # Add to analyze recs
                analyze_dict = {'task_type': actions_dicts[e]['task_type'], 'errs': next_step_dict_s[e]['errs'], 'action_pointer': list_of_actions_pointer_s[e], 'goal_found': goal_spotted_s[e],
                                'number_of_this_episode': number_of_this_episode}
                if not(args.test):
                    analyze_dict['success'] = envs.evaluate(e)[1][0]
                else:
                    analyze_dict['success'] = all_completed[e]
                analyze_recs.append(analyze_dict)
                # with open("results/analyze_recs/" + args.eval_split + "_anaylsis_recs_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as iii:
                with open(data_path + "results/analyze_recs/" + args.eval_split + "_anaylsis_recs_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as iii:
                    pickle.dump(analyze_recs, iii)


if __name__ == "__main__":
    main()
    print("All finsihed!")
