from bag_play.read_rosbag import PoseCameraListener
from agents.sem_exp_thor_ros import Sem_Exp_Env_Agent_Thor
from models.sem_mapping import Semantic_Mapping
from arguments import get_args
import torch
import numpy as np
import shutil
import os
import cv2
picture_folder_name = 'pictures/'
if __name__ == '__main__':
    if os.path.exists(picture_folder_name):
        shutil.rmtree(picture_folder_name)
        # ***********
    os.makedirs(picture_folder_name)
    os.makedirs(picture_folder_name + "/fmm_dist")
    os.makedirs(picture_folder_name + "/obstacles_pre_dilation")
    os.makedirs(picture_folder_name + "/Sem")
    os.makedirs(picture_folder_name + "/Sem_Map")
    os.makedirs(picture_folder_name + "/Sem_Map_Target")
    os.makedirs(picture_folder_name + "/rgb")
    os.makedirs(picture_folder_name + "/depth")
    os.makedirs(picture_folder_name + "/depth_thresholded")
    # *************
    os.makedirs(picture_folder_name + "/Sem_Map_new")
    os.makedirs(picture_folder_name + "/mask")
    args = get_args()
    args.picture_folder_name = picture_folder_name
    device = args.device = torch.device(
        "cuda:" + args.which_gpu if args.cuda else "cpu")
    
    # Initialize map variables
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5-: Semantic categories, as defined in sem_exp_thor.total_cat2idx
    # i.e. 'Knife': 0, 'SinkBasin': 1, 'ArmChair': 2, 'BathtubBasin': 3, 'Bed': 4, 'Cabinet': 5, 'Cart': 6, 'CoffeeMachine': 7, 'CoffeeTable': 8, 'CounterTop': 9, 'Desk': 10, 'DiningTable': 11, 'Drawer': 12, 'Dresser': 13, 'Fridge': 14, 'GarbageCan': 15, 'Microwave': 16, 'Ottoman': 17, 'Safe': 18, 'Shelf': 19, 'SideTable': 20, 'Sofa': 21, 'StoveBurner': 22, 'TVStand': 23, 'Toilet': 24, 'CellPhone': 25, 'FloorLamp': 26, 'None': 29
    num_scenes = 1
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
    
    env = Sem_Exp_Env_Agent_Thor(args, 'NULL', 0)
    env.reset_total_cat_new([])
    env.setup_scene(args)
    obs, infos = env.reset(0)
    obs,_ = env._preprocess_obs(obs)
    poses = torch.from_numpy(np.asarray(
        [infos['sensor_pose']], dtype=np.float32)
    ).float().to(device)
    _, local_map, _, local_pose,last_explore = \
        sem_map_module(torch.Tensor(obs).unsqueeze(0).to(device), poses, local_map, local_pose)
    sem_map_module.set_view_angles([infos['view_angle']])
    for steps_taken in range(100):
        obs, rew, done, infos = env.step('move')
        poses = torch.from_numpy(np.asarray(
            [infos['sensor_pose']], dtype=np.float32)
        ).float().to(device)
        obs,_ = env._preprocess_obs(obs)
        _, local_map, _, local_pose,last_explore = \
            sem_map_module(torch.Tensor(obs).unsqueeze(0).to(device), poses, local_map, local_pose,build_maps=True, no_update=False) 
        sem_map_module.set_view_angles([infos['view_angle']])
        env.steps_taken = steps_taken
        
        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            # cn = infos[e]['goal_cat_id'] + 4 
            
            cv2.imwrite(
                picture_folder_name + "rgb/"+ "rgb_" + str(steps_taken) + ".png",
                obs[:3, :, :].transpose((1, 2, 0)))
            
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = None
            p_input['found_goal'] = 0
            if True:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()
            env.update_visited(planner_pose_inputs[e])
            env._visualize(p_input)

    