import skimage.morphology
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sem_map_model import UNetMulti, CAP
from utils.model import Flatten
import math
import argparse
import torch.nn as nn
import copy
import os
import cv2
import random
import time
import json
import glob
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,0"
base_path = '/raid/lmy/maps/'
def find_files_with_numbers(directory):
    # Regex pattern to match files ending with numbers followed by ".p"
    pattern = re.compile(r"^map_step_\d+\.p$")
    # List to hold the names of matched files
    matched_files = []
    # Walking through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                matched_files.append(os.path.join(root, file))
    return matched_files
class DataLoadPreprocess(Dataset):
    def __init__(self,  mode='train', grid_size=15, uniform=True, mask=True):
        self.mode = mode
        if mode == 'train':
            # lists_path = "/raid/lmy/maps/train/cap_tmp_train_index.p"
            lists_path = base_path+'train/list_of_dicts_multitask.p'
            self.maps_dir = base_path+'train/'
        else:
            lists_path = base_path+'tests_unseen/list_of_dicts_tests_multitask.p'
            lists_path = base_path+'tests_unseen/list_of_dicts_tests.p'
            # lists_path = base_path+'tests_unseen/new_list_of_dicts_eval.pkl'
            self.maps_dir = base_path+'tests_unseen/'
        self.data_points_dir = self.maps_dir + 'data_points/'
        os.makedirs(self.data_points_dir, exist_ok=True)
        self.list_of_dicts = pickle.load(open(lists_path, 'rb'))
        print(len(self.list_of_dicts))
        # len list_of_dicts.p: 2766873, list_of_dicts_multitask.p:147788
        # # 726590
        
        self.all_objects = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan',
                       'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
        self.all_objects2idx = {o: i for i, o in enumerate(self.all_objects)}
        self.all_idx2objects = {i: o for i, o in enumerate(self.all_objects)}
        self.map_save_large_objects = ['ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable',
                                  'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toilet']
        self.large_objects2idx = {obj: i for i,
                                  obj in enumerate(self.map_save_large_objects)}
        self.grid_size = grid_size
        self.uniform = uniform
        self.mask = mask
        self.categories_counter = {obj:0 for obj in self.all_objects}
        self.new_to_original_mapping = {}  # Mapping from new_data_idx to (original_idx, channel)
        
        

    def save_data_point(self, data_point):
        """Saves a single data point to disk."""
        data_path = os.path.join(self.data_points_dir, f'data_point_{data_point["target_obj"]}_{self.categories_counter[data_point["target_obj"]]}.npz')
        np.savez(data_path, 
                 map_learned=np.array(data_point['map_learned']), 
                 map_gt=np.array(data_point['map_gt']),
                 map_gt_obs=np.array(data_point['map_gt_obs']))
        return data_path
    def process_gt_map_only(self):
        if self.mode == 'train':
            pass
        else:
            self.maps_dir = '/raid/lmy/maps/tests_unseen/tests_unseen_save_maps_0_30_GT'
            self.all_data_path = [self.maps_dir+'/'+x for x in os.listdir(self.maps_dir)]
        for cur_dir in self.all_data_path:
            map_gt_dir = find_files_with_numbers(cur_dir)
            if len(map_gt_dir)==0: 
                continue
            else:
                map_gt_dir = map_gt_dir[0]
            map_gt_reconst_dir = map_gt_dir.split(".p")[0] + "_reconst.p"
            if os.path.exists(map_gt_reconst_dir):
                continue
            print("Processing ", map_gt_dir)
            with open(map_gt_dir, 'rb') as file:
                map_gt = pickle.load(file).cpu()
            total_cat2idx_gt = pickle.load(open(
                '/'.join(map_gt_dir.split('/')[:-1]) + '/total_cat2idx.p', 'rb'))
            
            map_gt_reconst = torch.zeros((len(self.all_objects2idx), self.grid_size, self.grid_size))
            for cat, catid in total_cat2idx_gt.items():
                if cat in self.all_objects2idx:
                    map_gt_reconst[self.all_objects2idx[cat]] = self.into_grid(
                        map_gt[4+catid])
            with open(map_gt_reconst_dir, 'wb') as f:
                pickle.dump(map_gt_reconst, f)
                    # Store mapping from new_data_idx to original_idx and channel

    def process_data(self):
        data_points = []
        prev_map_gt_dir = ""
        prev_map_gt_info = []
        for original_idx, cur_dict in enumerate(self.list_of_dicts):
            if cur_dict['map_gt'] == prev_map_gt_dir:
                for obj, idx in prev_map_gt_info:
                    data_points.append({'map_gt_reconst':map_gt_reconst_dir, 'map_learned':cur_dict['map_learned'], 'cat':obj, 'cat_idx_in_map_gt':idx})
                continue
            print(original_idx)
            prev_map_gt_dir=cur_dict['map_gt']
            prev_map_gt_info=[]
            with open(self.maps_dir+cur_dict['map_gt'], 'rb') as file:
                map_gt = pickle.load(file).cpu()
            total_cat2idx_gt = pickle.load(open(
                self.maps_dir + '/'.join(cur_dict['map_gt'].split('/')[:-1]) + '/total_cat2idx.p', 'rb'))
            
            map_gt_reconst = torch.zeros((len(self.all_objects2idx), self.grid_size, self.grid_size))
            for cat, catid in total_cat2idx_gt.items():
                if cat in self.all_objects2idx:
                    map_gt_reconst[self.all_objects2idx[cat]] = self.into_grid(
                        map_gt[4+catid])
            map_gt_reconst_dir = cur_dict['map_gt'].split(".p")[0] + "_reconst.p"
            with open(self.maps_dir + map_gt_reconst_dir, 'wb') as f:
                pickle.dump(map_gt_reconst, f)
            for obj in self.all_objects:
                idx = self.all_objects2idx[obj]
                valid_indices = map_gt_reconst[idx] > 0
                if torch.any(valid_indices):
                    data_points.append({'map_gt_reconst':map_gt_reconst_dir, 'map_learned':cur_dict['map_learned'], 'cat':obj, 'cat_idx_in_map_gt':idx})
                    prev_map_gt_info.append((obj, idx))
                    # Store mapping from new_data_idx to original_idx and channel
        # Save the mapping
        with open(self.maps_dir + f'new_list_of_dicts_{self.mode}.pkl', 'wb') as f:
            pickle.dump(data_points, f)
    def show_data(self):
        cur_dict = self.list_of_dicts[0]
        scene_info = pickle.load(open(
                self.maps_dir + '/'.join(cur_dict['map_gt'].split('/')[:-1]) + '/scene_info.p', 'rb'))
        room_id = scene_info['floor_plan'].split("FloorPlan")[0]
        print(scene_info)
        quit()
        gt_dir = []
        for original_idx, cur_dict in enumerate(self.list_of_dicts):
            
            gt_dir.append(cur_dict['map_gt'])
        gt_dir = sorted(list(set(gt_dir)))
        print(gt_dir)
    # def re_tidy(self):
    #     with open(self.maps_dir + f'new_list_of_dicts_{self.mode}.pkl', 'rb') as f:
    #         data_points = pickle.load(f)
    #     for i, p in enumerate(data_points):
    #         if 'map_gt' in p:
    #             data_points[i]['map_gt_reconst']=p['map_gt'].split(".p")[0] + "_reconst.p"
    #             data_points[i].pop('map_gt')
    #     with open(self.maps_dir + f'new_list_of_dicts_{self.mode}.pkl', 'wb') as f:
    #         pickle.dump(data_points, f)
    # def process_data(self):
    #     for idx in range(len(self.list_of_dicts)):
    #         cur_dict = self.list_of_dicts[idx]
    #         # print(cur_dict)
    #         # quit()
    #         map_learned = pickle.load(
    #             open(self.maps_dir+cur_dict['map_learned'], 'rb')).cpu()
    #         total_cat2idx_learned = pickle.load(open(
    #             self.maps_dir + '/'.join(cur_dict['map_learned'].split('/')[:-1]) + '/total_cat2idx.p', 'rb'))
    #         map_gt = pickle.load(
    #             open(self.maps_dir+cur_dict['map_gt'], 'rb')).cpu()
    #         total_cat2idx_gt = pickle.load(open(
    #             self.maps_dir + '/'.join(cur_dict['map_gt'].split('/')[:-1]) + '/total_cat2idx.p', 'rb'))
    #         map_learned_reconst = torch.zeros(
    #             (4+len(self.large_objects2idx), 240, 240))
    #         map_learned_reconst[:4] = map_learned[:4]
    #         for cat, catid in total_cat2idx_learned.items():
    #             if cat in self.large_objects2idx:
    #                 map_learned_reconst[4+self.large_objects2idx[cat]] = map_learned[4+catid]

            
    #         # reconst map gt  [N, 62, 240, 240] -> [73, 15, 15]
    #         map_gt_reconst = torch.zeros((len(self.all_objects2idx), self.grid_size, self.grid_size))
    #         for cat, catid in total_cat2idx_gt.items():
    #             if cat in self.all_objects2idx:
    #                 map_gt_reconst[self.all_objects2idx[cat]] = self.into_grid(
    #                     map_gt[4+catid])

    #         for obj in self.all_objects:
    #             idx = self.all_objects2idx[obj]
    #             valid_indices = map_gt_reconst[idx] > 0
    #             if torch.any(valid_indices):
    #                 # Accumulate valid data points and corresponding ground truths
    #                 return_dict = {'map_learned': map_learned_reconst.tolist(), 'map_gt': map_gt_reconst[idx:idx+1].tolist(),
    #                             'map_gt_obs': map_gt[0].tolist(), 'target_obj': obj}
    #                 data_path = self.save_data_point(return_dict)
    #                 self.all_data_path.append({'path': data_path, 'target_obj': return_dict['target_obj']})
    #                 self.categories_counter[obj] += 1

    #     with open(self.maps_dir+f'cap_{self.mode}_index.json', 'w') as f:
    #         json.dump(self.all_data_path, f, indent=4)
    #     with open(self.maps_dir+f'cap_{self.mode}_info.json', 'w') as f:
    #         json.dump(self.categories_counter, f, indent=4)
    #     print(f"**********************For {self.mode}***************************")
    #     print(self.categories_counter)

    def into_grid(self, ori_grid):
        one_cell_size = math.ceil(240/self.grid_size)
        return_grid = torch.zeros(self.grid_size, self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if torch.sum(ori_grid[one_cell_size * i: one_cell_size*(i+1),  one_cell_size * j: one_cell_size*(j+1)].bool().float()) > 0:
                    return_grid[i, j] = 1
        # Do not normalize!! Consider it correct as long as give the right prediction.
        # if torch.sum(return_grid) != 0:
        #     return_grid = return_grid / torch.sum(return_grid)
        return return_grid

    def into_grid_simple(self, ori_grid):
        one_cell_size = math.ceil(240/self.grid_size)
        return_grid = torch.zeros(self.grid_size, self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if torch.sum(ori_grid[one_cell_size * i: one_cell_size*(i+1),  one_cell_size * j: one_cell_size*(j+1)].bool().float()) > 0:
                    return_grid[i, j] = 1

        return return_grid.unsqueeze(0)

    def mask_grid(self, ori_grid):
        ori_grid = ori_grid.cpu().numpy()
        ori_grid = skimage.morphology.binary_dilation(
            ori_grid, skimage.morphology.disk(3))
        center = (120, 120)
        connected_regions = skimage.morphology.label(
            1-ori_grid, connectivity=2)
        connected_lab = connected_regions[120, 120]
        mask = np.zeros((240, 240))
        mask[np.where(connected_regions == connected_lab)] = 1
        mask[np.where(ori_grid)] = 1
        mask = mask.astype(bool).astype(float)  # 0 for everywhere else

        # output mask in grid
        mask = torch.tensor(mask)
        mask = self.into_grid_simple(mask)
        return mask

if __name__ == '__main__':
    dataset_eval_processor = DataLoadPreprocess(mode='eval')
    # dataset_eval_processor.show_data()
    # dataset_eval_processor.process_gt_map_only()
    dataset_eval_processor.process_data()
    data_train_processor = DataLoadPreprocess()
    data_train_processor.process_data()
