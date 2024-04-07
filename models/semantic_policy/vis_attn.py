import skimage.morphology
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from models.semantic_policy.sem_map_model import UNetMulti, CAP
from utils.model import Flatten
import math
import argparse
import torch.nn as nn
import copy
import os
import cv2
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.morphology
import cv2
import json
import glob
import matplotlib.patches as mpatches
from alfred_utils.gen import constants
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 2, 0"
base_path = '/raid/lmy/maps/'


map_save_large_objects = ['ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toilet']
goal_idx2cat = {0: 'Knife', 1: 'SinkBasin', 2: 'ArmChair', 3: 'BathtubBasin', 4: 'Bed', 5: 'Cabinet', 6: 'Cart', 7: 'CoffeeMachine', 8: 'CoffeeTable', 9: 'CounterTop', 10: 'Desk', 11: 'DiningTable', 12: 'Drawer', 13: 'Dresser', 14: 'Fridge', 15: 'GarbageCan', 16: 'Microwave', 17: 'Ottoman', 18: 'Safe', 19: 'Shelf', 20: 'SideTable', 21: 'Sofa', 22: 'StoveBurner', 23: 'TVStand', 24: 'Toilet', 25: 'Pillow', 36: 'None'}
goal_cat2idx = {v: k for k, v in goal_idx2cat.items()}
map_id2cat = {i:obj for i,obj in enumerate(map_save_large_objects)}
all_objects2idx = {o: i for i, o in enumerate(constants.map_all_objects)}
def colorImage(sem_map, color_palette):
    semantic_img = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))

    semantic_img.putpalette(color_palette)
    #semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
    semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = np.asarray(semantic_img)
    semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGBA2BGR)

    return semantic_img
def visualize_sem_map(inputs, return_raw=False):

    map_pred = inputs['map_pred']
    exp_pred = inputs['exp_pred']
    sem_map = inputs['sem_map_pred'].copy()


    grid = np.rint(map_pred)
    explored = np.rint(exp_pred)

    sem_map += 5

    no_cat_mask = sem_map == 28
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1
    
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
    flattened = pickle.load(open("miscellaneous/flattened.p", "rb"))
    color_palette2 += flattened.tolist()
    color_palette2 = [int(x*255.) for x in color_palette2]
    color_palette = color_palette2

    semantic_img = colorImage(sem_map, color_palette)
    # from cluster_map import process_semantic_map
    # process_semantic_map(sem_map, self.picture_folder_name, str(self.steps_taken))
    if return_raw:
        return sem_map, semantic_img
    return semantic_img

def visualize_importance(importance, out_dname, title):
    importance = importance.cpu().numpy()  # Convert to numpy array
    # Normalize the importance map to [0, 1]
    importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + torch.finfo(torch.float32).tiny)
    plt.figure()
    plt.imshow(importance, cmap='hot', interpolation='nearest')
    plt.colorbar()
    # plt.title("Importance Map")
    # out_dname = cur_dir + cat_name + '_' + str(e_step) + '_grad_' + str(args.dn) + ".png"
    plt.title(title)
    plt.savefig(out_dname)
    plt.close()
def preprocess_importance(importance):
    importance = abs(importance.cpu().numpy())  # Convert to numpy array
    # Normalize the importance map to [0, 1]
    importance = (importance - np.min(importance)) / (np.max(importance) - np.min(importance) + np.finfo(np.float32).tiny)
    
    # Return the normalized importance map
    return importance
def create_custom_colormap():
    # Colors: start with transparent, move to red
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # (R, G, B, alpha)
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_colormap'

    # Create the colormap
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def overlay_maps(semantic_img, importance_map, alpha=0.7):
    importance_map = preprocess_importance(importance_map)
    x, y = np.unravel_index(importance_map.argmax(), importance_map.shape)
    # print("Hey", x//16,y//16)
    # Ensure the importance map has only one channel
    if len(importance_map.shape) > 2:
        importance_map = importance_map[0]

    # Create a color map in red (can be adjusted)
    colormap = plt.get_cmap('hot')
    importance_colored = colormap(importance_map)

    # Convert to RGBA to match the semantic image
    importance_rgba = (importance_colored[:, :, :3] * 255).astype(np.uint8)
    alpha_channel = (importance_colored[:, :, 3] * 255 * alpha).astype(np.uint8)
    importance_rgba = np.dstack((importance_rgba, alpha_channel))

    # Combine the images
    semantic_pil = Image.fromarray(semantic_img).convert("RGBA")
    combined_img = Image.alpha_composite(semantic_pil, Image.fromarray(importance_rgba))

    return np.array(combined_img)

def mix_importance_map(map_learned):
    # Calculate the importance of each spatial location in the input
    # by taking the mean absolute value of the gradients
    obj_gradients = map_learned.grad.data.squeeze(0)[4:]
    obj_importance = torch.sum(torch.abs(obj_gradients), dim=0)
    inputs = {}
    map_copy = map_learned.detach().cpu()
    map_copy[0, -1, :, :] = 1e-5
    inputs['map_pred'] = map_copy[0, 0, :, :].numpy()
    inputs['exp_pred'] = map_copy[0, 1, :, :].numpy()
    inputs['sem_map_pred'] = map_copy[0, 4:, :, :].argmax(0).numpy()
    sem_map = visualize_sem_map(inputs)
    # sem_map_file_name = cur_dir + cat_name + '_' + str(e_step) + '_sem' + ".png"
    # cv2.imwrite(sem_map_file_name, sem_map)
    obj_combined_image = overlay_maps(sem_map, obj_importance)
    return obj_combined_image

def mix_prob_map(map_learned, pred_probs):
    # Calculate the importance of each spatial location in the input
    # by taking the mean absolute value of the gradients
    obj_importance = pred_probs
    inputs = {}
    map_copy = map_learned.detach().cpu()
    map_copy[0, -1, :, :] = 1e-5
    inputs['map_pred'] = map_copy[0, 0, :, :].numpy()
    inputs['exp_pred'] = map_copy[0, 1, :, :].numpy()
    inputs['sem_map_pred'] = map_copy[0, 4:, :, :].argmax(0).numpy()
    sem_map = visualize_sem_map(inputs)
    # sem_map_file_name = cur_dir + cat_name + '_' + str(e_step) + '_sem' + ".png"
    # cv2.imwrite(sem_map_file_name, sem_map)
    obj_combined_image = overlay_maps(sem_map, obj_importance)
    return obj_combined_image


def save_labeled_sem_map(sem_map, out_fname):
    if len(sem_map.shape)>3:
        sem_map = sem_map.squeeze(0)
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
    flattened = pickle.load(open("miscellaneous/flattened.p", "rb"))
    color_palette2 += flattened.tolist()
    color_palette = [(color_palette2[i], color_palette2[i+1], color_palette2[i+2]) for i in range(0, len(color_palette2), 3)]
    sem_map = sem_map.astype(np.uint8)
    plt.figure(figsize=(10, 8))
    cmap = matplotlib.colors.ListedColormap(color_palette[:sem_map.max()+1])
    plt.imshow(sem_map, cmap=cmap, interpolation='nearest')
    # Create a custom legend
    # Assuming object IDs are sequential and start from 0
    valid_labels = np.unique(sem_map[sem_map>4]) # Remove unexplored and no category
    # valid_labels_dict = {int(i):goal_idx2cat[i-5] for i in valid_labels if self.goal_idx2cat[i-5] in constants.map_save_large_objects}
    valid_labels_dict = {int(i):constants.map_save_large_objects[i-5] for i in valid_labels}
    # base_patches = [mpatches.Patch(color=color_palette[1], label="Obstacle"), mpatches.Patch(color=color_palette[3], label="Visited"),
    #     mpatches.Patch(color=color_palette[4], label="Goal")]
    base_patches = [mpatches.Patch(color=color_palette[1], label="Obstacle")]
    patches = base_patches + [mpatches.Patch(color=color_palette[i], label=obj) for i, obj in valid_labels_dict.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.colorbar(img, cmap=cmap)
    plt.savefig(out_fname)
    plt.close()



def vis_evalset(dataloader_eval, model, cur_dir):
    softmax = nn.Softmax(dim=1)
    for e_step, eval_batched in enumerate(dataloader_eval):
        batch_size_cur = 1
        map_learned = eval_batched['map_learned'].to(device=device)
        
        map_learned.requires_grad_(True)
        model.zero_grad()
        obj = eval_batched['obj']
        cat_name = eval_batched['obj'][0]
        pred_probs = model(map_learned, obj)  # shape is N x 1 x 15 x 15
        if isinstance(model, UNetMulti):
            pred_probs = pred_probs[:, all_objects2idx[cat_name]] 
        pred_probs = pred_probs.view(batch_size_cur, -1) # N x 225
        pred_probs = softmax(pred_probs)    # N x 1 x 225
        
        pred_probs = pred_probs * 1000 # scale it to enlarge the gradients
        pred_probs.sum().backward()

        # Calculate the importance of each spatial location in the input
        # by taking the mean absolute value of the gradients
        obj_gradients = map_learned.grad.data.squeeze(0)[4:]
        obj_importance = torch.sum(torch.abs(obj_gradients), dim=0)
        obj_out_dname = cur_dir + cat_name + '_' + str(e_step) + '_grad_' + str(args.dn) + ".png"
        # visualize_importance(obj_importance, obj_out_dname, "Object Importance Map")
        obst_importance = map_learned.grad.data.squeeze(0)[0].abs()
        obst_out_dname = cur_dir + cat_name + '_' + str(e_step) + '_grad_' + str(args.dn) + "_obst" + ".png"
        # visualize_importance(obst_importance, obst_out_dname, "Obtacle Importance Map")
        
        # Judge whether rely on obstacle more or object conext more
        full_out_dname = cur_dir + cat_name + '_' + str(e_step) + '_grad_' + str(args.dn) + "_full" + ".png"
        full_importance = map_learned.grad.data.squeeze(0).abs().argmax(dim=0)
        full_importance[full_importance!=0] = 1 # Make it binary: 1 indicates rely on object context more
        # visualize_importance(full_importance, full_out_dname, "Full Importance Map")

        
        inputs = {}
        map_copy = map_learned.detach().cpu()
        map_copy[0, -1, :, :] = 1e-5
        inputs['map_pred'] = map_copy[0, 0, :, :].numpy()
        inputs['exp_pred'] = map_copy[0, 1, :, :].numpy()
        inputs['sem_map_pred'] = map_copy[0, 4:, :, :].argmax(0).numpy()
        sem_map_raw, sem_map = visualize_sem_map(inputs, return_raw=True)  
        # save_labeled_sem_map(sem_map_raw, cur_dir + cat_name + '_' + str(e_step) + "_a_sem.png")
        # sem_map_file_name = cur_dir + cat_name + '_' + str(e_step) + '_sem' + ".png"
        # cv2.imwrite(sem_map_file_name, sem_map)

        obj_combined_image = overlay_maps(sem_map, obj_importance)
        obj_combined_dname = cur_dir + cat_name + '_' + str(e_step) + '_mix_' + str(args.dn) + ".png"
        '''    
        map_gt = eval_batched['map_gt'][0].squeeze(0).numpy()
        gt_positions = np.argwhere(map_gt)
        # Find the minimum and maximum coordinates
        min_y, min_x = np.min(gt_positions, axis=0)
        max_y, max_x = np.max(gt_positions, axis=0)

        # Define the top-left and bottom-right corners of the bounding box
        top_left = (min_x * 16, min_y * 16)
        bottom_right = ((max_x + 1) * 16, (max_y + 1) * 16)  # +1 to include the entire area of the max point

        # Draw the bounding box on the image
        cv2.rectangle(obj_combined_image, top_left, bottom_right, color=(0, 255, 0), thickness=1)
        '''
        cv2.imwrite(obj_combined_dname, cv2.cvtColor(obj_combined_image, cv2.COLOR_RGBA2BGR))

room2qualified_objects = {
    "Kitchen": ["Scissors", "Peach", "CanOpener", "Whisk"],
    "LivingRoom": ["Umbrella", "HairDrier", "Scissors", "Comb", "Peach", "Magazine", "Eyeglasses"],
    "Bedroom": ["HairDrier", "Scissors", "Comb", "Magazine", "Eyeglasses"],
    "Bathroom": ["HairDrier", "Toothbrush", "Comb"],
}
def get_room_type(room_id):
    for t in constants.SCENE_TYPE.keys():
        if room_id in constants.SCENE_TYPE[t]:
            return t
def vis_evalset_extra(dataloader_eval, model, cur_dir, extra_objects):
    softmax = nn.Softmax(dim=1)
    extra_obj_recept_cnt = {obj:{} for obj in extra_objects}
    room_ids_cnt = {sn:0 for sn in constants.SCENE_NUMBERS}
    for e_step, eval_batched in enumerate(dataloader_eval):
        inputs = {}
        map_copy = eval_batched['map_learned'].detach().cpu()
        map_copy[0, -1, :, :] = 1e-5
        print(map_copy[0, 4:].shape)
        sem_map_flattened = map_copy[0, 4:].numpy().argmax(0) + 1   # plus 1 to set apart from blanks
        inputs['map_pred'] = map_copy[0, 0, :, :].numpy()
        inputs['exp_pred'] = map_copy[0, 1, :, :].numpy()
        inputs['sem_map_pred'] = map_copy[0, 4:, :, :].argmax(0).numpy()
        sem_map_raw, sem_map = visualize_sem_map(inputs, return_raw=True)  
        # save_labeled_sem_map(sem_map_raw, cur_dir + str(e_step) + '_a_sem_' + str(eval_batched['room_id'][0].item()) + ".png")
        room_id = int(eval_batched['room_id'][0])
        room_type = get_room_type(room_id)
        qualified_objs = room2qualified_objects[room_type]
        room_ids_cnt[room_id] += 1
        batch_size_cur = len(qualified_objs)
        # batch_size_cur = len(extra_objects)

        map_learned = eval_batched['map_learned'].to(device=device).repeat(batch_size_cur, 1, 1, 1)
        map_learned.requires_grad_(True)
        model.zero_grad()
        pred_probs = model(map_learned, qualified_objs)  # shape is N x 1 x 15 x 15
        pred_probs = pred_probs.view(batch_size_cur, -1) # N x 225
        pred_probs = softmax(pred_probs)    # N x 225
        print(model.guided_attention_layer.guidance_rate)
        shrink_sz = 240 // 15
        # up_sample_probs = pred_probs.detach().reshape(batch_size_cur, 1, 15, 15).repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2)
        up_sample_probs = nn.Upsample(scale_factor=shrink_sz, mode='nearest')(pred_probs.detach().reshape(batch_size_cur, 1, 15, 15))
        pred_probs_np = pred_probs.detach().cpu().numpy().reshape(batch_size_cur, 15, 15)

        pred_probs = pred_probs * 1000 # scale it to enlarge the gradients
        pred_probs.sum().backward()
        for i, cat_name in enumerate(qualified_objs):
            # Calculate the importance of each spatial location in the input
            # by taking the mean absolute value of the gradients
            obj_gradients = map_learned.grad.data[i, 4:]
            obj_importance = torch.sum(torch.abs(obj_gradients), dim=0)
            
            x, y = np.unravel_index(pred_probs_np[i].argmax(), pred_probs_np[i].shape)
            mask = np.zeros_like(sem_map_flattened)
            mask[x*16:(x+1)*16, y*16:(y+1)*16] = 1
            sem_map_masked = sem_map_flattened * mask
            recept_idxs = np.unique(sem_map_masked)
            recepts = [map_id2cat[idx-1] for idx in recept_idxs if idx!=0]
            for recept in recepts:
                if recept in extra_obj_recept_cnt[cat_name].keys():
                    extra_obj_recept_cnt[cat_name][recept] += 1
                else:
                    extra_obj_recept_cnt[cat_name][recept] = 1                  
            # sem_map_file_name = cur_dir + cat_name + '_' + str(e_step) + '_sem' + ".png"
            # cv2.imwrite(sem_map_file_name, sem_map)

            obj_combined_image = overlay_maps(sem_map, obj_importance)
            obj_combined_dname = cur_dir + str(e_step) + '_' + cat_name + '_mix_' + str(args.dn) + ".png"
            cv2.imwrite(obj_combined_dname, cv2.cvtColor(obj_combined_image, cv2.COLOR_RGBA2BGR))
            
            prob_combined_image = mix_prob_map(map_learned[i:i+1], up_sample_probs[i][0])
            prob_combined_dname = cur_dir + str(e_step) + '_' + cat_name + '_prob_' + str(args.dn) + ".png"
            x, y = np.unravel_index(pred_probs_np[i].argmax(), pred_probs_np[i].shape)
            # print(x,y, cat_name)
            # print(pred_probs_np[i,x,y], pred_probs_np[i,x,y])
            # quit()
            # print(pred_probs_np[i,x,y], up_sample_probs[i,0,x*16+1,y*16+1], up_sample_probs[i,0,(x + 1) * 16 -1, (y + 1) * 16 -1])
            top_left = (y * 16, x * 16)
            bottom_right = ((y + 1) * 16, (x + 1) * 16)  # +1 to include the entire area of the max point
            # Draw the bounding box on the image
            cv2.rectangle(prob_combined_image, top_left, bottom_right, color=(0, 255, 0), thickness=1)
            cv2.imwrite(prob_combined_dname, cv2.cvtColor(prob_combined_image, cv2.COLOR_RGBA2BGR))
    extra_obj_fname = cur_dir + 'extra_obj_recepts_' + str(args.dn) + ".json"
    room_ids_fname = cur_dir + f'room_ids_{args.mode}_' + str(args.dn) + ".json"
    extra_obj_recept_cnt_sorted = {obj:dict(sorted(extra_obj_recept_cnt[obj].items())) for obj in extra_objects}
    with open(extra_obj_fname, 'w') as f:
        json.dump(extra_obj_recept_cnt_sorted, f, sort_keys=True, indent=4)
    with open(room_ids_fname, 'w') as f:
        json.dump(room_ids_cnt, f, sort_keys=True, indent=4)

def extract_id(file_name):
    # Assuming the ID is the part of the filename after the last underscore and before the extension
    return int(file_name.split('_')[-1].split('.')[0])
class DataLoadPreprocess(Dataset):
    def __init__(self,  mode='train', grid_size=15, uniform=True, mask=True, start=0, end=650):
        self.mode = mode
        if mode == 'train':
            self.maps_dir = f'/raid/lmy/maps/train/train_save_maps_{start}_{end}_REAL'
            self.all_data_path = [x[0] for x in os.walk(self.maps_dir)]
            self.start = start
            self.end = end
        else:
            self.maps_dir = '/raid/lmy/maps/tests_unseen/tests_unseen_save_maps_0_30_REAL'
            self.all_data_path = [x[0] for x in os.walk(self.maps_dir)]
            self.start = 0
            self.end = 30
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
    def __len__(self):
        # exclude self
        return len(self.all_data_path)-1
    def __getitem__(self, idx):
        idx = idx - self.start
        map_learned_dir = sorted(glob.glob(f"{self.maps_dir}/{idx}/map_step*"), key=extract_id)[-1]
        print(f"loading from {map_learned_dir}")
        with open(map_learned_dir, 'rb') as file:
            map_learned = pickle.load(file).cpu()
        total_cat2idx_learned = pickle.load(open(
            '/'.join(map_learned_dir.split('/')[:-1]) + '/total_cat2idx.p', 'rb'))
        map_learned_reconst = torch.zeros(
            (4+len(self.large_objects2idx), 240, 240))
        map_learned_reconst[:4] = map_learned[:4]
        for cat, catid in total_cat2idx_learned.items():
            if cat in self.large_objects2idx:
                map_learned_reconst[4+self.large_objects2idx[cat]
                                    ] = map_learned[4+catid]
        scene_info = pickle.load(open('/'.join(map_learned_dir.split('/')[:-1]) + '/scene_info.p', 'rb'))
        room_id = scene_info['floor_plan'].split("FloorPlan")[-1]
        # return {'map_learned': map_learned_reconst, 'map_gt': map_gt_reconst[idx:idx+1],
        #                 'obj': cur_dict['cat'], 'map_gt_obs': map_gt[0]}
        return {'map_learned': map_learned_reconst, 'room_id':int(room_id)}

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
    torch.set_grad_enabled(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str, default='none')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--extra_obj', type=bool, default=False)

    args = parser.parse_args()
    device = torch.device('cuda')
    # if not args.extra_obj:
    #     from models.semantic_policy.my_train_map_multi import DataLoadPreprocess
    from models.semantic_policy.my_train_map_multi import DataLoadPreprocess
    dataset_eval = DataLoadPreprocess(mode=args.mode)
    dataloader_eval = DataLoader(dataset_eval, 1)
    attn_mode = args.dn.split("_v")[0]   # no_attn: film
    print(attn_mode)
    if args.dn == "no_attn":
        grid_size = 8
        model = UNetMulti(
            (240, 240), num_sem_categories=24).to(device=device)
        sd = torch.load(
            'models/semantic_policy/new_best_model.pt', map_location=device)
        model.load_state_dict(sd)
        del sd
    else:
        grid_size = 15
        model = CAP((240, 240), num_sem_categories=24, attn_mode=attn_mode, residual_connect=False, guidance_rate=0).to(device=device)
        # model = CAP((240, 240), num_sem_categories=24, attn_mode=attn_mode).to(device=device)
    
        model.load_state_dict(torch.load('models/semantic_policy/cap_dict/'+ args.dn +'.pt'))
        # model.load_state_dict(torch.load('map_pred_models_multi/'+ 'cap_mul_residual' +'_seed_1lr_0.001epoch_0_step_22700.pt'))
    model.eval()
    
    cur_dir = 'models/semantic_policy/data/grad_vis/'
    if args.mode == 'train':
        cur_dir += 'train/'
    if args.extra_obj:
        cur_dir += 'extra_obj/'
    else:
        cur_dir += 'in_domain_obj/'
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    if args.extra_obj:
        extra_objects = ["Umbrella", "HairDrier", "Scissors", "Toothbrush", "Comb", "Peach", "CanOpener", "Whisk", "Magazine", "Eyeglasses"]
        # extra_objects = ['HandTowelHolder', 'LaundryHamper', 'Boots', 'TowelHolder', 'ToiletPaperHanger', 'Mirror', 'Footstool', 'Toaster']
        # extra_objects = ['ShowerGlass', 'Bathtub', 'Blinds', 'HandTowelHolder', 'Sink', 'LaundryHamper', 'Boots', 'Painting', 'TowelHolder', 'ToiletPaperHanger', 'Window', 'Mirror', 'Chair', 'Footstool', 'Curtains', 'Poster', 'Toaster']
        vis_evalset_extra(dataloader_eval, model, cur_dir, sorted(extra_objects))
    else:
        vis_evalset(dataloader_eval, model, cur_dir)

