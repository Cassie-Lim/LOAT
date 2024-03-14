from array import array
import os
from tkinter.font import ROMAN
import torch
import einops
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px

from torchvision import transforms
import alfred_utils.gen.constants as constants

import skimage.morphology

# *************
import cv2
import pickle
from improve_search_v4.language_parse.utils import * #与alfred traj数据交互的函数
from improve_search_v4.semantic_policy.get_search_loc import *
import re

import torch.nn.functional as F
import json
from alfred_utils.gen import constants
from sentence_transformers import SentenceTransformer
from scipy.ndimage import convolve
SPLIT_PATH = "alfred_data_small/splits/oct21.json"
# ************


open_class = constants.OPENABLE_CLASS_LIST

np.random.seed(0)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UNet(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.deconv_main = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU()
        )

        # outsize is 15^2 (7208 total)
        self.linear1 = nn.Linear(out_size * 32 + 256, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256)  # 73 object categories
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.train()

    def forward(self, inputs, goal_cats):
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0])
        goal_emb = self.goal_emb(goal_cats).view(-1, 256)  # goal name
        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16, 16)
        x = self.deconv_main(x)  # WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return x


class UNetMulti(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNetMulti, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 73, 3, stride=1, padding=1),
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # nn.ReLU(),
            # Flatten()
        )

        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def plotSample(self, pl, fname, plot_type=None, names=None, wrap_sz=None, img_sz=(1280, 720), zmax=1):
        dname = os.path.split(fname)[0]
        os.makedirs(dname, exist_ok=True)

        fig_recep = px.imshow(
            pl, facet_col=0, facet_col_wrap=wrap_sz, zmin=0, zmax=zmax)

        if os.path.splitext(fname)[-1] == ".html":
            config = dict({'scrollZoom': True})
            fig_recep.write_html(fname, config=config)
        else:
            fig_recep.write_image(fname, width=img_sz[0], height=img_sz[1])

    def forward(self, inputs, target_name, out_dname=None, steps_taken=None, temperature=1):
        x = self.main(inputs)
        #x = self.flatten(x)
        #x = self.softmax(x)

        return x


class UNetDot(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, stride=1, padding=1),
            nn.ReLU()
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # nn.ReLU(),
            # Flatten()
        )

        self.deconv_main = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            nn.Conv2d(128, 64, 1, stride=1, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            nn.Conv2d(64, 32, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1, stride=1, padding=1),
            nn.ReLU(),
        )

        # self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size) #outsize is 15^2 (7208 total)
        #self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256)  # 73 object categories
        self.goal_lin = nn.Linear(256, 128)
        self.goal_lin2 = nn.Linear(128, 128)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs, goal_cats):
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0])
        goal_emb = self.goal_emb(goal_cats).view(-1, 256)  # goal name
        goal_emb = self.goal_lin(goal_emb)
        goal_emb = self.relu(goal_emb)
        goal_emb = self.goal_lin2(goal_emb)
        goal_emb = self.relu(goal_emb)

        # Tile goal_emb

        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16, 16)
        x = self.deconv_main(x)  # WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return


class MLM(nn.Module):

    def __init__(self, input_shape, output_shape, score_fname, device, small2indx=None, large2indx=None, options=list()):
        super(MLM, self).__init__()

        # input shape is (240, 240)
        self.input_h, self.input_w = input_shape
        self.output_h, self.output_w = output_shape
        self.options = options

        self.scores = pd.read_csv(score_fname)
        if small2indx is None:
            self.small2indx = {'AlarmClock': 0, 'Apple': 1, 'AppleSliced': 2, 'BaseballBat': 3, 'BasketBall': 4, 'Book': 5, 'Bowl': 6, 'Box': 7, 'Bread': 8, 'BreadSliced': 9, 'ButterKnife': 10, 'CD': 11, 'Candle': 12, 'CellPhone': 13, 'Cloth': 14, 'CreditCard': 15, 'Cup': 16, 'DeskLamp': 17, 'DishSponge': 18, 'Egg': 19, 'Faucet': 20, 'FloorLamp': 21, 'Fork': 22, 'Glassbottle': 23, 'HandTowel': 24, 'HousePlant': 25, 'Kettle': 26, 'KeyChain': 27, 'Knife': 28, 'Ladle': 29, 'Laptop': 30, 'LaundryHamperLid': 31, 'Lettuce': 32, 'LettuceSliced': 33, 'LightSwitch': 34, 'Mug': 35, 'Newspaper': 36,
                               'Pan': 37, 'PaperTowel': 38, 'PaperTowelRoll': 39, 'Pen': 40, 'Pencil': 41, 'PepperShaker': 42, 'Pillow': 43, 'Plate': 44, 'Plunger': 45, 'Pot': 46, 'Potato': 47, 'PotatoSliced': 48, 'RemoteControl': 49, 'SaltShaker': 50, 'ScrubBrush': 51, 'ShowerDoor': 52, 'SoapBar': 53, 'SoapBottle': 54, 'Spatula': 55, 'Spoon': 56, 'SprayBottle': 57, 'Statue': 58, 'StoveKnob': 59, 'TeddyBear': 60, 'Television': 61, 'TennisRacket': 62, 'TissueBox': 63, 'ToiletPaper': 64, 'ToiletPaperHanger':65, 'ToiletPaperRoll': 66, 'Tomato': 67, 'TomatoSliced': 68, 'Towel': 69, 'Vase': 70, 'Watch': 71, 'WateringCan': 72, 'WineBottle': 73}
            self.large2indx = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5, 'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                               'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15, 'Safe': 16, 'Shelf': 17, 'SideTable': 18, 'SinkBasin': 19, 'Sofa': 20, 'StoveBurner': 21, 'TVStand': 22, 'Toilet': 23}
        else:
            # self.small2indx = {' '.join(re.findall("[A-Z][a-z]+", k)).lower():v
            #                    for k, v in small2indx.items()}
            # self.large2indx = {' '.join(re.findall("[A-Z][a-z]+", k)).lower():v
            #                    for k, v in large2indx.items()}
            self.small2indx = small2indx
            self.large2indx = large2indx

        self.all2indx = {**self.small2indx, **{k: v+len(self.small2indx) for k, v in self.large2indx.items()}}

        self.indx2small = {v: k for k, v in self.small2indx.items()}
        self.indx2large = {v: k for k, v in self.large2indx.items()}

        self.scores["recep_indx"] = self.scores["receps"].map(self.large2indx)
        self.scores["object_indx"] = self.scores["object"].map(self.all2indx)
        self.scores = self.scores[self.scores.notnull().all(1)]
        self.scores["recep_indx"] = self.scores["recep_indx"].astype(int)
        self.scores["object_indx"] = self.scores["object_indx"].astype(int)
        self.score_mat = self.scores.pivot(
            index="recep_indx", columns="object_indx", values="scores")
        self.score_mat = torch.tensor(
            self.score_mat.values.astype(np.float32)).to(device=device)

        self.softmax_fn = torch.nn.Softmax(dim=0)
        # self.score_mat = softmax_fn(self.score_mat)
        # self.score_mat = torch.exp(self.score_mat)

    def split4Map(self, data):
        return data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :], data[:, 3, :, :], data[:, 4:, :, :]

    def plotSample(self, pl, fname, plot_type=None, names=None, wrap_sz=None, img_sz=(1280, 720), zmax=1):
        dname = os.path.split(fname)[0]
        os.makedirs(dname, exist_ok=True)

        if plot_type == "recep":
            names = [self.indx2large[indx]
                     for indx in range(len(self.large2indx))]
            wrap_sz = 6
        elif plot_type == "object":
            names = [self.indx2small[indx]
                     for indx in range(len(self.small2indx))]
            wrap_sz = 15

        # import cv2
        # for img, name in zip(pl, names):
            # cv2.imwrite(os.path.join(os.path.dirname(fname), f"{name}.png"), img.to('cpu').detach().numpy().copy() * 255)

        fig_recep = px.imshow(
            pl, facet_col=0, facet_col_wrap=wrap_sz, zmin=0, zmax=zmax)
        # fig_recep = px.imshow(pl, facet_col=0, facet_col_wrap=wrap_sz)
        fig_recep.for_each_annotation(lambda a: a.update(
            text=names[int(a.text.split("=")[-1])]))

        if os.path.splitext(fname)[-1] == ".html":
            config = dict({'scrollZoom': True})
            fig_recep.write_html(fname, config=config)
        else:
            fig_recep.write_image(fname, width=img_sz[0], height=img_sz[1])

    def forward(self, inputs, target_name, out_dname=None, steps_taken=None, temperature=1):
        # NOTE: lower the temperature, sharper the distribution will be

        # @inputs is (batch)x(4+24 receptacle categories)x(@input_shape)x(@input_shape)
        # Ex. 1x28x240x240
        # First 4 elements of 28 are:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        if "Sliced" in target_name and target_name not in self.all2indx.keys():
            target_name=target_name.replace("Sliced","")
        roi_indx = self.all2indx[target_name]
        score_roi = self.score_mat[:, roi_indx:roi_indx+1]

        # larger this value is, faster the softmax output will approach to a uniform distribution
        temperature_ratio = 0.5
        for opt_name in self.options:
            if "tempRatio" in opt_name:
                temperature_ratio = float(opt_name.replace("tempRatio", ""))

        temperature = 1 + (temperature - 1) * temperature_ratio
        score_mat_n = self.softmax_fn(score_roi / temperature)

        obstacles, explored, curr_loc, past_loc, large_pred = self.split4Map(
            inputs)

        # create score map
        if (self.input_h, self.input_w) == (self.output_h, self.output_w):
            large_pred_redux = large_pred.clone()
        else:
            large_pred_redux = einops.reduce(
                large_pred, "b r (s1 h) (s2 w) -> b r s1 s2", "max", s1=self.output_h, s2=self.output_w)

        # b r h w -> b h w r
        large_tmp = large_pred_redux.permute((0, 2, 3, 1))

        # normalize spatially per receptacle class, to reduce the effect of receptacle's physical size
        epsilon = torch.finfo(torch.float32).tiny
        if "spatial_norm" in self.options:
            large_tmp /= (large_tmp.sum(keepdim=True, dim=[0, 1, 2]) + epsilon)

        # combine large object predictions with object-object relationship score by...
        if "aggregate_sum" in self.options:
            # sum (i.e. matrix multiplication)
            # matrix multiplication is equivalent to: torch.sum(large_tmp.unsqueeze(-1) * score_mat_n, dim=-2)
            prob_scores = torch.matmul(large_tmp, score_mat_n)
        elif "aggregate_max" in self.options:
            # max
            prob_scores = torch.max(
                large_tmp.unsqueeze(-1) * score_mat_n, dim=-2)[0]
        elif "aggregate_sample" in self.options:
            recep_exist_indices = torch.nonzero(large_tmp.sum(dim=[0, 1, 2]))[:, 0]

            if len(recep_exist_indices) == 0:
                # an edge case in which no receptacles are observed yet
                select_recep_index = random.randint(0, len(score_mat_n) - 1)
            else:
                one_hot = torch.zeros_like(score_mat_n)
                one_hot[recep_exist_indices] = 1
                score_exist = one_hot * score_mat_n
                prob_exist = score_exist / score_exist.sum()

                # sample a receptacle index
                select_recep_index = torch.multinomial(prob_exist.squeeze(), 1)
            prob_scores = large_tmp[:, :, :, select_recep_index:select_recep_index+1]

        # b h w o -> b o h w
        prob_scores = prob_scores.permute((0, 3, 1, 2))

        # remove explored locations from the score
        exp_map = None
        if "explored" in self.options:
            exp_map = explored
        elif "past_loc" in self.options:
            exp_map = past_loc

        if exp_map is None:
            scores = prob_scores
        else:
            exp_map_s = einops.reduce(
                exp_map, "b (s1 h) (s2 w) -> b s1 s2", "mean", s1=self.output_h, s2=self.output_w)
            scores = prob_scores * (1 - exp_map_s)

        # scale the scores (should sum to 1 for h x w dimenstions)
        scores += epsilon
        scores_n = scores / torch.sum(scores, dim=[2, 3], keepdim=True)
        # scores_n = scores

        # the output should be 1x73x(self.output_h)x(self.output_w)
        # scores_redux = einops.reduce(scores_n, "b r (s1 h) (s2 w) -> b r s1 s2", "sum", s1=self.output_h, s2=self.output_w)
        scores_redux = scores_n

        # visualize the result
        if out_dname is not None:
            self.plotSample(
                large_pred[0].cpu(), os.path.join(
                    out_dname, "pred_receptacles", f"{steps_taken}_receptacles.html"),
                img_sz=(1920, 1080), plot_type="recep", zmax=1)

            # self.plotSample(
            #     large_tmp[0].permute((2, 0, 1)).cpu(), os.path.join(
            #         out_dname, "pred_receptacles", f"{steps_taken}_receptacles_norm.html"),
            #     img_sz=(1920, 1080), plot_type="recep", zmax=0.01)

            # plot the extraneous information included in map_learned
            self.plotSample(
                inputs[0, :4, :, :].cpu(), os.path.join(
                    out_dname, "extra_info", f"{steps_taken}.png"),
                plot_type=None, names=["obstacles", "explored", "current location", "past location"], wrap_sz=4)

        # import cv2
        # target_prob_name = os.path.join(out_dname, "pred_receptacles", f"{steps_taken}_{target_name}.png")
        # cv2.imwrite(target_prob_name, (scores_redux / scores_redux.max() * 255)[0, 0].to('cpu').detach().numpy().copy())

        return scores_redux

class SEQ_SEARCH(nn.Module):
    '''
    DONE 1. 将原本的queen改为list；2，前几步以探索环境为目的；去drawer里面找drewer?bug已修复
    '''
    # TODO 3.none搜索的时候还有问题，如何既考虑；3.探索一个地方总也探索不完；
    # 4. drop search list 传进来，还没用
    # 5. 目前那个共现关系还有点问题，应该去除掉那些数量比较少的，有可能是噪点
    # 6. 如果在找柜子的过程中发现的目标,还是会先开柜子,再拿目标
    '''
    搜索是单线程，不用考虑多线程
    '''
    def __init__(self, input_shape, output_shape, occur_fname,lan_gran, device,options=list(),split='tests_unseen',max_times=10, attn_mode="cap_avg_auto"):
        '''
        max_times:防止某些特殊情况，使得跳转目标一直无法得到满足
        TODO 如果在一个大物体处寻找了max_times次小物体，强制跳转到下一个大物体寻找；如果在房间里寻找了10次大物体或者房间被探索完，请求replan
        lan_gran:语言粒度
        '''
        super(SEQ_SEARCH, self).__init__()
        # self.cap =  CAP(
        #     (240, 240), num_sem_categories=24, attn_mode="cap_avg_auto").to(device=device)
        # sd = torch.load(
        #     f'models/semantic_policy/cap_dict/cap_avg_auto.pt', map_location=device)
        new = "_v2"
        new = ""
        self.cap =  CAP(
            (240, 240), num_sem_categories=24, attn_mode=attn_mode, guidance_rate=0).to(device=device)
        sd = torch.load(
            f'models/semantic_policy/cap_dict/{attn_mode}{new}.pt', map_location=device)
        self.cap.load_state_dict(sd)
        del sd
        # input shape is (240, 240)
        self.input_h, self.input_w = input_shape
        self.output_h, self.output_w = output_shape
        self.options = options
        self.device = device
        self.split = split
        # debug
        print(f"options is {options}")

        
        import json
        self.obj_occ = json.load(open(occur_fname, "r"))
        self.large2indx = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5, 'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                               'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15, 'Safe': 16, 'Shelf': 17, 'SideTable': 18, 'SinkBasin': 19, 'Sofa': 20, 'StoveBurner': 21, 'TVStand': 22, 'Toilet': 23}
        self.indx2large = {v: k for k, v in self.large2indx.items()}
        # 传进来的地图经过重构后，也是按照这个顺序排列

        # self.go_quene = queue.Queue()
        # 把quene的设置全都改为list
        self.go_list = list()
        # 该list按照优先顺序存放要找到东西
        self.search_target = None #当前要找的东西
        self.go_target = None #当前为了找search_target要去的地方
        self.explored_large = None 
        #这里和main函数里的explored不同，这里主要记录当前物体的探索程度，该变量在当前search target累加
        self.explored_all = None
        # 这里记录所有的探索过的位置，之前探索的将随时间慢慢遗忘
        self.max_times = max_times
        self.times = 0
        self.none_times = 0
        # 记录下主要探索的目标，和探索情况
        self.main_search_target = None #上一步搜索的目标
        # self.mian_go_quene = None #上一步搜索的队列
        self.main_go_list = None #NOTE 变为list()?
        self.main_go_target = None 
        self.main_explore_large = None
        self.main_explore_all = None
        self.main_times = 0
        self.main_none_times = 0

        # 和语言解析有关的一些变量
        self.language_granularity = lan_gran

        # 加载语言位置
        if "lan_locs" in self.options:
            self.use_lan_locs = True
            parse_data_path = "improve_search_v4/language_parse/data"
            file_name = "parse_results_"+self.split+"_"+ self.language_granularity+".json"
            print(f"use parse results of {self.language_granularity}, the file name is {file_name}")
            with open(os.path.join(parse_data_path,file_name), "r") as f:
                self.lan_loc_datas = json.load(f)
            with open(SPLIT_PATH,'r') as f:
                self.split_data = json.load(f)[split]
            if "tests" in self.split:
                unseen = 'unseen' in self.split
                self.test_dict = read_test_dict(self.split,self.language_granularity,unseen)
        else:
            self.use_lan_locs = False

    def reset_model(self,number_of_this_episode):
        '''
        每加载一条新轨迹，需要将搜索信息重置一下
        '''
        # self.go_quene = queue.Queue()
        self.go_list = list()
        self.search_target = None #当前要找的东西
        self.go_target = None #当前为了找search_target要去的地方
        self.explored_large = None 
        #这里和main函数里的explored不同，这里主要记录当前物体的探索程度，该变量在当前search target累加
        self.explored_all = None
        # 这里记录所有的探索过的位置，之前探索的将随时间慢慢遗忘
        self.times = 0
        self.none_times = 0
        # 记录下主要探索的目标，和探索情况
        self.main_search_target = None #上一步搜索的目标
        # self.mian_go_quene = None #上一步搜索的队列
        self.main_go_list = list()
        self.main_go_target = None 
        self.main_explore_large = None
        self.main_explore_all = None
        self.main_times = 0
        self.main_none_times = 0

        # 每次加载轨迹都会reset,因此可以在这里记录lan_search_info
        if self.use_lan_locs:
            lan_loc_item = self.lan_loc_datas[number_of_this_episode]
            assert lan_loc_item['idx'] == number_of_this_episode,'the idx and the number of this episode is not match'
            self.lan_loc_info = lan_loc_item['loc_info']
            split_item = self.split_data[number_of_this_episode]
            repeat_idx, task_num = split_item["repeat_idx"],split_item["task"]
            traj = get_traj(task_num,repeat_idx)
            if "tests" in self.split:
                task_type, self.pddl_params = get_pred_pddl_params(self.test_dict,traj)
            else:
                self.pddl_params = get_pddl_params(traj)
                task_type = get_task_type(traj)
            self.drop_location = get_drop_locs(task_type,self.pddl_params) #这里使用整个轨迹的参数计算了drop location，但是不一定使用
    
    def set_go_listby_occ(self,target_name):
        '''
        仅仅通过共现频率来设置go_list
        '''
        if target_name in self.obj_occ:
            search_list = [recp for recp in self.obj_occ[target_name] if recp in self.large2indx]
            # 将search_list的东西放入队列
            # self.go_quene = queue.Queue()
            self.go_list = list()
            for obj in search_list:
                # 把要开的东西放到最后
                if not obj in open_class:
                    self.go_list.append(obj)
            for obj in search_list:
                if obj in open_class:
                    self.go_list.append(obj)
            # 不把冰箱放在最后面
            #         if not obj == 'Fridge':
            #             self.go_quene.put(obj)
            # if 'Fridge' in search_list:
            #     self.go_quene.put('Fridge')
        else:
            # 要找的是大东西，或者场景里没有指示
            # self.go_quene = queue.Queue()
            self.go_list = list()
    
    def set_go_list_addlan(self,target_name):
        if target_name == self.pddl_params['object_target'] or target_name == self.pddl_params['mrecep_target']:
            drop_locs = self.drop_location
        else:
            drop_locs = []
        self.go_list = get_search_loc_for_target(target_name,self.obj_occ,self.lan_loc_info,drop_locs,self.large2indx)
        # debug
        print(f"self.lan_loc_info is {self.lan_loc_info}")
        print(f"self.go_list is {self.go_list}")
  
    def reset_target(self, target_name,record_before_reset=False,restore=False):
        '''
        record_before_reset:在reset之前是否要记录
        restore:是否要恢复之前的记录
        '''
        # #TODO 之后或许可以对于小物体，在队列里面加上所有的scene_obj

        # # 记录下上一步搜索的信息
        if record_before_reset:
            self.main_search_target = self.search_target
            # self.mian_go_quene = self.go_quene
            self.main_go_list = self.go_list
            self.main_go_target = self.go_target
            self.main_explore_large = self.explored_large
            self.main_explore_all = self.explored_all
            self.main_times = self.times
            self.main_none_times = self.none_times
        
        if restore: #恢复上一步的信息
            assert self.main_search_target == target_name, "restore set wrong"
            self.search_target = target_name
            # self.go_quene = self.mian_go_quene
            self.go_list = self.main_go_list
            self.go_target = self.main_go_target
            self.explored_large = self.main_explore_large
            self.explored_all = self.main_explore_all
            self.times = self.main_times
            self.none_times = self.main_none_times
        else:
            # 设置本次搜索的信息
            self.search_target = target_name
            if not self.use_lan_locs:
                self.set_go_listby_occ(target_name)
            else:
                self.set_go_list_addlan(target_name)
            self.none_times = 0

    def next_target(self,sem_map_objs,steps_taken,search_steps=30):
        '''
        前 search_steps 步，以探索环境为目的，返回的target都为None
        '''
        # TODO 应该有一定概率，选择None
        # 已经围绕当前目标转了一圈，没有发现，搜索下一个目标
        # 应该找当前队列里面有,并且地图里面也有的东西

        # # 决定是否要去None搜索
        # rate = 0.8*(1 / (1 + 0.1 * np.log(steps_taken+1))) #TODO 随时间线性衰减吧，不然太快了,0.1衰减太慢了，而且，如果设置为None的话，应该只找一次，然后重置目标的
        # search_none = np.random.choice([True, False], p=[rate, 1-rate])
        # if search_none:
        #     self.go_target = None
        #     print("random choice none")
        #     self.none_times = 0 #这是在随机寻找的次数，不算

        # 按照这样随机的感觉效果也不好？尝试让它在前几步以探索环境为目的
        if steps_taken<=search_steps:
            self.go_target = None
            self.none_times = 0 #是为了探索环境而找给的None，不算次数
        else:
            self.go_target = None #NOTE 如果要找之前的版本的原因的话，应该注释掉这里，选用下面这一条
            for recp in self.go_list:
                # self.go_target = None
                if recp in sem_map_objs:
                    # self.go_target = self.go_list.pop(recp)
                    self.go_target = recp
                    self.go_list.remove(recp)
                    # 已经得到self.go_target，应该跳出循环，否则还是会一直执行，直到队列中最后一个东西
                    break
        self.explored_large = None
        self.times = 0
            # go_targets = []
            # self.go_target = None
            # step = 0
            # while not self.go_quene.empty() and step<=10:
            #     # 有可能所有在队列中的物体都不在语义图中,这样就会陷入死循环
            #     go_target = self.go_quene.get()
            #     if go_target in sem_map_objs:
            #         self.go_target = go_target
            #         # 已经得到self.go_target，应该跳出循环，否则还是会一直执行，直到队列中最后一个东西
            #         break
            #     else:
            #         go_targets.append(go_target)
            #     step += 1
            # # 把go_targets放回队列
            # for go_target in go_targets:
            #     self.go_quene.put(go_target)
            # # # 因为不能判断是否完全找到，因此把这个再放回去。这样会导致其它地方没有被探索的机会
            # # self.go_quene.put(self.go_target)
            # self.explored_large = None
            # self.times = 0

    
    def split4Map(self, data):
        return data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :], data[:, 3, :, :], data[:, 4:, :, :]

    def plotSample(self, pl, fname, plot_type=None, names=None, wrap_sz=None, img_sz=(1280, 720), zmax=1):
        '''
        可供外部函数调用
        '''
        dname = os.path.split(fname)[0]
        os.makedirs(dname, exist_ok=True)

        if plot_type == "recep":
            names = [self.indx2large[indx]
                     for indx in range(len(self.large2indx))]
            wrap_sz = 6
        elif plot_type == "object":
            names = [self.indx2small[indx]
                     for indx in range(len(self.small2indx))]
            wrap_sz = 15

        # import cv2
        # for img, name in zip(pl, names):
            # cv2.imwrite(os.path.join(os.path.dirname(fname), f"{name}.png"), img.to('cpu').detach().numpy().copy() * 255)

        fig_recep = px.imshow(
            pl, facet_col=0, facet_col_wrap=wrap_sz, zmin=0, zmax=zmax) #文档里面没有写这里可以
        # fig_recep = px.imshow(pl, facet_col=0, facet_col_wrap=wrap_sz)
        fig_recep.for_each_annotation(lambda a: a.update(
            text=names[int(a.text.split("=")[-1])]))

        if os.path.splitext(fname)[-1] == ".html":
            config = dict({'scrollZoom': True})
            fig_recep.write_html(fname, config=config)
        else:
            fig_recep.write_image(fname, width=img_sz[0], height=img_sz[1])


    def forward(self, inputs, target_name,out_dname,last_explore,steps_taken=None,debug= False,drop_search_list=[],next_threshold=0.1):
        '''
        如果本身就是一个容器的话，是否会到另一个容器里面去寻找，会不会出现：为了找刀，去开Box，为了找box去开drawer?不会出现，因为给出的目标一定是语义图里面有的
        next_threshold:切换到下一个目标的阈值
        TODO 调节一下next_threshold试试？
        return: scores_redux: b o h w
        '''
        # @inputs is (batch)x(4+24 receptacle categories)x(@input_shape)x(@input_shape)
        # Ex. 1x28x240x240
        # First 4 elements of 28 are:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations

        # 对inputs进行一些处理
        inputs_new = inputs.clone()
        last_explore_new = last_explore.clone()
        # 1. reshape inputs
        if (self.input_h, self.input_w) != (self.output_h, self.output_w):
            inputs_new = einops.reduce(
                    inputs_new, "b r (s1 h) (s2 w) -> b r s1 s2", "max", s1=self.output_h, s2=self.output_w)
            last_explore_new =  einops.reduce(
                    last_explore_new, "b (s1 h) (s2 w) -> b s1 s2", "max", s1=self.output_h, s2=self.output_w)     
        obstacles, explored_all, curr_loc, past_loc, large_pred = self.split4Map(
                inputs_new)
        
        # obstacles, explored_all,curr_loc, past_loc,last_explore_new: b s1 s2
        # large_pred: b r h w

        # 获得场景中有的物体，TODO 或许这部分后面由main来完成，因为好多地方都要用到?如果用main的可能每一步都要算一下
        sem_map_obj_idx = large_pred[0].argmax(0).cpu().numpy()
        scene_obj_list = []
        for index_obj in range(len(self.indx2large)):
            if index_obj in sem_map_obj_idx:
                scene_obj_list.append(self.indx2large[index_obj])
        if debug:
            print("scene_obj_list:", scene_obj_list)

        target_name = target_name.replace("Sliced","") #NOTE 如果需要调试的话，要把这里去掉，是不是应该记下来物体的位置
        if not target_name == self.search_target:
            if target_name == self.go_target:#因为找杯子，去找微波炉，记下原来找杯子的情况
                record_before_reset = True
            else:
                record_before_reset = False
            if self.main_search_target is not None and target_name==self.main_search_target:
                restore = True
            else:
                restore = False
            # 现在要找的目标不等于之前要找的
            self.reset_target(target_name,record_before_reset,restore)
            if not restore:
                self.next_target(scene_obj_list,steps_taken)

        # 对explored_large进行累加
        if self.explored_large is None:
            self.explored_large = last_explore_new
        else:
            #  cat (b h w) (b h w)-> (b 1 h w) (b 1 h w)->b h w
            self.explored_large,_ = torch.max(torch.cat((self.explored_large.unsqueeze(1),last_explore_new.unsqueeze(1)),dim=1),1)
        
        # 对explore_all进行累加 TODO 修改这里的计算方式？
        if self.explored_all is None:
            self.explored_all = 0.2*explored_all+0.8*past_loc
        else:
            self.explored_all = 0.8*(self.explored_all)+0.2*(0.2*explored_all+0.8*past_loc)
        
        
        # 对每次目标都判断一下是否探索过，如果全探索过，则寻找下一个目标，直到找到为止
        # 判断是否应该请求下一个目标
        if self.go_target is None: #是None有可能是因为刚开始，或者当前场景里的东西不多
            next_goal = True
        elif self.go_target in ['Fridge', 'Microwave']: #TODO 关键还是要解决重复探索的问题
            next_goal = self.times > 0
            # 对于冰箱和微波炉，只开一次
        elif self.go_target in ['Cabinet', 'Drawer', 'Safe', 'Box']:
            next_goal = self.times >= self.max_times/2
            # 开东西，开一半次数就可以 TODO 之后可能需要做到开过的就不开
        elif self.times >= self.max_times:
            next_goal = True
        else: #计算探索率，TODO 之后修改为更好的指标
            go_tag_idx = self.large2indx[self.go_target]
            # large_pred: b r h w 
            recep = large_pred[:, go_tag_idx, :, :] #prob_scores: b h w NOTE 之后看一看这个为什么这么不规则
            if self.explored_large is not None:
                prob_scores = recep * (1 - self.explored_large)
                rate = torch.sum(prob_scores)/torch.sum(recep)
                next_goal = rate < next_threshold
            else:
                next_goal = False
            if debug:
                print(f'rate {rate}')
        
        if debug:
            print(f"times {self.times}, none times {self.none_times}, next_goal {next_goal}")
        
        if next_goal:
            self.next_target(scene_obj_list,steps_taken)
        
        if debug:
            print(f"current search goal is {self.go_target}")
        
        if self.go_target is not None:
            self.times += 1
            # 按照正常逻辑，应该explore_large是有的
            go_tag_idx = self.large2indx[self.go_target]
            # large_pred: b r h w 
            recep = large_pred[:, go_tag_idx, :, :]
            if self.explored_large is not None:
                prob_scores = recep * (1 - self.explored_large)
                # 可视化
                if out_dname is not None:
                    image_vis = torch.cat((recep,self.explored_large,prob_scores))
                    self.plotSample(
                            image_vis.cpu(), os.path.join(
                                out_dname, "large_search_info", f"{steps_taken}.png"),
                            plot_type=None, names=[self.go_target,'explored_large','prob_scores'], wrap_sz=3)
                    if debug:
                        pickle.dump(recep.cpu(),open(f"improve_search_v4/debug_data/recep_{steps_taken}.pkl",'wb'))
                        pickle.dump(self.explored_large.cpu(),open(f"improve_search_v4/debug_data/explored_large_{steps_taken}.pkl",'wb'))
            else:
                prob_scores = recep
                # 可视化
                if out_dname is not None:
                    self.plotSample(
                            recep.cpu(), os.path.join(
                                out_dname, "large_search_info", f"{steps_taken}.png"),
                            plot_type=None, names=[self.go_target], wrap_sz=1)
                    if debug:
                        pickle.dump(recep.cpu(),open(f"improve_search_v4/debug_data/recep_{steps_taken}.pkl","wb"))
            if debug:
                pickle.dump(curr_loc.cpu(),open(f"improve_search_v4/debug_data/curr_loc_{steps_taken}.pkl",'wb'))
        else: # go_target is None
            if target_name in constants.map_all_objects:
                # self.cap.set_guidance_rate(steps_taken)
                pred_probs = self.cap(inputs, [target_name])
                pred_probs = pred_probs.view(1, -1)
                pred_probs = F.softmax(pred_probs, dim=1)
                grid_sz = 15
                shrink_sz = 240 // grid_sz
                pred_probs = nn.Upsample(scale_factor=shrink_sz, mode='nearest')(pred_probs.detach().reshape(1, 1, grid_sz, grid_sz))[0][0]
            elif target_name in scene_obj_list:
                pred_probs = large_pred[:, self.large2indx[target_name]]
            else:
                pred_probs = 1
            self.none_times += 1
            for i in range(len(obstacles)): # b
                obstacle = obstacles[i].cpu()
                room_area = get_room_area(obstacle=obstacle)
                room_area = torch.from_numpy(room_area).unsqueeze(0).to(self.device)
                if i == 0:
                    room_areas = room_area
                else:
                    room_areas = torch.cat((room_areas,room_area),0)
            prob_scores = room_area * (1- obstacles) * pred_probs #TODO 之后可能选一个最远点
            # prob_scores = room_area * (1- obstacles) * (1- self.explored_all) * pred_probs #TODO 之后可能选一个最远点
            if out_dname is not None:
                image_vis = torch.cat((obstacles,room_area,self.explored_all,prob_scores))
                self.plotSample(
                        image_vis.cpu(), os.path.join(
                            out_dname, "None_search_info", f"{steps_taken}.png"),
                        plot_type=None, names=['obstacle','room_area','explore_all','prob_scores'], wrap_sz=4)

        prob_scores = prob_scores.unsqueeze(1)
        # 增加一个维度 b h w -> b o h w (也不知道为什么非要强行加一个维度)
        # 归一化分数
        scores = prob_scores
        # scale the scores (should sum to 1 for h x w dimenstions)
        epsilon = torch.finfo(torch.float32).tiny
        scores += epsilon
        scores_n = scores / torch.sum(scores, dim=[2, 3], keepdim=True)
        # scores_n = scores

        # the output should be 1x73x(self.output_h)x(self.output_w)
        # scores_redux = einops.reduce(scores_n, "b r (s1 h) (s2 w) -> b r s1 s2", "sum", s1=self.output_h, s2=self.output_w)
        scores_redux = scores_n

        # visualize the result
        if out_dname is not None:
            self.plotSample(
                large_pred[0].cpu(), os.path.join(
                    out_dname, "pred_receptacles", f"{steps_taken}_receptacles.html"),
                img_sz=(1920, 1080), plot_type="recep", zmax=1)


            # plot the extraneous information included in map_learned
            self.plotSample(
                inputs[0, :4, :, :].cpu(), os.path.join(
                    out_dname, "extra_info", f"{steps_taken}.png"),
                plot_type=None, names=["obstacles", "explored", "current location", "past location"], wrap_sz=4)
            

        return scores_redux, self.go_target

    
def get_room_area(obstacle):
    '''
    通过计算凸包的方式获得room_area
    obstacle: 240*240, numpy数组，在cpu上
    '''
    obstacle = obstacle>0.
    # selem = skimage.morphology.disk(1)
    # obstacle = skimage.morphology.binary_dilation(obstacle,selem)
    room_area = skimage.morphology.convex_hull_image(obstacle)
    room_area = skimage.morphology.binary_erosion(room_area,skimage.morphology.disk(7))
    # 只缩小二还是太小了？为什么感觉和没缩小一样？确实，一点都没变化
    return room_area
        
            

class CAP(nn.Module):
    class GuidedAttention(nn.Module):
        def __init__(self, attn_feat_dim=64, attn_mode="cap_mul", guidance_rate=0):
            super().__init__()
            self.attn_mode = attn_mode
            # Guidance Attention
            self.llm_object_attribute = json.load(open("/home/lmy/prompter-alfred/llm_info/llm_attr.json", "r"))
            self.map_all_objects2idx = {obj: i for i, obj in enumerate(constants.map_all_objects)}  # 73
            self.large_objects2idx = {obj: i for i, obj in enumerate(constants.map_save_large_objects)}  # 24
            self.large_idx2objects = {i: obj for obj,i in self.large_objects2idx.items()}  # 24
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.text_embed_dim = 384
            self.map_all_objects2embed = {obj: torch.from_numpy(model.encode(obj)).unsqueeze(0).cuda() for obj in constants.map_all_objects}
            self.large_objects_embed = torch.concatenate([torch.from_numpy(model.encode(self.large_idx2objects[i])).unsqueeze(0) for i in self.large_idx2objects.keys()], dim=0).cuda()  # 24*384
            del model
            
            guidance_rate = 0.5
            guidance_rate = 0
            # guidance_rate = 1
            self.guidance_rate = guidance_rate
            print("Guidance ratio set to ", guidance_rate)
            # self.goal_emb = nn.Linear(len(self.map_all_objects2idx), goal_emb_len)
            self.query_transform = nn.Linear(self.text_embed_dim, attn_feat_dim)
            self.key_transform = nn.Linear(self.text_embed_dim, attn_feat_dim)

        def forward(self, x, target_names, return_attn_scores=False):
            queries = torch.concatenate([self.map_all_objects2embed[obj] for obj in target_names], axis=0)
            transformed_queries = self.query_transform(queries)
            transformed_keys = self.key_transform(self.large_objects_embed)
            # learned_attention: shape [q_num, k_num]
            # learned_attention = F.relu(torch.matmul(transformed_queries, transformed_keys.transpose(-2, -1))/ np.sqrt(self.text_embed_dim))
            learned_attention = F.softmax(torch.matmul(transformed_queries, transformed_keys.transpose(-2, -1))/ np.sqrt(self.text_embed_dim), dim=1)
            guided_attention = self.guided_attention(x, target_names)
            # channel_weight = self.guidance_rate * guided_attention + (1-self.guidance_rate)*learned_attention
            # channel_weight = learned_attention 
            if self.attn_mode=="cap_mul":
                channel_weight = learned_attention * guided_attention 
            elif self.attn_mode=="no_llm":
                channel_weight = learned_attention
            elif self.attn_mode=="cap_avg":
                epsilon = torch.finfo(torch.float32).tiny
                guided_attention = guided_attention / (guided_attention.sum(dim=1, keepdim=True)+epsilon).expand_as(guided_attention)
                channel_weight = learned_attention * (1-self.guidance_rate) + guided_attention * self.guidance_rate 
            output = x * channel_weight.unsqueeze(-1).unsqueeze(-1).expand_as(x)
            if return_attn_scores:
                return channel_weight, output
            else:
                return output 
        def guided_attention(self, x, target_names):
            guidance_vector = torch.zeros((x.shape[0], len(self.large_objects2idx)), device=x.device)
            # for obj, idx in self.map_all_objects2idx.items():
            for i, obj in enumerate(target_names):
                idx = self.map_all_objects2idx[obj]
                related_objs = (set(self.llm_object_attribute[obj]['nearby_objects'])|(set(self.llm_object_attribute[obj]['containers']))) & set(constants.map_save_large_objects)
                for related_obj in related_objs:
                    related_idx = self.large_objects2idx[related_obj]
                    guidance_vector[i, related_idx] = 1
                # if len(related_objs)==0:
                #     print(obj)
            return guidance_vector
        def sample_xy_240(self, semantic_map, target_object, mask):
            sem_copy = semantic_map.detach().cpu().numpy().copy()
            sem_copy[1, :, :] = 1e-5
            sem_flattened = sem_copy[4:].argmax(0)
            all_recepts_idx = np.unique(sem_flattened)
            all_recepts = [self.large_idx2objects[idx] for idx in all_recepts_idx]
            # For large objects
            if target_object in constants.map_save_large_objects:
                if target_object in all_recepts:
                    mask = sem_flattened==self.large_objects2idx[target_object]
                    return searchArgmax(mask)
                else:
                    xy_list = np.argwhere(mask)
                    # Randomly choose one xy
                    xy_240 = xy_list[np.random.choice(xy_list.shape[0], size=1, replace=False)[0]]
                    return xy_240 
            # For small objects
            sem_flattened_masked = sem_flattened * mask
            all_recepts_idx = np.unique(sem_flattened_masked)
            all_recepts = [self.large_idx2objects[idx] for idx in all_recepts_idx]
            related_objs = (set(self.llm_object_attribute[target_object]['nearby_objects'])|(set(self.llm_object_attribute[target_object]['containers']))) & set(constants.map_save_large_objects)
            candidate_recepts = list(set(all_recepts) & related_objs)
            if len(candidate_recepts)>0:
                from random import sample
                recept = sample(candidate_recepts, 1)[0]
                # print(candidate_recepts, recept)
                xy_list = np.argwhere(sem_flattened_masked==self.large_objects2idx[recept])
                # Randomly choose one xy
                xy_240 = xy_list[np.random.choice(xy_list.shape[0], size=1, replace=False)[0]]
                return xy_240
            else:
                return None
        def step_growth(self, current_step, total_steps=1000, step_size=0.1):
            return min(1, step_size * (current_step // (total_steps * step_size)))
        def area_growth(self, explored_area, total_area=(240*240), step_size=0.1):
            return min(1, step_size * (explored_area // (total_area * step_size)))
    # class AutoGuidedAttention(GuidedAttention):
    #     class FeatureExtractor(nn.Module):
    #         def __init__(self, map_output_dim=128):
    #             super().__init__()
    #             # Define CNN layers
    #             self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    #             self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
    #             self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    #             self.conv4 = nn.Conv2d(64, map_output_dim, kernel_size=3, stride=2, padding=1)

    #         def forward(self, x):
    #             x = F.relu(self.conv1(x))
    #             x = F.relu(self.conv2(x))
    #             x = F.relu(self.conv3(x))
    #             x = F.relu(self.conv4(x))
    #             return x
    #     class DynamicFusionLayer(nn.Module):
    #         def __init__(self, feature_extractor, map_output_dim=128):
    #             super().__init__()
    #             # A flattened semantic map together with obstacle, the past locations： 3 channels
    #             self.map_conv = feature_extractor
    #             self.guidance_rate_proj = nn.Linear(map_output_dim, 1)
    #         def forward(self, semantic_map):
    #             # Process semantic map
    #             map_features = self.map_conv(semantic_map)
    #             # Global average pooling
    #             map_features = torch.mean(map_features, dim=[2, 3])  
    #             # Output guidance rate
    #             guidance_rate = torch.sigmoid(self.guidance_rate_proj(map_features))
    #             return guidance_rate

    #     # Make the guidance ratio adjustment an automatic process
    #     def __init__(self, attn_feat_dim=64, attn_mode="cap_avg_auto", map_output_dim=128):
    #         super().__init__(attn_feat_dim=attn_feat_dim, attn_mode=attn_mode)
    #         feature_extractor = self.FeatureExtractor(map_output_dim)
    #         self.dynamic_fusion_layer = self.DynamicFusionLayer(feature_extractor, map_output_dim)
    #     def split4Map(self, data):
    #         return data[:, 0:1, :, :], data[:, 1:2, :, :], data[:, 2:3, :, :], data[:, 3:4, :, :], data[:, 4:, :, :]
    #     def forward(self, x, target_names):
    #         obstacles, explored, curr_loc, past_loc, large_pred = self.split4Map(x)
    #         queries = torch.concatenate([self.map_all_objects2embed[obj] for obj in target_names], axis=0)
    #         transformed_queries = self.query_transform(queries)
    #         transformed_keys = self.key_transform(self.large_objects_embed)
    #         # learned_attention: shape [q_num, k_num]
    #         self.guidance_rate = self.dynamic_fusion_layer(torch.concatenate([obstacles, explored, past_loc], axis=1))
    #         print(self.guidance_rate)
    #         learned_attention = F.softmax(torch.matmul(transformed_queries, transformed_keys.transpose(-2, -1))/ np.sqrt(self.text_embed_dim), dim=1)
    #         guided_attention = self.guided_attention(large_pred, target_names)

    #         # Cap average auto
    #         epsilon = torch.finfo(torch.float32).tiny
    #         guided_attention = guided_attention / (guided_attention.sum(dim=1, keepdim=True)+epsilon).expand_as(guided_attention)
    #         channel_weight = learned_attention * (1-self.guidance_rate) + guided_attention * self.guidance_rate 

    #         results = large_pred * channel_weight.unsqueeze(-1).unsqueeze(-1).expand_as(large_pred)
    #         return results
    class AutoGuidedAttention(GuidedAttention):
        class FeatureExtractor(nn.Module):
            def __init__(self, map_output_dim=8):
                super().__init__()
                # Define CNN layers
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
                self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
                self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
                self.conv5 = nn.Conv2d(16, map_output_dim, kernel_size=3, stride=1, padding=1)
                self.maxpool = nn.MaxPool2d(2)
                self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.maxpool(F.relu(self.conv2(x)))
                x = self.maxpool(F.relu(self.conv3(x)))
                x = self.maxpool(F.relu(self.conv4(x)))
                x = self.avgpool(F.relu(self.conv5(x))).flatten(1)
                return x
        class DynamicFusionLayer(nn.Module):
            def __init__(self, feature_extractor, map_output_dim=8):
                super().__init__()
                # A flattened semantic map together with obstacle, the past locations： 3 channels
                self.map_conv = feature_extractor
                self.guidance_rate_proj = nn.Sequential(
                    nn.Linear(176, 32), # 128+24*2=176
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
            def forward(self, semantic_map, learned_attention_ft, guided_attention_ft):
                # Process semantic map
                map_features = self.map_conv(semantic_map)
                all_features = torch.concatenate([map_features, learned_attention_ft, guided_attention_ft], dim=1) 
                # Output guidance rate
                guidance_rate = torch.sigmoid(self.guidance_rate_proj(all_features))
                return guidance_rate
        def __init__(self, attn_feat_dim=64, attn_mode="cap_avg_auto", map_output_dim=8):
            super().__init__(attn_feat_dim=attn_feat_dim, attn_mode=attn_mode)
            feature_extractor = self.FeatureExtractor(map_output_dim)
            self.dynamic_fusion_layer = self.DynamicFusionLayer(feature_extractor, map_output_dim)
        def split4Map(self, data):
            return data[:, 0:1, :, :], data[:, 1:2, :, :], data[:, 2:3, :, :], data[:, 3:4, :, :], data[:, 4:, :, :]
        def forward(self, x, target_names):
            obstacles, explored, curr_loc, past_loc, large_pred = self.split4Map(x)
            queries = torch.concatenate([self.map_all_objects2embed[obj] for obj in target_names], axis=0)
            transformed_queries = self.query_transform(queries)
            transformed_keys = self.key_transform(self.large_objects_embed)
            # print(self.guidance_rate)
            learned_attention = F.softmax(torch.matmul(transformed_queries, transformed_keys.transpose(-2, -1))/ np.sqrt(self.text_embed_dim), dim=1)
            guided_attention = self.guided_attention(large_pred, target_names)
            self.guidance_rate = self.dynamic_fusion_layer(torch.concatenate([obstacles, explored, past_loc], axis=1), learned_attention, guided_attention)

            # Cap average auto
            epsilon = torch.finfo(torch.float32).tiny
            guided_attention = guided_attention / (guided_attention.sum(dim=1, keepdim=True)+epsilon).expand_as(guided_attention)
            channel_weight = learned_attention * (1-self.guidance_rate) + guided_attention * self.guidance_rate 

            results = large_pred * channel_weight.unsqueeze(-1).unsqueeze(-1).expand_as(large_pred)
            return results
    class CNNBackbone(nn.Module):
        def __init__(self, in_channels=24, out_channels=1):
            super().__init__()
            # Example: Simple CNN Backbone
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)  # Adjust channels as needed
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = self.conv6(x)
            # output n,1,15,15
            return x

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=24, attn_mode="cap_mul", guidance_rate=0):
        super(CAP, self).__init__()
        '''
        Do we need object embeddings? Such embedding might be a form of 'cluster', 
        closer objects might have closer embedding, but now that we have commonsense info, might skip this instead
        '''
        # map_all_objects: 73, map_save_large_objects: 24
        # self.guided_attention = GuidedAttention(sem_channels=num_sem_categories, in_channels=128)  # Adjust as per your backbone's output channels
        if attn_mode!="no_attn":
            if "auto" in attn_mode:
                print("Using AutoGuidedAttention")
                self.guided_attention_layer = self.AutoGuidedAttention(attn_mode=attn_mode)
            else:
                self.guided_attention_layer = self.GuidedAttention(attn_mode=attn_mode, guidance_rate=guidance_rate)  # Adjust as per your backbone's output channels
                self.set_guidance_rate = self.guided_attention_layer.step_growth
        self.backbone = self.CNNBackbone(in_channels=num_sem_categories+1, out_channels=1)
        self.attn_mode = attn_mode
        # self.backbone = self.CNNBackbone(in_channels=num_sem_categories+4, out_channels=1)
        
    def forward(self, inputs, target_names, out_dname=None, steps_taken=None, temperature=1):
        batch_size = inputs.shape[0]
        if self.attn_mode=="no_attn":
            obj_context = inputs[:, 4:]
        elif "auto" in self.attn_mode:
            obj_context = self.guided_attention_layer(inputs, target_names)
        else:
            obj_context = self.guided_attention_layer(inputs[:, 4:], target_names)
        x = self.backbone(torch.concatenate((inputs[:, 0:1], obj_context), dim=1))  # Only obstacle map are helpful in the judgment
       
        return x




