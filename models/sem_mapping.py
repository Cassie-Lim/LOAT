import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du

import cv2
import time

class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm//args.global_downscaling
        self.n_channels = 3  # TODO: add argument
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.print_time = args.print_time
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.no_straight_obs = args.no_straight_obs
        self.view_angles = [0.0]*args.num_processes

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height*100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi/2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(args.num_processes, 1 + self.num_sem_categories, vr, vr,
                                     self.max_height - self.min_height).float().to(self.device)
        self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories,
                               self.screen_h//self.du_scale * self.screen_w//self.du_scale
                               ).float().to(self.device)
        

    def set_view_angles(self, view_angles):
        self.view_angles = [-view_angle for view_angle in view_angles]

    def plot3D(self, ddd_pts, fname):
        # for debugging
        import pandas as pd
        import plotly.express as px
        aa = pd.DataFrame({"x": ddd_pts[:, :, 0].ravel(), "y": ddd_pts[:, :, 1].ravel(), "z": ddd_pts[:, :, 2].ravel()})
        fig = px.scatter_3d(aa, x="x", y="y", z="z")
        fig = fig.update_traces(marker_size=1)
        fig.write_html(fname)

    def removeMaskNearGround(self, mask, height, reliable, height_thresh=10):
        device = mask.device
        mask = mask.cpu().numpy()

        roi_mask = np.logical_and(reliable, mask)

        did_remove = False
        threshed_mask = np.zeros_like(mask)
        nlabels, labels = cv2.connectedComponents(roi_mask.astype(np.uint8))
        for lbl in range(1, nlabels):
            target_mask = labels == lbl
            roi_height = height[target_mask]
            top_height = np.percentile(roi_height, 90)
            keep = top_height >= height_thresh
            if keep:
                threshed_mask[target_mask] = 1
            did_remove |= not keep

        return torch.from_numpy(threshed_mask).to(device), did_remove

    def forward(self, obs, pose_obs, maps_last, poses_last, build_maps=True, no_update=False):
        # obs's channels are as follows:
        # 1-3: RGB
        # 4: Depth 
        # 5-end: categories
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        # depth -> point cloud (3D coordinates)
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        # Multiprocessing
        agent_view_t = du.transform_camera_view_t_multiple(
            point_cloud_t, self.agent_height, self.view_angles, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)
            # current_pose            : camera position (x, y, theta (radians))

        debug = False
        # # ******************
        # debug = True
        # # *********************
        if debug:
            self.plot3D(point_cloud_t[0].detach().cpu().numpy(), "point_cloud.html")
            self.plot3D(agent_view_t[0].detach().cpu().numpy(), "agent_view.html")
            self.plot3D(agent_view_centered_t[0].detach().cpu().numpy(), "agent_view_centered.html")
            cv2.imwrite("bw.png", obs[0, 0, :, :].detach().cpu().numpy())
            cv2.imwrite("depth.png", obs[0, 3, :, :].detach().cpu().numpy())
            # ****************
            import pickle
            with open("debug_data/point_cloud"+"/"+"point_cloud.pkl",'wb') as f:   
                pickle.dump(point_cloud_t[0].detach().cpu().numpy(),f)
            # ****************

        # remove segmentation results that are located near the ground
        height_info = agent_view_t[:, :, :, 2].cpu().numpy()
        reliable = (depth > 0).cpu().numpy()
        height_thresh = 10
        for mask_map, height_map, reliable_map in zip(obs, height_info, reliable):
            for cat_id in range(4, mask_map.shape[0]):
                mask_map[cat_id, :, :], did_remove = self.removeMaskNearGround(
                    mask_map[cat_id, :, :], height_map, reliable_map, height_thresh)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range//2.)/vision_range*2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h+min_h)//2.)/(max_h-min_h)*2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]
                                                          ).view(bs, c-4, h//self.du_scale * w//self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid*0., self.feat, XYZ_cm_std).transpose(2, 3)

        # min_z = int(5/z_resolution - min_h)
        min_z = int(10/z_resolution - min_h)
        max_z = int((self.agent_height + 1 + 50)/z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred/self.map_pred_threshold
        fp_exp_pred = fp_exp_pred/self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)


        if self.no_straight_obs:
            for vi, va in enumerate(self.view_angles):
                if abs(va - 0) <= 5:
                    fp_map_pred[vi, :, :, :] = 0.0

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm//self.resolution,
                                 self.map_size_cm//self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm//(self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :]/self.cat_pred_threshold,
            min=0.0, max=1.0)

        if self.cat_pred_threshold > 5.0:
            agent_view[:, 4:, y1:y2, x1:x2][np.where(
                agent_view[:, 4:, y1:y2, x1:x2].cpu().detach().numpy() < 0.5)] = 0.0

        if no_update:
            agent_view = torch.zeros(agent_view.shape).to(device=self.device)
        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2]/57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2]/57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2]/57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2]/57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2]*57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2]-180.0, 360.0)+180.0
            pose[:, 2] = torch.fmod(pose[:, 2]+180.0, 360.0)-180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0/self.resolution
                            - self.map_size_cm//(self.resolution*2)) /\
            (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True) # b c h w

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)  # b 2 c h w

        map_pred, _ = torch.max(maps2, 1)
        last_explore = translated[:,1,:,:] # b h w
        return fp_map_pred, map_pred, pose_pred, current_poses, last_explore
