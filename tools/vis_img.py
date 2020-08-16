# Created by silver at 2019/9/16 15:21
# Email: xiwuchencn[at]gmail[dot]com
import _init_path
import os
import numpy as np
import pickle
import torch
from torch.nn.functional import grid_sample

import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.kitti_dataset import KittiDataset
import argparse

from lib.datasets.kitti_rcnn_dataset import interpolate_img_by_xy

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type = str, default = './gt_database')
parser.add_argument('--class_name', type = str, default = 'Car')
parser.add_argument('--split', type = str, default = 'train')
args = parser.parse_args()

import cv2
from lib.config import cfg


class GTDatabaseGenerator(KittiDataset):
    def __init__(self, root_dir, split = 'train', classes = args.class_name):
        super().__init__(root_dir, split = split)
        self.gt_database = None
        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes
        self.velodyne_rgb_dir = os.path.join(root_dir, 'KITTI/object/training/velodyne_rgb')
        # if not os.path.exists(self.velodyne_rgb_dir):
        os.makedirs(self.velodyne_rgb_dir, exist_ok = True)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def vis_img(self):
        gt_database = []
        for idx, sample_id in enumerate(self.image_idx_list):
            sample_id = int(sample_id)
            print('process gt sample (id=%06d)' % sample_id)

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]

            # (H,W,3)
            img = self.get_image_rgb_with_normal(sample_id)
            img = ((img * self.std + self.mean) * 255)[:, :, ::-1]
            img = np.ascontiguousarray(img)
            pts_img, pts_depth = calib.rect_to_img(pts_rect)
            img_shape = self.get_image_shape(sample_id)
            pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_depth, img_shape)
            # (W,H)
            pts_img = pts_img[pts_valid_flag][:, :]
            # (H,W)
            # pts_img=pts_img[:,::-1]
            # print(pts_img)
            # print(pts_img[:,0].max(),pts_img[:,0].min(),pts_img[:,1].max(),pts_img[:,1].min())
            shape = self.get_image_shape_with_padding(sample_id)
            shape = np.array([shape[1], shape[0]])

            cur_pts_rgb = interpolate_img_by_xy(img, pts_img, shape)

            # from lib.net.point_rcnn import bili_interpolate
            # shape=shape.astype(np.float64)
            # img2 = torch.from_numpy(img.transpose((2, 0, 1)).reshape(1, 3, img.shape[0], img.shape[1]))
            # pts_img_2 = torch.from_numpy(pts_img).double().unsqueeze(0)
            # shape_2 = torch.from_numpy(shape).unsqueeze(0)
            # test_out = bili_interpolate(img2, pts_img_2, shape_2)
            # test_out = test_out.squeeze(0).numpy().transpose((1,0))
            # print((test_out - cur_pts_rgb).sum())

            # img=np.clip(img)
            # print(img.shape)
            img = img.astype(np.uint8)
            # cv2.imshow('origin',img)

            #
            # img_mask=(img).copy()
            img_mask = np.zeros_like(img).astype(np.uint8)

            cur_pts_rgb = ((cur_pts_rgb)).astype(np.uint8)
            pts_img = np.around(pts_img).astype(np.int)
            # cur_pts_rgb=np.clip(cur_pts_rgb,0,255)
            for i in range(pts_img.shape[0]):
                print(img[pts_img[i][1], pts_img[i][0], :], cur_pts_rgb[i, :])
                img_mask[pts_img[i][1], pts_img[i][0], :] = cur_pts_rgb[i, :]

            # cv2.imshow('gene',img_mask)
            # cv2.startWindowThread()
            # cv2.waitKey()
            input()

            #
            # cat_lidar = np.concatenate([pts_lidar, cur_pts_rgb], axis = 1)
            #
            #
            # # save
            # lidar_file = os.path.join(self.velodyne_rgb_dir, '%06d.bin' % sample_id)
            # cat_lidar.tofile(lidar_file,sep = '',format = '%f')


if __name__ == '__main__':
    # dataset = GTDatabaseGenerator(root_dir = '../../data/', split = args.split)
    # # os.makedirs(args.save_dir, exist_ok = True)
    #
    # dataset.generate_rgb_database()
    # # args.split=
    dataset = GTDatabaseGenerator(root_dir = '../../data/', split = 'train')
    dataset.vis_img()

    # gt_database = pickle.load(open('gt_database/train_gt_database.pkl', 'rb'))
    # print(gt_database.__len__())
    # import pdb
    # pdb.set_trace()
