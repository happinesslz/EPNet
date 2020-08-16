import torch
import numpy as np
from lib.config import cfg
import torch.nn.functional as F

def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
    return pc


def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                       get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    anchor_size = anchor_size.to(roi_box3d.get_device())
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    assert cfg.TRAIN.BBOX_AVG_BY_BIN == cfg.TEST.BBOX_AVG_BY_BIN

    if not cfg.TRAIN.BBOX_AVG_BY_BIN:
        # deafult: cfg.bbox_avg_by_bin = False
        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
        z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)

        pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
        pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope

        if get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
            z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
            x_res = x_res_norm * loc_bin_size
            z_res = z_res_norm * loc_bin_size

            pos_x += x_res
            pos_z += z_res
    else:
        # print('BBOX_AVG_BY_BIN: True')

        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        pred_x_bin = F.softmax(pred_reg[:, x_bin_l: x_bin_r], 1) # N x num_bin
        pred_z_bin = F.softmax(pred_reg[:, z_bin_l: z_bin_r], 1)

        # print(pred_x_bin[:10, :])
        # input()

        xz_bin_ind = torch.arange(per_loc_bin_num).float()
        xz_bin_center = xz_bin_ind * loc_bin_size + loc_bin_size / 2 - loc_scope # num_bin
        xz_bin_center = xz_bin_center.to(pred_x_bin.device)

        pred_x_abs = xz_bin_center
        pred_z_abs = xz_bin_center

        assert get_xz_fine, 'now only support bin format!'
        if get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            pred_x_reg = pred_reg[:, x_res_l: x_res_r] * loc_bin_size # N x num_bin
            pred_z_reg = pred_reg[:, z_res_l: z_res_r] * loc_bin_size

            pred_x_abs = pred_x_abs + pred_x_reg
            pred_z_abs = pred_z_abs + pred_z_reg

        pos_x = (pred_x_abs * pred_x_bin).sum(dim=1)
        pos_z = (pred_z_abs * pred_z_bin).sum(dim=1)


    # recover y localization
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_bin = torch.argmax(pred_reg[:, y_bin_l: y_bin_r], dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    assert cfg.TRAIN.RY_WITH_BIN == cfg.TEST.RY_WITH_BIN
    if not cfg.TEST.RY_WITH_BIN:
        ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
        ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
        else:
            angle_per_class = (2 * np.pi) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi
    else:
        # print("RY with BIN")
        ry_bin = F.softmax(pred_reg[:, ry_bin_l: ry_bin_r], 1)
        # print(ry_bin[:10, :])
        # input()
        ry_res_norm = pred_reg[:, ry_res_l: ry_res_r]
        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            # ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
            ry_bin_ind = torch.arange(num_head_bin).float().to(ry_res_norm.device)
            ry = (ry_bin_ind * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
            # [way1]
            # ry = (ry * ry_bin).sum(dim=1)

            # [way2]
            ry_bin_r = ry_bin.clone()
            ry_bin_r[ry<0] = 0 # [0, pi/4]
            p_rside = ry_bin_r.sum(dim=1, keepdim=True) + 1e-7 # B
            ry_bin_r =ry_bin_r/p_rside

            ry_bin_l = ry_bin.clone()
            ry_bin_l[ry>=0] = 0 #[-pi/4, 0]
            p_lside = ry_bin_l.sum(dim=1, keepdim=True) + 1e-7
            ry_bin_l =ry_bin_l/p_lside

            # assert 1 - (p_rside + p_lside) < p_lside.new().resize_(p_lside.size()).fill_(1e-4)
            ry_r = ry.clone()
            ry_r[ry_r<0] = 0
            ry_r = (ry_r * ry_bin_r).sum(dim=1)

            ry_l = ry.clone()
            ry_l[ry_l>=0] = 0
            ry_l = (ry_l * ry_bin_l).sum(dim=1)

            # flags
            use_r = p_rside.squeeze() >= p_lside.squeeze()
            use_l = p_rside.squeeze() < p_lside.squeeze()
            ry = ry_r * use_r.float() + ry_l * use_l.float()

        else:
            angle_per_class = (2 * np.pi) / num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            # ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry_bin_ind = torch.arange(num_head_bin).float().to(ry_res_norm.device)
            ry = (ry_bin_ind * angle_per_class + ry_res) % (2*np.pi)

            # [way1] to [0, pi]
            # ry[ry > np.pi] -= np.pi
            # ry = (ry * ry_bin).sum(dim=1)
            # ry[ry > np.pi] -= 2 * np.pi

            # [way2] ry [0, 2pi]
            ry_bin_r = ry_bin.clone()
            ry_bin_r[ry > np.pi] = 0 # [0, pi]
            p_rside = ry_bin_r.sum(dim=1, keepdim=True) + 1e-7 # B
            ry_bin_r =ry_bin_r/p_rside

            ry_bin_l = ry_bin.clone()
            ry_bin_l[ry <= np.pi] = 0 # (pi, 2*pi]
            p_lside = ry_bin_l.sum(dim=1, keepdim=True) + 1e-7
            ry_bin_l =ry_bin_l/p_lside

            ry_r = ry.clone()
            ry_r[ry_r > np.pi] = 0
            ry_r = (ry_r * ry_bin_r).sum(dim=1) # [0, pi]
            # print('ry_r', ry_r.size())

            ry_l = ry.clone()
            ry_l[ry_l <= np.pi] = 0
            ry_l = (ry_l * ry_bin_l).sum(dim=1) # (pi, 2*pi]
            # print('ry_l', ry_l.size())

            # flags
            use_r = p_rside.squeeze() >= p_lside.squeeze()
            use_l = p_rside.squeeze() < p_lside.squeeze()
            # print('use_r', use_r.size())
            ry = ry_r * use_r.float() + ry_l * use_l.float()

            # p_rside = ry_bin[ry <= np.pi].sum()
            # p_lside = ry_bin[ry > np.pi].sum()
            # assert 1 - (p_rside + p_lside).sum().data < 1e-4
            # if p_rside > p_lside:
            #     ws_r = ry_bin[ry <= np.pi]/ry_bin[ry <= np.pi].sum(dim=1, keepdim=True)
            #     ry_r = ry[ry<=np.pi]
            #     ry = (ry_r * ws_r).sum(dim=1) # [0, np.pi]
            # else:
            #     ws_l = ry_bin[ry>np.pi]/ry_bin[ry>np.pi].sum(dim=1, keepdim=True)
            #     ry_l = ry[ry>np.pi]
            #     ry = (ry_l * ws_l).sum(dim=1) # [np.pi, 2*np.pi]
            ry[ry>np.pi] -= 2*np.pi

            # print(ry.size())


    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l: size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, - roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]

    return ret_box3d