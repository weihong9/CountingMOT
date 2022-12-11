from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

from lib.models.losses import FocalLoss, TripletLoss, SSIM_Loss
from lib.models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.models.decode import mot_decode, _nms, _topk
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.utils.post_process import ctdet_post_process
from lib.utils.generate_anchors import generate_anchors
from .base_trainer import BaseTrainer

def fspecial_gaussian(size, sigma):
    '''
    Function to mimic the 'fspecial' gaussian MATLAB function
    '''
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d


class GaussianBlurConv(nn.Module):
    def __init__(self, truncate=4, sigma=1.5):
        super(GaussianBlurConv, self).__init__()
        size = 2 * int(truncate * sigma + 0.5) + 1
        self.kernel = fspecial_gaussian(size, sigma)
        self.size = size

    def forward(self, x):
        channels = x.shape[1]
        pad = (self.size - 1) // 2
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.kernel = np.repeat(self.kernel, channels, axis=0)
        # self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        self.kernel = self.kernel.to(x.device)
        x = F.conv2d(x, self.kernel, padding=pad, groups=channels)
        return x

def obj_cnt(heat, kernel=3, stride=1):
    pad = (kernel - 1) // 2

    havg = nn.functional.avg_pool2d(
        heat, (kernel, kernel), stride=stride, padding=pad)
    havg = havg*(kernel*kernel)
    return havg

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats

class CMotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CMotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.relu = nn.ReLU()
        self.mse_density = nn.MSELoss()
        self.ssim = SSIM_Loss(opt.num_classes)
        self.mse_dc1 = nn.MSELoss()
        self.mse_dc2 = nn.MSELoss()
        self.mse_dc_sum = nn.MSELoss()
        self.mse_dd_sum = nn.MSELoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

        # -2, -1, -1
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.s_cnt = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, cnt_loss = 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)
            if opt.cnt_weight > 0:
                # loss for density map estimation
                output['cnt'] = self.relu(output['cnt'])
                output['cnt'] = self.relu(output['cnt'])
                density_loss = self.mse_density(output['cnt'], opt.cnt_weight*batch['cm'])
                ssim_loss = self.ssim(output['cnt'], opt.cnt_weight*batch['cm'])

                heat = output['hm']
                heat_det = _nms(heat)
                heat_det[heat_det <= self.opt.conf_thres] = 0
                heat_det[heat_det > self.opt.conf_thres] = 1
                # blurconv = GaussianBlurConv(4, 1.5)
                # blurconv = GaussianBlurConv(4, 2)
                blurconv = GaussianBlurConv(4.5, 2)

                density_det = blurconv(heat_det)

                output['cnt'] = output['cnt']/opt.cnt_weight
                # cnt_heat = obj_cnt(output['cnt'], kernel=17)
                # cnt_det = obj_cnt(density_det, kernel=17)
                cnt_heat = obj_cnt(output['cnt'], kernel=19)
                cnt_det = obj_cnt(density_det, kernel=19)


                b, c, _, _ = batch['cm'].shape
                cnt_cm_sum = torch.sum(batch['cm'].view(b, c, -1), dim=2)
                cnt_det_sum = torch.sum(heat_det.view(b, c, -1), dim=2)
                cnt_density_sum = torch.sum(output['cnt'].view(b, c, -1), dim=2)

                dc_loss = self.mse_dc2(cnt_det, cnt_heat.detach()) + 0.01*self.mse_dd_sum(cnt_det_sum, cnt_cm_sum)
                cd_loss = self.mse_dc1(opt.cnt_weight*output['cnt'], opt.cnt_weight*density_det.detach()) + \
                          + 0.01 * self.mse_dc_sum(cnt_density_sum, cnt_cm_sum)

                # print(torch.sum(density_det))

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + dc_loss
        cnt_loss = ssim_loss + density_loss + cd_loss
        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + \
               torch.exp(-self.s_cnt) * cnt_loss + (self.s_det + self.s_id + self.s_cnt)

        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss,
                      'det_loss': det_loss,
                      'cnt_loss': cnt_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

class CMotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CMotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'cnt_loss', 'det_loss']
        loss = CMotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]