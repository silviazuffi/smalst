"""
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from . import geom_utils
import numpy as np
from ..smal_model.batch_lbs import batch_rodrigues

def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid)

    if vis_rend is not None:
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

        from ..utils import smal_vis
        for i in range(dist_transf.size(0)):
            rend_dt = vis_rend(verts[i], cams[i], dts[i])
            rend_img = smal_vis.tensor2im(tex_pred[i].data)            
            import matplotlib.pyplot as plt
            plt.ion()
            fig=plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(121)
            ax.imshow(rend_dt)
            ax = fig.add_subplot(122)
            ax.imshow(rend_img)
            import pdb; pdb.set_trace()

    return dist_transf.mean()


def texture_loss(img_pred, img_gt, mask_pred, mask_gt):
    """
    Input:
      img_pred, img_gt: B x 3 x H x W
      mask_pred, mask_gt: B x H x W
    """
    mask_pred = mask_pred.unsqueeze(1)
    mask_gt = mask_gt.unsqueeze(1)

    masked_rend = (img_pred * mask_pred)[0].data.cpu().numpy()
    masked_gt = (img_gt * mask_gt)[0].data.cpu().numpy()

    return torch.nn.L1Loss()(img_pred * mask_pred, img_gt * mask_gt)

def uv_flow_loss(uv_flow_pred, uv_flow_gt_w_mask):
    """
    Input:
      uv_flow_pred: B x 2 x H x W
      uv_flow_gt_w_mask: B x 3 x H x W
    """
    # We only have info for the uv_flow for the points where mask is one
    mask = uv_flow_gt_w_mask[:,2,:,:].unsqueeze(1)
    uv_flow_gt = uv_flow_gt_w_mask[:,:2,:,:]
    return torch.nn.L1Loss()(uv_flow_pred*mask, uv_flow_gt*mask)

def mask_loss(mask_pred, mask_gt):
    """
    Input:
      mask_pred: B x 3 x H x W
      mask_gt: B x H x W
    """

    return torch.nn.L1Loss()(mask_pred, mask_gt)

def delta_v_loss(delta_v, delta_v_gt):
    criterion = torch.nn.MSELoss()
    return criterion(delta_v, delta_v_gt)

def texture_map_loss(texture_map_pred, texture_map_gt, texture_map_mask, opts=None):
    """
    Input:
      texture_map_pred: B x 3 x tH x tW
      texture_gt: B x 3 x tH x tW
      texture_map_mask: tH x tW
    """
    mask = texture_map_mask[None,:,:,0]
    texture_map_pred = texture_map_pred*mask
    texture_map_gt = texture_map_gt*mask
    if opts.white_balance_for_texture_map:
        # do gray world normalization
        N = torch.sum(mask)
        B = texture_map_pred.shape[0]
        # gray values
        g_pred = torch.sum(texture_map_pred.view(B,3,-1),dim=2)/N
        g_gt = torch.sum(texture_map_gt.view(B,3,-1),dim=2)/N

        texture_map_pred = texture_map_pred / (g_pred.unsqueeze_(-1).unsqueeze_(-1))
        texture_map_gt = texture_map_gt / (g_gt.unsqueeze_(-1).unsqueeze_(-1))
    
    return torch.nn.L1Loss()(texture_map_pred, texture_map_gt)
    
def camera_loss(cam_pred, cam_gt, margin, normalized):
    """
    cam_* are B x 7, [sc, tx, ty, quat]
    Losses are in similar magnitude so one margin is ok.
    """

    # Only the first element as the rest is fixed
    if normalized:
        criterion = torch.nn.MSELoss()
        return criterion(cam_pred[:, 0], cam_gt[:, 0])
    else:
        st_loss = ((cam_pred[:, 0] - cam_gt[:, 0])/1e3)**2
        return st_loss.mean()

def model_trans_loss(trans_pred, trans_gt):
    """
    trans_pred: B x 3
    trans_gt: B x 3
    """
    criterion = torch.nn.MSELoss()
    return criterion(trans_pred, trans_gt)

def model_pose_loss(pose_pred, pose_gt, opts):
    """
    pose_pred: B x 115
    pose_gt: B x 115
    """
    if opts.use_pose_geodesic_loss:
        # Convert each angle in 
        R = torch.reshape( batch_rodrigues(torch.reshape(pose_pred, [-1, 3]), opts=opts), [-1, 35, 3, 3])
        # Loss is acos((tr(R'R)-1)/2)
        Rgt = torch.reshape( batch_rodrigues(torch.reshape(pose_gt, [-1, 3]), opts=opts), [-1, 35, 3, 3])
        RT = R.permute(0,1,3,2)
        A = torch.matmul(RT.view(-1,3,3),Rgt.view(-1,3,3))
        # torch.trace works only for 2D tensors

        n = A.shape[0]
        po_loss =  0    
        eps = 1e-7
        for i in range(A.shape[0]):
            T = (torch.trace(A[i,:,:])-1)/2.
            po_loss += torch.acos(torch.clamp(T, -1 + eps, 1-eps))
        po_loss = po_loss/(n*35)
        return po_loss
    else:
        criterion = torch.nn.MSELoss()
        return criterion(pose_pred, pose_gt)
    


def betas_loss(betas_pred, betas_gt=None, prec=None):
    """
    betas_pred: B x 10
    """
    if betas_gt is None:
        if prec is None:
            b_loss = betas_pred**2
        else:
            b_loss = betas_pred*prec
            return b_loss.mean()
    else:
        criterion = torch.nn.MSELoss()
        return criterion(betas_pred, betas_gt)


def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)
    return torch.max(loss - margin, zeros)

def kp_l2_loss(kp_pred, kp_gt):
    """
    L2 loss between visible keypoints.

    \Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()

    # This always has to be (output, target), not (target, output)
    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])

def keypoints_2D_loss(kp_pred, kp_gt):
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()

    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])

def MSE_texture_loss(img_pred, img_gt, mask_pred, mask_gt, background_imgs=None):
    mask_pred = mask_pred.unsqueeze(1)
    if mask_gt is None:
        M = torch.abs(mask_pred - 1.)
        img_pred = img_pred*mask_pred + background_imgs*M
        
        dist = torch.nn.MSELoss()(img_pred*mask_pred, img_gt*mask_pred)

        return dist 

class PerceptualTextureLoss(object):
    def __init__(self):
        from ..nnutils.perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt, mask_pred, mask_gt, background_imgs=None):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_pred = mask_pred.unsqueeze(1)
        
        # Add a background to img_pred. This is used for the optimization without the groundtruth mask, but could
        # be also used for the regular training
        if mask_gt is None:
            img_pred = img_pred*mask_pred + background_imgs*(torch.abs(mask_pred - 1.))

            dist = self.perceptual_loss(img_pred, img_gt)
            return dist.mean()
        
        mask_gt = mask_gt.unsqueeze(1)
        # Only use mask_gt..
        dist = self.perceptual_loss(img_pred * mask_gt, img_gt * mask_gt)
        return dist.mean()
