"""
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio

from ..nnutils import smal_mesh_net as mesh_net
from ..nnutils import geom_utils
from ..nnutils.nmr import NeuralRenderer
from ..utils import smal_vis
import pickle as pkl

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')

class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)

        # Load the texture map layers
        tex_masks = [None]*opts.number_of_textures
        self.vert2kp = torch.Tensor(pkl.load(open('smalst/zebra_data/verts2kp.pkl'))).cuda(device=opts.gpu_id)

        print('Setting up model..')
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat, tex_masks=tex_masks)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        # set the module in evaluation mode
        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)

        self.renderer = NeuralRenderer(opts.img_size, opts.projection_type, opts.norm_f, opts.norm_z, opts.norm_f0)

        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size, opts.projection_type, opts.norm_f, opts.norm_z, opts.norm_f0)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        self.mean_shape = self.model.get_mean_shape()

        # For visualization
        faces = self.model.faces.view(1, -1, 3)

        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.vis_rend = smal_vis.VisRenderer(opts.img_size,
                                             faces.data.cpu().numpy(), opts.projection_type, opts.norm_f, opts.norm_z, opts.norm_f0)
        self.vis_rend.set_bgcolor([1., 1., 1.])

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        network.load_state_dict(torch.load(save_path))

        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)

        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)

    def predict(self, batch, cam_gt=None, trans_gt=None, pose_gt=None, betas_gt=None, rot=0):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward(cam_gt, trans_gt, pose_gt, betas_gt, rot)
        return self.collect_outputs()

    def forward(self, cam_gt=None, trans_gt=None, pose_gt=None, betas_gt=None, rot=0):
        if self.opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes = self.model.forward(self.input_imgs)

        self.delta_v, scale, self.trans, self.pose, self.betas, self.kp_2D_pred = pred_codes

        # Rotate the view
        if rot != 0:
            import cv2
            r0 = self.pose[:,:3].detach().cpu().numpy()
            R0, _ = cv2.Rodrigues(r0)
            ry = np.array([0, rot, 0])
            Ry, _ = cv2.Rodrigues(ry)
            Rt = np.matrix(Ry)*np.matrix(R0)
            rt, _ = cv2.Rodrigues(Rt)
            self.pose[:,:3] = torch.Tensor(rt).permute(1,0)

        if cam_gt is not None:
            print('Setting gt cam')
            scale[:] = cam_gt
        if trans_gt is not None:
            print('Setting gt trans')
            self.trans[0,:] = torch.Tensor(trans_gt)
        if pose_gt is not None:
            print('Setting gt pose')
            self.pose[0,:] = torch.Tensor(pose_gt)
        if betas_gt is not None:
            print('Setting gt betas')
            self.betas[0,:] = torch.Tensor(betas_gt[:10])
            print('Removing delta_v')
            self.delta_v[:] = 0

        if True:
            if self.opts.projection_type == 'perspective':
                # The camera center does not change;
                cam_center = torch.Tensor([self.input_imgs.shape[2]//2, self.input_imgs.shape[3]//2]).cuda(device=self.opts.gpu_id)
                if scale.shape[0] == 1:
                    self.cam_pred = torch.cat([scale, cam_center[None,:]], 1)
                else:
                    self.cam_pred = torch.cat([scale.permute(1,0), cam_center.repeat(scale.shape[0],1).permute(1,0)]).permute(1,0)
            else:
                import pdb; pdb.set_trace()


        del_v = self.delta_v
        # Deform mean shape:
        if self.opts.ignore_pred_delta_v:
            del_v[:] = 0

        if self.opts.use_smal_pose:
            self.smal_verts = self.model.get_smal_verts(self.pose, self.betas, self.trans, del_v)
            self.pred_v = self.smal_verts
        else:
            # TODO
            import pdb; pdb.set_trace()

        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts,
                                                    self.cam_pred)
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces,
                                               self.cam_pred)

        # Render texture.
        if self.opts.texture: 
            if self.textures.size(-1) == 2:
                # Flow texture!
                self.texture_flow = self.textures
                self.textures = geom_utils.sample_textures(self.textures,
                                                           self.imgs)
            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1,
                                                                  tex_size, 1)

            # Render texture:
            self.texture_pred = self.tex_renderer.forward(
                self.pred_v, self.faces, self.cam_pred, textures=self.textures)

            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            self.uv_flows = uv_flows.permute(0, 2, 3, 1)

            self.uv_images = torch.nn.functional.grid_sample(self.imgs, self.uv_flows)
        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            'pose_pred': self.pose.data,
            'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
            'delta_v_pred': self.delta_v.data,
            'trans_pred':self.trans.data,
            'kp_2D_pred':self.kp_2D_pred,
            'shape_f': self.model.code_predictor.shape_predictor.shape_f.data,
            'f':self.faces,
            'v':self.smal_verts
        }
        if self.opts.use_smal_betas: 
            outputs['betas_pred'] = self.betas.data
        if self.opts.texture: 
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.texture_pred.data
            outputs['uv_image'] = self.uv_images.data
            outputs['uv_flow'] = self.uv_flows.data

        return outputs
