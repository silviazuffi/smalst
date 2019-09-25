"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import pickle as pkl
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from ..smal_model.smal_basics import load_smal_model
from ..smal_model.smal_torch import SMAL

from ..utils import mesh
from ..utils import geometry as geom_utils
from . import net_blocks as nb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('is_optimization', False, 'set to True to do refinement')
flags.DEFINE_boolean('is_refinement', False, 'set to True to do refinement')

flags.DEFINE_string('model_dir', 'smalst/smpl_models/', 'location of the SMAL model')
flags.DEFINE_string('model_name', 'my_smpl_00781_4_all.pkl', 'name of the model')
flags.DEFINE_boolean('symmetric', False, 'Use symmetric mesh or not')
flags.DEFINE_boolean('symmetric_texture', False, 'if true texture is symmetric!')
flags.DEFINE_integer('nz_feat', 1024, 'Encoded feature size')

flags.DEFINE_boolean('use_norm_f_and_z', True, 'if to use normalized f and z')
flags.DEFINE_float('camera_ref', '2700.', 'expected focal length value')
flags.DEFINE_float('trans_ref', '19.', 'expected model distance from camera (13 for linear)')
flags.DEFINE_float('norm_f', 2700., 'term in f=norm_f0+norm_f*x_f')  
flags.DEFINE_float('norm_f0', 2700., 'term in f=norm_f0+norm_f*x_f')  
flags.DEFINE_float('norm_z', 20., 'normalization term for depth')
flags.DEFINE_boolean('use_sym_idx', True, 'If to predict only half delta_v')

flags.DEFINE_bool('use_double_input', False, 'input the img and the fg')

flags.DEFINE_boolean('use_camera', True, 'if optimize camera focal length')
flags.DEFINE_boolean('use_delta_v', True, 'if predict vertex displacements')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('texture_map', True, 'if true uses texture map loss!')
flags.DEFINE_boolean('use_directional_light', True, 'if using directional light rather than ambient')

flags.DEFINE_integer('num_betas', 20, 'Number of betas variables')
flags.DEFINE_boolean('use_smal_pose', True, 'if using articulated shape')
flags.DEFINE_boolean('use_smal_betas', False, 'if using smal shape space')

flags.DEFINE_integer('scale_bias', 1, '1 or 0 for bias in nn.Linear') # Does not work for 0
flags.DEFINE_boolean('fix_trans', False, 'do not optimize trans')

flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face') 
flags.DEFINE_integer('texture_img_size', 256, 'Texture resolution per face')
flags.DEFINE_integer('number_of_textures', 4, 'Number of texture layers that compose the texture map')

flags.DEFINE_float('occlusion_map_scale', 1./16., 'division of the image')

flags.DEFINE_integer('bottleneck_size', 2048, 'Define bottleneck size')
flags.DEFINE_integer('channels_per_group', 16, 'number of channels per group in group normalization')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', True, 'If true, only the meanshape is symmetric')

flags.DEFINE_boolean('use_resnet50', True, 'otherwise use resnet18')

flags.DEFINE_string('uv_data_file', 'my_smpl_00781_4_all_template_w_tex_uv_001.pkl', 'ft and vt data of the obj file')
flags.DEFINE_string('projection_type', 'perspective', 'camera projection type (orth or perspective')

flags.DEFINE_integer('n_shape_feat', 40, 'number of shape features when we do not use the betas')

flags.DEFINE_float('depth_var', 2.0, 'see TransPred')
flags.DEFINE_float('x_var', 2.0, 'see TransPred')
flags.DEFINE_float('y_var', 1.0, 'see TransPred')
flags.DEFINE_float('pose_var', 1.0, 'see PosePred')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, opts=None):
        super(ResNetConv, self).__init__()
        if opts.use_resnet50:
            self.resnet = torchvision.models.resnet50(pretrained=True)
        else:
            self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks
        self.opts = opts
        if self.opts.use_double_input:
            self.fc = nb.fc_stack(512*16*8, 512*8*8, 2)

    def forward(self, x, y=None):
        if self.opts.use_double_input and y is not None:
            x = torch.cat([x, y], 2)
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        if self.opts.use_double_input and y is not None:
            x = x.view(x.size(0), -1)
            x = self.fc.forward(x)
            x = x.view(x.size(0), 512, 8, 8)
            
        return x

class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, opts, input_shape, n_blocks=4, nz_feat=100, bott_size=256):
        super(Encoder, self).__init__()
        self.opts = opts
        self.resnet_conv = ResNetConv(n_blocks=4, opts=opts)
        num_norm_groups = bott_size//opts.channels_per_group
        if opts.use_resnet50:
            self.enc_conv1 = nb.conv2d('group', 2048, bott_size, stride=2, kernel_size=4, num_groups=num_norm_groups)
        else:
            self.enc_conv1 = nb.conv2d('group', 512, bott_size, stride=2, kernel_size=4, num_groups=num_norm_groups)

        nc_input = bott_size * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2, 'batch')
        self.nenc_feat = nc_input

        nb.net_init(self.enc_conv1)

    def forward(self, img, fg_img):
        resnet_feat = self.resnet_conv.forward(img, fg_img)
        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)
        return feat, out_enc_conv1


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624, tex_masks=None,vt=None, ft=None):
        super(TexturePredictorUV, self).__init__()
        self.opts = opts
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        self.tex_masks = tex_masks

        # Convert texture masks into the nmr format
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        if opts.number_of_textures > 0:
            self.enc = nn.ModuleList([nb.fc_stack(nz_feat, self.nc_init*self.feat_H[i]*self.feat_W[i], 2, 'batch') for i in range(opts.number_of_textures)])
        else:
            self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2, 'batch')
        
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        if opts.number_of_textures > 0:
            num_groups = nc_init//opts.channels_per_group
            self.decoder = nn.ModuleList([nb.decoder2d(n_upconv, None, nc_init, 
                norm_type='group', num_groups=num_groups, init_fc=False, nc_final=nc_final,
                use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode) for _ in range(opts.number_of_textures)])
            self.uvimage_pred_layer = [None]*opts.number_of_textures
        else:
            num_groups = nc_init//opts.channels_per_group
            self.decoder = nb.decoder2d(n_upconv, None, nc_init, norm_type='group',
                num_groups=num_groups, init_fc=False, nc_final=nc_final,
                use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        if self.opts.number_of_textures > 0:
            tex_pred_layer = [None]*self.opts.number_of_textures
            uvimage_pred_layer = [None]*self.opts.number_of_textures
            for i in range(self.opts.number_of_textures):
                uvimage_pred_layer[i] = self.enc[i].forward(feat)
                uvimage_pred_layer[i] = uvimage_pred_layer[i].view(uvimage_pred_layer[i].size(0), self.nc_init, self.feat_H[i], self.feat_W[i])
                # B x 2 or 3 x H x W
                self.uvimage_pred_layer[i] = self.decoder[i].forward(uvimage_pred_layer[i])
                self.uvimage_pred_layer[i] = torch.nn.functional.tanh(self.uvimage_pred_layer[i])

            # Compose the predicted texture maps
            # Composition by tiling
            if self.opts.number_of_textures == 7:
                upper = torch.cat((uvimage_pred_layer[0], uvimage_pred_layer[1], uvimage_pred_layer[2]), 3)
                lower = torch.cat((uvimage_pred_layer[4], uvimage_pred_layer[5]), 3)
                right = torch.cat((uvimage_pred_layer[3], uvimage_pred_layer[6]), 2)
                uvimage_pred = torch.cat((torch.cat((upper, lower), 2), right), 3)

                upper = torch.cat((self.uvimage_pred_layer[0], self.uvimage_pred_layer[1], self.uvimage_pred_layer[2]), 3)
                lower = torch.cat((self.uvimage_pred_layer[4], self.uvimage_pred_layer[5]), 3)
                right = torch.cat((self.uvimage_pred_layer[3], self.uvimage_pred_layer[6]), 2)
                self.uvimage_pred = torch.cat((torch.cat((upper, lower), 2), right), 3)
            elif self.opts.number_of_textures == 4:
                uvimage_pred = torch.cat((torch.cat((uvimage_pred_layer[0],
                    torch.cat((uvimage_pred_layer[1], uvimage_pred_layer[2]), 3)), 2), uvimage_pred_layer[3]), 3)
                self.uvimage_pred = torch.cat((torch.cat((self.uvimage_pred_layer[0],
                    torch.cat((self.uvimage_pred_layer[1], self.uvimage_pred_layer[2]), 3)), 2), self.uvimage_pred_layer[3]), 3)
        else:
            uvimage_pred = self.enc.forward(feat)
            uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
            # B x 2 or 3 x H x W
            self.uvimage_pred = self.decoder.forward(uvimage_pred)
            self.uvimage_pred = torch.nn.functional.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler)
        tex_pred = tex_pred.view(self.uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts, opts, left_idx, right_idx, shapedirs):
        super(ShapePredictor, self).__init__()
        self.opts = opts
        if opts.use_delta_v:
            if opts.use_sym_idx:
                self.left_idx = left_idx
                self.right_idx = right_idx
                self.num_verts = num_verts
                B = shapedirs.reshape([shapedirs.shape[0], num_verts, 3])[:,left_idx]
                B = B.reshape([B.shape[0], -1])
                self.pred_layer = nn.Linear(nz_feat, len(left_idx) * 3)
            else:
                B = shapedirs
                self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

            if opts.use_smal_betas:
                # Initialize pred_layer weights to be small so initial def aren't so big
                self.pred_layer.weight.data.normal_(0, 0.0001)
            else:
                self.fc = nb.fc('batch', nz_feat, opts.n_shape_feat)
                n_feat = opts.n_shape_feat
                B = B.permute(1,0)
                A = torch.Tensor(np.zeros((B.size(0), n_feat)))
                n = np.min((B.size(1), n_feat))
                A[:,:n] = B[:,:n]
                self.pred_layer.weight.data = torch.nn.Parameter(A)
                self.pred_layer.bias.data.fill_(0.)

        else:
            self.ref_delta_v = torch.Tensor(np.zeros((opts.batch_size,num_verts,3))).cuda(device=opts.gpu_id) 


    def forward(self, feat):
        if self.opts.use_sym_idx:
            delta_v = torch.Tensor(np.zeros((self.opts.batch_size,self.num_verts,3))).cuda(device=self.opts.gpu_id)
            feat = self.fc(feat)
            self.shape_f = feat
     
            half_delta_v = self.pred_layer.forward(feat)
            half_delta_v = half_delta_v.view(half_delta_v.size(0), -1, 3)
            delta_v[:,self.left_idx,:] = half_delta_v
            half_delta_v[:,:,1] = -1.*half_delta_v[:,:,1]
            delta_v[:,self.right_idx,:] = half_delta_v
        else:
            delta_v = self.pred_layer.forward(feat)
            # Make it B x num_verts x 3
            delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v

class PosePredictor(nn.Module):
    """
    """
    def __init__(self, opts, nz_feat, num_joints=35):
        super(PosePredictor, self).__init__()
        self.opts = opts
        self.num_joints = num_joints
        self.pred_layer = nn.Linear(nz_feat, num_joints*3)

    def forward(self, feat):
        pose = self.opts.pose_var*self.pred_layer.forward(feat)

        # Add this to have zero to correspond to frontal facing
        pose[:,0] += 1.20919958
        pose[:,1] += 1.20919958
        pose[:,2] += -1.20919958
        return pose

class BetasPredictor(nn.Module):
    def __init__(self, opts, nz_feat, nenc_feat, num_betas=10):
        super(BetasPredictor, self).__init__()
        self.opts = opts
        self.pred_layer = nn.Linear(nenc_feat, num_betas)

    def forward(self, feat, enc_feat):
        betas = self.pred_layer.forward(enc_feat)

        return betas

class Keypoints2DPredictor(nn.Module):
    def __init__(self, opts, nz_feat, nenc_feat, num_keypoints=28):
        super(Keypoints2DPredictor, self).__init__()
        self.opts = opts
        self.num_keypoints = num_keypoints
        self.pred_layer = nn.Linear(nz_feat, 2*num_keypoints)

    def forward(self, feat, enc_feat):
        keypoints2D = self.pred_layer.forward(feat)
        return keypoints2D.view(-1,self.num_keypoints,2)



class ScalePredictor(nn.Module):
    '''
    In case of perspective projection scale is focal length
    '''
    def __init__(self, nz, opts):
        super(ScalePredictor, self).__init__()
        self.opts = opts
        if opts.use_camera:
            self.opts = opts
            self.pred_layer = nn.Linear(nz, opts.scale_bias)
        else:
            scale = np.zeros((opts.batch_size,1))
            scale[:,0] = 0.
            self.ref_camera = torch.Tensor(scale).cuda(device=opts.gpu_id) 

    def forward(self, feat):
        if not self.opts.use_camera:
            return self.ref_camera
        if self.opts.norm_f0 != 0:
            off = 0.
        else:
            off = 1.
        scale = self.pred_layer.forward(feat) + off   
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, projection_type, opts):
        super(TransPredictor, self).__init__()
        self.opts = opts
        if projection_type =='orth':
            self.pred_layer = nn.Linear(nz, 2)
        elif projection_type == 'perspective':
            self.pred_layer_xy = nn.Linear(nz, 2)
            self.pred_layer_z = nn.Linear(nz, 1)
            self.pred_layer_xy.weight.data.normal_(0, 0.0001)
            self.pred_layer_xy.bias.data.normal_(0, 0.0001)
            self.pred_layer_z.weight.data.normal_(0, 0.0001)
            self.pred_layer_z.bias.data.normal_(0, 0.0001)
        else:
            print('Unknown projection type')

    def forward(self, feat):
        trans = torch.Tensor(np.zeros((feat.shape[0],3))).cuda(device=self.opts.gpu_id)
        f = torch.Tensor(np.zeros((feat.shape[0],1))).cuda(device=self.opts.gpu_id)
        feat_xy = feat
        feat_z = feat
        trans[:,:2] = self.pred_layer_xy(feat_xy)
        trans[:,0] += 1.0
        trans[:,2] = 1.0+self.pred_layer_z(feat_z)[:,0]

        if self.opts.fix_trans:
            trans[:,2] = 1.

        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(self, nz_feat=100, nenc_feat=2048, num_verts=1000, opts=None, left_idx=None, right_idx=None, shapedirs=None):
        super(CodePredictor, self).__init__()
        self.opts = opts
        self.shape_predictor = ShapePredictor(nz_feat, num_verts=num_verts, opts=self.opts, left_idx=left_idx, right_idx=right_idx, shapedirs=shapedirs)
        self.scale_predictor = ScalePredictor(nz_feat, self.opts)
        self.trans_predictor = TransPredictor(nz_feat, self.opts.projection_type, self.opts)
        if opts.use_smal_pose:
            self.pose_predictor = PosePredictor(self.opts, nz_feat)
        if opts.use_smal_betas:
            self.betas_predictor = BetasPredictor(self.opts, nz_feat, nenc_feat, self.opts.num_betas)

    def forward(self, feat, enc_feat):
        if self.opts.use_delta_v:
            shape_pred = self.shape_predictor.forward(feat)
        else:
            shape_pred = self.shape_predictor.ref_delta_v
        if self.opts.use_camera:
            scale_pred = self.scale_predictor.forward(feat)
        else:
            scale_pred = self.scale_predictor.ref_camera

        trans_pred = self.trans_predictor.forward(feat)

        if self.opts.use_smal_pose:
            pose_pred = self.pose_predictor.forward(feat)
        else:
            pose_pred = None

        if self.opts.use_smal_betas:
            betas_pred = self.betas_predictor.forward(feat, enc_feat)
        else:
            betas_pred = None

        keypoints2D_pred = None

        return shape_pred, scale_pred, trans_pred, pose_pred, betas_pred, keypoints2D_pred

#------------ Mesh Net ------------#
#----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=28, sfm_mean_shape=None, tex_masks=None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture
        self.tex_masks = tex_masks

        self.op_features = None

        # Instantiate the SMAL model in Torch
        model_path = os.path.join(self.opts.model_dir, self.opts.model_name)
        self.smal = SMAL(pkl_path=model_path, opts=self.opts)

        self.left_idx = np.hstack((self.smal.left_inds, self.smal.center_inds))
        self.right_idx = np.hstack((self.smal.right_inds, self.smal.center_inds))

        pose = np.zeros((1,105))
        betas = np.zeros((1,self.opts.num_betas))
        V,J,R = self.smal(torch.Tensor(betas).cuda(device=self.opts.gpu_id), torch.Tensor(pose).cuda(device=self.opts.gpu_id))
        verts = V[0,:,:]
        verts = verts.data.cpu().numpy()
        faces = self.smal.f


        num_verts = verts.shape[0]

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces, self.smal.left_inds, self.smal.right_inds, self.smal.center_inds)
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])

            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = Variable(torch.ones(1, 3).cuda(device=self.opts.gpu_id), requires_grad=False)
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])            
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = num_verts
            faces = faces.astype(np.int32) 

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces).cuda(device=self.opts.gpu_id), requires_grad=False)
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        vert2kp_init = torch.Tensor(np.ones((num_kps, num_verts)) / float(num_verts))
        # Remember initial vert2kp (after softmax)
        self.vert2kp_init = torch.nn.functional.softmax(Variable(vert2kp_init.cuda(device=self.opts.gpu_id), requires_grad=False), dim=1)
        self.vert2kp = nn.Parameter(vert2kp_init)

        self.encoder = Encoder(self.opts, input_shape, n_blocks=4, nz_feat=nz_feat, bott_size=opts.bottleneck_size)
        nenc_feat = self.encoder.nenc_feat
        self.code_predictor = CodePredictor(nz_feat=nz_feat, nenc_feat=nenc_feat,
            num_verts=self.num_output, opts=opts, left_idx=self.left_idx, right_idx=self.right_idx, shapedirs=self.smal.shapedirs)

        if self.pred_texture:
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]
                self.num_sym_faces = 0

            # Instead of loading an obj file
            uv_data = pkl.load(open(os.path.join(self.opts.model_dir,opts.uv_data_file)))
            vt = uv_data['vt']
            ft = uv_data['ft']
            self.vt = vt
            self.ft = ft
            uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], vt, ft, tex_size=opts.tex_size)
            # F' x T x T x 2
            uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(device=self.opts.gpu_id), requires_grad=False)
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(self.opts.batch_size, 1, 1, 1, 1)
            if opts.number_of_textures > 0:
                    if opts.texture_img_size == 256:
                        if opts.number_of_textures == 7:
                            img_H = np.array([96, 96, 96, 96, 160, 160, 160])
                            img_W = np.array([64, 128, 32, 32, 128, 96, 32])
                        elif opts.number_of_textures == 4:
                            img_H = np.array([96, 160, 160, 256])
                            img_W = np.array([224, 128, 96, 32])
                    else:
                        print('ERROR texture')
                        import pdb; pdb.set_trace()
            else:
                img_H = opts.texture_img_size 
                img_W = opts.texture_img_size 


            self.texture_predictor = TexturePredictorUV(
              nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, predict_flow=True, symmetric=opts.symmetric_texture,
              num_sym_faces=self.num_sym_faces, tex_masks=self.tex_masks, vt=vt, ft=ft)
           
            nb.net_init(self.texture_predictor)

    def forward(self, img, masks=None):
        opts = self.opts
        if self.opts.is_optimization:
            if self.opts.is_var_opt:
                img_feat, enc_feat = self.encoder.forward(img, masks)
                if self.op_features is None:
                    codes_pred = self.code_predictor.forward(img_feat, enc_feat)
                    self.opts_scale = Variable(codes_pred[1].cuda(device=opts.gpu_id), requires_grad=True)
                    self.opts_pose = Variable(codes_pred[3].cuda(device=opts.gpu_id), requires_grad=True)
                    self.opts_trans = Variable(codes_pred[2].cuda(device=opts.gpu_id), requires_grad=True)
                    self.opts_delta_v= Variable(codes_pred[0].cuda(device=opts.gpu_id), requires_grad=True)
                    self.op_features = [self.opts_scale, self.opts_pose, self.opts_trans] 
                codes_pred = (self.opts_delta_v, self.opts_scale, self.opts_trans, self.opts_pose, None, None)
            else:
                # Optimization over the features
                if self.op_features is None:
                    img_feat, enc_feat = self.encoder.forward(img, masks)
                    self.op_features = Variable(img_feat.cuda(device=self.opts.gpu_id), requires_grad=True)
                codes_pred = self.code_predictor.forward(self.op_features, None)
                img_feat = self.op_features

        else:
            img_feat, enc_feat = self.encoder.forward(img, masks)
            codes_pred = self.code_predictor.forward(img_feat, enc_feat)
        if self.pred_texture:
            texture_pred = self.texture_predictor.forward(img_feat)
            return codes_pred, texture_pred
        else:
            return codes_pred

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_smal_verts(self, pose=None, betas=None, trans=None, del_v=None):
        if pose is None:
            pose = torch.Tensor(np.zeros((1,105))).cuda(device=self.opts.gpu_id)
        if betas is None:
            betas = torch.Tensor(np.zeros((1,self.opts.num_betas))).cuda(device=self.opts.gpu_id)
        if trans is None:
            trans = torch.Tensor(np.zeros((1,3))).cuda(device=self.opts.gpu_id)

        verts, _, _ = self.smal(betas, pose, trans, del_v)
        return verts

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)

