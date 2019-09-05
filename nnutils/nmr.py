from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import chainer
import torch

import neural_renderer

from ..nnutils import geom_utils

#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self):
        # setup renderer
        renderer = neural_renderer.Renderer()
        self.renderer = renderer

    def to_gpu(self, device=0):
        # self.renderer.to_gpu(device)
        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''
        self.faces = chainer.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        self.vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))

        self.masks = self.renderer.render_silhouettes(self.vertices, self.faces)

        masks = self.masks.data.get()
        return masks
    
    def backward_mask(self, grad_masks):
        ''' Compute gradient of vertices given mask gradients.
        Args:
            grad_masks: B X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
        '''
        self.masks.grad = chainer.cuda.to_gpu(grad_masks, self.cuda_device)
        self.masks.backward()
        return self.vertices.grad.get()

    def forward_img(self, vertices, faces, textures):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        self.faces = chainer.Variable(chainer.cuda.to_gpu(faces, self.cuda_device))
        self.vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, self.cuda_device))
        self.textures = chainer.Variable(chainer.cuda.to_gpu(textures, self.cuda_device))
        self.images = self.renderer.render(self.vertices, self.faces, self.textures)

        images = self.images.data.get()
        return images


    def backward_img(self, grad_images):
        ''' Compute gradient of vertices given image gradients.
        Args:
            grad_images: B X 3? X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
            grad_textures: B X F X T X T X T X 3 numpy array
        '''
        self.images.grad = chainer.cuda.to_gpu(grad_images, self.cuda_device)
        self.images.backward()
        return self.vertices.grad.get(), self.textures.grad.get()

########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render(torch.autograd.Function):
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        vs = vertices.cpu().numpy()
        vs[:, :, 1] *= -1
        fs = faces.cpu().numpy()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return convert_as(torch.Tensor(masks), vertices)
        else:
            self.mask_only = False
            ts = textures.cpu().numpy()
            imgs = self.renderer.forward_img(vs, fs, ts)
            return convert_as(torch.Tensor(imgs), vertices)

    def backward(self, grad_out):
        g_o = grad_out.cpu().numpy()
        if self.mask_only:
            grad_verts = self.renderer.backward_mask(g_o)
            grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
            grad_tex = None
        else:
            grad_verts, grad_tex = self.renderer.backward_img(g_o)
            grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
            grad_tex = convert_as(torch.Tensor(grad_tex), grad_out)

        grad_verts[:, :, 1] *= -1
        return grad_verts, None, grad_tex


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256, proj_type='perspective', norm_f=1., norm_z=0.,norm_f0=0.):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()

        self.norm_f = norm_f
        self.norm_f0 = norm_f0
        self.norm_z = norm_z

        # Adjust the core renderer
        self.renderer.renderer.image_size = img_size
        self.renderer.renderer.perspective = False

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.renderer.eye = [0, 0, -1.0]

        # Make it a bit brighter for vis
        self.renderer.renderer.light_intensity_ambient = 0.8

        self.renderer.to_gpu()

        # Silvia
        if proj_type == 'perspective':
            self.proj_fn = geom_utils.perspective_proj_withz
        else:
            print('unknown projection type')
            import pdb; pdb.set_trace()

        self.offset_z = -1.0

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def directional_light_only(self):
        # Make light only directional.
        self.renderer.renderer.light_intensity_ambient = 0.8
        self.renderer.renderer.light_intensity_directional = 0.8
        self.renderer.renderer.light_direction = [0, 1, 0]  # up-to-down, this is the default

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams, offset_z=self.offset_z, norm_f=self.norm_f, norm_z=self.norm_z, norm_f0=self.norm_f0)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None):
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z, norm_f=self.norm_f, norm_z=self.norm_z, norm_f0=self.norm_f0)

        if textures is not None:
            return Render(self.renderer)(verts, faces, textures)
        else:
            return Render(self.renderer)(verts, faces)


