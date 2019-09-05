"""
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)


def perspective_proj_withz(X, cam, offset_z=0, cuda_device=0,norm_f=1., norm_z=0.,norm_f0=0.):
    """
    X: B x N x 3
    cam: B x 3: [f, cx, cy] 
    offset_z is for being compatible with previous code and is not used and should be removed
    """

    # B x 1 x 1
    #f = norm_f * cam[:, 0].contiguous().view(-1, 1, 1)
    f = norm_f0+norm_f * cam[:, 0].contiguous().view(-1, 1, 1)
    # B x N x 1
    z = norm_z + X[:, :, 2, None]

    # Will z ever be 0? We probably should max it..
    eps = 1e-6 * torch.ones(1).cuda(device=cuda_device)
    z = torch.max(z, eps)
    image_size_half = cam[0,1]
    scale = f / (z*image_size_half)

    # Offset is because cam is at -1
    return torch.cat((scale * X[:, :, :2], z+offset_z),2)

