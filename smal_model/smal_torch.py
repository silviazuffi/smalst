"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from .smal_basics import align_smal_template_to_symmetry_axis, get_horse_template

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(object):
    def __init__(self, pkl_path, opts, dtype=torch.float):
        self.opts = opts
        # -- Load SMPL params --
        with open(pkl_path, 'r') as f:
            dd = pkl.load(f)

        self.f = dd['f']

        v_template = get_horse_template(model_name='my_smpl_00781_4_all.pkl', data_name='my_smpl_data_00781_4_all.pkl')
        v, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(v_template)

        # Mean template vertices
        self.v_template = Variable(
            torch.Tensor(v).cuda(device=self.opts.gpu_id),
            requires_grad=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = Variable(
            torch.Tensor(shapedir).cuda(device=self.opts.gpu_id), requires_grad=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = Variable(
            torch.Tensor(dd['J_regressor'].T.todense()).cuda(device=self.opts.gpu_id),
            requires_grad=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.Tensor(posedirs).cuda(device=self.opts.gpu_id), requires_grad=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])).cuda(device=self.opts.gpu_id),
            requires_grad=False)

    def __call__(self, beta, theta, trans=None, del_v=None, get_skin=True):

        if self.opts.use_smal_betas:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        # 1. Add shape blend shapes
        # (N x 10) x (10 x 3880*3) = N x 3889 x 3
        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + del_v 

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = torch.reshape( batch_rodrigues(torch.reshape(theta, [-1, 3]), opts=self.opts), [-1, 35, 3, 3])
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).cuda(device=self.opts.gpu_id), [-1, 306])


        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, opts=self.opts)


        # 5. Do skinning:
        num_batch = theta.shape[0]
        # W is N x 6890 x 24
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

            # (N x 6890 x 24) x (N x 24 x 16)
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).cuda(device=self.opts.gpu_id)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).cuda(device=self.opts.gpu_id)

        verts = verts + trans[:,None,:]

        # Get cocoplus or lsp joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints











def main():
    import os
    model_name='my_smpl_00781_4_all.pkl'
    model_dir = '/Users/silvia/Dropbox/Work/smalr/smpl_models/'
    model_path = os.path.join(model_dir, model_name)
    smpl = SMAL(model_path)

    theta = np.zeros((1,105))
    betas = np.zeros((1,10))
    theta[0,10] = 0.5
    betas[0,:] = .5
    V,J,R = smpl(torch.Tensor(betas), torch.Tensor(theta))

    from psbody.mesh.meshviewer import MeshViewer, MeshViewers
    from psbody.mesh import Mesh

    Mesh(v=V, f=smpl.f).show()


    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()

