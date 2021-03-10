"""
Evaluation on the testset.

python -m smalst.smal_eval --name=smal_net_600 --img_path='smalst/testset_zoo/' --num_train_epoch=186 --use_annotations=False --mirror=False --segm_eval=False --img_ext='.png' --bgval=0

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import numpy as np

import pickle as pkl

import torch
import scipy
import scipy.misc
#from .nnutils import test_utils
from .nnutils import smal_predictor as pred_util
from .utils import image as img_util
from glob import glob
import scipy.io as sio
import matplotlib.pyplot as plt

# Only necessary for running on the testset to pad the image with the original one
from .testset_shape_experiments_crops import bboxes
import os.path as osp
from os.path import exists
from os import makedirs

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, 'cachedir')

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')
flags.DEFINE_boolean('use_annotations', True, '')
flags.DEFINE_boolean('mirror', False, '')
flags.DEFINE_boolean('visualize', False, '')
flags.DEFINE_boolean('save_input', False, '')
flags.DEFINE_string('anno_path', '.', 'where the annotations are')
flags.DEFINE_string('img_ext', '.jpg', 'image extension')
flags.DEFINE_boolean('segm_eval', True, 'if we have gt segmentations and evaluate overlap')
flags.DEFINE_boolean('synthetic', False, '')
flags.DEFINE_integer('bgval', -1, '-1 means to pad the image with the original one (used for the testset)')
flags.DEFINE_boolean('test_optimization_results', False, '')
flags.DEFINE_string('optimization_dir', 'smalst/optimization_results', '')
flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_string('checkpoint_dir',
                    osp.join(cache_path, 'snapshots'),
                    'Directory where networks are saved')
flags.DEFINE_string('out_path', './smalst_results', 'where to save the result images')


opts = flags.FLAGS

def get_bbox(img_path, kp):
    # Load the mask
    mask_img = scipy.misc.imread(img_path.replace('images', 'bgsub')) / 255.
    # Load the image
    img = scipy.misc.imread(img_path) / 255.
    where = np.array(np.where(mask_img))
    xmin, ymin, _ = np.amin(where, axis=1)
    xmax, ymax, _ = np.amax(where, axis=1)
    mask_img = mask_img[xmin:xmax, ymin:ymax,:]
    img = img[xmin:xmax, ymin:ymax,:]
    kp[:,0] = kp[:,0] - ymin
    kp[:,1] = kp[:,1] - xmin
    return img, mask_img, kp

def preprocess_image(img_path, img_size=256, kp=None, border=5, bgval=-1, img=None, img_ext='.jpg'):

    if img is None:
        img = scipy.misc.imread(img_path) / 255.
    
    img = img[:,:,:3]
    img_in_shape = img.shape

    # Scale the max image size to be img_size
    scale_factor = float(img_size-2*border) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=bgval)

    
    # Replace the border with the real background
    img_name = osp.splitext(osp.basename(img_path))[0] 
    # Read the full image
    if bgval == -1:
        full_img_path = osp.join(osp.dirname(img_path), 'full_size', img_name+'*'+img_ext)
        full_img_path = glob(full_img_path)[0]
        if osp.exists(full_img_path):
            full_img = scipy.misc.imread(full_img_path)/255.
            bbox_orig = np.array(bboxes[img_name])
            sf = img_in_shape[0]/(1.0*bbox_orig[3])
            new_img, _ = img_util.resize_img(full_img, sf*scale_factor)
            center[0] = np.round((bbox_orig[2]/2. + bbox_orig[0])*sf*scale_factor).astype(int)
            center[1] = np.round((bbox_orig[3]/2. + bbox_orig[1])*sf*scale_factor).astype(int)
            bbox2 = np.hstack([center - img_size / 2., center + img_size / 2.])
            img = img_util.crop(new_img, bbox2, bgval=0)
 

    img, _ = img_util.resize_img(img, 256/257.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    if kp is not None:
        kp = kp*scale_factor
        kp[:,0] -= bbox[0]
        kp[:,1] -= bbox[1]

    return img, kp

def mirror_keypoints(keyp, vis, img_w):
    '''
    exchange keypoints from left to right and update x value
    names =['leftEye','rightEye','chin','frontLeftFoot','frontRightFoot',
            'backLeftFoot','backRightFoot','tailStart','frontLeftKnee',
            'frontRightKnee','backLeftKnee','backRightKnee','leftShoulder',
            'rightShoulder','frontLeftAnkle','frontRightAnkle','backLeftAnkle'
            'backRightAnkle','neck','TailTip','leftEar','rightEar',
            'nostrilLeft','nostrilRight','mouthLeft','mouthRight',
            'cheekLeft','cheekRight']
    '''

    dx2sx = [1,0,2,4,3,6,5,7,9,8,11,10,13,12,15,14,17,16,18,19,21,20,23,22,25,24,27,26]
        

    keyp_m = np.zeros_like(keyp)
    keyp_m[:,0] = img_w - keyp[dx2sx,0] - 1
    keyp_m[:,1] = keyp[dx2sx,1]

    vis_m = np.zeros_like(vis)
    vis_m[:] = vis[dx2sx]

    return keyp_m, vis_m

def mirror_image(img):
    if len(img.shape)==3:
        img_m = img[:,:,::-1].copy()
    else:
        img_m = img[:,::-1].copy()
    return img_m

def visualize_opt(img, predictor, renderer, data, out_path):

    pose = torch.Tensor(data['pose']).cuda(0)
    trans = torch.Tensor(data['trans']).cuda(0)
    del_v = torch.Tensor(data['delta_v']).cuda(0)
    vert = predictor.model.get_smal_verts(pose, None, trans, del_v)
    cam = 128*np.ones((3))
    cam[0] = data['scale'][0,:]
    cam = torch.Tensor(cam).cuda(0)
    shape_pred = renderer(vert, cam)
    img = np.transpose(img, (1, 2, 0))
    I = 0.3*255*img + 0.7*shape_pred
    scipy.misc.imsave(out_path, I)


def visualize(img, outputs, renderer, img_path=None, opts=None, cam_gt=0,
              z_gt=0, kp_gt=None, kp_pred=None, mask_gt=None, vis=None, out_path=None, tex_mask=None, visualize=False):

    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]
    mask = outputs['mask_pred'].cpu().detach().numpy()

    if 'texture' in outputs.keys():
        texture = outputs['texture'][0]
        uv_image = outputs['uv_image'][0].cpu().detach().numpy()
        T = uv_image.transpose(1,2,0)
        img_pred = renderer(vert, cam, texture=texture)
    if 'occ_pred' in outputs.keys():
        occ_pred = outputs['occ_pred'][0].cpu().detach().numpy()

    shape_pred = renderer(vert, cam)

    img = np.transpose(img, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))[:,:,0]

    Iov = 0.3*255*img + 0.7*shape_pred
    I = shape_pred
    scipy.misc.imsave(osp.join(opts.out_path, 'shape_ov_'+out_path), Iov)
    scipy.misc.imsave(osp.join(opts.out_path, 'shape_'+out_path), I)

    
    N = 0.6*np.abs(tex_mask[:,:,:3]-1)
    if 'texture' in outputs.keys():
        scipy.misc.imsave(osp.join(opts.out_path, 'tex_'+out_path), N+T*tex_mask[:,:,:3])
        scipy.misc.imsave(osp.join(opts.out_path, 'img_'+out_path), img_pred)
    
    if visualize:

        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(231)
        plt.imshow(img)
        plt.title('input')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(img)
        plt.imshow(shape_pred, alpha=0.7)
        if kp_gt is not None:
            idx = np.where(vis==True)
            plt.scatter(kp_gt[idx[0],0], kp_gt[idx[0],1])
            plt.scatter(kp_pred[idx,0], kp_pred[idx,1])

        plt.title('pred mesh')
        plt.axis('off')
        plt.subplot(233)
        if 'texture' in outputs.keys():
            plt.imshow(img_pred)
            plt.title('pred mesh w/texture')
            plt.axis('off')
            plt.subplot(234)
            plt.imshow(T)
            plt.axis('off')
            plt.subplot(235)
            plt.imshow(T*tex_mask[:,:,:3])
            plt.axis('off')
        plt.subplot(236)
        plt.imshow(mask)
        plt.axis('off')
        plt.draw()
        plt.show()
        plt.savefig(out_path, bbox_inches='tight')

        import pdb; pdb.set_trace()


def save_params(outputs, idx):
    data = {'pose': outputs['pose_pred'].data.detach().cpu().numpy()[0,:],
            'verts': outputs['verts'].data.detach().cpu().numpy()[0,:],
            'f': outputs['f'].data.detach().cpu().numpy()[0,:],
            'v': outputs['v'].data.detach().cpu().numpy()[0,:],}
    pkl.dump(data, open('data_'+str(idx)+'.pkl', 'wb'))


def main(_):

    texture_mask_path = 'smalst/zebra_data/texture_maps/my_smpl_00781_4_all_template_w_tex_uv_001_mask_small_256.png'
    tex_mask = scipy.misc.imread(texture_mask_path)/255. 

    if not exists(opts.out_path): makedirs(opts.out_path)

    images = sorted(glob(opts.img_path+'*'+opts.img_ext))
    print(str(len(images)))
    N = len(images)
    show = True
    segm_eval = opts.segm_eval
    use_annotations = opts.use_annotations
    mirror = opts.mirror
    if mirror:
        N = 2*N
    if show:
        predictor = pred_util.MeshPredictor(opts)
    tot_pose_err = 0
    annotations_path = opts.anno_path
    alpha = [0.01, 0.02, 0.05, 0.1, 0.15] 
    n_alpha = len(alpha)

    global_rotation = 0 
    mirr_global_rotation = 0 

    err_tot = np.zeros((N, n_alpha))

    shape_f = np.zeros((N,40))

    overlap = np.zeros((N))
    IOU = np.zeros((N))

    idx = 0

    for iidx, img_path in enumerate(images):

        if use_annotations:
            if opts.synthetic:
                anno_path = osp.join(annotations_path, osp.basename(img_path).replace(opts.img_ext, '.pkl'))
                res = pkl.load(open(anno_path))
                kp = res['keypoints']
                pose = res['pose']
                trans = res['trans']
                flength = res['flength']
                delta_v = res['delta_v']
                vis = np.ones((kp.shape[0],1),dtype=bool)
            else:
                anno_path = osp.join(annotations_path, osp.basename(img_path).replace(opts.img_ext, '_ferrari-tail-face.mat'))
                res = sio.loadmat(anno_path, squeeze_me=True, struct_as_record=False)
                res = res['annotation']
                kp = res.kp.astype(float)
                invisible = res.invisible
                vis = np.atleast_2d(~invisible.astype(bool)).T
                landmarks = np.hstack((kp, vis))
                names = [str(res.names[i]) for i in range(len(res.names))]
        else:
            kp = None
            kp_pred = None
            vis = None

        if opts.synthetic:
            img, mask_img, kp = get_bbox(img_path, kp=kp)
            mask_img, _ = preprocess_image(img_path.replace('images','bgsub'), img_size=opts.img_size, kp=None, bgval=0, img=mask_img, img_ext=opts.img_ext)

            img, kp = preprocess_image(img_path, img_size=opts.img_size, kp=kp, img=img, bgval=0, img_ext=opts.img_ext)
        else:
            img, kp = preprocess_image(img_path, img_size=opts.img_size, kp=kp, bgval=opts.bgval, img_ext=opts.img_ext)

        code = osp.splitext(osp.basename(img_path))[0]

        # Load the gt mask
        if segm_eval:
            if not opts.synthetic:
                mask_path = osp.join(opts.img_path, 'masks', osp.basename(img_path).replace('jpg','png'))
                mask_img, _ = preprocess_image(mask_path, img_size=opts.img_size, bgval=0, img_ext=opts.img_ext)

        if opts.save_input:
            scipy.misc.imsave(osp.join(opts.out_path, 'proc_'+osp.basename(img_path)), 255*np.transpose(img, (1, 2, 0)))
            if segm_eval:
                scipy.misc.imsave(osp.join(opts.out_path, 'mask_proc_'+osp.basename(img_path)), 255*np.transpose(mask_img, (1, 2, 0)))

        print(idx)
        print(code)

        if opts.test_optimization_results:
            res_file = osp.join(opts.optimization_dir, 'proc_'+code+'_best_res.pkl')
            mask_file = osp.join(opts.optimization_dir, 'proc_'+code+'_best_mask.png')

            print('look for file ' + res_file)
            if not osp.exists(res_file) or not osp.exists(mask_file):
                res_file = osp.join(opts.optimization_dir, 'proc_'+code+'_init_res.pkl')
                mask_file = osp.join(opts.optimization_dir, 'proc_'+code+'_init_mask.png')
            else:
                print('found optimization result')

            data = pkl.load(open(res_file))
            mask_pred = scipy.misc.imread(mask_file)/255. 

        else:
            batch = {'img': torch.Tensor(np.expand_dims(img, 0))}
            outputs = predictor.predict(batch, rot=global_rotation)
            mask_pred = outputs['mask_pred'].detach().cpu().numpy()[0,:,:]
            shape_f[idx,:] = outputs['shape_f'].detach().cpu().numpy()
    
        if segm_eval: 
            M_gt = mask_img[0,:,:]
            overlap[idx] = np.sum(M_gt*mask_pred)/(np.sum(M_gt)+np.sum(mask_pred))
            IOU[idx] = np.sum(M_gt*mask_pred)/(np.sum(M_gt)+np.sum(mask_pred)-np.sum(M_gt*mask_pred))
            print(overlap[idx])
            print(IOU[idx])


        if use_annotations:
            if opts.test_optimization_results:
                kp_pred = ((data['kp_pred'][0,:,:]+1.)*128).astype(int)
            else:
                kp_pred = ((outputs['kp_pred'].cpu().detach().numpy()[0,:,:]+1.)*128).astype(int)

            kp_diffs = np.linalg.norm(kp[vis[:,0],:]/256. - kp_pred[vis[:,0],:]/256., axis=1)
            for a in range(n_alpha):
                kp_err = np.mean(kp_diffs < alpha[a])
                print(kp_err)
                err_tot[idx, a] = kp_err

        if show and not opts.test_optimization_results:
            renderer = predictor.vis_rend
            renderer.set_light_dir([0, 1, -1], 0.4)
            visualize(img, outputs, predictor.vis_rend, img_path, opts,
                    kp_gt=kp, kp_pred=kp_pred, vis=vis, out_path=opts.name+'_test_%03d' % (idx) +'.png', tex_mask=tex_mask,
                    visualize=opts.visualize)

        if show and opts.test_optimization_results:
            visualize_opt(img, predictor, predictor.vis_rend, data, opts.name+'_opt_%03d' % (idx) +'.png')

        idx += 1

        if mirror:
            kp_m = None
            vis_m = None
            img_m = mirror_image(img)
            M_gt_m = mirror_image(M_gt)
            if opts.save_input:
                scipy.misc.imsave('proc_mirr_'+osp.basename(img_path), 255*np.transpose(img_m, (1, 2, 0)))

            if opts.test_optimization_results:
                res_file = osp.join(opts.optimization_dir, 'proc_mirr_'+code+'_best_res.pkl')
                mask_file = osp.join(opts.optimization_dir, 'proc_mirr_'+code+'_best_mask.png')
                if not osp.exists(res_file) or not osp.exists(mask_file):
                    res_file = osp.join(opts.optimization_dir, 'proc_mirr_'+code+'_init_res.pkl')
                    mask_file = osp.join(opts.optimization_dir, 'proc_mirr_'+code+'_init_mask.png')
                data = pkl.load(open(res_file))
                mask_pred = scipy.misc.imread(mask_file)/255. 
            else:
                batch = {'img': torch.Tensor(np.expand_dims(img_m, 0))}
                outputs = predictor.predict(batch, rot=mirr_global_rotation)
                shape_f[idx,:] = outputs['shape_f'].detach().cpu().numpy()
                mask_pred = outputs['mask_pred'].detach().cpu().numpy()[0,:,:]
            if segm_eval:
                M_gt = mask_img[0,:,:]
                overlap[idx] = np.sum(M_gt_m*mask_pred)/(np.sum(M_gt_m)+np.sum(mask_pred))
                IOU[idx] = np.sum(M_gt*mask_pred)/(np.sum(M_gt)+np.sum(mask_pred)-np.sum(M_gt*mask_pred))
                print(overlap[idx])
                print(IOU[idx])

            if use_annotations:
                if opts.test_optimization_results:
                    kp_m, vis_m = mirror_keypoints(kp, vis, mask_img.shape[1])
                    kp_pred = ((data['kp_pred'][0,:,:]+1.)*128).astype(int)
                else:
                    kp_m, vis_m = mirror_keypoints(kp, vis, img.shape[2])
                    kp_pred = ((outputs['kp_pred'].cpu().detach().numpy()[0,:,:]+1.)*128).astype(int)

                kp_diffs = np.linalg.norm(kp_m[vis[:,0],:]/256. - kp_pred[vis[:,0],:]/256., axis=1)
                for a in range(n_alpha):
                    kp_err = np.mean(kp_diffs < alpha[a])
                    print(kp_err)
                    err_tot[idx, a] = kp_err

            if show and not opts.test_optimization_results:
                visualize(img_m, outputs, predictor.vis_rend, img_path, opts,
                    kp_gt=kp_m, kp_pred=kp_pred, vis=vis_m, out_path=opts.name+'_test_%03d' % (idx) +'.png', tex_mask=tex_mask,
                    visualize=opts.visualize)

            if show and opts.test_optimization_results:
                visualize_opt(img_m, predictor, predictor.vis_rend, data, opts.name+'_opt_%03d' % (idx) +'.png')
            idx += 1

    

    if use_annotations:
        print('PCK')
        print(np.mean(err_tot, axis=0))
        print(np.median(err_tot, axis=0))
        print(np.std(err_tot, axis=0))
        print('Overlap')
        print(np.mean(overlap))
        print(np.median(overlap))
        print(np.std(overlap))
        print('IOU')
        print(np.mean(IOU))
        print(np.median(IOU))
        print(np.std(IOU))
    


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
