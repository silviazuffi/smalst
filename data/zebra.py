"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle as pkl

import scipy.io as sio
import scipy.misc

from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import smal_base as base_data
from ..utils import transformations

import pickle as pkl


# -------------- flags ------------- #
# ---------------------------------- #
#kData = 'nokap/zebra_data'
    
flags.DEFINE_string('zebra_dir', 'nokap/zebra_data', 'Zebra Data Directory')
flags.DEFINE_string('image_file_string', '*.png', 'String use to read the images')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('zebra_cache_dir', osp.join(cache_path, 'zebra'), 'Zebra Data Directory')
flags.DEFINE_integer('num_images', 3000, 'Number of training images')
flags.DEFINE_boolean('preload_image', False, '')
flags.DEFINE_boolean('preload_mask', False, '')
flags.DEFINE_boolean('preload_texture_map', False, '')
flags.DEFINE_boolean('preload_uvflow', False, '')
flags.DEFINE_boolean('use_per_file_texmap', True, 'use the file with updated change in color')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class ZebraDataset(base_data.BaseDataset):
    '''
    Zebra Data loader
    '''

    def __init__(self, opts, filter_key=None, filter_name=None):
        super(ZebraDataset, self).__init__(opts, filter_key=filter_key)
        self.data_cache_dir = opts.zebra_cache_dir
        self.filter_key = filter_key
        self.filter_name = filter_name 

        if True: 
            self.data_dir = opts.zebra_dir
            self.img_dir = osp.join(self.data_dir, 'images')
            from glob import glob
            if filter_name is None:
                images = glob(osp.join(self.img_dir, opts.image_file_string))
            else:
                images = glob(osp.join(self.img_dir, filter_name + opts.image_file_string))
            num_images = np.min([len(images), opts.num_images])
            images = images[:num_images]
            self.anno = [None]*num_images
            self.anno_camera = [None]*len(images)


            for i, img in enumerate(images):
                anno_path = osp.join(self.data_dir, 'annotations/%s.pkl' % osp.splitext(osp.basename(img))[0])
                if osp.exists(anno_path):
                    self.anno[i] = pkl.load(open(anno_path))
                    self.anno[i]['mask_path'] = osp.join(self.data_dir, 'bgsub/%s.png' % osp.splitext(osp.basename(img))[0])
                    self.anno[i]['img_path'] = img
                    uv_flow_path = osp.join(self.data_dir, 'uvflow/%s.pkl' % osp.splitext(osp.basename(img))[0])
                    if osp.exists(uv_flow_path):
                        self.anno[i]['uv_flow_path'] = uv_flow_path

                    # In case we have the texture map
                    if 'texture_map_filename' in self.anno[i].keys():
                        if opts.use_per_file_texmap:
                            self.anno[i]['texture_map'] = osp.join(self.data_dir, 'texmap/%s.png' % osp.splitext(osp.basename(img))[0])
                        else:
                            self.anno[i]['texture_map'] = osp.join(self.data_dir, 'texture_maps/%s' % self.anno[i]['texture_map_filename'])

                    # Add a column to the keypoints in case the visibility is not defined
                    if self.anno[i]['keypoints'].shape[1] < 3:
                        self.anno[i]['keypoints'] = np.column_stack([self.anno[i]['keypoints'], np.ones(self.anno[i]['keypoints'].shape[0])])

                    self.anno_camera[i]= {'flength': self.anno[i]['flength'], 'trans': np.zeros(2, dtype=float)}
                    self.kp_perm = np.array(range(self.anno[0]['keypoints'].shape[0]))

                    if opts.preload_image:
                        self.anno[i]['img'] = scipy.misc.imread(self.anno[i]['img_path']) / 255.0
                    if opts.preload_texture_map:
                        texture_map = scipy.misc.imread(self.anno[i]['texture_map']) / 255.0
                        self.anno[i]['texture_map_data'] = np.transpose(texture_map, (2, 0, 1))
                    if opts.preload_mask:
                        self.anno[i]['mask'] = scipy.misc.imread(self.anno[i]['mask_path']) / 255.0
                    if opts.preload_uvflow:
                        uvdata = pkl.load(open(self.anno[i]['uv_flow_path']))
                        uv_flow = uvdata['uv_flow'].astype(np.float32)
                        uv_flow[:,:,0] = uv_flow[:,:,0] /(uvdata['img_h']/2.)
                        uv_flow[:,:,1] = uv_flow[:,:,1] /(uvdata['img_w']/2.)
                        self.anno[i]['uv_flow'] = uv_flow

                else:
                    mask_path = osp.join(self.data_dir, 'bgsub/%s.png' % osp.splitext(osp.basename(img))[0])
                    if osp.exists(mask_path):
                        self.anno[i] = {'mask_path': mask_path, 'img_path':img, 'keypoints': None, 'uv_flow':None}
                    else:
                        self.anno[i] = {'mask_path': None, 'img_path':img, 'keypoints': None, 'uv_flow':None}
                    self.anno_camera[i] = {'flength': None, 'trans': None}

            self.num_imgs = len(self.anno)

        print('%d images' % self.num_imgs)

        #import pdb; pdb.set_trace()
        #self.debug_crop()


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True, filter_name=None):
    return base_data.base_loader(ZebraDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle, filter_name=filter_name)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(ZebraDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(ZebraDataset, batch_size, opts, filter_key='mask')

def texture_map_data_loader(batch_size, opts):
    return base_data.base_loader(ZebraDataset, batch_size, opts, filter_key='texture_map')
    
