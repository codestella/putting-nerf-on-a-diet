import os
import glob 
import json

import imageio
import cv2

import jax.numpy as np
from jax import jit


@jit
def get_rays(c2w, kinv, i, j):
#     i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    pixco = np.stack([i, j, np.ones_like(i)], -1)
    dirs = pixco @ kinv.T
#     dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], rays_d.shape)
    return np.stack([rays_o, rays_d], 0)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    return viewmatrix(vec2, up, center)


def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    """
    enumerate list of poses around a spiral
    used for test set visualization
    """
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def _parse_nerf_synthetic(pose_path, img_path, down):
    posedata = {}
    imgdata = {}

    for type in ['train', 'test', 'val']:
        imgs, poses = [], []
        posedata[type] = {}
        imgdata[type] = {}

        with open(os.path.join(pose_path, 'transforms_'+type+'.json'), 'r') as fp:
            meta = json.load(fp)

        img0 = imageio.imread(os.path.join(img_path, 'r_0.png')) # to get H, W

        H, W = img0.shape[0]//down, img0.shape[1]//down
        cy, cx = H/2., W/2.

        datalen = len(meta['frames'])

        for idx in np.arange(datalen):
            frame = meta['frames'][idx]
            fname = os.path.join(img_path, 'r_'+str(idx)+'.png')

            try:
                imgs.append(cv2.resize(imageio.imread(fname), (H,W)))
                poses.append(np.array(frame['transform_matrix']))
            except:
                continue

        focal = .5 * W / np.tan(.5 * float(meta['camera_angle_x']))/down 
        kinv = np.array([[1/focal, 0., -cx/focal], [0., 1/focal, -cy/focal], [0., 0., 1.]])
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        imgdata[type] = imgs[...,:3] * imgs[...,-1:] + 1-imgs[...,-1:]
        posedata[type]['c2w_mats'] = np.array(poses).astype(np.float32)
        posedata[type]['kinv_mats'] = np.tile(kinv, (datalen, 1, 1))
        posedata[type]['bds'] = np.tile(np.array([2.0, 6.0]), (datalen, 1))
        posedata[type]['res_mats'] = np.tile(np.array([H, W]), (datalen, 1))

    return imgdata, posedata

def _parse_phototourism(pose_path, img_path):
    posedata = {}
    imgdata = {}

    imgfiles = sorted(glob.glob(img_path + '/*.jpg'))

    for type in ['train', 'test', 'val']:
        posedata[type] = {}
        imgdata[type] = {}
        imgs = []

        if type == 'train':
            start, end = 25, len(os.listdir(pose_path))
        elif type == 'test':
            start, end = 0, 20
        elif type == 'val':
            start, end = 20, 25

        for iter, f in enumerate(os.listdir(pose_path)[start:end]):
            if '.npy' not in f:
                continue

            z = np.load(os.path.join(pose_path, f))
            posedata[type][f.split('.')[0]] = z
            _im = imageio.imread(imgfiles[iter])[..., :3] / 255.
            imgs.append(np.array(_im))
            # imgs.append(imageio.imread(imgfiles[iter])[..., :3] / 255.)

        print(iter)
        print(imageio.imread(imgfiles[iter]).shape)
        print(imageio.imread(imgfiles[iter])[..., :3] / 255.)
    
        imgdata[type] = np.array(imgs)

        # if imgs.shape[3]==4: # alpha channel
        #     imgdata[type] = imgs[...,:3] * imgs[...,-1:] + 1-imgs[...,-1:]
        # else:
        #     imgdata[type] = imgs

    return imgdata, posedata

def data_loader(select_data, abspath, preload=True, down=1):
    '''
    input:
        select_data: 'data_class/dataname'
            e.g.) 'nerf_synthetic/lego', 'phototourism/sacre', 'shapenet/chair'
        abspath: a directory which contains all dataset
        preload: whether pre-loading the images at onces OR loading whenever get_example() called
    output:
        imgfiles
    '''

    data_class, data_name = select_data.split('/')

    if data_class == 'nerf_synthetic':
        pose_path = os.path.join(abspath, data_class, data_name) # 'transforms_'+type+'.json'
        img_path = os.path.join(abspath, data_class, data_name, "scene0", "train") # 'r_'+str(idx)+'.png')
        if preload:
            return _parse_nerf_synthetic(pose_path, img_path, down)

    elif data_class == 'phototourism':
        ## temporary setting to test;
        temp_data_class = "pull-phototourism-images"
        pose_path = os.path.join(abspath, data_class, data_class, data_name) # Directory condtains [bds.npy, c2w_mats.npy, kinv_mats.npy, res_mats.npy]
        img_path = os.path.join(abspath, temp_data_class, data_name+'_coeur', 'dense', 'images') # Directory of images
        if preload:
            return _parse_phototourism(pose_path, img_path)

    elif data_class == 'shapenet':
        raise "NOT IMPLEMENTED"

    else:
        raise NameError('Wrong data class. check `select_data` variable')
    
