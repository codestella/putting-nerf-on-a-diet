import os
import glob

import imageio
import jax.numpy as np
from jax import jit


DATASET = 'sacre'
posedir = f'/mnt/hdd1/stella/inerf/learnit_Data/phototourism/sacre' # Directory condtains [bds.npy, c2w_mats.npy, kinv_mats.npy, res_mats.npy]
imgdir = f'/mnt/hdd1/stella/inerf/learnit_Data/phototourism/original/sacre/sacre_coeur/dense/images/' # Directory of images
posedata = {}
for f in os.listdir(posedir):
    if '.npy' not in f:
        continue
    z = np.load(os.path.join(posedir, f))
    posedata[f.split('.')[0]] = z

imgfiles = sorted(glob.glob(imgdir + '/*.jpg'))
print(f'{len(imgfiles)} images')
print('Pose data loaded - ', posedata.keys())
imgfiles = sorted(glob.glob(imgdir + '/*.jpg'))


def get_example(img_idx, split='train', downsample=4):
    sc = .05

    # first 20 are test, next 5 are validation, the rest are training:
    # https://github.com/tancik/learnit/issues/3
    if 'train' in split:
        img_idx = img_idx + 25
    if 'val' in split:
        img_idx = img_idx + 20

    # uint8 --> float
    img = imageio.imread(imgfiles[img_idx])[... ,:3 ] /255.

    # WHAT DO THESE MATRICES MEAN???
    # (4, 4)
    c2w = posedata['c2w_mats'][img_idx]
    # (3, 3)
    kinv = posedata['kinv_mats'][img_idx]
    c2w = np.concatenate([c2w[:3 ,:3], c2w[:3 ,3:4 ] *sc], -1)
    # (2, )
    bds = posedata['bds'][img_idx] * np.array([.9, 1.2]) * sc
    H, W = img.shape[:2]

    # (0, 4, 8, ..., H)
    # WHAT ARE THE PURPOSES OF THIS MATRIX???
    i, j = np.meshgrid(np.arange(0 ,W ,downsample), np.arange(0 ,H ,downsample), indexing='xy')

    test_images = img[j, i]
    test_rays = get_rays(c2w, kinv, i, j)
    return test_images, test_rays, bds


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
