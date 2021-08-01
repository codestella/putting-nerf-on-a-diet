import os
from functools import partial
import jax
from jax import random
import numpy as np
from PIL import Image

from nerf import clip_utils
from nerf import utils
from demo.src.config import NerfConfig
from demo.src.models import init_model

model, _ = init_model()


def render_predict_from_pose(state, theta, phi, radius):
    rng = random.PRNGKey(0)
    partial_render_fn = partial(render_pfn, state.optimizer.target)
    rays = _render_rays_from_pose(theta, phi, radius)
    pred_color, pred_disp, _ = utils.render_image(
        partial_render_fn, rays,
        rng, False, chunk=NerfConfig.CHUNK)
    return pred_color, pred_disp


def predict_to_image(pred_out) -> Image:
    image_arr = np.array(np.clip(pred_out, 0., 1.) * 255.).astype(np.uint8)
    return Image.fromarray(image_arr)


def _render_rays_from_pose(theta, phi, radius):
    camtoworld = np.array(clip_utils.pose_spherical(radius, theta, phi))
    rays = _camtoworld_matrix_to_rays(camtoworld)
    return rays


def _camtoworld_matrix_to_rays(camtoworld):
    """ render one instance of rays given a camera to world matrix (4, 4) """
    pixel_center = 0.
    w, h = NerfConfig.W, NerfConfig.H
    focal, downsample = NerfConfig.FOCAL, NerfConfig.DOWNSAMPLE
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(0, w, downsample, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(0, h, downsample, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")
    camera_dirs = np.stack([(x - w * 0.5) / focal,
                            -(y - h * 0.5) / focal,
                            -np.ones_like(x)],
                           axis=-1)
    directions = (camera_dirs[..., None, :] * camtoworld[None, None, :3, :3]).sum(axis=-1)
    origins = np.broadcast_to(camtoworld[None, None, :3, -1], directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return utils.Rays(origins=origins, directions=directions, viewdirs=viewdirs)


def _render_fn(variables, key_0, key_1, rays):
    return jax.lax.all_gather(model.apply(
        variables, key_0, key_1, rays, False),
        axis_name="batch")


render_pfn = jax.pmap(_render_fn, in_axes=(None, None, None, 0),
                      donate_argnums=3, axis_name="batch")
