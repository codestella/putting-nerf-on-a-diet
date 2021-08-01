import os
from functools import partial
import jax
from jax import random
import jax.numpy as jnp
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
    pred_color, pred_disp, _ = render_image_for_inference(partial_render_fn, rays,
                                                          rng, False, chunk=NerfConfig.CHUNK)
    return pred_color, pred_disp


def render_image_for_inference(render_fn, rays, rng, normalize_disp, chunk=8192):
    """Render all the pixels of an image (in test mode).

    Args:
        render_fn: function, jit-ed render function.
        rays: a `Rays` namedtuple, the rays to be rendered.
        rng: jnp.ndarray, random number generator (used in training mode only).
        normalize_disp: bool, if true then normalize `disp` to [0, 1].
        chunk: int, the size of chunks to render sequentially.

    Returns:
        rgb: jnp.ndarray, rendered color image.
        disp: jnp.ndarray, rendered disparity image.
        acc: jnp.ndarray, rendered accumulated weights per pixel.
    """
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)
    unused_rng, key_0, key_1 = jax.random.split(rng, 3)
    host_id = jax.host_id()
    results = []
    for i in range(0, num_rays, chunk):
        # pylint: disable=cell-var-from-loop
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
        chunk_size = chunk_rays[0].shape[0]
        rays_remaining = chunk_size % jax.device_count()
        if rays_remaining != 0:
            padding = jax.device_count() - rays_remaining
            chunk_rays = utils.namedtuple_map(
                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        rays_per_host = chunk_rays[0].shape[0] // jax.process_count()
        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
        chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]), chunk_rays)
        chunk_results = render_fn(key_0, key_1, chunk_rays)[-1]
        results.append([utils.unshard(x[0], padding) for x in chunk_results])
        # pylint: enable=cell-var-from-loop
    rgb, disp, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
    # Normalize disp for visualization for ndc_rays in llff front-facing scenes.
    if normalize_disp:
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    return (rgb.reshape((height, width, -1)), disp.reshape(
        (height, width, -1)), acc.reshape((height, width, -1)))


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
