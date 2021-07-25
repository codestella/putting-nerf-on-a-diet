# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for Nerf."""
import math
import glob
import os
from os import path
import functools

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import tensorflow as tf

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

from nerf import datasets
from nerf import models
from nerf import utils
from nerf import clip_utils

FLAGS = flags.FLAGS
utils.define_flags()



def compute_lpips(image1, image2, model):
    """Compute the LPIPS metric."""
    # The LPIPS model expects a batch dimension.
    return model(
        tf.convert_to_tensor(image1[None, Ellipsis]),
        tf.convert_to_tensor(image2[None, Ellipsis]))[0]


def predict_to_image(pred_out):
    image_arr = np.array(np.clip(pred_out, 0., 1.) * 255.).astype(np.uint8)
    return Image.fromarray(image_arr)


def main(unused_argv):
    # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
    # LPIPS computation or dataset loading.
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.config.experimental.set_visible_devices([], "TPU")

    #wandb.init(project="hf-flax-clip-nerf", entity="wandb", sync_tensorboard=True)

    rng = random.PRNGKey(20200823)

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)
    if FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    dataset = datasets.get_dataset("test", FLAGS)
    rng, key = random.split(rng)
    model, init_variables = models.get_model(key, dataset.peek(), FLAGS)
    optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates "speckle" artifacts.
    def render_fn(variables, key_0, key_1, rays):
        return model.apply(variables, key_0, key_1, rays, False)

    # pmap over only the data input.
    render_pfn = jax.pmap(
        render_fn,
        in_axes=(None, None, None, 0),
        donate_argnums=3,
        axis_name="batch",
    )

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = jax.jit(
        functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

    last_step = 0
    out_dir = path.join(FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds")
    os.makedirs(out_dir, exist_ok=True)
    if FLAGS.save_output:
        print(f'eval output will be saved: {out_dir}')
    else:
        print(f'eval output will not be saved')

    if not FLAGS.eval_once:
        summary_writer = tensorboard.SummaryWriter(
            path.join(FLAGS.train_dir, "eval"))

    def generate_spinning_gif(radius, phi, output_dir, frame_n):
        _rng = random.PRNGKey(0)
        partial_render_fn = functools.partial(render_pfn, state.optimizer.target)
        gif_images = []
        gif_images2 = []
        for theta in tqdm(np.linspace(-math.pi, math.pi, frame_n)):
            camtoworld = np.array(clip_utils.pose_spherical(radius, theta, phi))
            rays = dataset.camtoworld_matrix_to_rays(camtoworld, downsample=4)
            _rng, key0, key1 = random.split(_rng, 3)
            color, disp, _ = utils.render_image(partial_render_fn, rays,
                                             _rng, False, chunk=4096)
            image = predict_to_image(color)
            image2 = predict_to_image(disp[Ellipsis, 0])
            gif_images.append(image)
            gif_images2.append(image2)

        gif_fn = os.path.join(output_dir, 'rgb_spinning.gif')
        gif_fn2 = os.path.join(output_dir, 'disp_spinning.gif')
        gif_images[0].save(gif_fn, save_all=True,
                           append_images=gif_images,
                           duration=100, loop=0)
        gif_images2[0].save(gif_fn2, save_all=True,
                           append_images=gif_images2,
                           duration=100, loop=0)

        #return gif_images, gif_images2

    if FLAGS.generate_gif_only:
        print('generate GIF file only')
        _radius = 4.
        _phi = (30 * math.pi) / 180
        generate_spinning_gif(_radius, _phi, out_dir, frame_n=30)
        print('GIF file for spinning views written)')
        return
    else:
        print('generate GIF file AND evaluate model performance')

    is_gif_written = False
    while True:
        step = int(state.optimizer.state.step)
        if step <= last_step:
            continue
        if FLAGS.save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)
        psnr_values = []
        ssim_values = []

        #lpips_values = []
        if not FLAGS.eval_once:
            showcase_index = np.random.randint(0, dataset.size)
        for idx in range(dataset.size):
            print(f"Evaluating {idx + 1}/{dataset.size}")
            batch = next(dataset)
            pred_color, pred_disp, pred_acc = utils.render_image(
                functools.partial(render_pfn, state.optimizer.target),
                batch["rays"],
                rng,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk)
            if jax.host_id() != 0:  # Only record via host 0.
                continue
            if not FLAGS.eval_once and idx == showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                if not FLAGS.render_path:
                    showcase_gt = batch["pixels"]
            if not FLAGS.render_path:
                psnr = utils.compute_psnr(((pred_color - batch["pixels"]) ** 2).mean())
                ssim = ssim_fn(pred_color, batch["pixels"])
                #lpips = compute_lpips(pred_color, batch["pixels"], lpips_model)
                print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
                psnr_values.append(float(psnr))
                ssim_values.append(float(ssim))
                #lpips_values.append(float(lpips))
            if FLAGS.save_output:
                utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
                utils.save_img(pred_disp[Ellipsis, 0],
                               path.join(out_dir, "disp_{:03d}.png".format(idx)))
        if (not FLAGS.eval_once) and (jax.host_id() == 0):
            summary_writer.image("pred_color", showcase_color, step)
            summary_writer.image("pred_disp", showcase_disp, step)
            summary_writer.image("pred_acc", showcase_acc, step)
            if not FLAGS.render_path:
                summary_writer.scalar("psnr", np.mean(np.array(psnr_values)), step)
                summary_writer.scalar("ssim", np.mean(np.array(ssim_values)), step)
                #summary_writer.scalar("lpips", np.mean(np.array(lpips_values)), step)
                summary_writer.image("target", showcase_gt, step)

        if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
            with utils.open_file(path.join(out_dir, f"psnrs_{step}.txt"), "w") as f:
                f.write(" ".join([str(v) for v in psnr_values]))
            with utils.open_file(path.join(out_dir, f"ssims_{step}.txt"), "w") as f:
                f.write(" ".join([str(v) for v in ssim_values]))
            #with utils.open_file(path.join(out_dir, f"lpips_{step}.txt"), "w") as f:
                #f.write(" ".join([str(v) for v in lpips_values]))
            with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
                f.write("{}".format(np.mean(np.array(psnr_values))))
            with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
                f.write("{}".format(np.mean(np.array(ssim_values))))
            #with utils.open_file(path.join(out_dir, "lpips.txt"), "w") as f:
                #f.write("{}".format(np.mean(np.array(lpips_values))))
            print(f'performance metrics written as txt files: {out_dir}')

            imglist = glob.glob(os.path.join(out_dir, "[0-9][0-9][0-9].png"))
            sorted_files = sorted(imglist, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = 10.0
            img = cv2.imread(sorted_files[0], cv2.IMREAD_COLOR)
            video_fn = os.path.join(out_dir, "rendering_video.mp4")
            out = cv2.VideoWriter(video_fn, fourcc, fps,
                                  (img.shape[1], img.shape[0]))

            for i in range(len(sorted_files)):
                img = cv2.imread(sorted_files[i], cv2.IMREAD_COLOR)
                out.write(img)
            out.release()
            print(f'video file written: {video_fn}')

            # write gif file for spinning views of a scene
            if not is_gif_written:
                _radius = 4.
                _phi = (30 * math.pi) / 180
                generate_spinning_gif(_radius, _phi, out_dir, frame_n=30)
                print(f'GIF file for spinning views written')
                is_gif_written = True

        if FLAGS.eval_once:
            break
        if int(step) >= FLAGS.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)
