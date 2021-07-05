import os
import pickle
from tqdm import tqdm
import imageio
import jax
from jax import random
import jax.numpy as np
from livelossplot import PlotLosses
import matplotlib.pyplot as plt

from src.step_utils import (update_network_weights, inner_update_steps,
                        update_model_single, update_model, N_samples,
                        test_inner_steps, render_fn, psnr_fn,
                        batch_size, model, params, opt_state, inner_step_size, lr)

from src.data_utils import DATASET, get_example, imgfiles, posedata, poses_avg, render_path_spiral, get_rays


def train(
    max_iters: int = 150000, 
    ):

    exp_name = f'{DATASET}_ius_{inner_update_steps}_ilr_{inner_step_size}_olr_{lr}_bs_{batch_size}'
    exp_dir = f'checkpoint/phototourism_checkpoints/{exp_name}/'

    temp_eval_result_dir = f'temp/temp_eval_result_dir/{exp_name}/'

    plt_groups = {'Train PSNR': [], 'Test PSNR': []}
    plotlosses_model = PlotLosses(groups=plt_groups)
    plt_groups['Train PSNR'].append(exp_name + f'_train')
    plt_groups['Test PSNR'].append(exp_name + f'_test')
    step = 0

    train_psnrs = []
    rng = jax.random.PRNGKey(0)

    train_steps = []
    train_psnrs_all = []
    test_steps = []
    test_psnrs_all = []

    for step in tqdm(range(max_iters)):
        try:
            rng, rng_input = random.split(rng)
            img_idx = random.randint(rng_input, shape=(), minval=0, maxval=len(imgfiles) - 25)
            images, rays, bds = get_example(img_idx, downsample=1)
        except:
            print('data loading error')
            raise

        images = np.reshape(images, (-1, 3))
        rays = np.reshape(rays, (2, -1, 3))

        if inner_update_steps == 1:
            rng, rng_input = random.split(rng)
            idx = random.randint(rng_input, shape=(batch_size,), minval=0, maxval=images.shape[0])
            rng, params, opt_state, loss = update_model_single(step, rng, params, opt_state,
                                                            images[idx, :], rays[:, idx, :], bds)
        else:
            rng, params, opt_state, loss = update_model(step, rng, params, opt_state,
                                                        images, rays, bds)

        train_psnrs.append(-10 * np.log10(loss))

        if step % 250 == 0:
            plotlosses_model.update({exp_name + '_train': np.mean(np.array(train_psnrs))}, current_step=step)
            train_steps.append(step)
            train_psnrs_all.append(np.mean(np.array(train_psnrs)))
            train_psnrs = []

        if step % 500 == 0 and step != 0:
            test_psnr = []
            for ti in range(5):
                test_images, test_rays, bds = get_example(ti, split='val', downsample=2)

                test_images, test_holdout_images = np.split(test_images, [test_images.shape[1] // 2], axis=1)
                test_rays, test_holdout_rays = np.split(test_rays, [test_rays.shape[2] // 2], axis=2)

                test_images_flat = np.reshape(test_images, (-1, 3))
                test_rays = np.reshape(test_rays, (2, -1, 3))

                rng, test_params, test_inner_loss = update_network_weights(rng, test_images_flat, test_rays, params,
                                                                        test_inner_steps, bds)

                test_result = np.clip(
                    render_fn(rng, model, test_params, None, test_holdout_rays, bds[0], bds[1], N_samples, rand=False)[0],
                    0, 1)
                test_psnr.append(psnr_fn(test_holdout_images, test_result))
            test_psnr = np.mean(np.array(test_psnr))

            test_steps.append(step)
            test_psnrs_all.append(test_psnr)

            plotlosses_model.update({exp_name + '_test': test_psnr}, current_step=step)
            plotlosses_model.send()

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(test_images)
            plt.subplot(1, 3, 2)
            plt.imshow(test_holdout_images)
            plt.subplot(1, 3, 3)
            plt.imshow(test_result)
            plt.savefig(os.path.join(temp_eval_result_dir, "{:06d}.png".format(step)))

        if step % 10000 == 0 and step != 0:
            test_images, test_rays, bds = get_example(0, split='test')
            test_images_flat = np.reshape(test_images, (-1, 3))
            test_rays = np.reshape(test_rays, (2, -1, 3))
            rng, test_params_1, test_inner_loss = update_network_weights(rng, test_images_flat, test_rays, params,
                                                                        test_inner_steps, bds)

            test_images, test_rays, bds = get_example(1, split='test')
            test_images_flat = np.reshape(test_images, (-1, 3))
            test_rays = np.reshape(test_rays, (2, -1, 3))
            rng, test_params_2, test_inner_loss = update_network_weights(rng, test_images_flat, test_rays, params,
                                                                        test_inner_steps, bds)

            poses = posedata['c2w_mats']
            c2w = poses_avg(poses)
            focal = .8
            render_poses = render_path_spiral(c2w, c2w[:3, 1], [.1, .1, .05], focal, zrate=.5, rots=2, N=120)

            bds = np.array([5., 25.]) * .05
            H = 128
            W = H * 3 // 2
            f = H * 1.
            kinv = np.array([
                1. / f, 0, -W * .5 / f,
                0, -1. / f, H * .5 / f,
                0, 0, -1.
            ]).reshape([3, 3])
            i, j = np.meshgrid(np.arange(0, W), np.arange(0, H), indexing='xy')
            renders = []
            for p, c2w in enumerate(tqdm(render_poses)):
                rays = get_rays(c2w, kinv, i, j)
                interp = p / len(render_poses)
                interp_params = jax.tree_multimap(lambda x, y: y * p / len(render_poses) + x * (1 - p / len(render_poses)),
                                                test_params_1, test_params_2)
                result = render_fn(rng, model, interp_params, None, rays, bds[0], bds[1], N_samples, rand=False)[0]
                renders.append(result)

            renders = (np.clip(np.array(renders), 0, 1) * 255).astype(np.uint8)
            imageio.mimwrite(f'{exp_dir}render_sprial_{step}.mp4', renders, fps=30, quality=8)

            plt.plot(train_steps, train_psnrs_all)
            plt.savefig(f'{exp_dir}train_curve_{step}.png')

            plt.plot(test_steps, test_psnrs_all)
            plt.savefig(f'{exp_dir}test_curve_{step}.png')

            with open(f'{exp_dir}checkpount_{step}.pkl', 'wb') as file:
                pickle.dump(params, file)