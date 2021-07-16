"""
command line example:
$ python -i -m jaxnerf.nerf.precompute --data_dir {path-to-data-dir} --split train \
                                       --dataset blender --factor 4 --dtype float16
"""
import os
import argparse
from typing import Optional

import jax.numpy as np

from jaxnerf.nerf import utils
from jaxnerf.nerf import clip_utils
from jaxnerf.nerf import datasets


def precompute_image_features(data_dir: str, split: str, dataset: str, factor: int, dtype: str,
                              model_name: Optional[str], render_path: Optional[str]):
    if dataset == "blender":
        if render_path:
            raise ValueError("render_path cannot be used for the blender dataset.")

        # image in numpy.ndarray
        _, images, _ = datasets.Blender.load_files(data_dir, split, factor)
        clip_model = clip_utils.init_CLIP(dtype, model_name)

        # CLIP output in jax.numpy.ndarray
        images = np.stack(images).transpose(0, 3, 1, 2)
        images = images[:, :3, :, :]
        images = clip_utils.preprocess_for_CLIP(images)
        embeddings = clip_model.get_image_features(pixel_values=images)
        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        print(f'completed precomputing CLIP embeddings: ({embeddings.shape[0]} images)')

        # write as pickle
        write_path = os.path.join(data_dir, f'clip_cache_{split}_factor{factor}_{dtype}.pkl')
        utils.write_pickle(embeddings, write_path)
        print(f'precompute written as pickle: {write_path}')

    elif dataset == "llff":
        raise NotImplementedError
    else:
        raise ValueError(f"invalid dataset: {dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, help="train/val/test")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--factor", type=int, required=True,
                        help="downsampling factor: 0/2/4")
    parser.add_argument("--dtype", type=str, required=True,
                        help="float32/float16 (float16 is used to save memory)")
    parser.add_argument("--model_name", type=str, required=False, default=None)
    parser.add_argument("--render_path", type=str, required=False, default=None)
    args = parser.parse_args()
    precompute_image_features(args.data_dir, args.split, args.dataset, args.factor,
                              args.dtype, args.model_name, args.render_path)
